import os
import pickle
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from itertools import combinations

from utils.circuits import graph_encoding_circuit
from utils.layers import EquivariantLayer, SoftmaxInverseTemp
from utils.noise_models import with_decoherence_and_readout_noise
from utils.storage import ScoresCheckpoint, MetaCallback
from utils.tsp_env import TspEnv


class TrainTspEnv:
    def __init__(
            self,
            hyperparams,
            save=True,
            save_as=None,
            test=False,
            path='data/'):

        self.save = save
        self.save_as = save_as
        self.path = path
        self.test = test

        self.n_vars = hyperparams.get('n_vars')
        self.episodes = hyperparams.get('episodes', 5000)
        self.batch_size = 1  # @TODO: higher batch sizes not implemented
        self.gamma = hyperparams.get('gamma', 0.99)
        self.beta_init = hyperparams.get('beta_init', 1)
        self.n_layers = hyperparams.get('n_layers')
        self.n_shots = hyperparams.get('n_shots', None)
        self.n_trajectories = hyperparams.get('n_trajectories', 0)
        self.single_qb_depol_error = hyperparams.get('single_qb_depol_error', 0)
        self.two_qb_depol_error = hyperparams.get('two_qb_depol_error', 0)
        self.amplitude_damp_error = hyperparams.get('amplitude_damp_error', 0)
        self.bitflip_error = hyperparams.get('bitflip_error', 0)
        self.num_instances = hyperparams.get('num_instances', 100)
        self.param_perturbation = hyperparams.get('param_perturbation', 0)

        self.learning_rate = hyperparams.get('learning_rate', 0.01)
        self.learning_rate_beta = hyperparams.get('learning_rate_beta', 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer_beta = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_beta)

        self.optimizers = [self.optimizer, self.optimizer_beta]
        self.w_idx = [0, 1]

        self.loss_fun = tf.keras.losses.mse

        self.initialize_save_dir()

        self.meta = self.generate_meta_data_dict()

        self.fully_connected_qubits = list(combinations(list(range(self.n_vars)), 2))
        self.qubits = cirq.GridQubit.rect(1, self.n_vars)
        self.readout_op = self.initialize_readout()
        self.model = self.initialize_model()
        self.data_path = hyperparams.get('data_path')

    def generate_meta_data_dict(self):
        meta = {key: str(value) for key, value in self.__dict__.items() if
                not key.startswith('__') and not callable(key)}

        del meta['optimizer']
        del meta['loss_fun']

        return meta

    def initialize_save_dir(self):
        if self.save:
            check_path = Path(self.path)
            if not check_path.exists():
                os.makedirs(self.path)

    def save_data(self, meta, tour_lengths, optimal_tour_lengths):
        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_tour_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(tour_lengths, file)

        with open(self.path + '{}_optimal_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(optimal_tour_lengths, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))

    def initialize_readout(self):
        readout_ops = []
        for edge in self.fully_connected_qubits:
            readout_ops.append(cirq.Z(self.qubits[edge[0]]) * cirq.Z(self.qubits[edge[1]]))
        return readout_ops

    def generate_pg_model(self):
        num_edges_in_graph = len(self.fully_connected_qubits)
        n_input_params = self.n_vars + num_edges_in_graph

        # symbols to encode nodes and edges
        data_symbols = [
            [
                sympy.Symbol(f'd_{layer}_{qubit}')
                for qubit in range(len(self.qubits))] +
            [[sympy.Symbol(f'd_{layer}_e_{ew}') for ew in range(num_edges_in_graph)]]
            for layer in range(self.n_layers)]

        circuit = graph_encoding_circuit(
            self.fully_connected_qubits, self.qubits, self.n_layers, data_symbols)

        input_data = tf.keras.Input(shape=n_input_params, dtype=tf.dtypes.float32, name='input')
        input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

        flattened_data_symbols = []
        for layer in data_symbols:
            for item in layer:
                if type(item) == list:
                    for symbol in item:
                        flattened_data_symbols.append(str(symbol))
                else:
                    flattened_data_symbols.append(str(item))

        encoding_layer = EquivariantLayer(
            num_input_params=n_input_params, n_vars=self.n_vars,
            n_edges=num_edges_in_graph, circuit_depth=self.n_layers,
            params=flattened_data_symbols, param_perturbation=self.param_perturbation)

        if self.single_qb_depol_error > 0 \
                or self.two_qb_depol_error > 0 \
                or self.amplitude_damp_error > 0 \
                or self.bitflip_error > 0:

            assert self.n_trajectories > 0, "Number of trajectories for noisy simulation must be > 0"

            circuit = with_decoherence_and_readout_noise(
                circuit,
                self.qubits,
                one_qubit_depol_p=self.single_qb_depol_error,
                two_qubit_depol_p=self.two_qb_depol_error,
                damping_gamma=self.amplitude_damp_error,
                bitflip_readout=self.bitflip_error)

            expectation_layer = tfq.layers.NoisyControlledPQC(
                circuit, self.readout_op, sample_based=False, repetitions=self.n_trajectories)
        else:
            expectation_layer = tfq.layers.ControlledPQC(
                circuit, differentiator=tfq.differentiators.Adjoint(),
                operators=self.readout_op, name="PQC")

        expectation_values = expectation_layer(
            [input_q_state, encoding_layer(input_data)])

        softmax_layer = SoftmaxInverseTemp(
            trainable_beta=True, beta_init=self.beta_init, n_shots=self.n_shots)
        output = softmax_layer(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_data],
            outputs=output, name='pg_model')

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.mse)

        return model

    def initialize_model(self):
        model = self.generate_pg_model()

        if self.test:
            model.summary()

        return model

    @staticmethod
    def compute_tour_length_from_edge_weights(tour, edge_weights):
        tour_len = 0
        for i in range(len(tour)-1):
            try:
                tour_len += edge_weights[(tour[i], tour[i+1 % len(tour)])]
            except KeyError:
                tour_len += edge_weights[(tour[i + 1 % len(tour)], tour[i])]

        return tour_len

    @staticmethod
    def graph_to_list(
            nodes, fully_connected_edges, edge_weights, available_nodes):
        vals = []
        for i, node in enumerate(nodes):
            vals.append(int(i in available_nodes) * np.pi)

        for edge in fully_connected_edges:
            vals.append(np.arctan(edge_weights[edge]))

        return vals

    def get_action(self, action_probs, available_nodes, partial_tour):
        action_probs = action_probs.numpy()[0]
        edge_action_probs = []

        for i in range(self.n_vars):
            if i in partial_tour:
                edge_action_probs.append(0)
            else:
                try:
                    ix = self.fully_connected_qubits.index((partial_tour[-1], i))
                except:
                    ix = self.fully_connected_qubits.index((i, partial_tour[-1]))
                edge_action_probs.append(np.clip(action_probs[ix], 10e-40, 1))

        edge_action_probs = np.asarray(edge_action_probs)
        edge_action_probs /= edge_action_probs.sum()  # fixes issue of numpy when probs sum to 0.999999 or the like
        action = np.random.choice(self.n_vars, p=edge_action_probs)

        return action

    def compute_edge_weights(self, instance):
        edge_weights = {}
        for edge in self.fully_connected_qubits:
            edge_distance = np.linalg.norm(np.asarray(instance[edge[0]]) - np.asarray(instance[edge[1]]))
            edge_weights[edge] = edge_distance
        return edge_weights

    def gather_episodes(self, model, n_episodes):
        trajectories = [defaultdict(list) for _ in range(n_episodes)]
        envs = [TspEnv(
            n_cities=self.n_vars,
            num_instances=self.num_instances,
            base_path=self.data_path) for _ in range(n_episodes)]

        done = [False for _ in range(n_episodes)]
        states = [e.reset() for e in envs]
        edge_weights_list = [self.compute_edge_weights(instance) for instance, available_nodes in states]
        optimal_tours = [e.get_current_optimal_tour() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(n_episodes) if not done[i]]

            state_lists = []
            for i, (instance, edge_weights) in enumerate(zip(states, edge_weights_list)):
                if not done[i]:
                    nodes, available_nodes = instance
                    state_list = self.graph_to_list(
                        nodes, self.fully_connected_qubits, edge_weights,
                        available_nodes)
                    state_lists.append(state_list)

            for i, state in zip(unfinished_ids, state_lists):
                trajectories[i]['states'].append(state)

            states = tf.convert_to_tensor(state_lists)
            action_probs = model(
                [tfq.convert_to_tensor([cirq.Circuit() for _ in range(self.batch_size)]), states])

            states = [None for i in range(n_episodes)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = self.get_action(action_probs, envs[i].available_nodes, envs[i].tour)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        ratios = []
        for i, trajectory in enumerate(trajectories):
            prop_tour_len = self.compute_tour_length_from_edge_weights(
                [0] + trajectory['actions'] + [0], edge_weights_list[i])
            optimal_tour_len = self.compute_tour_length_from_edge_weights(
                [x-1 for x in optimal_tours[i]], edge_weights_list[i])
            ratios.append(prop_tour_len/optimal_tour_len)

        return trajectories, ratios

    def compute_returns(self, rewards_history):
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        return returns

    def train_step(self, states, actions, returns, model, batch_size, optimizers):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns, dtype=np.float32)
        returns = tf.expand_dims(returns, 1)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model([tfq.convert_to_tensor([cirq.Circuit() for _ in range(states.shape[0])]), states])

            action_probs = tf.gather_nd(logits, actions)
            action_probs = tf.clip_by_value(action_probs, 10e-20, 1)
            action_log_probs = tf.math.log(action_probs)
            action_log_probs = tf.expand_dims(action_log_probs, 1)
            loss = tf.math.reduce_sum(-action_log_probs * returns) / batch_size

        grads = tape.gradient(loss, model.trainable_variables)

        if len(optimizers) == 1:
            optimizers[0].apply_gradients(zip(grads, model.trainable_variables))
        else:
            for optimizer, w in zip(optimizers, range(len(model.trainable_variables))):
                optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

        return loss

    def perform_episodes(self, num_instances):
        if self.save:
            self.meta['num_instances'] = num_instances
            self.meta['best_tour_length'] = 100000
            self.meta['best_tour'] = []
            self.meta['best_tour_ix'] = 0
            self.meta['env_solved_after'] = 0
            self.meta['env_solved'] = False

            scores_checkpoint = ScoresCheckpoint(self.path, self.save_as)
            meta_callback = MetaCallback(self.meta, self.path, self.save_as)

        tsp_env = TspEnv(
            n_cities=self.n_vars, num_instances=num_instances, base_path=self.data_path)
        tsp_env.reset()

        ratio_history = []
        running_avgs = []
        episodes_reward = []
        episode_history = []
        loss_history = []

        for episode in range(self.episodes):
            episodes, ep_ratios = self.gather_episodes(self.model, self.batch_size)

            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)
            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            step_loss = self.train_step(
                states, id_action_pairs, returns, self.model, self.batch_size, self.optimizers)

            for ep_rwds in rewards:
                tot_rew = np.sum(ep_rwds)
                episode_history.append(tot_rew)
                episodes_reward.append(tot_rew)
            ratio_history.append(statistics.mean(ep_ratios))
            loss_history.append(step_loss)

            if len(ratio_history) >= 100:
                running_avg = np.mean(ratio_history[-100:])
            else:
                running_avg = np.mean(ratio_history)

            running_avgs.append(running_avg)

            if self.save:
                self.model.save_weights(self.path + self.save_as)
                scores_checkpoint.on_epoch_end(episode, {'scores': ratio_history})
                meta_callback.on_epoch_end(episode)

            if self.test:
                print(f"Episode {episode}, running reward: {ratio_history[-1]}, running avg: {running_avg}")

            if len(ratio_history) >= 100 and running_avg <= 1.05:
                print(f"Environment solved in {episode+1} episodes!")
                if self.save:
                    self.meta['env_solved'] = True
                    self.meta['env_solved_after'] = episode+1
                    meta_callback.on_epoch_end(episode, self.meta)
                break

        if self.test:
            import matplotlib.pyplot as plt

            print(self.model.trainable_variables)

            plt.plot(running_avgs)
            plt.ylabel("Approximation ratio")
            plt.xlabel("Episode")
            plt.show()
