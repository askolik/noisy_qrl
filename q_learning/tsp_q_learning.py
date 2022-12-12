import copy
import pickle
import random
from enum import Enum
from itertools import combinations
from random import choice, choices

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from config import BASE_PATH
from collections import namedtuple

from q_learning.q_learning import QLearning
from utils.circuits import graph_encoding_circuit
from utils.layers import EquivariantLayer, TrainableRescaling
from utils.noise_models import with_decoherence_and_readout_noise
from utils.tsp_env import TspEnv


class CircuitType(Enum):
    EQC = 'eqc'


class QLearningTsp(QLearning):
    def __init__(
            self,
            hyperparams,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningTsp, self).__init__(hyperparams, save, save_as, test, path)

        self.circuit_type = hyperparams.get('circuit_type', CircuitType.EQC)
        self.fully_connected_qubits = list(combinations(list(range(self.n_vars)), 2))
        self.qubits = cirq.GridQubit.rect(1, self.n_vars)
        self.is_multi_instance = hyperparams.get('is_multi_instance')
        self.readout_op = self.initialize_readout()
        self.interaction = namedtuple(
            'interaction', ('state', 'action', 'reward', 'next_state', 'done', 'partial_tour', 'edge_weights'))
        self.model, self.target_model = self.initialize_models()
        self.data_path = hyperparams.get('data_path')

    def save_data(self, meta, tour_lengths, optimal_tour_lengths, shot_history=[]):
        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_tour_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(tour_lengths, file)

        with open(self.path + '{}_optimal_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(optimal_tour_lengths, file)

        if shot_history:
            with open(self.path + '{}_shots.pickle'.format(self.save_as), 'wb') as file:
                pickle.dump(shot_history, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))

    def initialize_readout(self):
        readout_ops = []
        for edge in self.fully_connected_qubits:
            readout_ops.append(cirq.Z(self.qubits[edge[0]]) * cirq.Z(self.qubits[edge[1]]))
        return readout_ops

    def generate_eqc_model(self, is_target_model=False):
        name_prefix = ''
        if is_target_model:
            name_prefix = 'target_'

        num_edges_in_graph = len(self.fully_connected_qubits)
        n_input_params = self.n_vars + num_edges_in_graph

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

            assert self.n_trajectories > 0, "Number of trajectories for noise simulation needs to be > 0"

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

        output_extension_layer = tf.keras.Sequential([
            TrainableRescaling(
                len(self.readout_op),
                trainable_obs_weight=self.trainable_obs_weight)
        ])

        output = output_extension_layer(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_data],
            outputs=output, name=f'{name_prefix}q_model')

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.mse)

        if self.test:
            model.summary()

        return model

    def initialize_models(self):
        model = self.generate_eqc_model()
        target_model = self.generate_eqc_model(is_target_model=True)
        target_model.set_weights(model.get_weights())

        return model, target_model

    @staticmethod
    def graph_to_list(
            nodes, fully_connected_edges, edge_weights, available_nodes, node_to_qubit_map):
        vals = []
        for node in nodes:
            vals.append(int(node_to_qubit_map[node] in available_nodes) * np.pi)

        for edge in fully_connected_edges:
            vals.append(np.arctan(edge_weights[edge]))

        return vals

    def q_vals_from_expectations(self, partial_tours, edge_weights, expectations, shot_noise=None):
        expectations = expectations.numpy()
        indexed_expectations = []

        if shot_noise is not None:
            shot_noise = shot_noise.numpy()
            indexed_shot_noise = []

        for i, exps in enumerate(expectations):
            batch_ix_exp = {}
            batch_ix_shots = {}
            for edge, exp_val in zip(self.fully_connected_qubits, exps):
                batch_ix_exp[edge] = exp_val

            indexed_expectations.append(batch_ix_exp)

            if shot_noise is not None:
                for edge, noise in zip(self.fully_connected_qubits, shot_noise[i]):
                    batch_ix_shots[edge] = noise

                indexed_shot_noise.append(batch_ix_shots)

        batch_q_vals = []
        for tour_ix, partial_tour in enumerate(partial_tours):
            q_vals = []
            for i in range(self.n_vars):
                node_in_tour = False
                for edge in partial_tour:
                    if i in edge:
                        node_in_tour = True

                if not node_in_tour:
                    next_edge = None
                    if partial_tour:
                        next_edge = (partial_tour[-1][1], i)
                    else:
                        if i > 0:
                            next_edge = (0, i)

                    if next_edge is not None:
                        try:
                            if shot_noise is None:
                                q_val = edge_weights[tour_ix][next_edge] * indexed_expectations[tour_ix][next_edge]
                            else:
                                q_val = edge_weights[tour_ix][next_edge] * indexed_expectations[tour_ix][next_edge] \
                                        + edge_weights[tour_ix][next_edge] * indexed_shot_noise[tour_ix][next_edge]
                        except KeyError:
                            if shot_noise is None:
                                q_val = edge_weights[tour_ix][
                                    (next_edge[1], next_edge[0])] * indexed_expectations[tour_ix][
                                            (next_edge[1], next_edge[0])]
                            else:
                                q_val = edge_weights[tour_ix][
                                            (next_edge[1], next_edge[0])] * indexed_expectations[tour_ix][
                                            (next_edge[1], next_edge[0])] + edge_weights[tour_ix][
                                            (next_edge[1], next_edge[0])] * indexed_shot_noise[tour_ix][
                                    (next_edge[1], next_edge[0])]
                    else:
                        q_val = -10000
                else:
                    q_val = -10000
                q_vals.append(q_val)

            batch_q_vals.append(q_vals)

        return np.asarray(batch_q_vals)

    def get_action(self, state_tensor, available_nodes, partial_tour, edge_weights):
        n_shots = self.n_shots
        if np.random.uniform() < self.epsilon:
            action = choice(available_nodes)
        else:
            state_tensor = tf.convert_to_tensor(state_tensor)
            state_tensor = tf.expand_dims(state_tensor, 0)
            expectations = self.model([tfq.convert_to_tensor([cirq.Circuit()]), state_tensor])
            q_vals = self.q_vals_from_expectations([partial_tour], [edge_weights], expectations)[0]

            if self.ucb_alg:
                n_shots = self.ucb_alg_init_shots
                sorted_qvals = np.sort(q_vals)[::-1]
                delta_q = abs(sorted_qvals[0] - sorted_qvals[1])
                while delta_q < 2/np.sqrt(n_shots) and n_shots < self.ucb_alg_max_shots:
                    n_shots += self.ucb_alg_shot_increment

            if self.ucb_alg or self.n_shots > 0:
                # Simulate shot noise with Gaussians
                shot_noise = tf.random.normal(expectations.shape) * (1/np.sqrt(n_shots))
                q_vals = self.q_vals_from_expectations([partial_tour], [edge_weights], expectations, shot_noise)[0]

            action = np.argmax(q_vals)

        return action, n_shots

    @staticmethod
    def get_masks_for_actions(edge_weights, partial_tours):
        batch_masks = []
        for tour_ix, partial_tour in enumerate(partial_tours):
            mask = []
            for edge, weight in edge_weights[tour_ix].items():
                if edge in partial_tour or (edge[1], edge[0]) in partial_tour:
                    mask.append(weight)
                else:
                    mask.append(0)

            batch_masks.append(mask)

        return np.asarray(batch_masks)

    @staticmethod
    def cost(nodes, tour):
        return - TspEnv.compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)

    def train_step(self):
        training_batch = choices(self.memory, k=self.batch_size)
        training_batch = self.interaction(*zip(*training_batch))

        states = [x for x in training_batch.state]
        rewards = np.asarray([x for x in training_batch.reward], dtype=np.float32)
        next_states = [x for x in training_batch.next_state]
        done = np.asarray([x for x in training_batch.done])
        partial_tours = [x for x in training_batch.partial_tour]
        edge_weights = [x for x in training_batch.edge_weights]

        states = tf.convert_to_tensor(states)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        exp_values_future = self.target_model([tfq.convert_to_tensor([cirq.Circuit()] * self.batch_size), next_states])

        future_rewards = tf.convert_to_tensor(self.q_vals_from_expectations(
            partial_tours, edge_weights, exp_values_future), dtype=tf.float32)

        n_shots_update = self.n_shots * self.batch_size

        if self.n_shots or self.ucb_alg:
            n_shots_for_estimate = tf.ones(shape=future_rewards.shape[0], dtype=tf.float32) * self.n_shots
            if self.ucb_alg:
                sorted_qvals = tf.sort(future_rewards, axis=-1, direction='DESCENDING')
                q_deltas = tf.math.abs(sorted_qvals[:, 0] - sorted_qvals[:, 1])
                n_shots = tf.clip_by_value(2 / q_deltas ** 2, 0, self.ucb_alg_max_shots)  # m = 1/eps²

                # Compute how many shots would be required when incrementing by self.ucb_alg_shot_increment,
                # basically "rounding up" n_shots to the next increment.
                n_increments = tf.math.ceil((n_shots - self.ucb_alg_init_shots) / self.ucb_alg_shot_increment)
                n_increments = tf.clip_by_value(n_increments, 0, self.ucb_alg_max_shots)
                n_shots_ucb = self.ucb_alg_init_shots + n_increments * self.ucb_alg_shot_increment
                n_shots_update = tf.math.reduce_sum(n_shots_ucb).numpy()

                n_shots_for_estimate = n_shots_ucb

            # Based on n_shots_ucb, apply Gaussian perturbations to the model output to simulate
            # shot noise. Using the fact that N(0, 1) * D + M = N(M, D) for ease of implementation.
            # Being very explicit w/ data types as tf.random.normal is buggy and everything has to be float32.
            n_shots_for_estimate = tf.reshape(n_shots_for_estimate, [-1, 1])
            n_shots_for_estimate = tf.tile(n_shots_for_estimate, [1, exp_values_future.shape[-1]])
            std_term = np.ones(shape=exp_values_future.shape) / tf.sqrt(n_shots_for_estimate)
            std_term = tf.convert_to_tensor(std_term, dtype=tf.float32)
            gaussians = tf.random.normal(exp_values_future.shape, dtype=tf.float32) * std_term

            future_rewards = tf.convert_to_tensor(self.q_vals_from_expectations(
                partial_tours, edge_weights, exp_values_future, gaussians), dtype=tf.float32)

        target_q_values = rewards + (
                self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done))

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            exp_values = self.model([tfq.convert_to_tensor([cirq.Circuit()]*self.batch_size), states])
            exp_val_masks = self.get_masks_for_actions(edge_weights, partial_tours)

            if self.n_shots > 0 or self.ucb_alg:
                q_values_masked = tf.reduce_sum(
                    tf.add(tf.multiply(exp_values, exp_val_masks), exp_val_masks * gaussians ), axis=1)
            else:
                q_values_masked = tf.reduce_sum(tf.multiply(exp_values, exp_val_masks), axis=1)

            loss = self.loss_fun(target_q_values, q_values_masked)

        grads = tape.gradient(loss, self.model.trainable_variables)

        try:
            tf.debugging.check_numerics(grads, message='Gradient contains NaN or Inf')
            self.optimizers[0].apply_gradients(zip(grads, self.model.trainable_weights))
        except Exception as e:
            print(e.message)

        return loss.numpy(), n_shots_update

    def perform_episodes(self, num_instances):
        self.meta['num_instances'] = num_instances
        self.meta['best_tour_length'] = 100000
        self.meta['best_tour'] = []
        self.meta['best_tour_ix'] = 0
        self.meta['env_solved'] = False

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)

        if self.is_multi_instance:
            x_train = data['x_train'][:num_instances]
            y_train = data['y_train'][:num_instances]
        else:
            x_train = data['x_train']
            y_train = data['y_train']

        tour_length_history = []
        optimal_tour_length_history = []
        ratio_history = []
        running_avgs = []
        running_avg = 0
        shot_history = []

        for episode in range(self.episodes):
            n_shots_episode = 0
            instance_number = random.randint(0, num_instances-1)
            tsp_graph_nodes = x_train[instance_number]
            optimal_tour_length = TspEnv.compute_tour_length(
                tsp_graph_nodes, [int(x - 1) for x in y_train[instance_number][:-1]])
            node_to_qubit_map = {}
            for i, node in enumerate(tsp_graph_nodes):
                node_to_qubit_map[node] = i

            fully_connected_edges = []
            edge_weights = {}
            edge_weights_ix = {}
            for edge in self.fully_connected_qubits:
                fully_connected_edges.append((tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]]))
                edge_distance = np.linalg.norm(
                    np.asarray(tsp_graph_nodes[edge[0]]) - np.asarray(tsp_graph_nodes[edge[1]]))
                edge_weights[(tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]])] = edge_distance
                edge_weights_ix[edge] = edge_distance

            tour = [0]  # w.l.o.g. we always start at city 0
            tour_edges = []
            step_rewards = []
            available_nodes = list(range(1, self.n_vars))

            for i in range(self.n_vars):
                prev_tour = copy.deepcopy(tour)
                state_list = self.graph_to_list(
                    tsp_graph_nodes, fully_connected_edges, edge_weights,
                    available_nodes, node_to_qubit_map)

                next_node, ucb_n_shots = self.get_action(state_list, available_nodes, tour_edges, edge_weights_ix)
                n_shots_episode += ucb_n_shots
                tour_edges.append((tour[-1], next_node))
                new_tour_edges = copy.deepcopy(tour_edges)
                tour.append(next_node)

                remove_node_ix = available_nodes.index(next_node)
                del available_nodes[remove_node_ix]

                if len(tour) > 1:
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    done = 0 if len(available_nodes) > 1 else 1
                    transition = (state_list, next_node, reward, self.graph_to_list(
                        tsp_graph_nodes, fully_connected_edges, edge_weights,
                        available_nodes, node_to_qubit_map), done, new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)

                if len(available_nodes) == 1:
                    prev_tour = copy.deepcopy(tour)
                    tour_edges.append((tour[-1], available_nodes[0]))
                    tour_edges.append((available_nodes[0], tour[0]))
                    new_tour_edges = copy.deepcopy(tour_edges)
                    tour.append(available_nodes[0])
                    tour.append(tour[0])
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    transition = (state_list, next_node, reward, self.graph_to_list(
                        tsp_graph_nodes, fully_connected_edges, edge_weights,
                        available_nodes, node_to_qubit_map), 1, new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)
                    break

            tour_length = TspEnv.compute_tour_length(tsp_graph_nodes, tour)
            tour_length_history.append(tour_length)
            optimal_tour_length_history.append(optimal_tour_length)

            if tour_length < self.meta.get('best_tour_length'):
                self.meta['best_tour_length'] = tour_length
                self.meta['best_tour'] = tour
                self.meta['best_tour_ix'] = instance_number

            if len(self.memory) >= self.batch_size:
                if episode % self.update_after == 0:
                    loss, n_shots_update = self.train_step()
                    n_shots_episode += n_shots_update
                    print(
                        f"Episode {episode}, loss {loss}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")
                else:
                    print(
                            f"Episode {episode}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")

                if self.test:
                    print("\tn_shots_episode", n_shots_episode)

                if episode % self.update_target_after == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if self.epsilon_schedule == 'fast':
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            shot_history.append(n_shots_episode)

            if self.save:
                self.save_data(self.meta, tour_length_history, optimal_tour_length_history)

            ratio_history.append(tour_length_history[-1] / optimal_tour_length)

            if len(ratio_history) >= 100:
                running_avg = np.mean(ratio_history[-100:])
            else:
                running_avg = np.mean(ratio_history)

            running_avgs.append(running_avg)

            if len(ratio_history) >= 100 and running_avg <= 1.05:
                print(f"Environment solved in {episode+1} episodes!")
                self.meta['env_solved'] = True
                if self.save:
                    self.save_data(self.meta, tour_length_history, optimal_tour_length_history)
                break

        if self.test:
            import matplotlib.pyplot as plt
            plt.plot(running_avgs)
            plt.ylabel("Ratio to optimal tour length")
            plt.xlabel("Episode")
            plt.show()
