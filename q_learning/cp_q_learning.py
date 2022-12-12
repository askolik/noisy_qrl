import gym
import pickle
import random

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np

from config import BASE_PATH
from q_learning.q_learning import QLearning
from collections import namedtuple

from utils.circuits import generate_cp_q_circuit, empty_circuits
from utils.layers import ScalableDataReuploadingController, TrainableRescaling


class QLearningCartpole(QLearning):
    def __init__(
            self,
            hyperparams,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningCartpole, self).__init__(hyperparams, save, save_as, test, path)

        self.env = gym.make('CartPole-v0')
        self.max_steps = 200
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.readout_op = self.initialize_readout()
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.observation_space)]
        self.n_qubits = len(self.qubits)

        self.interaction = namedtuple('interaction', ('state', 'action', 'reward', 'next_state', 'done'))
        self.output_factor = hyperparams.get('output_factor', 1)
        self.noise_p = hyperparams.get('noise_p', 0)

        self.w_input, self.w_var, self.w_output = 1, 0, 2
        self.optimizers = [self.optimizer, self.optimizer_input, self.optimizer_output]

        self.model, self.target_model, self.circuit = self.initialize_models()

        self.meta = self.generate_meta_data_dict()

    def save_data(self, meta, scores, shot_history=None):
        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))

        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_scores.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(scores, file)

        if shot_history:
            with open(self.path + '{}_shots.pickle'.format(self.save_as), 'wb') as file:
                pickle.dump(shot_history, file)

    def initialize_readout(self):
        qubits = [cirq.GridQubit(0, i) for i in range(self.observation_space)]
        return [
            cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
            cirq.Z(qubits[2]) * cirq.Z(qubits[3])]

    def generate_model(self, circuit, num_params, params, inputs, observables, target):
        input_tensor = tf.keras.Input(shape=(len(self.qubits)), dtype=tf.dtypes.float32, name='input')
        input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

        encoding_layer = ScalableDataReuploadingController(
            num_input_params=len(self.qubits), num_params=num_params, circuit_depth=self.n_layers,
            params=[str(param) for param in params] + [str(x) for x in inputs],
            trainable_scaling=self.trainable_scaling, use_reuploading=self.use_reuploading,
            param_perturbation=self.param_perturbation)

        if self.noise_p > 0:
            assert self.n_trajectories > 0, "Number of trajectories for noisy simulation must be > 0"
            expectation_layer = tfq.layers.NoisyControlledPQC(circuit, observables,
                                                              sample_based=False, repetitions=self.n_trajectories)
        else:
            expectation_layer = tfq.layers.ControlledPQC(
                circuit, differentiator=tfq.differentiators.Adjoint(),
                operators=observables, name="PQC")

        prepend = ""
        if target:
            prepend = "Target"

        expectation_values = expectation_layer(
            [input_q_state, encoding_layer(input_tensor)])

        process = tf.keras.Sequential([
            TrainableRescaling(len(observables), self.trainable_obs_weight)
        ], name=prepend + "Q-values")

        q_values = process(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_tensor],
            outputs=q_values,
            name=prepend + "Q-function")

        if not target:
            model.summary()

        return model

    def initialize_models(self):
        circuit, param_dim, param_symbols, input_symbols = self.create_circuit()
        model = self.generate_model(
            circuit, param_dim, param_symbols, input_symbols, self.readout_op, False)
        target_model = self.generate_model(
            circuit, param_dim, param_symbols, input_symbols, self.readout_op, True)
        target_model.set_weights(model.get_weights())

        return model, target_model, circuit

    def create_circuit(self):
        circuit, param_dim, param_symbols, input_symbols = generate_cp_q_circuit(
            self.observation_space, self.n_layers, self.qubits,
            use_reuploading=self.use_reuploading)

        if self.noise_p >= 1e-5:
            circuit = circuit.with_noise(cirq.depolarize(self.noise_p))

        return circuit, param_dim, param_symbols, input_symbols

    def save_env_data(self, state_history):
        with open(self.path + '{}_states.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(state_history, file)

    def add_to_memory(self, state, action, reward, next_state, done):
        transition = self.interaction(
            state, action, reward, next_state, float(done))
        self.memory.append(transition)

    def perform_action(self, state):
        # We count n_shots even when a random action is selected, to get a fair
        # comparison between algorithms that is not influenced by the randomness
        # of the epsilon-greedy policy.
        n_shots = self.n_shots
        action_type = 'random'
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            q_vals = self.model([empty_circuits(1), state])

            if self.ucb_alg:
                n_shots = self.ucb_alg_init_shots
                delta_q = abs(q_vals.numpy()[0, 0] - q_vals.numpy()[0, 1])
                while delta_q < 2/np.sqrt(n_shots) and n_shots < self.ucb_alg_max_shots:
                    n_shots += self.ucb_alg_shot_increment

            if self.ucb_alg or self.n_shots > 0:
                # Simulate shot noise with Gaussians, accounting for the fact that the model output is (<o>+1)/2 * w
                q_vals = q_vals + (
                        tf.random.normal(q_vals.shape) * (1/np.sqrt(n_shots))
                        * tf.reverse(self.model.trainable_variables[self.w_output], axis=[1]) / 2)

            action = int(tf.argmax(q_vals[0]).numpy())
            action_type = 'argmax'

        return action, action_type, n_shots

    def train_step(self):
        training_batch = random.choices(self.memory, k=self.batch_size)
        training_batch = self.interaction(*zip(*training_batch))

        states = np.asarray([x for x in training_batch.state])
        rewards = np.asarray([x for x in training_batch.reward], dtype=np.float32)
        next_states = np.asarray([x for x in training_batch.next_state])
        done = np.asarray([x for x in training_batch.done], dtype=np.float32)

        states = tf.convert_to_tensor(states)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        future_rewards = self.target_model([empty_circuits(self.batch_size), next_states])

        n_shots_update = self.n_shots * self.batch_size

        if self.n_shots or self.ucb_alg:
            n_shots_for_estimate = tf.constant(self.n_shots, dtype=tf.float32)
            if self.ucb_alg:
                q_deltas = tf.math.abs(future_rewards[:, 0] - future_rewards[:, 1])
                n_shots = 2 / q_deltas**2  # m = 1/eps²

                # Compute how many shots would be required when incrementing by self.ucb_alg_shot_increment,
                # basically "rounding up" n_shots to the next increment.
                n_increments = tf.math.ceil((n_shots - self.ucb_alg_init_shots) / self.ucb_alg_shot_increment)
                n_increments = tf.clip_by_value(n_increments, 0, self.ucb_alg_max_shots)
                n_shots_ucb = self.ucb_alg_init_shots + n_increments * self.ucb_alg_shot_increment
                n_shots_update = tf.math.reduce_sum(n_shots_ucb).numpy()

                n_shots_for_estimate = n_shots_ucb

            # Based on n_shots_for_estimate, apply Gaussian perturbations to the model output to simulate
            # shot noise. Using the fact that N(0, 1) * D + M = N(M, D) for ease of implementation.
            std_term = np.ones(shape=future_rewards.shape)
            std_term[:, 0] *= 1 / tf.sqrt(n_shots_for_estimate)
            std_term[:, 1] *= 1 / tf.sqrt(n_shots_for_estimate)
            gaussians = tf.random.normal(future_rewards.shape) * std_term

            # Our model outputs are (<o>+1)/2 * w for some observable o, weight w.
            # Consequently, for shot error eps, ((<o>+eps)*w + w) / 2 = (<o>+1)/2 * w + w*eps/2.
            output_weights = tf.reverse(self.target_model.trainable_variables[self.w_output], axis=[1])

            # future_rewards is a tensor of a batch of Q-values, where we need to multiply
            # the first column with the first noisy output weight, second column with second
            # noisy output weight.
            shot_noise = np.ones(shape=gaussians.shape)
            shot_noise[:, 0] *= output_weights[0, 0] / 2
            shot_noise[:, 1] *= output_weights[0, 1] / 2
            shot_noise = gaussians * shot_noise

            future_rewards = future_rewards + shot_noise

        target_q_values = rewards + (
                self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done))
        masks = tf.one_hot(training_batch.action, self.action_space)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([empty_circuits(self.batch_size), states])

            if self.n_shots > 0 or self.ucb_alg:
                n_shots_for_estimate = tf.constant(self.n_shots, dtype=tf.float32)
                if self.ucb_alg:
                    q_deltas = tf.math.abs(q_values[:, 0] - q_values[:, 1])
                    n_shots = 2 / q_deltas ** 2  # m = 1/eps²

                    # Compute how many shots would be required when incrementing by self.ucb_alg_shot_increment,
                    # basically "rounding up" n_shots to the next increment.
                    n_increments = tf.math.ceil((n_shots - self.ucb_alg_init_shots) / self.ucb_alg_shot_increment)
                    n_increments = tf.clip_by_value(n_increments, 0, self.ucb_alg_max_shots)
                    n_shots_ucb = self.ucb_alg_init_shots + n_increments * self.ucb_alg_shot_increment
                    n_shots_update = n_shots_update + tf.math.reduce_sum(n_shots_ucb).numpy()

                    n_shots_for_estimate = n_shots_ucb
                else:
                    n_shots_update += self.n_shots * self.batch_size

                std_term = np.ones(shape=q_values.shape)
                std_term[:, 0] *= 1 / tf.sqrt(n_shots_for_estimate)
                std_term[:, 1] *= 1 / tf.sqrt(n_shots_for_estimate)
                gaussians = tf.random.normal(q_values.shape) * std_term

                output_weights = tf.reverse(self.model.trainable_variables[self.w_output], axis=[1])

                shot_noise = np.ones(shape=gaussians.shape)
                shot_noise[:, 0] *= output_weights[0, 0] / 2
                shot_noise[:, 1] *= output_weights[0, 1] / 2
                shot_noise = gaussians * shot_noise

                q_values = q_values + shot_noise

            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_fun(target_q_values, q_values_masked)

        optimizers = [self.optimizer]
        weights = [self.w_var]

        if self.trainable_scaling:
            optimizers.append(self.optimizer_input)
            weights.append(self.w_input)

        if self.trainable_obs_weight:
            optimizers.append(self.optimizer_output)
            weights.append(self.w_output if self.trainable_scaling else self.w_input)

        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip(optimizers, weights):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

        return n_shots_update

    def perform_episodes(self):
        scores = []
        recent_scores = []
        shot_history = []

        if self.epsilon_schedule == 'linear':
            eps_values = list(np.linspace(self.epsilon, self.epsilon_min, self.episodes)[::-1])

        solved = False
        for episode in range(self.episodes):
            if solved:
                break

            n_shots_episode = 0
            state = self.env.reset()
            for iteration in range(self.max_steps):
                old_state = state
                action, action_type, ucb_n_shots = self.perform_action(state)
                n_shots_episode += ucb_n_shots

                state, reward, done, _ = self.env.step(action)
                self.add_to_memory(old_state, action, reward, state, done)

                if done:
                    scores.append(iteration + 1)
                    self.meta['last_epsilon'] = self.epsilon

                    if len(scores) > 100:
                        recent_scores = scores[-100:]

                    avg_score = np.mean(recent_scores) if recent_scores else np.mean(scores)

                    if self.test:
                        print(
                            "\rEpisode {:03d} , epsilon={:.4f}, action type={}, score={:03d}, avg score={:.3f}, shots={}".format(
                                episode, self.epsilon, action_type, iteration + 1, avg_score, n_shots_episode))

                    if self.n_shots > 0 or self.ucb_alg:
                        shot_history.append(n_shots_episode)

                    break

                if len(self.memory) >= self.batch_size and iteration % self.update_after == 0:
                    n_shots_update = self.train_step()
                    n_shots_episode += n_shots_update

                if iteration % self.update_target_after == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if np.mean(recent_scores) >= 195:
                print("\nEnvironment solved in {} episodes.".format(episode), end="")
                self.meta['env_solved_at'].append(episode)
                solved = True

            if self.epsilon_schedule == 'fast':
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            elif self.epsilon_schedule == 'linear':
                self.epsilon = eps_values.pop()

            if self.save:
                self.save_data(self.meta, scores, shot_history)