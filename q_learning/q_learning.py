import os
from pathlib import Path

import tensorflow as tf
from collections import deque
from config import BASE_PATH


class QLearning:
    def __init__(self, hyperparams, save=True, save_as=None, test=False, path=BASE_PATH):
        self.save = save
        self.save_as = save_as
        self.path = path
        self.test = test

        self.n_vars = hyperparams.get('n_vars')
        self.episodes = hyperparams.get('episodes')
        self.batch_size = hyperparams.get('batch_size')
        self.gamma = hyperparams.get('gamma')
        self.n_layers = hyperparams.get('n_layers')
        self.update_after = hyperparams.get('update_after')
        self.update_target_after = hyperparams.get('update_target_after')
        self.memory_length = hyperparams.get('memory_length')
        self.use_reuploading = hyperparams.get('use_reuploading', False)
        self.trainable_scaling = hyperparams.get('trainable_scaling', False)
        self.trainable_obs_weight = hyperparams.get('trainable_obs_weight', False)
        self.n_shots = hyperparams.get('n_shots', None)
        self.n_trajectories = hyperparams.get('n_trajectories', None)
        self.param_perturbation = hyperparams.get('param_perturbation', 0)
        self.single_qb_depol_error = hyperparams.get('single_qb_depol_error', 0)
        self.two_qb_depol_error = hyperparams.get('two_qb_depol_error', 0)
        self.amplitude_damp_error = hyperparams.get('amplitude_damp_error', 0)
        self.bitflip_error = hyperparams.get('bitflip_error', 0)
        self.ucb_alg = hyperparams.get('ucb_alg', False)
        self.ucb_alg_init_shots = hyperparams.get('ucb_alg_init_shots', 100)
        self.ucb_alg_shot_increment = hyperparams.get('ucb_alg_shot_increment', 100)
        self.ucb_alg_max_shots = hyperparams.get('ucb_alg_max_shots', 10000)

        self.epsilon = hyperparams.get('epsilon')
        self.epsilon_schedule = hyperparams.get('epsilon_schedule')
        self.epsilon_min = hyperparams.get('epsilon_min')
        self.epsilon_decay = hyperparams.get('epsilon_decay')

        self.learning_rate = hyperparams.get('learning_rate', 0.01)
        self.learning_rate_out = hyperparams.get('learning_rate_out', 0.01)
        self.learning_rate_in = hyperparams.get('learning_rate_in', 0.01)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer_output = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_out)
        self.optimizer_input = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_in)

        self.optimizers = []
        self.w_idx = []

        if self.trainable_scaling:
            self.optimizers.append(self.optimizer_input)
            self.w_idx.append(self.optimizers.index(self.optimizer_input))

        if self.trainable_obs_weight:
            self.optimizers.append(self.optimizer_output)
            self.w_idx.append(self.optimizers.index(self.optimizer_output))

        self.loss_fun = tf.keras.losses.mse

        self.memory = self.initialize_memory()
        self.initialize_save_dir()

        self.meta = self.generate_meta_data_dict()

    def generate_meta_data_dict(self):
        meta = {key: str(value) for key, value in self.__dict__.items() if
                not key.startswith('__') and not callable(key)}

        del meta['optimizer']
        del meta['optimizer_output']
        del meta['loss_fun']
        del meta['memory']

        return meta

    def initialize_memory(self):
        memory = deque(maxlen=self.memory_length)
        return memory

    def add_to_memory(self, state, action, reward, next_state, done):
        transition = self.interaction(
            state, action, reward, next_state, float(done))
        self.memory.append(transition)

    def initialize_save_dir(self):
        if self.save:
            check_path = Path(self.path)
            if not check_path.exists():
                os.makedirs(self.path)
