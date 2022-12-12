import tensorflow as tf
from utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from config import BASE_PATH
from run import run_tsp


hyperparams = {
    'n_vars': 10,
    'episodes': 1000,
    'batch_size': 10,
    'epsilon': 1,
    'epsilon_decay': 0.99,
    'epsilon_min': 0.01,
    'gamma': 0.9,
    'update_after': 10,
    'update_target_after': 30,
    'learning_rate': 0.0001,
    'learning_rate_out': 0.001,
    'learning_rate_in': 0.001,
    'n_layers': 1,
    'epsilon_schedule': 'fast',
    'memory_length': 10000,
    'num_instances': 100,
    'use_reuploading': True,
    'trainable_scaling': True,
    'n_shots': 0,
    'n_trajectories': 0,
    'param_perturbation': 0,
    'single_qb_depol_error': 0,
    'two_qb_depol_error': 0,
    'bitflip_error': 0,
    'amplitude_damp_error': 0,
    'data_path': BASE_PATH + 'tsp/tsp_10_train/tsp_10_reduced_train.pickle',
    'repetitions': 1,
    'save': False,
    'test': True
}


if __name__ == '__main__':
    path = BASE_PATH
    run_tsp(hyperparams, path)
