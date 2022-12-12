import tensorflow as tf
from utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from run import run_tsp_pg
from config import BASE_PATH


hyperparams = {
    'n_vars': 10,
    'episodes': 1000,
    'gamma': 0.9,
    'beta_init': 1,
    'learning_rate': 0.01,
    'learning_rate_beta': 0.1,
    'n_layers': 1,
    'num_instances': 100,
    'n_shots': 0,
    'n_trajectories': 0,
    'single_qb_depol_error': 0,
    'two_qb_depol_error': 0,
    'bitflip_error': 0,
    'amplitude_damp_error': 0,
    'param_perturbation': 0,
    'data_path': BASE_PATH + 'tsp/tsp_10_train/tsp_10_reduced_train.pickle',
    'repetitions': 1,
    'save': True,
    'test': True
}


if __name__ == '__main__':
    run_tsp_pg(hyperparams, BASE_PATH)
