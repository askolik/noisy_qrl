import tensorflow as tf

from config import BASE_PATH
from utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from policy_gradients.tsp_pg import TrainTspEnv

import random
import time


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
    'save': False,
    'test': True
}


if __name__ == '__main__':
    save_path = BASE_PATH
    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    for i in range(hyperparams.get('repetitions', 1)):
        save_as_i = save_as + str(i)
        tsp = TrainTspEnv(
            hyperparams, save=hyperparams.get('save'),
            save_as=save_as_i, test=hyperparams.get('test'), path=save_path)
        tsp.perform_episodes(num_instances=hyperparams.get('num_instances'))
