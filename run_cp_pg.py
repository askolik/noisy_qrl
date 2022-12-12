import tensorflow as tf
from utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from config import BASE_PATH
from run import run_cp_pg


hyperparams = {
    'noise_p': 0,
    'noise_type': 'depolarize',
    'n_qubits': 4,
    'n_layers': 5,
    'beta': 1,
    'gamma': 0.99,
    'batch_size': 10,
    'max_episodes': 5000,
    'param_perturbation': 0,
    'save': False
}


if __name__ == '__main__':
    run_cp_pg(hyperparams, BASE_PATH)