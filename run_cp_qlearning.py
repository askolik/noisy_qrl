import tensorflow as tf
from utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from config import BASE_PATH
from run import run_cp_qlearning


hyperparams = {
    'episodes': 1000,
    'batch_size': 16,
    'epsilon': 1,
    'epsilon_decay': 0.99,
    'epsilon_min': 0.01,
    'gamma': 0.99,
    'update_after': 1,
    'update_target_after': 1,
    'learning_rate': 0.0001,
    'learning_rate_in': 0.0001,
    'learning_rate_out': 0.01,
    'n_layers': 5,
    'epsilon_schedule': 'fast',
    'use_reuploading': True,
    'trainable_scaling': True,
    'trainable_obs_weight': True,
    'output_factor': 1,
    'n_shots': 0,
    'n_trajectories': 0,
    'ucb_alg': True,
    'ucb_alg_init_shots': 100,
    'ucb_alg_shot_increment': 100,
    'ucb_alg_max_shots': 1000,
    'param_perturbation': 0.0,
    'noise_p': 0,
    'reps': 1,
    'save': False,
    'test': True
}


if __name__ == '__main__':
    save_path = BASE_PATH
    run_cp_qlearning(hyperparams, save_path)
