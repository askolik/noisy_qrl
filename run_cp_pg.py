import tensorflow as tf
from src.utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

import random
import time
from Stefano_files.scripts.policy_gradients_cartpole import main

if __name__ == '__main__':
    noise_p = 0.0
    noisy_type = 'depolarize'
    n_qubits = 4
    n_layers = 5
    beta = 1.
    param_perturbation = 0.1

    # Discount factor in returns
    gamma = 0.99

    # Training hyperparams
    BATCH_SIZE = 10
    MAX_EPISODES = 5000

    # Sanity function to check that noise behaves correctly (works for depolarizing noise)
    # noise_check()

    path = '/data/skolika/quantum-actor-critics/data/cpv0_pp_pg/'
    # path = '/home/andrea/BAK/quantum-actor-critics/data/cpv0_pp_pg/'
    # path = '/Users/stefanomangini/Desktop/PhD/Research/Quantum-Actor-Critic/quantum-actor-critics/data/param_perturbation_pg/local/'

    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    for i in range(10):
        save_as_i = save_as + str(i)
        main(noise_p=noise_p, noise_type=noisy_type, n_qubits=n_qubits, n_layers=n_layers,
             beta=beta, gamma=gamma, batch_size=BATCH_SIZE, max_episodes=MAX_EPISODES,
             lr_in=0.1, lr_var=0.01, lr_out=0.01, param_perturbation=param_perturbation, add_baseline=False,
             checkpoint=True, checkpoint_path=path, save_as=save_as_i)
