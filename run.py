import random
import time
from copy import copy

from policy_gradients.cp_pg import train_cp_pg
from policy_gradients.tsp_pg import TrainTspEnv
from q_learning.cp_q_learning import QLearningCartpole
from q_learning.tsp_q_learning import QLearningTsp


def run_tsp_qlearning(hyperparams, path):
    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    for i in range(hyperparams.get('repetitions', 1)):
        save_as_instance = copy(save_as)
        if hyperparams.get('repetitions', 1) > 1:
            save_as_instance += f'_{i}'

        tsp = QLearningTsp(
            hyperparams=hyperparams,
            save=save,
            save_as=save_as_instance,
            path=path,
            test=test)

        tsp.perform_episodes(hyperparams.get('num_instances'))


def run_tsp_pg(hyperparams, path):
    save_path = path
    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    for i in range(hyperparams.get('repetitions', 1)):
        save_as_i = save_as + str(i)
        tsp = TrainTspEnv(
            hyperparams, save=hyperparams.get('save'),
            save_as=save_as_i, test=hyperparams.get('test'), path=save_path)
        tsp.perform_episodes(num_instances=hyperparams.get('num_instances'))


def run_cp_qlearning(hyperparams, path):
    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    for i in range(hyperparams.get('reps', 1)):
        save_as_instance = copy(save_as)
        if hyperparams.get('reps', 1) > 1:
            save_as_instance += f'_{i}'

        cpq = QLearningCartpole(
            hyperparams=hyperparams,
            save=save,
            save_as=save_as_instance,
            path=path,
            test=test)

        cpq.perform_episodes()


def run_cp_pg(hyperparams, path):
    save = hyperparams.get('save', False)
    timestamp = time.localtime()
    save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    noise_p = hyperparams.get('noise_0', 0)
    noise_type = hyperparams.get('noise_type', 'depolarize')
    n_qubits = hyperparams.get('n_qubits', 4)
    n_layers = hyperparams.get('n_layers', 5)
    beta = hyperparams.get('beta', 1)
    gamma = hyperparams.get('gamma', 0.99)
    batch_size = hyperparams.get('batch_size', 16)
    max_episodes = hyperparams.get('max_episodes', 1000)
    param_perturbation = hyperparams.get('param_perturbation', 0)

    for i in range(hyperparams.get('repetitions', 1)):
        save_as_i = save_as + str(i)
        train_cp_pg(noise_p=noise_p, noise_type=noise_type, n_qubits=n_qubits, n_layers=n_layers,
                    beta=beta, gamma=gamma, batch_size=batch_size, max_episodes=max_episodes,
                    lr_in=0.1, lr_var=0.01, lr_out=0.01, param_perturbation=param_perturbation,
                    checkpoint=save, checkpoint_path=path, save_as=save_as_i)
