import random
import time
from copy import copy

from q_learning.tsp_q_learning import QLearningTsp


def run_tsp(hyperparams, path):
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
