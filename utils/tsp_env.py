import pickle
import random
from copy import copy

import numpy as np


class TspEnv:

    def __init__(self, n_cities=5, num_instances=100, base_path='../../../data/'):
        with open(base_path, 'rb') as file:
            data = pickle.load(file)
            x_train = data['x_train'][:num_instances]
            y_train = data['y_train'][:num_instances]

        self.n_cities = n_cities
        self.instances = x_train
        self.optimal_solutions = y_train
        self.current_instance_ix = None
        self.tour = None
        self.available_nodes = None
        self.done = False

    def get_current_instance(self):
        return self.instances[self.current_instance_ix]

    def get_current_optimal_tour(self):
        return self.optimal_solutions[self.current_instance_ix]

    @staticmethod
    def compute_tour_length(nodes, tour):
        """
        Compute length of a tour, including return to start node.
        (If start node is already added as last node in tour, 0 will be added to tour length.)
        :param nodes: all nodes in the graph in form of (x, y) coordinates
        :param tour: list of node indices denoting a (potentially partial) tour
        :return: tour length
        """
        tour_length = 0
        for i in range(len(tour)):
            if i < len(tour) - 1:
                tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i + 1]]))
            else:
                tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

        return tour_length

    def cost(self, nodes, tour):
        return -self.compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)

    def reset(self):
        self.current_instance_ix = random.randint(0, len(self.instances)-1)
        self.tour = [0]
        self.available_nodes = list(range(1, self.n_cities))
        return self.instances[self.current_instance_ix], self.available_nodes

    def step(self, action):
        assert action in self.available_nodes, "City is already in tour"
        if len(self.tour) == len(self.instances[self.current_instance_ix]) - 1:
            prev_tour = copy(self.tour)
            self.tour.append(self.available_nodes[-1])
            reward = self.compute_reward(
                self.instances[self.current_instance_ix], prev_tour, self.tour)
            self.reset()
            self.done = True
        else:
            self.done = False
            prev_tour = copy(self.tour)
            self.tour.append(action)
            remove_node_ix = self.available_nodes.index(action)
            del self.available_nodes[remove_node_ix]
            reward = self.compute_reward(
                self.instances[self.current_instance_ix], prev_tour, self.tour)

        return (self.instances[self.current_instance_ix], self.available_nodes), \
               reward, self.done, []
