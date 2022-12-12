from typing import Iterable
from tensorflow.python.keras.callbacks import Callback

import h5py
import json
import numpy as np


class MetaCallback(Callback):
    def __init__(self, meta_data_dict, checkpoint_path, save_as):
        self.meta_data_dict = meta_data_dict
        self.saved = False
        self.checkpoint_path = checkpoint_path
        self.save_as = save_as
        super(MetaCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not self.saved:
            with open(self.checkpoint_path + '{}_meta.json'.format(self.save_as), 'w') as file:
                json.dump(self.meta_data_dict, file)

            self.saved = True
        elif logs:
            with open(self.checkpoint_path + '{}_meta.json'.format(self.save_as), 'w') as file:
                json.dump(logs, file)


class ScoresCheckpoint(Callback):
    def __init__(self, checkpoint_path, save_as):
        self.scores = []
        self.checkpoint_path = checkpoint_path
        self.save_as = save_as

        super(ScoresCheckpoint, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.scores = logs['scores']

        save_model_data(
            self.scores,
            '{}_scores'.format(self.save_as),
            self.checkpoint_path)


def save_model_data(
        data: Iterable,
        model_name: str,
        path: str = 'data/models/'):
    hf = h5py.File(path + model_name + '.h5', 'w')
    hf.create_dataset('data', data=data)
    hf.close()


def load_model_data(
        model_name: str,
        path: str) -> np.array:
    hf = h5py.File(path + model_name + '.h5', 'r')
    data = np.array(hf.get('data'))
    return data


def load_json(file_name, path):
    if file_name[-5:] != '.json':
        file_name += '.json'
    with open(path + file_name, 'r') as file:
        data = json.load(file)
    return data
