from collections import defaultdict

import numpy as np


class GeneratorDataset:
    def __init__(self, gen):
        self._gen = gen

    def get_dataset_iter(self):
        yield from self._gen()


class InfiniteDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_dataset_iter(self):
        while True:
            yield from self._dataset.get_dataset_iter()


class EpochDataset:
    def __init__(self, dataset, instances_in_epoch):
        self._iter = InfiniteDataset(dataset).get_dataset_iter()
        self._instances_in_epoch = instances_in_epoch

    def get_dataset_iter(self):
        for i in range(self._instances_in_epoch):
            yield next(self._iter)


class MinibatchDataset:
    def __init__(self, dataset, minibatch_size, truncate=True):
        self._minibatch_size = minibatch_size
        self._dataset = dataset
        self._truncate = truncate

    def get_dataset_iter(self):
        list_dict = defaultdict(list)
        _count = 0

        for data in self._dataset.get_dataset_iter():
            _count += 1
            for k, v in data.items():
                list_dict[k].append(v)
            if _count == self._minibatch_size:
                kv = {}
                for k, vl in list_dict.items():
                    kv[k] = np.array(vl)
                yield kv
                list_dict.clear()
                _count = 0

        if _count > 0 and not self._truncate:
            kv = {}
            for k, vl in list_dict.items():
                kv[k] = np.array(vl)
            yield kv
