import os
import pathlib
from typing import Dict, Iterable, List
import logging
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from itertools import chain, cycle, islice

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def round_robin(*iterables: [Iterable]):
    """round_robin('ABC', 'D', 'EF') --> A D E B F C """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def add_framework_label_to_instances(framework: str, iterator: Iterable[Instance]) -> Iterable[Instance]:
    it = iter(iterator)
    while True:
        try:
            value = next(it)
            value.add_field("framework", MetadataField(framework))
            yield value
        except StopIteration:
            return


@DatasetReader.register("multi-task")
class MultiTaskDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    dataset_readers : ``Dict[str, DatasetReader]``,
        Dataset reader for each framework. Dictionary key is framework label.
    """

    def __init__(self,
                 dataset_readers: Dict[str, DatasetReader],
                 sort_by_length: bool = True,
                 filter_by_length: int = None,
                 epoch_size_per_framework: int = None,
                 dataset_extension: str = 'train.aug.mrp') -> None:
        # DatasetReader need to be lazy in order to support epoch_size_per_framework.
        # In practice, the dataset is not lazy as it saves all the instances in memory
        is_lazy = epoch_size_per_framework is not None
        super().__init__(lazy=is_lazy)
        self._dataset_readers = dataset_readers
        assert all(not dataset_reader.lazy for dataset_reader in dataset_readers.values()), \
            "Do not support sub dataset readers laziness"

        self._sort_by_length = sort_by_length
        self._filter_by_length = filter_by_length
        self._epoch_size_per_framework = epoch_size_per_framework
        self._dataset_extension = dataset_extension

        # A dict of type {framework_label : {dataset_dir, instance list}}
        self._cached_instances: Dict[str, Dict[str, List[Instance]]] = \
            {framework_label: {} for framework_label in self._dataset_readers.keys()}

        if epoch_size_per_framework is not None:
            self._next_instances_to_read: Dict[str, Dict[str, List[Instance]]] = \
                {framework_label: {} for framework_label in self._dataset_readers.keys()}

    @overrides
    def _read(self, path: str):
        assert os.path.isdir(path), "MultiTaskDatasetReader requires a directory as dataset path."

        self._read_if_not_in_cache(path)

        current_epoch_instance_list = []
        for framework_label in self._cached_instances.keys():
            assert path in self._cached_instances[framework_label]
            cached_instances = self._cached_instances[framework_label][path]

            if self._epoch_size_per_framework is not None:
                if path not in self._next_instances_to_read[framework_label]:
                    self._next_instances_to_read[framework_label][path] = []

                next_instances_to_read = self._next_instances_to_read[framework_label][path]
                temp_instance_list = []
                left_to_read_count = self._epoch_size_per_framework

                while left_to_read_count > 0:
                    if len(next_instances_to_read) == 0:
                        next_instances_to_read = cached_instances.copy()

                    to_read_count = min(left_to_read_count, len(next_instances_to_read))
                    temp_instance_list.extend(next_instances_to_read[:to_read_count])
                    del next_instances_to_read[:to_read_count]
                    left_to_read_count = left_to_read_count - to_read_count
                current_epoch_instance_list.append(temp_instance_list)
            else:
                current_epoch_instance_list.append(cached_instances)

        yield from round_robin(*current_epoch_instance_list)

    def text_to_instance(self, *inputs) -> Instance:
        raise NotImplementedError

    def _read_if_not_in_cache(self, path: str):
        for framework_label in self._dataset_readers.keys():
            if path in self._cached_instances[framework_label]:
                continue

            dataset_list = list(pathlib.Path(path).glob(f'{framework_label}.{self._dataset_extension}'))

            assert len(dataset_list) > 0, \
                f'MultiTaskDatasetReader did not find any dataset files for {framework_label} in {path}'

            iterators = [add_framework_label_to_instances(framework_label,
                                                          self._dataset_readers[framework_label]
                                                          .read(pathlib.Path(path)))
                         for path in dataset_list]
            iterator = chain(*iterators)
            self._cached_instances[framework_label][path] = list(iterator)

            if self._sort_by_length:
                self._cached_instances[framework_label][path].sort(
                    key=lambda instance: instance["tokens"].sequence_length())
            if self._filter_by_length:
                self._cached_instances[framework_label][path] = \
                    list(filter(lambda instance: instance["tokens"].sequence_length() <= self._filter_by_length,
                                self._cached_instances[framework_label][path].copy()))
