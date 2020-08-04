import random
from collections import defaultdict
from typing import Iterable, Dict, List, Any
import logging
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)


@DataIterator.register("multi-task")
class HomogeneousBatchIterator(DataIterator):
    """
    An iterator that takes instances of various types
    and yields single-type batches of them.
    """

    def __init__(
        self,
        type_field_name: str = "framework",
        batch_size: int = 32,
        buffer_size_multiplier: int = 2,
        shuffle: bool = True,
    ) -> None:
        super().__init__(batch_size)
        self._type_field_name = type_field_name
        self._buffer_size = batch_size * buffer_size_multiplier
        self._shuffle = shuffle

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        hoppers: Dict[Any, List[Instance]] = defaultdict(list)

        for instance in instances:
            # Which hopper do we put this instance in?
            instance_type = instance.fields[self._type_field_name].metadata

            hoppers[instance_type].append(instance)

            # If the hopper is full, yield up the batch and clear it.
            if len(hoppers[instance_type]) >= self._buffer_size:
                if self._shuffle:
                    random.shuffle(hoppers[instance_type])
                yield Batch(hoppers[instance_type][:self._batch_size])
                del hoppers[instance_type][:self._batch_size]

        # Deal with leftovers
        for remaining in hoppers.values():
            if remaining:
                yield Batch(remaining)
