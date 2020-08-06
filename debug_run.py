import os
import shutil
import sys
from allennlp.commands import main
# !!! The following imports are needed for AllenNLP to register the modules !!! #
from multi_task.multi_task_dataset_reader import MultiTaskDatasetReader
from multi_task.multi_task_model import MultiTaskModel
from metrics.mrp_score import MCESScore
from multi_task.multi_task_iterator import HomogeneousBatchIterator
from modules.multi_task.transition_parser_ucca_multi import UccaTransitionParserMultiTaskWrapper
from modules.multi_task import transition_parser_eds_multi
from modules.stack_rnn import StackRnn

if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(__file__))
    config_file = f'{dirname}/config/multi_task_stack_buffer.json'
    serialization_dir = f'{dirname}/checkpoints/'

    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

    # Assemble the command into sys.argv

    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "utils",
        "--include-package","multi_task" ,
        "--include-package", "modules",
        "--include-package", "metrics",
        "--file-friendly-logging",
    ]

    main()
