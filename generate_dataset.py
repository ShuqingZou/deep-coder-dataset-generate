import numpy as np
import random
import os
from tqdm import tqdm_notebook as tqdm

from src.dsl import to_function, Program
from src.deepcoder_utils import generate_io_samples
from src.generate_dataset import generate_dataset, DatasetSpec, EquivalenceCheckingSpec, IteratorDecorator
from src.program_simplifier import remove_redundant_variables, remove_redundant_expressions, remove_dependency_between_variables

runtime = "local"

#@title Parameters
#@markdown |Name            |Description|
#@markdown |:---            |:---|
#@markdown |`seed`|The random seed|
seed = 3984 #@param {type: "number"}

#@markdown ### `deep-coder` Repositories
#@markdown |Name            |Description|
#@markdown |:---            |:---|
#@markdown |`repository_url`|The URL of `deep-coder` git repository (enabled only in the host runtime)|
#@markdown |`branch_name`   |The branch name (enabled only in the host runtime)|
repository_url = "https://github.com/HiroakiMikami/deep-coder" #@param {type: "string"}
branch_name = "master" #@param {type: "string"}

#@markdown ### Dataset Configurations
#@markdown |Name          |Description|
#@markdown |:---          |:---|
#@markdown |`num_dataset` |The total number of programs in the dataset. If it is -1, the program will enumerate all valid source code.|
#@markdown |`num_valid`   |The number of programs used for validation.|
#@markdown |`value_range` |The largest absolute value used in the dataset.|
#@markdown |`max_list_length` |The maximum length of lists used in the dataset.|
#@markdown |`num_examples`|The number of I/O examples per program|
#@markdown |`min_length`  |The minimum length of the program body|
#@markdown |`max_length`  |The maximum length of the program body|
#@markdown |`num_examples_for_pruning`|The number of examples used to prune the identical programs.|
num_dataset = -1 #@param {type: "number"}
num_valid = 10 #@param {type: "number"}
value_range = 256 #@param {type: "number"}
max_list_length = 20 #@param {type: "number"}
num_examples = 5 #@param {type: "number"}
min_length = 1 #@param {type: "number"}
max_length = 2 #@param {type: "number"}
num_examples_for_pruning = 100 #@param {type: "number"}

#@markdown ### Filepath
#@markdown |Name                   |Description|
#@markdown |:---                   |:---|
#@markdown |`destination_dir_path` |The directory of the directory that will contain the dataset.|
destination_dir_path = "dataset/" #@param {type: "string"}


SEED_MAX = 2**32 - 1

root_rng = np.random.RandomState(seed)
random.seed(root_rng.randint(SEED_MAX))
np.random.seed(root_rng.randint(SEED_MAX))

LINQ, _ = generate_io_samples.get_language(value_range)
LINQ = [f for f in LINQ if not "IDT" in f.src]

MINIMUM = to_function([f for f in LINQ if f.src == "MINIMUM"][0])
MAXIMUM = to_function([f for f in LINQ if f.src == "MAXIMUM"][0])


def simplify(program):
    program = remove_redundant_expressions(program)
    program = remove_redundant_variables(program)
    program = remove_dependency_between_variables(program, MINIMUM, MAXIMUM)
    return program


# TODO: tqdm_notebook does not work in a local runtime
program_iterator = lambda iterator: tqdm(iterator, desc="Program Generation")
entry_iterator = lambda iterator: tqdm(iterator, desc="Prune Entries")
decorator = IteratorDecorator(program_iterator, entry_iterator)

generate_dataset(LINQ,
             DatasetSpec(value_range, max_list_length,
                         num_examples, min_length, max_length),
             EquivalenceCheckingSpec(0, num_examples_for_pruning, np.random.RandomState(
                 root_rng.randint(SEED_MAX))),
             "./dataset.pickle", num_dataset if num_dataset > 0 else None,
             simplify=simplify, decorator=decorator)
