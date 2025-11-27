import numpy as np
import random
import os
from tqdm import tqdm

import argparse

from src.dsl import to_function, Program
from src.deepcoder_utils import generate_io_samples
from src.generate_dataset import generate_dataset, DatasetSpec, EquivalenceCheckingSpec, IteratorDecorator
from src.program_simplifier import remove_redundant_variables, remove_redundant_expressions, remove_dependency_between_variables

from pathlib import Path
from typing import List, Tuple, Union, Dict, Callable, Iterator, Iterable

class DCGenerator:
    @staticmethod
    def _from_cli_args(args: argparse.Namespace):
        return DCGenerator(
            args.num_programs,
            args.num_valid,
            args.dir,
            args.min_length,
            args.max_length,
            args.value_range,
            args.num_examples,
            args.num_prune,
            args.list_max,
            args.pickle_name,
            args.seed,
            args.enumerate
        )
    
    def __init__(self, num_programs, num_valid, dataset_dir, min_length, max_length,
                 value_range, num_examples, num_prune, list_max, pickle_name, seed, enumerate):
        # just storing attribs for fun
        self.seed = seed
        self.num_programs = num_programs
        self.num_valid = num_valid
        self.dataset_dir = Path(dataset_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.value_range = value_range
        self.num_examples = num_examples
        self.num_prune = num_prune
        self.list_max = list_max
        self.pickle_name = pickle_name + ".pickle"
        self.enumerate = enumerate

        # rng setup
        self.SEED_MAX = 2**32 - 1

        self.root_rng = np.random.RandomState(seed)

        # globally set random and np.random seeds
        random.seed(self.root_rng.randint(self.SEED_MAX))
        np.random.seed(self.root_rng.randint(self.SEED_MAX))

        # get language object (list of generate_io_samples.Function)
        LINQ, _ = generate_io_samples.get_language(value_range)

        # remove the MAP IDT case as it does not improve expressiveness
        self.LINQ = [f for f in LINQ if not "IDT" in f.src]

        # extract minimum and maximum function used in simplification
        self.MINIMUM = to_function([f for f in LINQ if f.src == "MINIMUM"][0])
        self.MAXIMUM = to_function([f for f in LINQ if f.src == "MAXIMUM"][0])

        # setup iteration tracking stuff
        program_iterator = lambda iterator: tqdm(iterator, desc="Total programs generated", unit="program", bar_format='{l_bar}{bar:40}{r_bar}')
        entry_iterator = lambda iterator: tqdm(iterator, desc="Prune Entries")
        self.iterator_decorator = IteratorDecorator(program_iterator, entry_iterator)

        # setup dataset and pruning heuristic
        self.dataset_spec = DatasetSpec(value_range, list_max, num_examples, min_length, max_length)
        self.eq_spec = EquivalenceCheckingSpec(0, num_prune, np.random.RandomState(seed))

    def __repr__(self):
        printables = [
            "num_programs",
            "num_valid",
            "dataset_dir",
            "min_length",
            "max_length",
            "value_range",
            "num_examples",
            "num_prune",
            "list_max",
            "seed",
            "enumerate"
        ]

        s = '\n'.join([f"{k} = {v}" for k, v in vars(self).items() if k in printables])
        return s

    def simplify(self, program):
        # everything from program_simplifier.py
        program = remove_redundant_expressions(program)
        program = remove_redundant_variables(program)
        program = remove_dependency_between_variables(program, self.MINIMUM, self.MAXIMUM)
        return program
    
    def generate(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)  # create parent dirs if not exist
        out_file = self.dataset_dir / self.pickle_name
        generate_dataset(self.LINQ,
                    self.dataset_spec,
                    self.eq_spec,
                    str(out_file), self.num_programs if self.num_programs > 0 else None,
                    simplify=self.simplify,
                    decorator=self.iterator_decorator,
                    enumerate=self.enumerate)

def _debug_dump_dcg(dcg):
    print("======DEBUG=======")
    print("Created DeepCoder dataset generator.")
    print(repr(dcg))
    print("==================")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Generates sample datasets for DeepCoder paper."
    )

    parser.add_argument("-n", "--num-programs", type=int, default=1000000, help = "Number of programs to generate.")
    parser.add_argument("--num-valid", type=int, default=10, help="Number of validation programs.")
    parser.add_argument("--dir", type=str, default="dataset/", help="Directory to store output datasets in.")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum program length.")
    parser.add_argument("--max-length", type=int, default=5, help="Maximum program length.")
    parser.add_argument("--value-range", type=int, default=256, help="Largest absolute value used in the dataset.")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of I/O examples per program.")
    parser.add_argument("--num-prune", type=int, default=100, help="Number of examples used to prune the identical programs.")
    parser.add_argument("--list-max", type=int, default=20, help="The maximum length of lists used in the dataset.")
    parser.add_argument("--pickle-name", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument('--enumerate', action='store_true')

    args = parser.parse_args()

    dcg: DCGenerator = DCGenerator._from_cli_args(args)
    _debug_dump_dcg(dcg)

    dcg.generate()

