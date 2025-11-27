import dataclasses
import copy
import pickle
import os
import contextlib
import numpy as np

from typing import List, Tuple, Union, Dict, Callable, Iterator, Iterable

from .dataset import Primitive, Example, Entry, Dataset, dataset_metadata
from .deepcoder_utils import generate_io_samples
from .dsl import Function, Program, Type, to_function, Signature
from .program_simplifier import normalize
from .program_generator import programs, random_programs

from tqdm import tqdm

@dataclasses.dataclass
class IntermediateEntry:
    source_code: str
    program: generate_io_samples.Program
    examples: List[Example]
    attribute: Dict[str, bool]

    def f(self, *args):
        return self.program.fun(*args)

def get_signature(program: Program):
    input = []
    for i in program.inputs:
        input.append(i.t)
    output = program.body[-1].expression.function.signature.output_type if len(
        program.body) > 0 else None
    return Signature(input, output)

def generate_intermediate_entry(program: Program, LINQ, dataset_spec, functions_dsl) -> None | IntermediateEntry:
    """
    Generates intermediate entry for a program.

    Args:
        program (Program): Program to create intermediate entry for.

    Returns:
        None | IntermediateEntry: IntermediateEntry or None if either 
                                    compilatin or IO generation failed.
    """
    # last newline should be removed to compile source code
    code = program.to_string()[:-1]

    # Compile the source code
    with contextlib.redirect_stdout(None):  # ignore stdout
        p = generate_io_samples.compile_faster(
            code, LINQ, V=dataset_spec.value_range, L=dataset_spec.max_list_length)
    if p is None:
        # Compilation is failed
        return None

    try:
        # Generate IO examples
        with contextlib.redirect_stdout(None):  # ignore stdout
            examples = generate_io_samples.generate_IO_examples(
                p, N=dataset_spec.num_examples, L=dataset_spec.max_list_length, V=dataset_spec.value_range)
    except ValueError:
        return None

    # Generate binary attribute
    ss = set()
    for statement in program.body:
        for symbol in statement.expression.function.name.split(" "):
            ss.add(symbol)
    attribute = dict()
    for f in functions_dsl:
        for symbol in f.name.split(" "):
            if not symbol in attribute:
                attribute[symbol] = False
            attribute[symbol] |= symbol in ss

    return IntermediateEntry(code, p, list(map(lambda x: Example(x[0], x[1]), examples)), attribute)


@dataclasses.dataclass
class DatasetSpec:
    """
    The specification of the dataset.

    Attributes:
        value_range : int
            The maximum absolute value allowed for numbers within the
            generated programs (e.g., 0 to N).
        max_list_length : int
            The maximum length allowed for any list or array generated
            as part of the data.
        num_examples : int
            The total number of input/output examples to be generated
            for the dataset.
        min_program_length : int
            The minimum program length.
        max_program_length : int
            The maximum program length

    Note:
        Program length is equal to number of statements in program body.
    """
    value_range: int
    max_list_length: int
    num_examples: int
    min_program_length: int
    max_program_length: int


@dataclasses.dataclass
class EquivalenceCheckingSpec:
    """The specification used to determine pruning heuristics."""
    ratio_of_examples: float
    num_of_examples: int
    rng: Union[np.random.RandomState, None]

type SimplifyFunction = Callable[[Program], Program]

@dataclasses.dataclass
class IteratorDecorator:
    """
    The functions to wrap the iterators (such as tqdm)
    
    Attributes
    ----------
    program_decorator : Callable[[Iterable], Iterable]
        Keeps track of total programs enumerated.
    entry_decorator : Callable[[Iterable], Iterable]
        Keeps track of valid entries saved.
    """
    program_decorator: Callable[[Iterable], Iterable]
    entry_decorator: Callable[[Iterable], Iterable]

class Canon:
    # setup canonical mapper
    # signature -> input -> output -> set of IntermediateEntry
    # if input doesn't exist, no canonical program
    # if output doesn't exist for input, no canonical program
    # if output exists for input, get shorter of two programs

    def __init__(self, LINQ, dataset_spec, functions_dsl, max_compare_ios = 10):
        self.iotable: dict[Signature, dict[list[Primitive], dict[Primitive, set[IntermediateEntry]]]] = {}
        self.LINQ = LINQ
        self.dataset_spec = dataset_spec
        self.functions_dsl = functions_dsl
        self.max_compare_ios = max_compare_ios

        self._seen_programs: set[str] = set()
        self._valid_intermediates: set[IntermediateEntry] = set()
        self._valid_entries: set[Entry] = set()

    def add_program(self, program) -> bool:
        # mark program as seen
        prog_str = program.to_string()
        self._mark_seen_str(prog_str)

        # add signature to iotable
        signature = get_signature(program)
        self._add_signature(signature)

        # try create entry
        # TODO implement better error prop
        intermediate = generate_intermediate_entry(program, self.LINQ, self.dataset_spec, self.functions_dsl)
        if intermediate is None:
            # compilation failed or io failed
            return False

        collision_set: set[IntermediateEntry] = set(self._valid_intermediates)

        # tmp store for calculated io pairs in case we're adding this entry
        i_list = []
        o_list = []

        io_map = self.iotable[Signature]
        for i, o_map in io_map:
            o_entry = intermediate.f(i)
            i_list.append(i)
            o_list.append(o_entry)

            if o_entry in o_map:
                # output for this input exists
                # reduce collision set to only those with this output
                collision_set &= o_map[o_entry] 
            else:
                collision_set = set()

        # no full collisions
        if not collision_set:
            # build final entry too
            entry = Entry(intermediate.source_code, intermediate.examples, intermediate.attribute)
            
            # add to valids
            self._valid_intermediates.add(intermediate)
            self._valid_entries.add(entry)
            
            # add entry's results to map
            for i, o in zip(i_list, o_list):
                # i GURANTEED to be in iomap
                if o not in io_map[i]:
                    io_map[i][o] = set(intermediate)
                else:
                    io_map[i][o].add(intermediate)

            # no full collisions
            if len(io_map) < self.max_compare_ios:
                # io_map not full, add unique's first example
                first_example = intermediate.examples[0]

                i_to_add = first_example.inputs

                # add only if this input doens't exist
                if i_to_add not in io_map:
                    # add as new entry
                    output_map: dict[Primitive, set[IntermediateEntry]] = dict()

                    # propagate this to all entries
                    for valid_intermediate in self._valid_intermediates:
                        o_intermediate = valid_intermediate.f(i_to_add)
                        if o_intermediate in output_map:
                            output_map[o_intermediate].add(valid_intermediate)
                        else:
                            output_map[o_intermediate] = set(valid_intermediate)

                    io_map[i_to_add] = output_map

    def _add_io_examples(self,
                         entry: IntermediateEntry,
                         signature: Signature,
                         inputs: list[list[Primitive]],
                         outputs: list[Primitive]):

        # the number to add is the min between free space in the table 
        # as defined by the max_compare_ios attribute, or the number of
        # ios we have provided
        n_free_space = self.max_compare_ios - len(iotable_sig)
        n_io = len(inputs)
        assert len(inputs) == len(outputs)

        n = min(n_free_space, n_io)
        if n == 0:
            # early exit
            return

        iotable_sig = self.iotable[signature]


        for k in range(n):
            i = inputs[k]
            o = outputs[k]

            if i not in iotable_sig:
                iotable_sig[i] = dict()
                iotable_sig[i][o] = set(entry)
            elif o not in iotable_sig[i]:
                # one less line to compute type beat
                iotable_sig[i][o] = set(entry)
            else:
                iotable_sig[i][o]
    
    def _is_prune_unique(self,
                         e: IntermediateEntry,
                         signature: Signature) -> bool:
        # get signature specific iotable
        iotable_sig = self.iotable[signature]

        ilist: list[list[Primitive]] = []
        olist: list[Primitive] = []

        # work set to keep track which programs are perfect matches
        # initiated as all possible collisions (valid intermediates)
        collision_set: set[IntermediateEntry] = set(self._valid_intermediates)

        # handle empty case
        # populate io examples with own and return true
        if len(iotable_sig) == 0:
            # no existing cases for this signature
            # ACCEPT
            for ex in e.examples:
                ilist.append(ex.inputs)
                olist.appenx(ex.output)
        else:
            # cases exist
            # we must do comparison,
            for i, output_map in iotable_sig.items():
                o_e = e.f(i)
                ilist.append(i)
                olist.append(o_e)

                # NOTE guard because i'm unsure of executor behavior
                # NOTE there is also no checking of oob behavior based on V here
                if o_e is None:
                    raise ValueError("o_e is None in canon")
                
                if o_e not in output_map:
                    collision_set = set()
                else:
                    collision_set &= output_map[o_e]

            # collision_set holds entries that we 100% match with
            if len(collision_set) == 0:
                # no collisions, safe to accept
                
        ...

    def _add_signature(self, signature: Signature) -> bool:
        if not signature in self.iotable:
            self.iotable[signature] = dict()
            return True
        else:
            return False
        
    def has_seen(self, program: Program) -> bool:
        return program.to_string() in self._seen_programs

    def _has_seen_str(self, program_str: str) -> bool:
        return program_str in self._seen_programs

    def _mark_seen_str(self, program_str: str) -> bool:
        if program_str in self._seen_programs:
            return False
        else:
            self._seen_programs.add(program_str)
            return True


# As defined from generate_io_samples.py
# Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])
# Program = namedtuple('Program', ['src', 'ins', 'out', 'fun', 'bounds'])

def generate_dataset(functions: List[generate_io_samples.Function],
                     spec: DatasetSpec,
                     equivalence_spec: EquivalenceCheckingSpec,
                     destination: str,
                     num_dataset: Union[None, int] = None,
                     simplify: Union[None, SimplifyFunction] = None,
                     decorator: Union[None, IteratorDecorator] = None,
                     enumerate: bool = False):
    """
    Generate dataset to the file

    Parameters
    ----------
    functions : list of generate_io_samples.Function
        The set of functions that can be used in the dataset
    spec : DatasetSpec
        The specification of generated dataset
    equivalence_spec: EquivalenceCheckingSpec
        The specification used to check equivalence of programs
    destination : str
        The destination of the dataset file
    num_dataset : int or None
        The number of dataset to be created.
        If this argument is None, the function enumerate all source code
    simplify : function or None
        The function to simplify the source code
    decorator: IteratorDecorator or None
        The decorator of iterators. It is maily used to show the progress (e.g., tqdm)
    enumerate: bool
        Whether to enumerate programs by length or randomly generate.
    Notes
    -----
    Currently this function generates and prunes source code in memory.
    It might be a problem if the program size is large.
    """

    def _saturate_simplify(program: Program, simplify: SimplifyFunction) -> Program:
        # loop until simplification saturated
        # simplification function should be convergent
        while True:
            prev_program = program.to_string()
            program = simplify(program)

            if program.to_string() == prev_program:
                return program

    
    # setup simplification and normalization function
    if simplify is not None:
        _simplify_and_normalize = lambda p: normalize(_saturate_simplify(p, simplify))
    else:
        _simplify_and_normalize = lambda p: normalize(p)

    # setup helper structures
    functions_dsl = [to_function(f) for f in functions]
    invalid_program = set()
    entries = dict()  # Signature -> dict(str -> IntermediateEntry)

    # setup decorators
    progd = decorator.program_decorator if decorator is not None else lambda x: x
    entryd = decorator.entry_decorator if decorator is not None else lambda x: x

    # setup canonical mapper
    # signature -> input -> output -> program
    # if input doesn't exist, no canonical program
    # if output doesn't exist for input, no canonical program
    # if output exists for input, get shorter of two programs
    canon = Canon(LINQ=functions, dataset_spec=spec, functions_dsl=functions_dsl)

    # choose program generator based on style (enumerate / random)
    if enumerate:
        program_generator = progd(programs(functions_dsl, spec.min_program_length, spec.max_program_length))
    else:
        program_generator = progd(random_programs(functions_dsl, spec.min_program_length, spec.max_program_length))

    # define break condition
    # in the current implementation it's just the num_dataset
    if num_dataset is not None:
        should_break = lambda count: count >= num_dataset
    else:
        should_break = lambda count: False

    n_entries = 0
    with tqdm(total=num_dataset, desc="Creating pruned entries", unit="entry", bar_format='{l_bar}{bar:40}{r_bar}') as entry_progress:
        for program in program_generator:
            # simplify and normalize first
            program = _simplify_and_normalize(program)

            # this program has already been seen, skip
            if canon.has_seen(program):
                continue

            canon.add_program(program)
            
            # False if num_dataset is None, implying never breaks here
            if should_break(n_entries):
                break

        dataset = canon._valid_entries

    # Create metadata
    metadata = dataset_metadata(
        dataset, spec.value_range, spec.max_list_length)
    
    print(f"generated {len(dataset)} examples")

    # Dump the dataset to the file
    with open(destination, "wb") as f:
        pickle.dump(Dataset(dataset, metadata), f)
