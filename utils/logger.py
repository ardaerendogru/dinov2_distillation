# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import functools
import logging
import os
import sys
import time
from collections import Counter

from termcolor import colored

import functools

import numpy as np
import torch
import torch.distributed as tdist

_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call comm.create_local_process_group("
    "num_workers_per_machine) after calling torch.distributed.init_process_group()."
)


def get_world_size() -> int:
    if not tdist.is_available():
        return 1
    if not tdist.is_initialized():
        return 1
    return tdist.get_world_size()


def get_rank() -> int:
    if not tdist.is_available():
        return 0
    if not tdist.is_initialized():
        return 0
    return tdist.get_rank()


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Anyma's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine)
        )
        pg = tdist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not tdist.is_available():
        return 0
    if not tdist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return tdist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not tdist.is_available():
        return 1
    if not tdist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return tdist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not tdist.is_available():
        return
    if not tdist.is_initialized():
        return
    world_size = tdist.get_world_size()
    if world_size == 1:
        return
    if tdist.get_backend() == tdist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        tdist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        tdist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if tdist.get_backend() == "nccl":
        return tdist.new_group(backend="gloo")
    else:
        return tdist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = (
            _get_global_gloo_group()
        )  # use CPU group by default, to reduce GPU RAM usage.
    world_size = tdist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    tdist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = tdist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = tdist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        tdist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        tdist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        tdist.reduce(values, dst=0)
        if tdist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


LOGGER_NAME = "dinov2" # Changed from "anyma" to "dinov2" to align with codebase
LOGGER = None  # set up at end of the file
EXT_LOGGER = ["detectron2", "fvcore"] # Consider removing or adapting EXT_LOGGER

D2_LOG_BUFFER_SIZE_KEY: str = "D2_LOG_BUFFER_SIZE"

DEFAULT_LOG_BUFFER_SIZE: int = 1024 * 1024  # 1MB


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# cache the opened file object, so that different calls
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=_get_log_stream_buffer_size(filename))
    atexit.register(io.close)
    return io


def _get_log_stream_buffer_size(filename: str) -> int:
    if "://" not in filename:
        # Local file, no extra caching is necessary
        return -1
    # Remote file requires a larger cache to avoid many small writes.
    if D2_LOG_BUFFER_SIZE_KEY in os.environ:
        return int(os.environ[D2_LOG_BUFFER_SIZE_KEY])
    return DEFAULT_LOG_BUFFER_SIZE


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if "dinov2/logging/my_logger.py" not in code.co_filename: # Changed path to current file
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "dinov2" # Changed default module name
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg,)

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n(lvl, msg, n=1, *, name=None):
    """
    Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time



# Cache it so we avoid adding multiple handlers to the same logger.
@functools.lru_cache()
def set_logging(name="dinov2", verbose=True, color=True, rank=0, output_dir=None): # Modified function signature to accept output_dir and default name to "dinov2"
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the dinov2 library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger. Defaults to "dinov2".
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise. Defaults to True.
        color (bool): Flag to enable colorful output. Defaults to True.
        rank (int): Rank of the current process in distributed training. Defaults to 0.
        output_dir (str): Path to the output directory to save log files. Defaults to None (no file logging).

    Examples:
        >>> set_logging(name="dinov2", verbose=True)
        >>> logger = logging.getLogger("dinov2")
        >>> logger.info("This is an info message")
    """
    level = (
        logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    )  # rank in world for Multi-GPU trainings

    format_string = "%(message)s"
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + format_string,
            datefmt="%m/%d %H:%M",
            root_name=name,
            abbrev_name=str(name),
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: " + format_string, datefmt="%m/%d %H:%M"
        )

    # Set up stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)


    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []
    logger.addHandler(stream_handler)
    logger.propagate = False

    if output_dir: # Add file logging if output_dir is provided
        add_file_logging(output_dir=output_dir, verbose=verbose, rank=rank, logger=logger) # Pass logger instance

    return logger


def add_file_logging(output_dir="log.txt", verbose=True, rank=0, logger=None): # Modified to accept output_dir and logger, default output_dir to "log.txt"
    level = logging.DEBUG if verbose else logging.INFO
    if output_dir.endswith(".txt") or output_dir.endswith(".log"):
        output = output_dir
    else:
        output = os.path.join(output_dir, "log.txt")
    distributed_rank = rank
    if distributed_rank > 0:
        output = output + ".rank{}".format(distributed_rank)
    dirname = os.path.dirname(output)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    fh = logging.StreamHandler(_cached_log_stream(output))
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M"
        )
    )
    if logger is None:
        logger = logging.getLogger(LOGGER_NAME) # Fallback to default logger if not provided
    logger.addHandler(fh)
    for ext_logger in EXT_LOGGER: # Use ext_logger instead of logger
        logging.getLogger(ext_logger).addHandler(fh)


# Set logger
LOGGER = set_logging(LOGGER_NAME, verbose=True, rank=get_rank()) # Use distributed.get_rank()
for ext_logger in EXT_LOGGER: # Use ext_logger instead of logger
    set_logging(ext_logger, verbose=True, rank=get_rank())


def get_logger():
    return LOGGER