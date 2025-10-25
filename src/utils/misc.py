import ast
import cProfile
import logging
import os
import pstats
import random
import socket
import uuid
from functools import wraps
from typing import Dict

import numpy as np
import torch


# Update prevision of pstats module
def f8(x):
    return "%12.6f" % x


pstats.f8 = f8


logger = logging.getLogger(__name__)


def fix_random_seed(seed=46) -> None:
    """Fix random seed"""
    logger.info("Fix random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_file_utils(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]


def get_server_name() -> str:
    mac_address = hex(uuid.getnode())
    host_name = socket.gethostname()
    if host_name == "0c9770c5202d":
        return "bengio"
    elif host_name == "63f7aabe37ae":
        return "dlbox"
    elif host_name == "de77647750c3":
        return "feifei"
    elif host_name == "df09df35217e":
        return "dlbox2"
    elif host_name == "a0aeb5beb501":
        return "hinton"
    elif host_name == "774aa67a1d37":
        return "lecun"
    elif host_name == "fcb9da9721e2":
        return "efros"
    elif host_name == "44c94f3124ec":
        return "yoshimitsu"
    elif host_name == "a764d1ab222d":
        return "ashikaga"
    elif host_name == "977e72b97e91":
        return "he"
    elif host_name == "a2198d1e5198":
        return "ian"
    elif mac_address == "0x242ac11001f" or host_name == "22cd4afb415e":
        return "adam"
    return "unknown"


def read_flow_error_text(filename: str, abs_val: bool = False) -> tuple:
    """Read per-frame error file and returns it with statistics.

    Args:
        filename (str): [description]
        abs_val (bool): If True, calculate statistics etc on abs value.

    Returns:
        error_per_frame (dict): {key: np.ndarray}
        stats_over_frame (dict): {key: {"mean", "std", "min", "max", "n_data"}}
    """
    file = open(filename, "r")
    cnt = 0
    while 1:
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            line = line.replace("nan", "0.0")
            data = ast.literal_eval(line[line.find("::") + 2 : line.rfind("\n")])
            if isinstance(data, dict):   # dict: value, normal case
                if cnt == 0:
                    error_metrics_list = data.keys()
                    error_per_frame: Dict[str, list] = {k: [] for k in error_metrics_list}
                for k in error_metrics_list:
                    error_per_frame[k].append(data[k])
            elif isinstance(data, float):   # only one value
                if cnt == 0:
                    error_per_frame: Dict[str, list] = {"result": []}
                    error_metrics_list = ["result"]
                else:
                    error_per_frame["result"].append(data)
            else:
                raise ValueError(f"data type is not supported {data = }")
            cnt += 1

    error_per_frame = {k: np.array(error_per_frame[k]) for k in error_metrics_list}
    if abs_val:
        error_per_frame = {k: np.abs(v) for (k, v) in error_per_frame.items()}
    # Convert FWL to inverse
    for k in error_metrics_list:
        if "FWL" in k:
            error_per_frame[k] = 1.0 / error_per_frame[k]
        if k in ["1PE", "2PE", "3PE", "5PE", "10PE", "20PE"]:
            error_per_frame[k] *= 100.0
        if "runtime" in k:
            min_5 = np.percentile(error_per_frame[k], 5)
            max_5 = np.percentile(error_per_frame[k], 95)
            error_per_frame[k] = error_per_frame[k][error_per_frame[k] >= min_5]
            error_per_frame[k] = error_per_frame[k][error_per_frame[k] <= max_5]
    # error_per_frame = {k: v[v > 0] for (k, v) in error_per_frame.items()}
    # error_per_frame = {k: v[9910:10710] for (k, v) in error_per_frame.items()}   # only for outdoor_day1
    # error_per_frame = {k: v[50:] for (k, v) in error_per_frame.items()}   # slider_depth to see the difference

    stats_over_frame: Dict[str, dict] = {k: {} for k in error_metrics_list}
    for k in error_metrics_list:
        stats_over_frame[k]["mean"] = np.mean(error_per_frame[k])
        stats_over_frame[k]["rms"] = np.sqrt(np.mean(np.square(error_per_frame[k])))
        stats_over_frame[k]["std"] = np.std(error_per_frame[k])
        stats_over_frame[k]["min"] = float(np.min(error_per_frame[k]))
        stats_over_frame[k]["max"] = float(np.max(error_per_frame[k]))
        stats_over_frame[k]["n_data"] = len(error_per_frame[k])
    return error_per_frame, stats_over_frame


def save_stats_text(save_file_name: str, nth_frame: int, stats_dict: dict):
    with open(save_file_name, "a") as f:
        f.write(f"frame {nth_frame}::" + str(stats_dict) + "\n")


class Stats(pstats.Stats):
    # Override class for more flexible pstats
    # The parent class is https://github.com/python/cpython/blob/3.11/Lib/pstats.py
    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = " " * 12  # this is customized
        for func in self.top_level:
            print(indent, pstats.func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end=" ", file=self.stream)
        print("in %.3f seconds" % self.total_tt, file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self


def profile(output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/

    Usage is,
    ```
    @profile(output_file= ...)
    def your_function():
        ...
    ```
    Then you will get the profile automatically after the function call is finished.

    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner
