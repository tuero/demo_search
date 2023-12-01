from __future__ import annotations

import multiprocessing as mp
import os
import signal
from typing import TYPE_CHECKING, Callable

from _hptspy import _common  # type: ignore

from . import env, util

if TYPE_CHECKING:
    from .env import SimpleEnv, SimpleEnvT

# Bring names into current scope
ObservationShape = _common.ObservationShape


def load_problems(
    envT: SimpleEnvT, path: str, max_instances: int
) -> tuple[list[SimpleEnv], list[str]]:
    """Load list of problems from problem string file into a list of Env objects."""
    env_mapping = {
        env.RNDSimpleState: _common.load_problems_rnd_simple,
        env.BoxWorldState: _common.load_problems_boxworld,
        env.CraftWorldState: _common.load_problems_craftworld,
        env.SokobanState: _common.load_problems_sokoban,
    }
    if envT not in env_mapping:
        raise TypeError("Unknown environment.")
    if not os.path.exists(path):
        raise Exception("File {} does not exist".format(path))
    return env_mapping[envT](path, max_instances, mp.cpu_count())


def _make_signal_hander(stop_token: util.StopToken) -> Callable:
    def signal_handler(sig, frame):
        _ = (sig, frame)
        stop_token.stop()

    return signal_handler


def signal_installer() -> util.StopToken:
    """Creates a StopToken with SIGINT signal handler installed to signal to C++ code."""
    stop_token = util.StopToken()
    signal_handler = _make_signal_hander(stop_token)
    signal.signal(signal.SIGINT, signal_handler)
    return stop_token
