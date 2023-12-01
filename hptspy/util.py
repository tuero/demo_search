from multiprocessing import Manager, Pool
from typing import Any, Callable, TypeVar

from _hptspy import _util  # type: ignore

# Bring names into current scope
NumericLimits = _util.NumericLimits
StopToken = _util.StopToken

T = TypeVar("T")


class ThreadPool:
    @staticmethod
    def _call(args):
        func = args[0]
        idx = args[1]
        manager_dict = args[2]
        manager_dict[idx] = func(*args[3])

    @staticmethod
    def run(num_threads: int, func: Callable[..., T], args: list[Any]) -> list[T]:
        manager = Manager()
        data = manager.dict()
        with Pool(num_threads) as p:
            p.map(
                ThreadPool._call,
                [(data, i, func, *args) for i in range(len(args))],
            )
        return [data[i] for i in range(len(args))]
