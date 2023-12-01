from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Type, TypeAlias, overload

from _hptspy import _astar  # type: ignore

from .. import env

if TYPE_CHECKING:
    from ..util import StopToken

# Bring names into current scope
SearchInputRNDSimple = _astar.astar_search_input_rnd_simple
SearchInputBoxWorld = _astar.astar_search_input_boxworld
SearchInputCraftWorld = _astar.astar_search_input_craftworld
SearchInputSokoban = _astar.astar_search_input_sokoban
SearchOutputRNDSimple = _astar.astar_search_output_rnd_simple
SearchOutputBoxWorld = _astar.astar_search_output_boxworld
SearchOutputCraftWorld = _astar.astar_search_output_craftworld
SearchOutputSokoban = _astar.astar_search_output_sokoban


# Type Hint typedefs
SearchInput: TypeAlias = (
    SearchInputRNDSimple
    | SearchInputBoxWorld
    | SearchInputCraftWorld
    | SearchInputSokoban
)
SearchInputT: TypeAlias = (
    Type[SearchInputRNDSimple]
    | Type[SearchInputBoxWorld]
    | Type[SearchInputCraftWorld]
    | Type[SearchInputSokoban]
)
SearchOutput: TypeAlias = (
    SearchOutputRNDSimple
    | SearchOutputBoxWorld
    | SearchOutputCraftWorld
    | SearchOutputSokoban
)
SearchOutputT: TypeAlias = (
    Type[SearchOutputRNDSimple]
    | Type[SearchOutputBoxWorld]
    | Type[SearchOutputCraftWorld]
    | Type[SearchOutputSokoban]
)

# mapping for environment/model evaulator -> search input/algorithms
_ASTAR_MAPPING_ATTRIBUTES = namedtuple(
    "_ASTAR_MAPPING_ATTRIBUTES",
    ["search_input", "astar_search", "astar_batched_search"],
)
_astar_mapping = {
    env.RNDSimpleState: _ASTAR_MAPPING_ATTRIBUTES(
        _astar.astar_search_input_rnd_simple,
        _astar.astar_rnd_simple,
        _astar.astar_batched_rnd_simple,
    ),
    env.BoxWorldState: _ASTAR_MAPPING_ATTRIBUTES(
        _astar.astar_search_input_boxworld,
        _astar.astar_boxworld,
        _astar.astar_batched_boxworld,
    ),
    env.CraftWorldState: _ASTAR_MAPPING_ATTRIBUTES(
        _astar.astar_search_input_craftworld,
        _astar.astar_craftworld,
        _astar.astar_batched_craftworld,
    ),
    env.SokobanState: _ASTAR_MAPPING_ATTRIBUTES(
        _astar.astar_search_input_sokoban,
        _astar.astar_sokoban,
        _astar.astar_batched_sokoban,
    ),
}


def create_search_input(
    puzzle_name: str,
    state: env.SimpleEnv,
    search_budget: int,
    stop_token: StopToken,
) -> SearchInput:
    StateT = state.__class__
    if StateT not in _astar_mapping:
        raise TypeError("Unknown environment.")

    return _astar_mapping[StateT].search_input(
        puzzle_name, state, search_budget, stop_token
    )


@overload
def astar_search(search_input: SearchInput, num_threads: int = 1) -> SearchOutput:
    ...


@overload
def astar_search(
    search_input: list[SearchInput], num_threads: int = 1
) -> list[SearchOutput]:
    ...


def _check_env_(StateT):
    if StateT not in _astar_mapping:
        raise TypeError("Unknown environment.")


def astar_search(
    search_input: SearchInput | list[SearchInput],
    num_threads: int = 1,
) -> SearchOutput | list[SearchOutput]:
    if isinstance(search_input, list):
        if len(search_input) == 0:
            return []
        StateT = search_input[0].state.__class__
        _check_env_(StateT)
        return _astar_mapping[StateT].astar_batched_search(search_input, num_threads)
    else:
        StateT = search_input.state.__class__
        _check_env_(StateT)
        return _astar_mapping[StateT].astar_search(search_input)
