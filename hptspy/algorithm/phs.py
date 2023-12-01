from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Type, TypeAlias, overload

from _hptspy import _phs  # type: ignore

from .. import env, model

if TYPE_CHECKING:
    from ..util import StopToken

# Bring names into current scope
# Search Inputs
SearchInputPolicyConvNetRNDSimple = _phs.phs_search_input_policy_convnet_craftworld
SearchInputTwoHeadedConvNetRNDSimple = (
    _phs.phs_search_input_twoheaded_convnet_rnd_simple
)
SearchInputPolicyConvNetBoxWorld = _phs.phs_search_input_policy_convnet_boxworld
SearchInputTwoHeadedConvNetBoxWorld = _phs.phs_search_input_twoheaded_convnet_boxworld
SearchInputPolicyConvNetCraftWorld = _phs.phs_search_input_policy_convnet_craftworld
SearchInputTwoHeadedConvNetCraftWorld = (
    _phs.phs_search_input_twoheaded_convnet_craftworld
)
SearchInputPolicyConvNetSokoban = _phs.phs_search_input_policy_convnet_sokoban
SearchInputTwoHeadedConvNetSokoban = _phs.phs_search_input_twoheaded_convnet_sokoban

# Search Outputs
SearchOutputRNDSimple = _phs.phs_search_output_rnd_simple
SearchOutputBoxWorld = _phs.phs_search_output_boxworld
SearchOutputCraftWorld = _phs.phs_search_output_craftworld
SearchOutputSokoban = _phs.phs_search_output_sokoban


# Type Hint typedefs
SearchInput: TypeAlias = (
    SearchInputPolicyConvNetRNDSimple
    | SearchInputTwoHeadedConvNetRNDSimple
    | SearchInputPolicyConvNetBoxWorld
    | SearchInputTwoHeadedConvNetBoxWorld
    | SearchInputPolicyConvNetCraftWorld
    | SearchInputTwoHeadedConvNetCraftWorld
    | SearchInputPolicyConvNetSokoban
    | SearchInputTwoHeadedConvNetSokoban
)
SearchInputT: TypeAlias = (
    Type[SearchInputPolicyConvNetRNDSimple]
    | Type[SearchInputTwoHeadedConvNetRNDSimple]
    | Type[SearchInputPolicyConvNetBoxWorld]
    | Type[SearchInputTwoHeadedConvNetBoxWorld]
    | Type[SearchInputPolicyConvNetCraftWorld]
    | Type[SearchInputTwoHeadedConvNetCraftWorld]
    | Type[SearchInputPolicyConvNetSokoban]
    | Type[SearchInputTwoHeadedConvNetSokoban]
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
_PHS_MAPPING_ATTRIBUTES = namedtuple(
    "_PHS_MAPPING_ATTRIBUTES", ["search_input", "phs_search", "phs_batched_search"]
)
_phs_mapping = {
    env.RNDSimpleState: {
        model.PolicyConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_policy_convnet_rnd_simple,
            _phs.phs_policy_convnet_rnd_simple,
            _phs.phs_batched_policy_convnet_rnd_simple,
        ),
        model.TwoHeadedConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_twoheaded_convnet_rnd_simple,
            _phs.phs_twoheaded_convnet_rnd_simple,
            _phs.phs_batched_twoheaded_convnet_rnd_simple,
        ),
    },
    env.BoxWorldState: {
        model.PolicyConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_policy_convnet_boxworld,
            _phs.phs_policy_convnet_boxworld,
            _phs.phs_batched_policy_convnet_boxworld,
        ),
        model.TwoHeadedConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_twoheaded_convnet_boxworld,
            _phs.phs_twoheaded_convnet_boxworld,
            _phs.phs_batched_twoheaded_convnet_boxworld,
        ),
    },
    env.CraftWorldState: {
        model.PolicyConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_policy_convnet_craftworld,
            _phs.phs_policy_convnet_craftworld,
            _phs.phs_batched_policy_convnet_craftworld,
        ),
        model.TwoHeadedConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_twoheaded_convnet_craftworld,
            _phs.phs_twoheaded_convnet_craftworld,
            _phs.phs_batched_twoheaded_convnet_craftworld,
        ),
    },
    env.SokobanState: {
        model.PolicyConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_policy_convnet_sokoban,
            _phs.phs_policy_convnet_sokoban,
            _phs.phs_batched_policy_convnet_sokoban,
        ),
        model.TwoHeadedConvNetEvaluator: _PHS_MAPPING_ATTRIBUTES(
            _phs.phs_search_input_twoheaded_convnet_sokoban,
            _phs.phs_twoheaded_convnet_sokoban,
            _phs.phs_batched_twoheaded_convnet_sokoban,
        ),
    },
}


def _check_env_model_eval(StateT, EvaluatorT):
    if StateT not in _phs_mapping:
        raise TypeError("Unknown environment.")
    if EvaluatorT not in _phs_mapping[StateT]:
        raise TypeError("Unknown model evaluator.")


def create_search_input(
    puzzle_name: str,
    state: env.SimpleEnv,
    search_budget: int,
    stop_token: StopToken,
    model_eval: model.PolicyConvNetEvaluator | model.TwoHeadedConvNetEvaluator,
) -> SearchInput:
    StateT = state.__class__
    EvaluatorT = model_eval.__class__
    _check_env_model_eval(StateT, EvaluatorT)

    return _phs_mapping[StateT][EvaluatorT].search_input(
        puzzle_name, state, search_budget, stop_token, model_eval
    )


@overload
def phs_search(search_input: SearchInput, num_threads: int = 1) -> SearchOutput:
    ...


@overload
def phs_search(
    search_input: list[SearchInput], num_threads: int = 1
) -> list[SearchOutput]:
    ...


def phs_search(
    search_input: SearchInput | list[SearchInput],
    num_threads: int = 1,
) -> SearchOutput | list[SearchOutput]:
    if isinstance(search_input, list):
        if len(search_input) == 0:
            return []
        StateT = search_input[0].state.__class__
        EvaluatorT = search_input[0].model_eval.__class__
        _check_env_model_eval(StateT, EvaluatorT)
        return _phs_mapping[StateT][EvaluatorT].phs_batched_search(
            search_input, num_threads
        )
    else:
        StateT = search_input.state.__class__
        EvaluatorT = search_input.model_eval.__class__
        _check_env_model_eval(StateT, EvaluatorT)
        return _phs_mapping[StateT][EvaluatorT].phs_search(search_input)
