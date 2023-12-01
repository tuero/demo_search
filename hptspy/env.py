from typing import Type, TypeAlias

from _hptspy import _env  # type: ignore

# Bring names into current scope
RNDSimpleState = _env.RNDSimpleState
BoxWorldState = _env.BoxWorldState
CraftWorldState = _env.CraftWorldState
SokobanState = _env.SokobanState


SimpleEnv: TypeAlias = RNDSimpleState | BoxWorldState | CraftWorldState | SokobanState
SimpleEnvT: TypeAlias = (
    Type[RNDSimpleState]
    | Type[BoxWorldState]
    | Type[CraftWorldState]
    | Type[SokobanState]
)


ENV_STR_MAP = {
    RNDSimpleState.name: RNDSimpleState,
    BoxWorldState.name: BoxWorldState,
    CraftWorldState.name: CraftWorldState,
    SokobanState.name: SokobanState,
}


def env_from_str(env_name: str) -> SimpleEnvT:
    if env_name not in ENV_STR_MAP:
        raise TypeError("Unknown environment name: {}.".format(env_name))
    return ENV_STR_MAP[env_name]
