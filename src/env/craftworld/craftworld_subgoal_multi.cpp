// File: craftworld_subgoal_multi.cpp
// Description: craftworld environment with generic multi subgoals

#include "env/craftworld/craftworld_subgoal_multi.h"

namespace hpts::env::cw {

using namespace craftworld;

auto CraftWorldSubgoalMultiState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto CraftWorldSubgoalMultiState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    return state.observation_shape_environment();
}

auto CraftWorldSubgoalMultiState::observation_shape_subgoal() const noexcept -> ObservationShape {
    return observation_shape();
}

// ---------------------------------------------------------

auto CraftWorldSubgoalMultiState::get_observation_low() const noexcept -> Observation {
    return get_observation();
}

auto CraftWorldSubgoalMultiState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept
    -> Observation {
    return state.get_observation_environment();
}

auto CraftWorldSubgoalMultiState::get_observation_subgoal() const noexcept -> Observation {
    return get_observation();
}

// ---------------------------------------------------------

auto operator<<(std::ostream &os, const CraftWorldSubgoalMultiState &state) -> std::ostream & {
    state.print(os);
    return os;
}

}    // namespace hpts::env::cw
