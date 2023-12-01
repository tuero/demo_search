// File: rnd_subgoal_key_door_multi.cpp
// Description: rnd environment with key/door subgoals as separate independent obvservations

#include "env/rnd/rnd_subgoal_key_door_multi.h"

namespace hpts::env::rnd {

auto RNDSubgoalKeyDoorMultiState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto RNDSubgoalKeyDoorMultiState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto RNDSubgoalKeyDoorMultiState::observation_shape_subgoal() const noexcept -> ObservationShape {
    return observation_shape();
}

// ---------------------------------------------------------

auto RNDSubgoalKeyDoorMultiState::get_observation_low() const noexcept -> Observation {
    return get_observation();
}

auto RNDSubgoalKeyDoorMultiState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept
    -> Observation {
    return get_observation();
}

auto RNDSubgoalKeyDoorMultiState::get_observation_subgoal() const noexcept -> Observation {
    return get_observation();
}

// ---------------------------------------------------------

auto operator<<(std::ostream &os, const RNDSubgoalKeyDoorMultiState &state) -> std::ostream & {
    state.print(os);
    return os;
}

}    // namespace hpts::env::rnd
