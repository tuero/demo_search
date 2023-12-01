// File: rnd_subgoal_key_door_visual.cpp
// Description: rnd environment with key/door subgoals and additional channel highlighted observation

#include "env/rnd/rnd_subgoal_key_door_visual.h"

namespace hpts::env::rnd {

// ---------------------------------------------------------

auto RNDSubgoalKeyDoorVisualState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto RNDSubgoalKeyDoorVisualState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    const auto obs_shape = observation_shape();
    return {obs_shape.c + 1, obs_shape.h, obs_shape.w};
}

auto RNDSubgoalKeyDoorVisualState::observation_shape_subgoal() const noexcept -> ObservationShape {
    return observation_shape();
}

// ---------------------------------------------------------

auto RNDSubgoalKeyDoorVisualState::get_observation_low() const noexcept -> Observation {
    return RNDSimpleState::get_observation();
}

auto RNDSubgoalKeyDoorVisualState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept
    -> Observation {
    Observation obs = state.get_observation();
    const ObservationShape obs_shape = state.observation_shape();
    Observation subgoal_channel = Observation(obs_shape.w * obs_shape.h, 0);
    // Get all indices this subgoal exists on
    std::vector<int> indices;
    for (const auto &cell_type : SUBGOAL_TYPE_MAP.at(static_cast<Subgoal>(subgoal))) {
        const auto temp_indices = state.get_indices(cell_type);
        indices.insert(indices.end(), temp_indices.begin(), temp_indices.end());
    }
    for (auto const &idx : indices) {
        subgoal_channel[idx] = 1;
    }
    obs.insert(std::end(obs), std::begin(subgoal_channel), std::end(subgoal_channel));
    return obs;
}

auto RNDSubgoalKeyDoorVisualState::get_observation_subgoal() const noexcept -> Observation {
    return RNDSimpleState::get_observation();
}

// ---------------------------------------------------------

auto operator<<(std::ostream &os, const RNDSubgoalKeyDoorVisualState &state) -> std::ostream & {
    state.print(os);
    return os;
}

}    // namespace hpts::env::rnd
