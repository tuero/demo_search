// File: boxworld_subgoal_variable.cpp
// Description: Subgoal boxworld_cpp environment

#include "env/boxworld/boxworld_subgoal_variable.h"

#include <absl/strings/str_format.h>

namespace hpts::env::bw {

// ---------------------------------------------------------

auto BoxWorldSubgoalVariableState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto BoxWorldSubgoalVariableState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    const ObservationShape obs_shape = state.observation_shape_environment();
    return {static_cast<int>(obs_shape.c) + 1, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
}

auto BoxWorldSubgoalVariableState::observation_shape_subgoal() const noexcept -> ObservationShape {
    const ObservationShape obs_shape = state.observation_shape();
    return {static_cast<int>(obs_shape.c) + 1, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
}

// ---------------------------------------------------------

auto BoxWorldSubgoalVariableState::get_subgoal_observation(std::size_t subgoal) const noexcept -> Observation {
    Observation obs = state.get_observation();
    const ObservationShape obs_shape = state.observation_shape();
    Observation subgoal_channel = Observation(obs_shape.w * obs_shape.h, 0);
    subgoal_channel[subgoal] = 1;
    obs.insert(obs.end(), subgoal_channel.begin(), subgoal_channel.end());
    return obs;
}

auto BoxWorldSubgoalVariableState::get_observation_low() const noexcept -> Observation {
    return BoxWorldBaseState::get_observation();
}

auto BoxWorldSubgoalVariableState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept -> Observation {
    Observation obs = state.get_observation_environment();
    const ObservationShape obs_shape = state.observation_shape();
    Observation subgoal_channel = Observation(obs_shape.w * obs_shape.h, 0);
    subgoal_channel[subgoal] = 1;
    obs.insert(obs.end(), subgoal_channel.begin(), subgoal_channel.end());
    return obs;
}

auto BoxWorldSubgoalVariableState::get_observation_subgoal() const noexcept -> std::vector<Observation>  {
    std::vector<Observation> observations;
    for (const auto& subgoal : child_subgoals()) {
        observations.push_back(get_subgoal_observation(subgoal));
    }
    return observations;
}

// ---------------------------------------------------------

void BoxWorldSubgoalVariableState::apply_action(std::size_t action) {
    state.apply_action(static_cast<boxworld::Action>(action));
    reward_signal = state.get_reward_signal();
}

auto BoxWorldSubgoalVariableState::is_subgoal_done(std::size_t subgoal) const noexcept -> bool {
    // reward signal is > 0 representing index agent collects target
    // We implicitly add 1 in reward_signal to distinguish between 0 (not solved) vs solved
    return reward_signal == (subgoal + 1);
}

auto BoxWorldSubgoalVariableState::is_any_subgoal_done() const noexcept -> bool {
    return reward_signal > 0;
}

const std::vector<std::size_t> ALL_SUBGOALS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
auto BoxWorldSubgoalVariableState::child_subgoals() const noexcept -> const std::vector<std::size_t> {
    // Single keys + lock indices
    return state.get_target_indices();
}

auto BoxWorldSubgoalVariableState::subgoal_to_str(std::size_t subgoal) const noexcept -> std::string {
    return std::to_string(subgoal);
}

auto BoxWorldSubgoalVariableState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto operator<<(std::ostream& os, const BoxWorldSubgoalVariableState& state) -> std::ostream& {
    os << state.state;
    os << std::endl << state.reward_signal;
    return os;
}
}    // namespace hpts::env::bw
