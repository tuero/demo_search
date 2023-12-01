// File: boxworld_subgoal_visual.cpp
// Description: Subgoal boxworld_cpp environment

#include "env/boxworld/boxworld_subgoal_visual.h"

#include <absl/strings/str_format.h>
#include <algorithm>

namespace hpts::env::bw {

// ---------------------------------------------------------

auto BoxWorldSubgoalVisualState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto BoxWorldSubgoalVisualState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    const ObservationShape obs_shape = state.observation_shape_environment();
    return {static_cast<int>(obs_shape.c) + 1, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
}

auto BoxWorldSubgoalVisualState::observation_shape_subgoal() const noexcept -> ObservationShape {
    return observation_shape();
}

// ---------------------------------------------------------

auto BoxWorldSubgoalVisualState::get_subgoal_observation(std::size_t subgoal) const noexcept -> Observation {
    Observation obs = state.get_observation();
    const ObservationShape obs_shape = state.observation_shape();
    Observation subgoal_channel = Observation(obs_shape.w * obs_shape.h, 0);
    subgoal_channel[subgoal] = 1;
    obs.insert(obs.end(), subgoal_channel.begin(), subgoal_channel.end());
    return obs;
}

auto BoxWorldSubgoalVisualState::get_observation_low() const noexcept -> Observation {
    return BoxWorldBaseState::get_observation();
}

auto BoxWorldSubgoalVisualState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept -> Observation {
    Observation obs = state.get_observation_environment();
    const ObservationShape obs_shape = state.observation_shape_environment();
    Observation subgoal_channel = Observation(obs_shape.w * obs_shape.h, 0);
    for (const auto & idx : state.get_indices(static_cast<boxworld::Element>(subgoal))) {
        subgoal_channel[idx] = 1;
    }
    obs.insert(obs.end(), subgoal_channel.begin(), subgoal_channel.end());
    return obs;
}

auto BoxWorldSubgoalVisualState::get_observation_subgoal() const noexcept -> Observation {
    return BoxWorldBaseState::get_observation();
}

// ---------------------------------------------------------

void BoxWorldSubgoalVisualState::apply_action(std::size_t action) {
    state.apply_action(static_cast<boxworld::Action>(action));
    reward_signal = state.get_reward_signal(true);
}

auto BoxWorldSubgoalVisualState::is_subgoal_done(std::size_t subgoal) const noexcept -> bool {
    // reward signal is > 0 representing index agent collects target
    // We implicitly add 1 in reward_signal to distinguish between 0 (not solved) vs solved
    return reward_signal == (subgoal + 1);
}

auto BoxWorldSubgoalVisualState::is_any_subgoal_done() const noexcept -> bool {
    return reward_signal > 0;
}

auto create_vec(std::size_t n) {
    std::vector<std::size_t> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}
const std::vector<std::size_t> ALL_SUBGOALS = create_vec(BoxWorldSubgoalVisualState::num_subgoals);
auto BoxWorldSubgoalVisualState::child_subgoals() const noexcept -> const std::vector<std::size_t> {
    return ALL_SUBGOALS;
}

auto BoxWorldSubgoalVisualState::subgoal_to_str(std::size_t subgoal) const noexcept -> std::string {
    return absl::StrFormat("idx: %d, color: %s", subgoal, state.get_item_str(subgoal));
}

auto BoxWorldSubgoalVisualState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto operator<<(std::ostream& os, const BoxWorldSubgoalVisualState& state) -> std::ostream& {
    os << state.state;
    os << std::endl << state.reward_signal;
    return os;
}
}    // namespace hpts::env::bw
