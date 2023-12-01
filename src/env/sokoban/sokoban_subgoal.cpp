// File: sokoban_subgoal.cpp
// Description: Subgoal sokoban_cpp environment

#include "env/sokoban/sokoban_subgoal.h"

#include <absl/strings/str_format.h>

#include "env/sokoban/sokoban_base.h"

namespace hpts::env::sokoban {

namespace soko = ::sokoban;

constexpr std::size_t WIDTH = 10;
constexpr std::size_t HEIGHT = 10;

constexpr auto subgoal_to_box(std::size_t subgoal_id) -> std::size_t {
    return subgoal_id % (WIDTH * HEIGHT);
}

constexpr auto subgoal_to_goal(std::size_t subgoal_id) -> std::size_t {
    return subgoal_id / (WIDTH * HEIGHT);
}

constexpr auto to_subgoal(std::size_t box_id, std::size_t goal_idx) {
    return goal_idx * (WIDTH * HEIGHT) + box_id;
}

// ---------------------------------------------------------

auto SokobanSubgoalState::observation_shape_low() const noexcept -> ObservationShape {
    return observation_shape();
}

auto SokobanSubgoalState::observation_shape_conditional_low() const noexcept -> ObservationShape {
    const ObservationShape obs_shape = state.observation_shape();
    return {static_cast<int>(obs_shape.c) + 2, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
}

auto SokobanSubgoalState::observation_shape_subgoal() const noexcept -> ObservationShape {
    const ObservationShape obs_shape = state.observation_shape();
    // return {static_cast<int>(obs_shape.c) + 2, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
    return {static_cast<int>(obs_shape.c) + 1, static_cast<int>(obs_shape.h), static_cast<int>(obs_shape.w)};
}

// ---------------------------------------------------------

auto SokobanSubgoalState::get_subgoal_observation(std::size_t subgoal, bool remove_agent) const noexcept -> Observation {
    Observation obs = state.get_observation();
    auto obs_shape = state.observation_shape();
    const std::size_t channel_length = obs_shape[1] * obs_shape[2];
    if (remove_agent) {
        std::vector<decltype(obs)::value_type>(obs.begin() + (int)channel_length, obs.end()).swap(obs);
    }
    Observation box_channel = Observation(channel_length, 0);
    Observation goal_channel = Observation(channel_length, 0);

    std::size_t box_id = subgoal_to_box(subgoal);
    std::size_t box_index = state.get_box_index(static_cast<int>(box_id));
    std::size_t goal_index = subgoal_to_goal(subgoal);
    box_channel[box_index] = 1;
    goal_channel[goal_index] = 1;
    obs.insert(obs.end(), box_channel.begin(), box_channel.end());
    obs.insert(obs.end(), goal_channel.begin(), goal_channel.end());
    return obs;
}

auto SokobanSubgoalState::get_observation_low() const noexcept -> Observation {
    return SokobanBaseState::get_observation();
}

auto SokobanSubgoalState::get_observation_conditional_low([[maybe_unused]] std::size_t subgoal) const noexcept -> Observation {
    return get_subgoal_observation(subgoal, false);
}

auto SokobanSubgoalState::get_observation_subgoal() const noexcept -> std::vector<Observation> {
    std::vector<Observation> observations;
    for (const auto& subgoal : child_subgoals()) {
        // observations.push_back(get_observation(subgoal));
        observations.push_back(get_subgoal_observation(subgoal, true));
    }
    return observations;
}

// ---------------------------------------------------------

void SokobanSubgoalState::apply_action(std::size_t action) {
    state.apply_action(static_cast<soko::Action>(action));
    reward_signal = state.get_reward_signal();
}

auto SokobanSubgoalState::is_subgoal_done(std::size_t subgoal) const noexcept -> bool {
    // reward signal is > 0 if box gets pushed on goal
    // We implicitly add 1 in reward_signal to distinguish between 0 (not solved) vs solved
    // Need to check if box/goal combo matches the signal
    return reward_signal == (subgoal + 1);
}

auto SokobanSubgoalState::is_any_subgoal_done() const noexcept -> bool {
    return reward_signal > 0;
}

auto SokobanSubgoalState::child_subgoals() const noexcept -> const std::vector<std::size_t> {
    // all pairs of box/goals which are not already pairwise solved
    std::vector<std::size_t> subgoals;
    subgoals.reserve(4 * 4);
    for (const auto& goal_idx : state.get_all_goal_indices()) {
        for (const auto& box_id : state.get_all_box_ids()) {
            if (state.get_box_index(box_id) != goal_idx) {
                subgoals.push_back(to_subgoal(box_id, goal_idx));
            }
        }
    }
    return subgoals;
}

auto SokobanSubgoalState::subgoal_to_str(std::size_t subgoal) const noexcept -> std::string {
    std::size_t box_id = subgoal_to_box(subgoal);
    std::size_t box_index = state.get_box_index(static_cast<int>(box_id));
    return absl::StrFormat("(B: %d, G: %d)", box_index, subgoal_to_goal(subgoal));
}

[[nodiscard]] auto SokobanSubgoalState::to_image() const -> std::vector<uint8_t> {
    return state.to_image();
}

auto SokobanSubgoalState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto operator<<(std::ostream& os, const SokobanSubgoalState& state) -> std::ostream& {
    os << state.state;
    os << std::endl << state.reward_signal;
    return os;
}

}    // namespace hpts::env::sokoban
