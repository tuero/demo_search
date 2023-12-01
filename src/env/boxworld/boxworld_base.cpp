// File: boxworld_base.cpp
// Description: Base wrapper around boxworld_cpp standalone environment

#include "env/boxworld/boxworld_base.h"

#include <sstream>

namespace hpts::env::bw {

namespace {

 boxworld::GameParameters init_params(const std::string &board_str) {
    boxworld::GameParameters params = boxworld::kDefaultGameParams;
    params["game_board_str"] = boxworld::GameParameter(board_str);
    return params;
}

}    // namespace

BoxWorldBaseState::BoxWorldBaseState(const std::string &board_str) : state(init_params(board_str)) {}

void BoxWorldBaseState::apply_action(std::size_t action) {
    state.apply_action(static_cast<boxworld::Action>(action));
}

const std::vector<std::size_t> all_actions = {0, 1, 2, 3};

auto BoxWorldBaseState::child_actions() const noexcept -> const std::vector<std::size_t> & {
    return all_actions;
}

auto BoxWorldBaseState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto BoxWorldBaseState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto BoxWorldBaseState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto BoxWorldBaseState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto BoxWorldBaseState::get_heuristic() const noexcept -> double {
    return 0;
}

auto BoxWorldBaseState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto BoxWorldBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto BoxWorldBaseState::operator==(const BoxWorldBaseState &rhs) const -> bool {
    return state == rhs.state;
}
auto operator<<(std::ostream &os, const BoxWorldBaseState &state) -> std::ostream & {
    os << state.state;
    return os;
}

}    // namespace hpts::env::bw
