// File: sokoban_base.cpp
// Description: Base wrapper around sokoban_cpp standalone environment

#include "env/sokoban/sokoban_base.h"

#include <sstream>

namespace hpts::env::sokoban {

namespace soko = ::sokoban;

soko::GameParameters init_params(const std::string &board_str) {
    soko::GameParameters params = soko::kDefaultGameParams;
    params["game_board_str"] = soko::GameParameter(board_str);
    return params;
}

SokobanBaseState::SokobanBaseState(const std::string &board_str) : state(init_params(board_str)) {}

void SokobanBaseState::apply_action(std::size_t action) {
    state.apply_action(static_cast<soko::Action>(action));
}

const std::vector<std::size_t> all_actions = {0, 1, 2, 3};
auto SokobanBaseState::child_actions() const noexcept -> const std::vector<std::size_t> & {
    return all_actions;
}

auto SokobanBaseState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto SokobanBaseState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto SokobanBaseState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto SokobanBaseState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto SokobanBaseState::get_heuristic() const noexcept -> double {
    return 0;
}

auto SokobanBaseState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto SokobanBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto SokobanBaseState::operator==(const SokobanBaseState &rhs) const -> bool {
    return state == rhs.state;
}
auto operator<<(std::ostream &os, const SokobanBaseState &state) -> std::ostream & {
    os << state.state;
    return os;
}

}    // namespace hpts::env::sokoban
