// File: craftworld_base.cpp
// Description: Base wrapper around craftworld_cpp standalone environment

#include "env/craftworld/craftworld_base.h"

#include <sstream>

namespace hpts::env::cw {

using namespace craftworld;

namespace {

GameParameters init_params(const std::string &board_str) {
    GameParameters params = kDefaultGameParams;
    params["game_board_str"] = GameParameter(board_str);
    return params;
}

}    // namespace

CraftWorldBaseState::CraftWorldBaseState(const std::string &board_str) : state(init_params(board_str)) {}

void CraftWorldBaseState::apply_action(std::size_t action) {
    state.apply_action(static_cast<Action>(action));
}

const std::vector<std::size_t> all_actions = {0, 1, 2, 3, 4};
auto CraftWorldBaseState::child_actions() const noexcept -> const std::vector<std::size_t> & {
    return all_actions;
}

auto CraftWorldBaseState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto CraftWorldBaseState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto CraftWorldBaseState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto CraftWorldBaseState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto CraftWorldBaseState::get_heuristic() const noexcept -> double {
    return 0;
}

auto CraftWorldBaseState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto CraftWorldBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto CraftWorldBaseState::operator==(const CraftWorldBaseState &rhs) const -> bool {
    return state == rhs.state;
}
auto operator<<(std::ostream &os, const CraftWorldBaseState &state) -> std::ostream & {
    os << state.state;
    return os;
}

}    // namespace hpts::env::cw
