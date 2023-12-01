// File: rnd_base.h
// Description: Base wrapper around stonesngems_cpp standalone environment

#include "env/rnd/rnd_base.h"

#include <sstream>

namespace hpts::env::rnd {

using namespace stonesngems;

namespace {

GameParameters init_params(const std::string &board_str) {
    GameParameters params = kDefaultGameParams;
    params["game_board_str"] = GameParameter(board_str);
    params["gravity"] = GameParameter(true);
    return params;
}

}    // namespace

RNDBaseState::RNDBaseState(const std::string &board_str) : state(init_params(board_str)) {}

RNDBaseState::RNDBaseState(const stonesngems::GameParameters &params) : state(params) {}

void RNDBaseState::apply_action(std::size_t action) {
    state.apply_action(static_cast<Action>(action));
}

const std::vector<std::size_t> all_actions = {0, 1, 2, 3, 4};

auto RNDBaseState::child_actions() const noexcept -> const std::vector<std::size_t> & {
    return all_actions;
}

auto RNDBaseState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto RNDBaseState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto RNDBaseState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto RNDBaseState::is_terminal() const noexcept -> bool {
    return state.is_terminal();
}

auto RNDBaseState::get_heuristic() const noexcept -> double {
    return 0;
}

auto RNDBaseState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto RNDBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

auto RNDBaseState::operator==(const RNDBaseState &rhs) const -> bool {
    return state == rhs.state;
}
auto operator<<(std::ostream &os, const RNDBaseState &state) -> std::ostream & {
    os << state.state;
    return os;
}

}    // namespace hpts::env::rnd
