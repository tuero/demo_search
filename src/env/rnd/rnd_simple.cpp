// File: rnd_simple.cpp
// Description: rnd environment with no gravity and reduced action set

#include "env/rnd/rnd_simple.h"

namespace hpts::env::rnd {

using namespace stonesngems;

namespace {

stonesngems::GameParameters init_params(const std::string &board_str) {
    stonesngems::GameParameters params = kDefaultGameParams;
    params["game_board_str"] = GameParameter(board_str);
    params["gravity"] = stonesngems::GameParameter(false);
    return params;
}

}    // namespace

RNDSimpleState::RNDSimpleState(const std::string &board_str) : RNDBaseState(init_params(board_str)) {}

void RNDSimpleState::apply_action(std::size_t action) {
    state.apply_action(static_cast<Action>(action + 1));
}

const std::vector<std::size_t> all_actions = {0, 1, 2, 3};
auto RNDSimpleState::child_actions() const noexcept -> const std::vector<std::size_t> & {
    return all_actions;
}

auto operator<<(std::ostream &os, const RNDSimpleState &state) -> std::ostream & {
    os << state.state;
    return os;
}

}    // namespace hpts::env::rnd
