// File: sokoban_base.h
// Description: Base wrapper around sokoban_cpp standalone environment
#ifndef HPTS_ENV_SOKOBAN_BASE_H_
#define HPTS_ENV_SOKOBAN_BASE_H_

#include <sokoban/sokoban.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts::env::sokoban {

class SokobanBaseState {
public:
    SokobanBaseState(const std::string &board_str);
    virtual ~SokobanBaseState() = default;

    SokobanBaseState(const SokobanBaseState &) noexcept = default;
    SokobanBaseState(SokobanBaseState &&) noexcept = default;
    auto operator=(const SokobanBaseState &) noexcept -> SokobanBaseState & = default;
    auto operator=(SokobanBaseState &&) noexcept -> SokobanBaseState & = default;

    virtual void apply_action(std::size_t action);

    [[nodiscard]] virtual auto child_actions() const noexcept -> const std::vector<std::size_t> &;
    [[nodiscard]] virtual auto get_observation() const noexcept -> Observation;
    [[nodiscard]] virtual auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] virtual auto is_solution() const noexcept -> bool;
    [[nodiscard]] virtual auto is_terminal() const noexcept -> bool;
    [[nodiscard]] virtual auto get_heuristic() const noexcept -> double;
    [[nodiscard]] virtual auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] virtual auto to_str() const -> std::string;
    [[nodiscard]] auto operator==(const SokobanBaseState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const SokobanBaseState &state) -> std::ostream &;

    inline static const std::string name{"sokoban"};
    inline static const int num_actions = 4;

protected:
    ::sokoban::SokobanGameState state;    // NOLINT
};

}    // namespace hpts::env::sokoban

namespace std {
template <>
struct hash<hpts::env::sokoban::SokobanBaseState> {
    size_t operator()(const hpts::env::sokoban::SokobanBaseState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_SOKOBAN_BASE_H_
