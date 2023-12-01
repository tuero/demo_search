// File: boxworld_base.h
// Description: Base wrapper around boxworld_cpp standalone environment
#ifndef HPTS_ENV_BOXWORLD_BASE_H_
#define HPTS_ENV_BOXWORLD_BASE_H_

#include <boxworld/boxworld.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts::env::bw {

class BoxWorldBaseState {
public:
    BoxWorldBaseState(const std::string &board_str);
    virtual ~BoxWorldBaseState() = default;

    BoxWorldBaseState(const BoxWorldBaseState &) noexcept = default;
    BoxWorldBaseState(BoxWorldBaseState &&) noexcept = default;
    auto operator=(const BoxWorldBaseState &) noexcept -> BoxWorldBaseState & = default;
    auto operator=(BoxWorldBaseState &&) noexcept -> BoxWorldBaseState & = default;

    virtual void apply_action(std::size_t action);

    [[nodiscard]] virtual auto child_actions() const noexcept -> const std::vector<std::size_t> &;
    [[nodiscard]] virtual auto get_observation() const noexcept -> Observation;
    [[nodiscard]] virtual auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] virtual auto is_solution() const noexcept -> bool;
    [[nodiscard]] virtual auto is_terminal() const noexcept -> bool;
    [[nodiscard]] virtual auto get_heuristic() const noexcept -> double;
    [[nodiscard]] virtual auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] virtual auto to_str() const -> std::string;
    [[nodiscard]] auto operator==(const BoxWorldBaseState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const BoxWorldBaseState &state) -> std::ostream &;

    inline static const std::string name{"boxworld"};
    inline static const int num_actions = 4;

protected:
    boxworld::BoxWorldGameState state;    // NOLINT
};

}    // namespace hpts::env::bw

namespace std {
template <>
struct hash<hpts::env::bw::BoxWorldBaseState> {
    size_t operator()(const hpts::env::bw::BoxWorldBaseState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_BOXWORLD_BASE_H_
