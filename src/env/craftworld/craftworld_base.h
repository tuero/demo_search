// File: craftworld_base.h
// Description: Base wrapper around craftworld_cpp standalone environment
#ifndef HPTS_ENV_CRAFTWORLD_BASE_H_
#define HPTS_ENV_CRAFTWORLD_BASE_H_

#include <craftworld/craftworld.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts::env::cw {

class CraftWorldBaseState {
public:
    CraftWorldBaseState(const std::string &board_str);
    virtual ~CraftWorldBaseState() = default;

    CraftWorldBaseState(const CraftWorldBaseState &) noexcept = default;
    CraftWorldBaseState(CraftWorldBaseState &&) noexcept = default;
    auto operator=(const CraftWorldBaseState &) noexcept -> CraftWorldBaseState & = default;
    auto operator=(CraftWorldBaseState &&) noexcept -> CraftWorldBaseState & = default;

    virtual void apply_action(std::size_t action);

    [[nodiscard]] virtual auto child_actions() const noexcept -> const std::vector<std::size_t> &;
    [[nodiscard]] virtual auto get_observation() const noexcept -> Observation;
    [[nodiscard]] virtual auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] virtual auto is_solution() const noexcept -> bool;
    [[nodiscard]] virtual auto is_terminal() const noexcept -> bool;
    [[nodiscard]] virtual auto get_heuristic() const noexcept -> double;
    [[nodiscard]] virtual auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] virtual auto to_str() const -> std::string;
    [[nodiscard]] auto operator==(const CraftWorldBaseState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const CraftWorldBaseState &state) -> std::ostream &;

    inline static const std::string name{"craftworld"};
    inline static const int num_actions = craftworld::kNumActions;

protected:
    craftworld::CraftWorldGameState state;    // NOLINT
};

}    // namespace hpts::env::cw

namespace std {
template <>
struct hash<hpts::env::cw::CraftWorldBaseState> {
    size_t operator()(const hpts::env::cw::CraftWorldBaseState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_CRAFTWORLD_BASE_H_
