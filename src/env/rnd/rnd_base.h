// File: rnd_base.h
// Description: Base wrapper around stonesngems_cpp standalone environment
#ifndef HPTS_ENV_RND_BASE_H_
#define HPTS_ENV_RND_BASE_H_

#include <rnd/stonesngems.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts::env::rnd {

class RNDBaseState {
public:
    RNDBaseState(const std::string &board_str);
    RNDBaseState(const stonesngems::GameParameters &params);
    virtual ~RNDBaseState() = default;

    RNDBaseState(const RNDBaseState &) noexcept = default;
    RNDBaseState(RNDBaseState &&) noexcept = default;
    auto operator=(const RNDBaseState &) noexcept -> RNDBaseState & = default;
    auto operator=(RNDBaseState &&) noexcept -> RNDBaseState & = default;

    virtual void apply_action(std::size_t action);

    [[nodiscard]] virtual auto child_actions() const noexcept -> const std::vector<std::size_t> &;
    [[nodiscard]] virtual auto get_observation() const noexcept -> Observation;
    [[nodiscard]] virtual auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] virtual auto is_solution() const noexcept -> bool;
    [[nodiscard]] virtual auto is_terminal() const noexcept -> bool;
    [[nodiscard]] virtual auto get_heuristic() const noexcept -> double;
    [[nodiscard]] virtual auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] virtual auto to_str() const -> std::string;
    [[nodiscard]] auto operator==(const RNDBaseState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const RNDBaseState &state) -> std::ostream &;

    inline static const std::string name{"rnd"};
    inline static const int num_actions = 5;

protected:
    stonesngems::RNDGameState state;    // NOLINT
};

}    // namespace hpts::env::rnd

namespace std {
template <>
struct hash<hpts::env::rnd::RNDBaseState> {
    size_t operator()(const hpts::env::rnd::RNDBaseState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_RND_BASE_H_
