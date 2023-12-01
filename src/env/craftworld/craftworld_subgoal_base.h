// File: craftworld_subgoal_base.h
// Description: Base wrapper around craftworld_cpp for subgoals

#ifndef HPTS_ENV_CRAFTWORLD_SUBGOAL_BASE_H_
#define HPTS_ENV_CRAFTWORLD_SUBGOAL_BASE_H_

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include "env/craftworld/craftworld_base.h"

namespace hpts::env::cw {

class CraftWorldSubgoalBaseState : public CraftWorldBaseState {
public:
    using CraftWorldBaseState::CraftWorldBaseState;

    [[nodiscard]] auto is_subgoal_done(std::size_t subgoal) const noexcept -> bool;
    [[nodiscard]] auto is_any_subgoal_done() const noexcept -> bool;

    void apply_action(std::size_t action) override;
    [[nodiscard]] auto child_subgoals() const noexcept -> const std::vector<std::size_t>;
    auto subgoal_to_str(std::size_t subgoal) const noexcept -> std::string;

    auto print(std::ostream &os) const -> std::ostream &;
    [[nodiscard]] auto to_str() const -> std::string override;

    static constexpr int num_subgoals = 9;

protected:
    static const std::vector<std::size_t> ALL_SUBGOALS;
    static const absl::flat_hash_map<craftworld::Subgoal, absl::flat_hash_set<craftworld::Element>> SUBGOAL_TYPE_MAP;
    static const absl::flat_hash_map<std::size_t, craftworld::Subgoal> SUBGOAL_MAP;

    uint64_t reward_signal;    // NOLINT(*-non-private-member-variables-in-classes)
};

}    // namespace hpts::env::cw

#endif    // HPTS_ENV_CRAFTWORLD_SUBGOAL_BASE_H_
