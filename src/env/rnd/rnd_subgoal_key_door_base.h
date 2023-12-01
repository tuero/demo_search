// File: rnd_subgoal_key_door_base.h
// Description: rnd environment with key/door subgoals

#ifndef HPTS_ENV_RND_SUBGOAL_KEY_DOOR_BASE_H_
#define HPTS_ENV_RND_SUBGOAL_KEY_DOOR_BASE_H_

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "env/rnd/rnd_simple.h"

namespace hpts::env::rnd {

class RNDSubgoalKeyDoorBaseState : public RNDSimpleState {
public:
    using RNDSimpleState::RNDSimpleState;

    [[nodiscard]] auto is_subgoal_done(std::size_t subgoal) const noexcept -> bool;
    [[nodiscard]] auto is_any_subgoal_done() const noexcept -> bool;

    void apply_action(std::size_t action) override;
    [[nodiscard]] auto child_subgoals() const noexcept -> const std::vector<std::size_t>;
    auto subgoal_to_str(std::size_t subgoal) const noexcept -> std::string;

    auto print(std::ostream &os) const -> std::ostream &;
    [[nodiscard]] auto to_str() const -> std::string override;

    static constexpr int num_actions = 4;
    static constexpr int num_subgoals = 6;

protected:
    enum class Subgoal {
        WalkThroughExit = 0,
        CollectDiamond = 1,
        CollectKeyRed = 2,
        CollectKeyBlue = 3,
        CollectKeyGreen = 4,
        CollectKeyYellow = 5,
    };

    static const std::vector<std::size_t> ALL_SUBGOALS;
    static const absl::flat_hash_map<Subgoal, stonesngems::RewardCodes> SUBGOAL_SIGNAL_MAP;
    static const absl::flat_hash_map<Subgoal, std::string> SUBGOAL_STR_MAP;
    static const absl::flat_hash_map<Subgoal, absl::flat_hash_set<stonesngems::HiddenCellType>> SUBGOAL_TYPE_MAP;

    uint64_t reward_signal;    // NOLINT(*-non-private-member-variables-in-classes)
};

}    // namespace hpts::env::rnd

#endif    // HPTS_ENV_RND_SUBGOAL_KEY_DOOR_BASE_H_
