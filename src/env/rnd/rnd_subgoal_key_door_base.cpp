// File: rnd_subgoal_key_door_base.cpp
// Description: rnd environment with key/door subgoals

#include "env/rnd/rnd_subgoal_key_door_base.h"

namespace hpts::env::rnd {

using namespace stonesngems;

const std::vector<std::size_t> RNDSubgoalKeyDoorBaseState::ALL_SUBGOALS = {
    static_cast<std::size_t>(Subgoal::WalkThroughExit), static_cast<std::size_t>(Subgoal::CollectDiamond),
    static_cast<std::size_t>(Subgoal::CollectKeyRed),   static_cast<std::size_t>(Subgoal::CollectKeyBlue),
    static_cast<std::size_t>(Subgoal::CollectKeyGreen), static_cast<std::size_t>(Subgoal::CollectKeyYellow),
};

const absl::flat_hash_map<RNDSubgoalKeyDoorBaseState::Subgoal, RewardCodes> RNDSubgoalKeyDoorBaseState::SUBGOAL_SIGNAL_MAP{
    {Subgoal::WalkThroughExit, RewardCodes::kRewardWalkThroughExit},
    {Subgoal::CollectDiamond, RewardCodes::kRewardCollectDiamond},
    {Subgoal::CollectKeyRed, RewardCodes::kRewardCollectKeyRed},
    {Subgoal::CollectKeyBlue, RewardCodes::kRewardCollectKeyBlue},
    {Subgoal::CollectKeyGreen, RewardCodes::kRewardCollectKeyGreen},
    {Subgoal::CollectKeyYellow, RewardCodes::kRewardCollectKeyYellow},
};
const absl::flat_hash_map<RNDSubgoalKeyDoorBaseState::Subgoal, std::string> RNDSubgoalKeyDoorBaseState::SUBGOAL_STR_MAP{
    {Subgoal::WalkThroughExit, "Exit"},    {Subgoal::CollectDiamond, "Diamond"},    {Subgoal::CollectKeyRed, "Red Key"},
    {Subgoal::CollectKeyBlue, "Blue Key"}, {Subgoal::CollectKeyGreen, "Green Key"}, {Subgoal::CollectKeyYellow, "Yellow Key"},
};
const absl::flat_hash_map<RNDSubgoalKeyDoorBaseState::Subgoal, absl::flat_hash_set<HiddenCellType>>
    RNDSubgoalKeyDoorBaseState::SUBGOAL_TYPE_MAP = {
        {Subgoal::WalkThroughExit, {HiddenCellType::kExitOpen}}, {Subgoal::CollectDiamond, {HiddenCellType::kDiamond}},
        {Subgoal::CollectKeyRed, {HiddenCellType::kKeyRed}},     {Subgoal::CollectKeyBlue, {HiddenCellType::kKeyBlue}},
        {Subgoal::CollectKeyGreen, {HiddenCellType::kKeyGreen}}, {Subgoal::CollectKeyYellow, {HiddenCellType::kKeyYellow}},
};
constexpr uint64_t SIGNAL_MASK = RewardCodes::kRewardWalkThroughExit | RewardCodes::kRewardCollectDiamond |
                                 RewardCodes::kRewardCollectKeyRed | RewardCodes::kRewardCollectKeyBlue |
                                 RewardCodes::kRewardCollectKeyGreen | RewardCodes::kRewardCollectKeyYellow;

// ---------------------------------------------------------

auto RNDSubgoalKeyDoorBaseState::is_subgoal_done(std::size_t subgoal) const noexcept -> bool {
    return reward_signal & SUBGOAL_SIGNAL_MAP.at(static_cast<Subgoal>(subgoal));
}

auto RNDSubgoalKeyDoorBaseState::is_any_subgoal_done() const noexcept -> bool {
    return reward_signal > 0;
}

// ---------------------------------------------------------

void RNDSubgoalKeyDoorBaseState::apply_action(std::size_t action) {
    state.apply_action(static_cast<Action>(action + 1));
    reward_signal = state.get_reward_signal() & SIGNAL_MASK;
}

auto RNDSubgoalKeyDoorBaseState::child_subgoals() const noexcept -> const std::vector<std::size_t> {
    return ALL_SUBGOALS;
}

auto RNDSubgoalKeyDoorBaseState::subgoal_to_str(std::size_t subgoal) const noexcept -> std::string {
    return std::to_string(subgoal);
}

// ---------------------------------------------------------

auto RNDSubgoalKeyDoorBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    print(stream);
    return stream.str();
}

auto RNDSubgoalKeyDoorBaseState::print(std::ostream &os) const -> std::ostream & {
    os << state;
    return os;
}

}    // namespace hpts::env::rnd
