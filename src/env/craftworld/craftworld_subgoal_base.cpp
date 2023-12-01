// File: craftworld_subgoal_base.h
// Description: Base wrapper around craftworld_cpp for subgoals

#include "env/craftworld/craftworld_subgoal_base.h"

namespace hpts::env::cw {

using namespace craftworld;

const absl::flat_hash_map<std::size_t, Subgoal> CraftWorldSubgoalBaseState::SUBGOAL_MAP{
    {0, Subgoal::kCollectTin},  {1, Subgoal::kCollectCopper}, {2, Subgoal::kCollectWood},
    {3, Subgoal::kCollectIron}, {4, Subgoal::kCollectGem},    {5, Subgoal::kUseStation1},
    {6, Subgoal::kUseStation2}, {7, Subgoal::kUseStation3},   {8, Subgoal::kUseFurnace},
};

const std::vector<std::size_t> CraftWorldSubgoalBaseState::ALL_SUBGOALS{0, 1, 2, 3, 4, 5, 6, 7, 8};

const absl::flat_hash_map<Subgoal, uint64_t> SUBGOAL_SIGNAL_MAP{
    {Subgoal::kCollectTin, static_cast<uint64_t>(RewardCode::kRewardCodeCollectTin)},
    {Subgoal::kCollectCopper, static_cast<uint64_t>(RewardCode::kRewardCodeCollectCopper)},
    {Subgoal::kCollectWood, static_cast<uint64_t>(RewardCode::kRewardCodeCollectWood)},
    {Subgoal::kCollectIron, static_cast<uint64_t>(RewardCode::kRewardCodeCollectIron)},
    {Subgoal::kCollectGem, static_cast<uint64_t>(RewardCode::kRewardCodeCollectGem)},
    {Subgoal::kUseStation1, static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation1)},
    {Subgoal::kUseStation2, static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation2)},
    {Subgoal::kUseStation3, static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation3)},
    {Subgoal::kUseFurnace, static_cast<uint64_t>(RewardCode::kRewardCodeUseAtFurnace)},
};

const absl::flat_hash_map<Subgoal, absl::flat_hash_set<Element>> CraftWorldSubgoalBaseState::SUBGOAL_TYPE_MAP = {
    {Subgoal::kCollectTin, {Element::kTin}},        {Subgoal::kCollectCopper, {Element::kCopper}},
    {Subgoal::kCollectWood, {Element::kWood}},      {Subgoal::kCollectIron, {Element::kIron}},
    {Subgoal::kCollectGem, {Element::kGem}},        {Subgoal::kUseStation1, {Element::kWorkshop1}},
    {Subgoal::kUseStation2, {Element::kWorkshop2}}, {Subgoal::kUseStation3, {Element::kWorkshop3}},
    {Subgoal::kUseFurnace, {Element::kFurnace}},
};
constexpr uint64_t SIGNAL_MASK =
    static_cast<uint64_t>(RewardCode::kRewardCodeCollectTin) | static_cast<uint64_t>(RewardCode::kRewardCodeCollectCopper) |
    static_cast<uint64_t>(RewardCode::kRewardCodeCollectWood) | static_cast<uint64_t>(RewardCode::kRewardCodeCollectIron) |
    static_cast<uint64_t>(RewardCode::kRewardCodeCollectGem) | static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation1) |
    static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation2) |
    static_cast<uint64_t>(RewardCode::kRewardCodeUseAtWorkstation3) | static_cast<uint64_t>(RewardCode::kRewardCodeUseAtFurnace);

// ---------------------------------------------------------

auto CraftWorldSubgoalBaseState::is_subgoal_done(std::size_t subgoal) const noexcept -> bool {
    return reward_signal & SUBGOAL_SIGNAL_MAP.at(SUBGOAL_MAP.at(subgoal));
}

auto CraftWorldSubgoalBaseState::is_any_subgoal_done() const noexcept -> bool {
    return reward_signal > 0;
}

// ---------------------------------------------------------

void CraftWorldSubgoalBaseState::apply_action(std::size_t action) {
    CraftWorldBaseState::apply_action(action);
    reward_signal = state.get_reward_signal() & SIGNAL_MASK;
}

auto CraftWorldSubgoalBaseState::child_subgoals() const noexcept -> const std::vector<std::size_t> {
    return ALL_SUBGOALS;
}

auto CraftWorldSubgoalBaseState::subgoal_to_str(std::size_t subgoal) const noexcept -> std::string {
    return std::to_string(subgoal);
}

// ---------------------------------------------------------

auto CraftWorldSubgoalBaseState::to_str() const -> std::string {
    std::ostringstream stream;
    print(stream);
    return stream.str();
}

auto CraftWorldSubgoalBaseState::print(std::ostream &os) const -> std::ostream & {
    os << state;
    return os;
}

}    // namespace hpts::env::cw
