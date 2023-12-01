// File: boxworld_subgoal_variable.h
// Description: Subgoal boxworld_cpp environment

#ifndef HPTS_ENV_BOXWORLD_SUBGOAL_VARIABLE_H_
#define HPTS_ENV_BOXWORLD_SUBGOAL_VARIABLE_H_

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "env/boxworld/boxworld_base.h"

namespace hpts::env::bw {

class BoxWorldSubgoalVariableState : public BoxWorldBaseState {
public:
    using BoxWorldBaseState::BoxWorldBaseState;

    [[nodiscard]] auto observation_shape_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_conditional_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_subgoal() const noexcept -> ObservationShape ;

    [[nodiscard]] auto get_observation_low() const noexcept -> Observation;
    [[nodiscard]] auto get_observation_conditional_low(std::size_t subgoal) const noexcept -> Observation;
    [[nodiscard]] auto get_observation_subgoal() const noexcept -> std::vector<Observation> ;

    [[nodiscard]] auto is_subgoal_done(std::size_t subgoal) const noexcept -> bool;
    [[nodiscard]] auto is_any_subgoal_done() const noexcept -> bool;

    void apply_action(std::size_t action) override;
    [[nodiscard]] auto child_subgoals() const noexcept -> const std::vector<std::size_t>;
    [[nodiscard]] auto to_str() const -> std::string override;
    [[nodiscard]] auto subgoal_to_str(std::size_t subgoal) const noexcept -> std::string;

    friend auto operator<<(std::ostream &os, const BoxWorldSubgoalVariableState &state) -> std::ostream &;

    inline static const std::string name{"boxworld_subgoal_variable"};
    inline static const int num_actions = 4;
    inline static const int num_subgoals = boxworld::kNumColours - 1;

private:
    auto get_subgoal_observation(std::size_t subgoal) const noexcept -> Observation;
    uint64_t reward_signal = 0;
};

}    // namespace hpts::env::bw

namespace std {
template <>
struct hash<hpts::env::bw::BoxWorldSubgoalVariableState> {
    size_t operator()(const hpts::env::bw::BoxWorldSubgoalVariableState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_BOXWORLD_SUBGOAL_MULTI_H_
