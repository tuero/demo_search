// File: craftworld_subgoal_visual.h
// Description: craftworld environment with visual subgoals

#ifndef HPTS_ENV_CRAFTWORLD_SUBGOAL_VISUAL_H_
#define HPTS_ENV_CRAFTWORLD_SUBGOAL_VISUAL_H_

#include <functional>

#include "env/craftworld/craftworld_subgoal_base.h"

namespace hpts::env::cw {

class CraftWorldSubgoalVisualState : public CraftWorldSubgoalBaseState {
public:
    using CraftWorldSubgoalBaseState::CraftWorldSubgoalBaseState;

    [[nodiscard]] auto observation_shape_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_conditional_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_subgoal() const noexcept -> ObservationShape;

    [[nodiscard]] auto get_observation_low() const noexcept -> Observation;
    [[nodiscard]] auto get_observation_conditional_low(std::size_t subgoal) const noexcept -> Observation;
    [[nodiscard]] auto get_observation_subgoal() const noexcept -> Observation;

    friend auto operator<<(std::ostream &os, const CraftWorldSubgoalVisualState &state) -> std::ostream &;

    inline static const std::string name{"craftworld_subgoal_visual"};
};

}    // namespace hpts::env::cw

namespace std {
template <>
struct hash<hpts::env::cw::CraftWorldSubgoalVisualState> {
    size_t operator()(const hpts::env::cw::CraftWorldSubgoalVisualState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_CRAFTWORLD_SUBGOAL_VISUAL_H_
