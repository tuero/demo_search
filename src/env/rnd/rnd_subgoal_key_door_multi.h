// File: rnd_subgoal_key_door_multi.h
// Description: rnd environment with key/door subgoals as separate independent obvservations

#ifndef HPTS_ENV_RND_SUBGOAL_KEY_DOOR_MULTI_H_
#define HPTS_ENV_RND_SUBGOAL_KEY_DOOR_MULTI_H_

#include <functional>

#include "env/rnd/rnd_subgoal_key_door_base.h"

namespace hpts::env::rnd {

class RNDSubgoalKeyDoorMultiState : public RNDSubgoalKeyDoorBaseState {
public:
    using RNDSubgoalKeyDoorBaseState::RNDSubgoalKeyDoorBaseState;

    [[nodiscard]] auto observation_shape_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_conditional_low() const noexcept -> ObservationShape;
    [[nodiscard]] auto observation_shape_subgoal() const noexcept -> ObservationShape;

    [[nodiscard]] auto get_observation_low() const noexcept -> Observation;
    [[nodiscard]] auto get_observation_conditional_low(std::size_t subgoal) const noexcept -> Observation;
    [[nodiscard]] auto get_observation_subgoal() const noexcept -> Observation;

    friend auto operator<<(std::ostream &os, const RNDSubgoalKeyDoorMultiState &state) -> std::ostream &;

    inline static const std::string name{"rnd_subgoal_key_door_multi"};
};

}    // namespace hpts::env::rnd

namespace std {
template <>
struct hash<hpts::env::rnd::RNDSubgoalKeyDoorMultiState> {
    size_t operator()(const hpts::env::rnd::RNDSubgoalKeyDoorMultiState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_RND_SUBGOAL_KEY_DOOR_MULTI_H_
