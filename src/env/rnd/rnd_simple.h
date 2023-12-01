// File: rnd_simple.h
// Description: rnd environment with no gravity and reduced action set

#ifndef HPTS_ENV_RND_SIMPLE_H_
#define HPTS_ENV_RND_SIMPLE_H_

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "env/rnd/rnd_base.h"

namespace hpts::env::rnd {

class RNDSimpleState : public RNDBaseState {
public:
    RNDSimpleState(const std::string &board_str);
    ~RNDSimpleState() override = default;

    RNDSimpleState(const RNDSimpleState &) noexcept = default;
    RNDSimpleState(RNDSimpleState &&) noexcept = default;
    auto operator=(const RNDSimpleState &) noexcept -> RNDSimpleState & = default;
    auto operator=(RNDSimpleState &&) noexcept -> RNDSimpleState & = default;

    void apply_action(std::size_t action) override;
    [[nodiscard]] auto child_actions() const noexcept -> const std::vector<std::size_t> & override;

    friend auto operator<<(std::ostream &os, const RNDSimpleState &state) -> std::ostream &;

    inline static const std::string name{"rnd_simple"};
    inline static const int num_actions = 4;
};

}    // namespace hpts::env::rnd

namespace std {
template <>
struct hash<hpts::env::rnd::RNDSimpleState> {
    size_t operator()(const hpts::env::rnd::RNDSimpleState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_RND_SIMPLE_H_
