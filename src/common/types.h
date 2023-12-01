// File: types.h
// Description: common types

#ifndef HPTS_COMMON_TYPES_H_
#define HPTS_COMMON_TYPES_H_

#include <array>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "model/model_evaluator.h"
#include "util/stop_token.h"

namespace hpts {

// using Observation = std::vector<float>;
// struct ObservationShape {
//     int c = 0;    // Number of channels NOLINT (misc-non-private-member-variable-in-classes)
//     int h = 0;    // Height of observation NOLINT (misc-non-private-member-variable-in-classes)
//     int w = 0;    // Width of observation NOLINT (misc-non-private-member-variable-in-classes)
//     ObservationShape() = default;
//     ~ObservationShape() = default;
//     ObservationShape(int c, int h, int w) : c(c), h(h), w(w) {}
//     ObservationShape(const std::array<int, 3> &shape) : c(shape[0]), h(shape[1]), w(shape[2]) {}
//     ObservationShape(const std::array<std::size_t, 3> &shape)
//         : c(static_cast<int>(shape[0])), h(static_cast<int>(shape[1])), w(static_cast<int>(shape[2])) {}
//     ObservationShape(const ObservationShape &) = default;
//     ObservationShape(ObservationShape &&) = default;
//     ObservationShape &operator=(const ObservationShape &) = default;
//     ObservationShape &operator=(ObservationShape &&) = default;
//     auto operator==(const ObservationShape &rhs) const -> bool {
//         return c == rhs.c && h == rhs.h && w == rhs.w;
//     }
//     auto operator!=(const ObservationShape &rhs) const -> bool {
//         return c != rhs.c || h != rhs.h || w != rhs.w;
//     }
//     [[nodiscard]] auto flat_size() const -> int {
//         return c * h * w;
//     }
// };

// Args should be list of required specializations of ModelEvaluator
template <typename T, typename... Args>
    requires(!std::is_pointer_v<Args> && ...)
struct SearchInput {
    SearchInput() = delete;
    SearchInput(const std::string &puzzle_name, T state, int search_budget, StopToken *stop_token, std::shared_ptr<Args>... args)
        : puzzle_name(std::move(puzzle_name)),
          state(std::move(state)),
          search_budget(search_budget),
          stop_token(stop_token),
          model_evals(std::move(args)...) {}
    // NOLINTBEGIN (misc-non-private-member-variable-in-classes)
    std::string puzzle_name;
    T state;
    int search_budget;
    StopToken *stop_token;
    std::optional<std::mt19937> rng = std::nullopt;
    std::tuple<std::shared_ptr<Args>...> model_evals;
    // NOLINTEND (misc-non-private-member-variable-in-classes)
};

}    // namespace hpts

#endif    // HPTS_COMMON_TYPES_H_
