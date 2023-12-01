// File: search_input.h
// Description: Common search algorithm input type

#ifndef HPTS_COMMON_SEARCH_INPUT_H_
#define HPTS_COMMON_SEARCH_INPUT_H_

#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

#include "util/stop_token.h"

namespace hpts {

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
