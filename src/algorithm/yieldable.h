// File: yieldable.h
// Description: Yieldable concept

#ifndef HPTS_ALGORITHM_YIELDABLE_H_
#define HPTS_ALGORITHM_YIELDABLE_H_

#include <concepts>

namespace hpts::algorithm {

enum class Status {
    INIT,
    OK,
    ERROR,
    TIMEOUT,
    SOLVED,
};

// Concept for search algorithm output to satisfy requirements for testing
template <typename T>
concept IsYieldable = requires(T t, const T ct) {
    { t.init() } -> std::same_as<void>;
    { t.reset() } -> std::same_as<void>;
    { t.step() } -> std::same_as<void>;
    { t.get_status() } -> std::same_as<Status>;
};

}    // namespace hpts::algorithm

#endif    // HPTS_ALGORITHM_YIELDABLE_H_
