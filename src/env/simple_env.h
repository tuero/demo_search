// File: simple_env.h
// Simple environment requirements for generic algorithms

#ifndef HPTS_ENV_SIMPLE_STATE_H_
#define HPTS_ENV_SIMPLE_STATE_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"
#include "util/concepts.h"

namespace hpts::env {

// Concept for simple states for flat search
template <typename T>
concept SimpleEnv = IsSTDHashable<T> && requires(T t, const T ct, std::ostream &os, const std::string &s) {
    T(s);
    { t.apply_action(makeval<std::size_t>()) } -> std::same_as<void>;
    { ct.get_observation() } -> std::same_as<Observation>;
    { ct.observation_shape() } -> std::same_as<ObservationShape>;
    { ct.child_actions() } -> std::same_as<const std::vector<std::size_t> &>;
    { ct.get_heuristic() } -> std::same_as<double>;
    { ct.get_hash() } -> std::same_as<uint64_t>;
    { ct.is_solution() } -> std::same_as<bool>;
    { ct.is_terminal() } -> std::same_as<bool>;
    { ct.to_str() } -> std::same_as<std::string>;
    { os << ct } -> std::convertible_to<std::ostream &>;
    *(&T::name) == makeval<std::string>();
    *(&T::num_actions) == makeval<int>();
};

}    // namespace hpts::env

#endif    // HPTS_ENV_SIMPLE_STATE_H_
