// File: utility.h
// Description: Utility helper functions

#include "util/utility.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "util/zip.h"

namespace hpts {

constexpr double SMALL_E = 1e-8;    // Ensure log(0) doesn't happen
const double SQRT_2 = std::sqrt(2.0);
constexpr double PI = 3.14159265358979;

template <typename T>
auto vec_sum(const std::vector<T> &values) -> T {
    T sum{};
    for (const auto &value : values) {
        sum += value;
    }
    return sum;
}

auto scalar_mul(const std::vector<double> &values, double alpha) -> std::vector<double> {
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto &value : values) {
        result.push_back(value * alpha);
    }
    return result;
}
auto scalar_mul(std::vector<double> &&values, double alpha) -> std::vector<double> {
    for (auto &value : values) {
        value *= alpha;
    }
    return values;
}

auto log(const std::vector<double> &values) -> std::vector<double> {
    std::vector<double> result;
    result.resize(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](double v) { return std::log(v + SMALL_E); });
    return result;
}
auto log(std::vector<double> &&values) -> std::vector<double> {
    for (auto &v : values) {
        v = std::log(v + SMALL_E);
    }
    return values;
}

auto exp(const std::vector<double> &values) -> std::vector<double> {
    std::vector<double> result;
    result.resize(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](double v) { return std::exp(v); });
    return result;
}
auto exp(std::vector<double> &&values) -> std::vector<double> {
    for (auto &v : values) {
        v = std::exp(v);
    }
    return values;
}

auto policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> result;
    result.reserve(policy.size());
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (const auto p : policy) {
        result.push_back(((1.0 - epsilon) * p) + (epsilon * noise));
    }
    return result;
}
auto policy_noise(std::vector<double> &&policy, double epsilon) -> std::vector<double> {
    if (epsilon == 0) {
        return policy;
    }
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (auto &p : policy) {
        p = ((1.0 - epsilon) * p) + (epsilon * noise);
    }
    return policy;
}
auto log_policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> log_policy;
    log_policy.reserve(policy.size());
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (const auto p : policy) {
        log_policy.push_back(std::log(((1.0 - epsilon) * p) + (epsilon * noise) + SMALL_E));
    }
    return log_policy;
}
auto log_policy_noise(std::vector<double> &&policy, double epsilon) -> std::vector<double> {
    if (epsilon == 0) {
        return policy;
    }
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (auto &p : policy) {
        p = std::log(((1.0 - epsilon) * p) + (epsilon * noise) + SMALL_E);
    }
    return policy;
}

auto softmax(const std::vector<double> &values, double temperature) -> std::vector<double> {
    std::vector<double> new_values = values;
    for (double &v : new_values) {
        v *= temperature;
    }
    const double max_value = *std::max_element(std::begin(new_values), std::end(new_values));
    const double sum{std::accumulate(std::begin(new_values), std::end(new_values), 0.0,
                                     [&](double left_sum, double next) { return left_sum + std::exp(next - max_value); })};
    const double k = max_value + std::log(sum);
    std::vector<double> output;
    output.reserve(new_values.size());
    for (auto const &v : new_values) {
        output.push_back(std::exp(v - k));
    }
    return output;
}

auto sum(const std::vector<double> &lhs, const std::vector<double> &rhs) -> std::vector<double> {
    assert(lhs.size() == rhs.size());
    std::vector<double> result;
    result.reserve(lhs.size());
    for (auto &&[l, r] : zip(lhs, rhs)) {
        result.push_back(l + r);
    }
    return result;
}

auto mix_policy(const std::vector<double> &lhs, const std::vector<double> &rhs, double alpha) -> std::vector<double> {
    assert(lhs.size() == rhs.size());
    assert(alpha >= 0 && alpha <= 1);
    std::vector<double> result;
    result.reserve(lhs.size());

    for (auto &&[l, r] : zip(lhs, rhs)) {
        result.push_back((alpha * l) + ((1.0 - alpha) * r));
    }

    return result;
}

auto geo_mix_policy(const std::vector<double> &lhs, const std::vector<double> &rhs, double alpha) -> std::vector<double> {
    assert(lhs.size() == rhs.size());
    assert(alpha >= 0 && alpha <= 1);
    const auto temp = exp(sum(scalar_mul(log(lhs), alpha), scalar_mul(log(rhs), 1.0 - alpha)));
    return scalar_mul(temp, 1.0 / vec_sum(temp));
}
auto geo_mix_policy(std::vector<double> &&lhs, std::vector<double> &&rhs, double alpha) -> std::vector<double> {
    assert(lhs.size() == rhs.size());
    assert(alpha >= 0 && alpha <= 1);
    const auto temp = exp(sum(scalar_mul(log(lhs), alpha), scalar_mul(log(rhs), 1.0 - alpha)));
    return scalar_mul(temp, 1.0 / vec_sum(temp));
}

auto geo_mix_policy(const std::vector<std::vector<double>> &vs, const std::vector<double> &alphas, std::size_t policy_size,
                    bool normalize) -> std::vector<double> {
    assert(vs.size() == alphas.size());
    std::vector<double> result(policy_size, 0);
    for (std::size_t i = 0; i < policy_size; ++i) {
        for (std::size_t j = 0; j < alphas.size(); ++j) {
            result[i] += std::log(vs.at(j).at(i) + SMALL_E) * alphas.at(j);
        }
        result[i] = std::exp(result[i]);
    }
    return normalize ? scalar_mul(result, 1.0 / vec_sum(result)) : result;
}

auto geo_mix_heuristic(const std::vector<double> &vs, const std::vector<double> &alphas) -> double {
    assert(vs.size() == alphas.size());
    double result = 0;
    for (std::size_t j = 0; j < alphas.size(); ++j) {
        result += std::log(std::max(vs[j], SMALL_E)) * alphas[j];
    }
    return std::exp(result);
}

}    // namespace hpts
