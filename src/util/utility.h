// File: utility.h
// Description: Utility helper functions

#ifndef HPTS_UTIL_UTILITY_H_
#define HPTS_UTIL_UTILITY_H_

#include <spdlog/spdlog.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpts {

/**
 * Split a list of items into a train and validate set
 * @param items The entire list of items to split
 * @param num_train Number of items to place in the train set
 * @param num_validate Number of items to place into the validation set
 * @param seed The seed used on the data shuffling
 * @return Pair of train and validation sets
 */
template <typename T>
auto split_train_validate(std::vector<T> &items, std::size_t num_train, std::size_t num_validate, int seed) {
    if (items.size() < num_train + num_validate) {
        SPDLOG_ERROR("Input items {:d} is less than num_train {:d} + num_validate {:d}", items.size(), num_train, num_validate);
        std::exit(1);
    }
    assert(items.size() >= num_train + num_validate);
    std::mt19937 rng(seed);
    std::shuffle(items.begin(), items.end(), rng);
    return std::make_pair(std::vector<T>(items.begin(), items.begin() + num_train),
                          std::vector<T>(items.begin() + num_train, items.begin() + num_train + num_validate));
}

/**
 * Split a vector if items into batches
 * @param items The tiems to split
 * @param batch_size Size of each batch
 * @return Vector with each item containing batch_size number of items from the input vector
 */
template <typename T>
auto split_to_batch(const std::vector<T> &items, int batch_size) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> batches;
    std::vector<T> batch;

    for (auto const &item : items) {
        batch.push_back(item);
        if ((int)batch.size() == batch_size) {
            batches.push_back(batch);
            batch.clear();
        }
    }

    // Final batch spill over
    if (!batch.empty()) {
        batches.push_back(batch);
    }
    return batches;
}

template <typename T>
auto vec_to_str(const std::vector<T> &vec) -> std::string {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << vec[i];
    }
    ss << "]";
    return ss.str();
}

/**
 * Apply the log function to all values in the given vector
 */
auto log(const std::vector<double> &values) -> std::vector<double>;
auto log(std::vector<double> &&values) -> std::vector<double>;

/**
 * Apply the exp function to all values in the given vector
 */
auto exp(const std::vector<double> &values) -> std::vector<double>;
auto exp(std::vector<double> &&values) -> std::vector<double>;

/**
 * Apply log + uniform mixture to policy
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 * @return Vector of policy with log + uniform mixture applied
 */
auto log_policy_noise(const std::vector<double> &policy, double epsilon = 0) -> std::vector<double>;
auto policy_noise(const std::vector<double> &policy, double epsilon = 0) -> std::vector<double>;
// auto log_policy_noise(std::vector<double> &&policy, double epsilon = 0) -> std::vector<double>;

/**
 * Apply softmax to vector of values
 * @param values The values
 * @return Softmax of given values
 */
auto softmax(const std::vector<double> &values, double temperature = 1) -> std::vector<double>;

/**
 * Compute the sum of two vectors
 * @param lhs the left-hand operand of the sum
 * @param rhs the right-hand operand of the sum
 * @return The sum of lhs and rhs
 */
auto sum(const std::vector<double> &lhs, const std::vector<double> &rhs) -> std::vector<double>;

/**
 * Compute the mixture of two policies
 * @param lhs the left-hand operand of the mixture
 * @param rhs the right-hand operand of the mixture
 * @param alpha the mixture coefficient of alpha * lhs + (1-alpha) * rhs
 * @return The mixture of lhs and rhs
 */
auto mix_policy(const std::vector<double> &lhs, const std::vector<double> &rhs, double alpha) -> std::vector<double>;
auto mix_policy(std::vector<double> &&lhs, std::vector<double> &&rhs, double alpha) -> std::vector<double>;
auto geo_mix_policy(const std::vector<double> &lhs, const std::vector<double> &rhs, double alpha) -> std::vector<double>;
auto geo_mix_policy(std::vector<double> &&lhs, std::vector<double> &&rhs, double alpha) -> std::vector<double>;

auto geo_mix_policy(const std::vector<std::vector<double>> &vs, const std::vector<double> &alphas, std::size_t policy_size,
                    bool normalize = true) -> std::vector<double>;
auto geo_mix_heuristic(const std::vector<double> &vs, const std::vector<double> &alphas) -> double;

}    // namespace hpts

#endif    // HPTS_UTIL_UTILITY_H_
