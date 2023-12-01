// File: replay_buffer.h
// Description: Simple replay buffer

#ifndef HPTS_UTIL_REPLAY_BUFFER_H_
#define HPTS_UTIL_REPLAY_BUFFER_H_

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace hpts {

// Simple uniform random replay buffer
template <typename T>
class ReplayBuffer {
public:
    ReplayBuffer(std::size_t capacity, std::size_t min_sample_size) : capacity(capacity), min_sample_size(min_sample_size) {
        if (capacity == 0) {
            SPDLOG_ERROR("Buffer capacity {:d} must be > 0", capacity);
            std::exit(1);
        }
        if (capacity < min_sample_size) {
            SPDLOG_ERROR("Capacity {:d} must be > minimum sample size {:d}", capacity, min_sample_size);
            std::exit(1);
        }
    };

    /**
     * Sample from the buffer uniform randomly
     * @param batch_size Size of batch to sample
     * @param rng The source of randomness
     * @return Vector of samples
     */
    [[nodiscard]] auto sample(std::size_t batch_size, std::mt19937 &rng) const noexcept -> std::vector<T> {
        std::vector<T> sample;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(sample), batch_size, rng);
        return sample;
    }

    /**
     * Insert item into the buffer
     * @param item Item to add
     */
    void insert(const T &item) noexcept {
        if (buffer.size() >= capacity) {
            buffer[idx] = item;
        } else {
            buffer.push_back(item);
        }
        idx = (idx + 1) % capacity;
        // buffer[idx] = item;
        // items_stored = std::min((int)buffer.size(), items_stored + 1);
        // idx = (idx + 1) % buffer.size();
    }

    /**
     * Emplace item into the buffer
     * @param item Item to add
     */
    template <typename... Args>
    void emplace(Args &&...args) noexcept {
        if (buffer.size() >= capacity) {
            buffer[idx] = T{std::forward<Args>(args)...};
        } else {
            buffer.emplace_back(std::forward<Args>(args)...);
        }
        idx = (idx + 1) % capacity;
    }

    /**
     * Get number of items stored
     * @return Count of items saves
     */
    [[nodiscard]] auto count() const noexcept -> std::size_t {
        return buffer.size();
    }

    /*
     * Check if buffer can be sampled from
     * Return true if buffer has more items than minimum sample size
     */
    [[nodiscard]] auto can_sample() const noexcept -> bool {
        return count() > min_sample_size;
    }

    /*
     * Clear all contents from the buffer
     */
    void clear() noexcept {
        idx = 0;
        buffer.clear();
    }

private:
    std::size_t idx = 0;
    std::size_t capacity;
    std::size_t min_sample_size;
    std::vector<T> buffer;
};

}    // namespace hpts

#endif    // HPTS_UTIL_REPLAY_BUFFER_H_
