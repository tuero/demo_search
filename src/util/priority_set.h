// File: timer.h
// Description: Measures user space time and signals a timeout

#ifndef HPTS_UTIL_PRIORITY_QUEUE_H_
#define HPTS_UTIL_PRIORITY_QUEUE_H_

#include <absl/container/flat_hash_map.h>

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

namespace hpts {

// Priority set with index tracking for random access removal and updating
template <typename T, typename CompareT, typename HashT, typename KeyEqualT>
class PrioritySet {
public:
    PrioritySet() = default;

    /**
     * Insert the value.
     * @note If the value already exists, then no change occurs
     * @param t The value to insert
     */
    template <typename U>
    void push(U &&u) {
        if (contains(u)) {
            return;
        }
        const std::size_t idx = size();
        indices[u] = idx;
        data.emplace_back(std::move(u));
        swim(idx);
    }

    /**
     * Removes the top element
     * @note If the priority set is empty, then no change occurs
     */
    void pop() {
        if (empty()) {
            return;
        }
        swap_elements(0, size() - 1);
        indices.erase(data.back());
        data.pop_back();
        sink(0);
    }

    /**
     * Removes the top element and returns
     * @note If the priority set is empty, then no change occurs
     */
    std::optional<T> pop_and_move() {
        if (empty()) {
            return {};
        }
        swap_elements(0, size() - 1);
        indices.erase(data.back());
        std::optional<T> value = std::move(data.back());
        data.pop_back();
        sink(0);
        return value;
    }

    /**
     * Removes the top element
     * @note If the priority set is empty, then no change occurs
     * @param t The value to remove
     */
    void erase(const T &t) {
        if (!contains(t)) {
            return;
        }
        const std::size_t idx = indices.at(t);
        swap_elements(idx, size() - 1);
        indices.erase(data.back());
        data.pop_back();
        swim(idx);
        sink(idx);
    }

    /**
     * Get a reference to the top element
     * @note Since the reference is non-const, modifying the underlying element could lead to the priority set being in
     * an undetermined state
     * @return The top element
     */
    [[nodiscard]] auto top() -> T & {
        return data.front();
    }

    /**
     * Get a const reference to the top element
     * @return The top element
     */
    [[nodiscard]] auto top() const -> const T & {
        return data.front();
    }

    /**
     * Get a reference to the element stored which matches the given element
     * @note Since the reference is non-const, modifying the underlying element could lead to the priority set being in
     * an undetermined state
     * @param T The element to search for
     * @return The stored element matching the input
     */
    [[nodiscard]] auto get(const T &t) -> T & {
        assert(contains(t));
        return data[indices.at(t)];
    }

    /**
     * Get a reference to the element stored which matches the given element
     * @param T The element to search for
     * @return The stored element matching the input
     */
    // [[nodiscard]] auto get(const T &t) const -> const T & {
    //     assert(contains(t));
    //     return data[indices.at(t)];
    // }

    /**
     * Check if a element exists in the priority set
     * @return True if the element is contained in the priority set, false otherwise
     */
    // [[nodiscard]] auto contains(const T &t) const -> bool {
    //     return indices.find(t) != indices.end();
    // }

    /**
     * Check if a element exists in the priority set
     * @return True if the element is contained in the priority set, false otherwise
     */
    template <typename U>
    [[nodiscard]] auto contains(const U &u) const -> bool {
        return indices.find(u) != indices.end();
    }

    /**
     * Update the element's priority
     * @note If the value already exists, then no change occurs
     * @note The element to update must share the same hash as the new element being passed
     * @param T The element to replace the existing element which shares the same hash
     */
    void update(T t) {
        if (!contains(t)) {
            return;
        }
        const std::size_t idx = indices.at(t);
        data[idx] = std::move(t);
        swim(idx);
        sink(idx);
    }

    /**
     * Remove all elements from the priority set
     */
    void clear() {
        data.clear();
        indices.clear();
    }

    /**
     * Check if the priority set is empty
     * @return True if the priority set is empty, false otherwise
     */
    [[nodiscard]] auto empty() const -> bool {
        return data.empty();
    }

    /**
     * Get the number of elements stored in the priority set
     * @return The number of elements stored in the priority set
     */
    [[nodiscard]] auto size() const -> std::size_t {
        return data.size();
    }

private:
    // Parent index from child
    [[nodiscard]] auto get_par(std::size_t idx) const -> std::size_t {
        return (idx - 1) / 2;
    }

    // Left child index from parent
    [[nodiscard]] auto get_left(std::size_t idx) const -> std::size_t {
        return idx * 2 + 1;
    }

    // Right child index from parent
    [[nodiscard]] auto get_right(std::size_t idx) const -> std::size_t {
        return idx * 2 + 2;
    }

    void swap_elements(std::size_t idx1, std::size_t idx2) {
        std::swap(data[idx1], data[idx2]);
        std::swap(indices.at(data[idx1]), indices.at(data[idx2]));
    }

    void swim(std::size_t idx) {
        std::size_t par_idx = get_par(idx);
        while (idx > 0 && comper(data[idx], data[par_idx])) {
            swap_elements(idx, par_idx);
            idx = par_idx;
            par_idx = get_par(idx);
        }
    }

    void sink(std::size_t idx) {
        while (true) {
            const std::size_t left = get_left(idx);
            const std::size_t right = get_right(idx);
            std::size_t swap_idx = idx;

            // Check children
            if (left < size() && comper(data[left], data[swap_idx])) {
                swap_idx = left;
            }
            if (right < size() && comper(data[right], data[swap_idx])) {
                swap_idx = right;
            }

            // No swap, done fixing heap
            if (idx == swap_idx) {
                return;
            }

            swap_elements(idx, swap_idx);
            idx = swap_idx;
        }
    }

    // Required because T may have hash defined, but pointers are stored
    class ItemPtrHash {
    public:
        auto operator()(const T *item) const -> std::size_t {
            return hasher(*item);
        }

    private:
        HashT hasher;
    };

    // Required because T may have equal_to defined, but pointers are stored
    class ItemPtrEqual {
    public:
        auto operator()(const T *left, const T *right) const -> bool {
            return equal_to(*left, *right);
        }

    private:
        KeyEqualT equal_to;
    };

    CompareT comper;
    std::vector<T> data;                                              // Data storage
    absl::flat_hash_map<T, std::size_t, HashT, KeyEqualT> indices;    // Mapping of element to index in queue
    // absl::flat_hash_map<const T *, std::size_t, ItemPtrHash, ItemPtrEqual> indices;    // Mapping of element to index in queue
};

}    // namespace hpts

#endif    // HPTS_UTIL_PRIORITY_QUEUE_H_
