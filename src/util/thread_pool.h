// File: thread_pool.h
// Description: Simple thread pool class to dispatch threads continuously on input

#ifndef HPTS_UTIL_THREAD_POOL_H_
#define HPTS_UTIL_THREAD_POOL_H_

#include <absl/synchronization/mutex.h>

#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <thread>
#include <vector>

#include "util/queue.h"
#include "util/thread_mapper.h"

namespace hpts {

// Create a thread pool object.
template <typename InputT, typename OutputT>
class ThreadPool {
public:
    ThreadPool() = delete;

    /**
     * Create a thread pool object.
     * @param num_threads Number of threads the pool should run
     */
    ThreadPool(std::size_t num_threads)
        : num_threads(num_threads)
    //   queue_input_(std::numeric_limits<uint16_t>::max()),
    //   queue_output_(std::numeric_limits<uint16_t>::max())
    {
        if (num_threads == 0) {
            throw std::invalid_argument("Expected at least one thread count");
        }
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto run(std::function<OutputT(InputT)> func, const std::vector<InputT>& inputs) noexcept
        -> std::vector<OutputT> {
        // Populate queue
        int id = -1;
        {
            std::queue<QueueItemInput> empty;
            std::swap(queue_input_, empty);
        }
        for (auto const& job : inputs) {
            // queue_input_.Push({job, ++id});
            queue_input_.emplace(job, ++id);
        }

        // Start N threads
        threads_.clear();
        threads_.reserve(num_threads);
        thread_mapper::clear();
        for (std::size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this, func, i]() { this->thread_runner(func, i); });
        }

        // Wait for all to complete
        for (auto& t : threads_) {
            t.join();
        }
        threads_.clear();

        // Compile results, such that the id is in order to match passed order
        std::vector<OutputT> results;
        results.reserve(inputs.size());
        std::map<int, OutputT> result_map;
        // while (!queue_output_.Empty()) {
        //     absl::optional<QueueItemOutput> result = queue_output_.Pop();
        //     result_map.emplace(result->id, result->output);
        // }
        while (!queue_output_.empty()) {
            const auto result = queue_output_.front();
            queue_output_.pop();
            result_map.emplace(result.id, std::move(result.output));
        }
        for (auto const& result : result_map) {
            results.push_back(std::move(result.second));
        }

        return results;
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @param workers Number of threads to use
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto run(std::function<OutputT(InputT)> func, const std::vector<InputT>& inputs, std::size_t workers) noexcept
        -> std::vector<OutputT> {
        std::size_t old_count = workers;
        num_threads = workers;
        const auto results = run(func, inputs);
        num_threads = old_count;
        return results;
    }

private:
    struct QueueItemInput {    // Wrapper for input type with id
        InputT input;
        int id;
    };

    struct QueueItemOutput {    // Wrapper for output type with id
        OutputT output;
        int id;
    };

    // Runner for each thread, runs given function and pulls next item from input jobs if available
    void thread_runner(std::function<OutputT(InputT)> func, std::size_t thread_idx) noexcept {
        // Ideally we should loop until StopToken signals, but then we need to save state of a search which may be
        // complex. Idea will be to let threadpool finish then let the caller do cleanup and prepare for stoppage
        // while (true) {
        //     absl::optional<QueueItemInput> item;
        //     {
        //         absl::MutexLock lock(&queue_input_m_);

        //         // Jobs are done, thread can stop
        //         if (queue_input_.Empty()) {
        //             break;
        //         }

        //         absl::Time deadline = absl::InfiniteFuture();
        //         item = queue_input_.Pop(deadline);
        //     }

        //     // Hit the deadline.
        //     if (!item) {
        //         continue;
        //     }

        //     // Run job
        //     OutputT result = func(item->input);

        //     // Store result
        //     {
        //         absl::MutexLock lock(&queue_output_m_);
        //         queue_output_.Push({result, item->id});
        //     }
        // }
        thread_mapper::add(thread_idx);
        while (true) {
            std::optional<QueueItemInput> item;
            {
                std::lock_guard<std::mutex> lock(queue_input_m_);

                // Jobs are done, thread can stop
                if (queue_input_.empty()) {
                    break;
                }

                item = queue_input_.front();
                queue_input_.pop();
            }

            // Run job
            OutputT result = func(item->input);

            // Store result
            {
                std::lock_guard<std::mutex> lock(queue_output_m_);
                queue_output_.emplace(std::move(result), item->id);
            }
        }
    }

    std::size_t num_threads;              // How many threads in the pool
    std::vector<std::thread> threads_;    // Threads in the pool
    std::queue<QueueItemInput> queue_input_;
    std::queue<QueueItemOutput> queue_output_;
    // ThreadedQueue<QueueItemInput> queue_input_;      // Queue for input argument for job
    // ThreadedQueue<QueueItemOutput> queue_output_;    // Queue for output return values for job
    // absl::Mutex queue_input_m_;                      // Mutex for the input queue
    // absl::Mutex queue_output_m_;                     // Musted for the output queue
    std::mutex queue_input_m_;
    std::mutex queue_output_m_;
};

}    // namespace hpts

#endif    // HPTS_UTIL_THREAD_POOL_H_
