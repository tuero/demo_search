// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// --------------------------------------------------------------------
// File: model_evaluator.h
// Description: Interfaces between user code and device manager/ModelWrapper
//              Can spawn threads for inference if

#ifndef HPTS_MODEL_EVALUATOR_H_
#define HPTS_MODEL_EVALUATOR_H_

// NOLINTBEGIN
#include <absl/synchronization/mutex.h>
#ifdef DEBUG_PRINT
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif
#include <spdlog/spdlog.h>
// NOLINTEND

#include <concepts>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "model/device_manager.h"
#include "util/concepts.h"
#include "util/queue.h"
#include "util/stop_token.h"
#include "util/thread_mapper.h"
#include "util/zip.h"

namespace hpts::model {

// Forward declaration
template <ModelWrapper ModelWrapperT>
class ModelEvaluator;

// Concept to check if specialization of ModelEvaluator
template <typename T>
concept IsModelEvaluator = IsSpecialization<T, ModelEvaluator>;

// Handles threaded queries for the model
template <ModelWrapper ModelWrapperT>
class ModelEvaluator {
public:
    // Expose types from model wrapper to users of the evaluator
    using InferenceInput = ModelWrapperT::InferenceInput;
    using InferenceOutput = ModelWrapperT::InferenceOutput;
    using LearningInput = ModelWrapperT::LearningInput;
    using BaseType = ModelWrapperT::BaseType;

    /**
     * @param device_manager Pointer to device manager (holds the models on devices)
     * @param search_threads Number of threads which have a handle on the evaulator
     */
    explicit ModelEvaluator(std::unique_ptr<DeviceManager<ModelWrapperT>> device_manager, int search_threads)
        : device_manager_(std::move(device_manager)), queue_(search_threads * 4) {
        // Reserve space and spawn threads on inference runner
        // One thread per device
        inference_threads_.reserve(device_manager_->Count());
        for (int i = 0; i < static_cast<int>(device_manager_->Count()); ++i) {
            inference_threads_.emplace_back([this, i]() { this->BatchedInferenceRunner(i); });
        }
    }

    ~ModelEvaluator() {
        // Clear the incoming queues and stop oustanding threads
        stop_token_.stop();
        queue_.BlockNewValues();
        queue_.Clear();
        for (auto& t : inference_threads_) {
            t.join();
        }
    }

    // Doesn't make sense to copy/move
    ModelEvaluator(const ModelEvaluator&) = delete;
    ModelEvaluator(ModelEvaluator&&) = delete;
    ModelEvaluator operator=(const ModelEvaluator&) = delete;
    ModelEvaluator operator=(ModelEvaluator&&) = delete;

    /**
     * Perform inference for a group of observations, single-threaded
     * @param inference_inputs inputs for inference
     * @return inference outputs
     */
    [[nodiscard]] auto Inference(std::vector<InferenceInput>& inference_inputs) -> std::vector<InferenceOutput> {
        return device_manager_->Get(1)->Inference(inference_inputs);
    }

    /**
     * Perform inference for a group of observations, single-threaded
     * @param inference_inputs inputs for inference
     * @return inference outputs
     */
    [[nodiscard]] auto InferenceBatched(std::vector<InferenceInput>&& inference_inputs) -> std::vector<InferenceOutput> {
        std::promise<std::vector<InferenceOutput>> prom;
        std::future<std::vector<InferenceOutput>> fut = prom.get_future();
        queue_.Push(QueueItem{inference_inputs, inference_inputs.size(), &prom});
        return fut.get();
    }

    [[nodiscard]] auto get_device_manager() -> DeviceManager<ModelWrapperT>* {
        return device_manager_.get();
    }

    /**
     * Print the model
     */
    void print() {
        get_device_manager()->Get(0, 0)->print();
    }

    /**
     * Load the model and optimizer from the given checkpoint step
     * @param step checkpoint step to load from
     */
    void load(long long int step = -1) {
        get_device_manager()->load_all(step);
    }

    /**
     * Load the model from the given checkpoint step
     * @param step checkpoint step to load from
     */
    void load_without_optimizer(long long int step = -1) {
        get_device_manager()->load_all_without_optimizer(step);
    }

    /**
     * Checkpoint the model/optimizer and sync on all devices
     * @param step Checkpoint number to save as
     */
    void checkpoint_and_sync(long long int step = -1) {
        get_device_manager()->checkpoint_and_sync(step);
    }

    /**
     * Checkpoint the model and sync on all devices
     * @param step Checkpoint number to save as
     */
    void checkpoint_and_sync_without_optimizer(long long int step = -1) {
        get_device_manager()->checkpoint_and_sync_without_optimizer(step);
    }

    /**
     * Checkpoint the model/optimizer
     * @param step Checkpoint number to save as
     */
    void save_checkpoint(long long int step = -1) {
        get_device_manager()->Get(0, 0)->SaveCheckpoint(step);
    }

    /**
     * Checkpoint the model
     * @param step Checkpoint number to save as
     */
    void save_checkpoint_without_optimizer(long long int step = -1) {
        get_device_manager()->Get(0, 0)->SaveCheckpointWithoutOptimizer(step);
    }

    /**
     * Increment number of threads which may be requesting to run inference
     */
    void increment_batch_size() {
        absl::MutexLock lock(&batch_size_lock_);
        ++batch_size_;
    }

    /**
     * Decrement number of threads which may be requesting to run inference
     */
    void decrement_batch_size() {
        absl::MutexLock lock(&batch_size_lock_);
        --batch_size_;
    }

private:
    const int WAIT_TIME = 10;                // NOLINT(*-avoid-const-or-ref-data-members)
    const std::size_t MAX_BATCH_SIZE = 8;    // NOLINT(*-avoid-const-or-ref-data-members)

    // Runner to perform inference queries if using threading on the model (not used currently)
    void BatchedInferenceRunner(int device_id) {
        std::vector<InferenceInput> inference_inputs;    // Collapsed inference inputs
        std::vector<std::promise<std::vector<InferenceOutput>>*> promises;
        std::vector<std::size_t> Ns;    // Each batch item has N inference inputs
        while (!stop_token_.stop_requested()) {
            absl::Time deadline = absl::InfiniteFuture();

            std::size_t batch_size{};
            {
                absl::MutexLock lock(&batch_size_lock_);
                batch_size = batch_size_;
            }
            // skip if no search threads have been fired
            if (batch_size == 0) {
                continue;
            }

            // Try to get a few items to send to device
            // Once we have inputs to send off, we only wait for a short time
            for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(MAX_BATCH_SIZE), batch_size); ++i) {
                absl::optional<QueueItem> item = queue_.Pop(deadline);
                if (!item) {
                    break;
                }
                // At least one search thread is ready for inference.
                // Only wait for WAIT_TIME for other search threads before sending out to device
                if (inference_inputs.empty()) {
                    deadline = absl::Now() + absl::Milliseconds(WAIT_TIME);
                }
                Ns.push_back(item->inputs.size());
                promises.push_back(item->prom);
                for (auto& input : item->inputs) {
                    inference_inputs.push_back(std::move(input));
                }
            }

            // Send batch to network
            auto results = device_manager_->Get(1, device_id)->Inference(inference_inputs);

            assert(promises.size() == Ns.size());
            std::size_t start_idx = 0;
            for (auto&& [promise, N] : zip(promises, Ns)) {
                std::vector<InferenceOutput> inference_outputs;
                for (std::size_t i = 0; i < N; ++i) {
                    inference_outputs.push_back(std::move(results[start_idx + i]));
                }
                promise->set_value(std::move(inference_outputs));
                start_idx += N;
            }

            inference_inputs.clear();
            promises.clear();
            Ns.clear();
        }
    }

    std::unique_ptr<DeviceManager<ModelWrapperT>> device_manager_;    // Sole owner of the device manager

    // Struct for holding promised value for inference queries
    struct QueueItem {
        std::vector<InferenceInput> inputs;    // List of inputs for the current request
        std::size_t N{};                       // Number of inputs for the curent request
        std::promise<std::vector<InferenceOutput>>* prom{};
    };

    ThreadedQueue<QueueItem> queue_;                // Queue for inference requests
    StopToken stop_token_;                          // Stop token flag to signal to quit the inference thread
    std::vector<std::thread> inference_threads_;    // Threads for inference requests
    absl::Mutex batch_size_lock_;                   // Lock for checking batch size on inference thread
    std::size_t batch_size_ = 0;                    // Batch size which corresponds to how many search threads are running
};

}    // namespace hpts::model

#endif    // HPTS_MODEL_EVALUATOR_H_
