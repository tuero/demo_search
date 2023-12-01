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
// File: device_manager.h
// Description: Holds multiple models and gives out when requested with load balancing

#ifndef HPTS_DEVICE_MANAGER_H_
#define HPTS_DEVICE_MANAGER_H_

#include <absl/strings/str_split.h>
#include <absl/synchronization/mutex.h>

#include <memory>
#include <string>
#include <vector>

#include "model/base_model_wrapper.h"

namespace hpts::model {

// Keeps track of multiple BaseModelWrappers models, intended to be one per device, and
// gives them out based on usage. When you request a device you specify how much
// work you're going to give it, which is assumed done once the loan is
// returned.
template <ModelWrapper ModelWrapperT>
class DeviceManager {
public:
    // Not thread safe
    void AddDevice(std::unique_ptr<ModelWrapperT> model) {
        devices.emplace_back(std::move(model));
        multiple_devices_ = devices.size() > 1;
    }

    // Acts as a pointer to the model, but lets the manager know when you're done.
    class DeviceLoan {
    public:
        // DeviceLoan is not public constructible and is move only.
        DeviceLoan(DeviceLoan&& other) = default;
        DeviceLoan& operator=(DeviceLoan&& other) = default;
        DeviceLoan(const DeviceLoan&) = delete;
        DeviceLoan& operator=(const DeviceLoan&) = delete;
        ~DeviceLoan() {
            manager_->Return(device_id_, requests_);
        }

        ModelWrapperT* operator->() {
            return model_;
        }

    private:
        DeviceLoan(DeviceManager<ModelWrapperT>* manager, ModelWrapperT* model, int device_id, int requests)
            : manager_(manager), model_(model), device_id_(device_id), requests_(requests) {}

        DeviceManager<ModelWrapperT>* manager_;
        ModelWrapperT* model_;
        int device_id_;
        int requests_;
        friend DeviceManager;
    };

    // Gives the device with the fewest outstanding requests.
    // If learning, device_id=0 should be requested
    [[nodiscard]] auto Get(int requests, int device_id = -1) -> DeviceLoan {
        if (device_id >= 0) {
            return {this, devices[device_id].model.get(), device_id, requests};
        }
        const absl::MutexLock lock(&m_);
        if (device_id < 0) {
            // The starting device changes depending on if we are allowed to
            // use the first device or not.
            device_id = 0 + (learning_ && multiple_devices_);
            for (int i = 1 + (learning_ && multiple_devices_); i < (int)devices.size(); ++i) {
                if (devices[i].requests < devices[device_id].requests) {
                    device_id = i;
                }
            }
        }
        devices[device_id].requests += requests;
        return {this, devices[device_id].model.get(), device_id, requests};
    }

    // A member to ensure that when device:0 is learning and there are
    // multiple devices available, that device:0 does not take on any
    // inference requests from the actors and evaluators. These inference
    // requests should be dealt with by the other available devices.
    // @note This is only required when running inference + learning concurrently
    void SetLearning(bool value) {
        learning_ = value;
    }

    [[nodiscard]] auto Count() const -> std::size_t {
        return devices.size();
    }

    // Checkpoint model and sync all models to that saved checkpoint
    void checkpoint_and_sync(long long int step, int device_id = 0) {
        const std::string checkpoint_path = this->Get(0, device_id)->SaveCheckpoint(step);
        for (int i = 0; i < static_cast<int>(this->Count()); ++i) {
            if (i != device_id) {
                this->Get(0, i)->LoadCheckpoint(checkpoint_path);
            }
        }
    }
    void checkpoint_and_sync_without_optimizer(long long int step, int device_id = 0) {
        const std::string checkpoint_path = this->Get(0, device_id)->SaveCheckpointWithoutOptimizer(step);
        for (int i = 0; i < static_cast<int>(this->Count()); ++i) {
            if (i != device_id) {
                this->Get(0, i)->LoadCheckpointWithoutOptimizer(checkpoint_path);
            }
        }
    }

    // Load all models to a given checkpoint step
    void load_all(long long int step) {
        for (int i = 0; i < static_cast<int>(this->Count()); ++i) {
            this->Get(0, i)->LoadCheckpoint(step);
        }
    }
    void load_all_without_optimizer(long long int step) {
        for (int i = 0; i < static_cast<int>(this->Count()); ++i) {
            this->Get(0, i)->LoadCheckpointWithoutOptimizer(step);
        }
    }

private:
    void Return(int device_id, int requests) {
        const absl::MutexLock lock(&m_);
        devices[device_id].requests -= requests;
    }

    struct Device {
        std::unique_ptr<ModelWrapperT> model;
        int requests = 0;
    };

    bool learning_ = false;
    bool multiple_devices_ = false;
    std::vector<Device> devices;
    absl::Mutex m_;
};

}    // namespace hpts::model

#endif    // HPTS_DEVICE_MANAGER_H_
