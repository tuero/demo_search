// File: heuristic_convnet_wrapper.h
// Description: Convnet for Heuristic predictions

#ifndef HPTS_WRAPPER_HEURISTIC_CONVNET_H_
#define HPTS_WRAPPER_HEURISTIC_CONVNET_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/heuristic_convnet/heuristic_convnet.h"

namespace hpts::model::wrapper {

struct HeuristicConvNetConfig {
    ObservationShape observation_shape;
    int resnet_channels;
    int resnet_blocks;
    int heuristic_channels;
    std::vector<int> heuristic_mlp_layers;
    bool use_batchnorm;
};

struct HeuristicConvNetLearningInput {
    Observation observation;
    double target_cost_to_goal;
};

class HeuristicConvNetWrapperBase : public BaseModelWrapper {
public:
    constexpr static std::string ModelType = "heuristic";

    struct InferenceInput {
        Observation observation;
    };

    struct InferenceOutput {
        double heuristic;
    };

    HeuristicConvNetWrapperBase(const HeuristicConvNetConfig& config, double learning_rate, double l2_weight_decay,
                                const std::string& device, const std::string& output_path,
                                const std::string& checkpoint_base_name = "");

    void print() const override;

    auto SaveCheckpoint(long long int step = -1) -> std::string override;
    auto SaveCheckpointWithoutOptimizer(long long int step = -1) -> std::string override;

    using BaseModelWrapper::LoadCheckpoint;
    using BaseModelWrapper::LoadCheckpointWithoutOptimizer;
    void LoadCheckpoint(const std::string& path) override;
    void LoadCheckpointWithoutOptimizer(const std::string& path) override;

    /**
     * Perform inference
     * @param inputs Batched observations (implementation defined)
     * @returns Implementation defined output
     */
    [[nodiscard]] auto Inference(std::vector<InferenceInput>& batch) -> std::vector<InferenceOutput>;

protected:
    // NOLINTBEGIN(*-non-private-member-variables-in-classes)
    network::HeuristicConvNet model_;
    torch::optim::Adam model_optimizer_;
    HeuristicConvNetConfig config;
    int input_flat_size;
    // NOLINTEND(*-non-private-member-variables-in-classes)
};

// Implementations for various loss types

class HeuristicConvNetWrapperMSE : public HeuristicConvNetWrapperBase {
public:
    using BaseType = HeuristicConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        double target_cost_to_goal = 0;
    };

    using HeuristicConvNetWrapperBase::HeuristicConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_HEURISTIC_CONVNET_H_
