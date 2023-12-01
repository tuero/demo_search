// File: policy_convnet_wrapper.h
// Description: Convnet wrapper for policy convnet

#ifndef HPTS_WRAPPER_POLICY_CONVNET_H_
#define HPTS_WRAPPER_POLICY_CONVNET_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/policy_convnet/policy_convnet.h"
#include "util/concepts.h"

namespace hpts::model::wrapper {

struct PolicyConvNetConfig {
    ObservationShape observation_shape;
    int num_actions;
    int resnet_channels;
    int resnet_blocks;
    int policy_channels;
    std::vector<int> policy_mlp_layers;
    bool use_batchnorm;
};

class PolicyConvNetWrapperBase : public BaseModelWrapper {
public:
    constexpr static std::string ModelType = "policy";
    constexpr static std::string LevinLoss = "levin";
    constexpr static std::string PolicyGradientLoss = "policy_gradient";
    constexpr static std::string PHSLoss = "phs";

    struct InferenceInput {
        Observation observation;
    };

    struct InferenceOutput {
        std::vector<double> logits;
        std::vector<double> policy;
        std::vector<double> log_policy;
    };

    PolicyConvNetWrapperBase(const PolicyConvNetConfig& config, double learning_rate, double l2_weight_decay,
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
    network::PolicyConvNet model_;
    torch::optim::Adam model_optimizer_;
    PolicyConvNetConfig config;
    int input_flat_size;
    // NOLINTEND(*-non-private-member-variables-in-classes)
};

// Implementations for various loss types

class PolicyConvNetWrapperLevin : public PolicyConvNetWrapperBase {
public:
    using BaseType = PolicyConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        int solution_expanded = 0;
    };

    using PolicyConvNetWrapperBase::PolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class PolicyConvNetWrapperPolicyGradient : public PolicyConvNetWrapperBase {
public:
    using BaseType = PolicyConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        double reward = 0;
    };

    using PolicyConvNetWrapperBase::PolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class PolicyConvNetWrapperPHS : public PolicyConvNetWrapperBase {
public:
    using BaseType = PolicyConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        double solution_cost = 0;
        int solution_expanded = 0;
        double solution_log_pi = 0;
    };

    using PolicyConvNetWrapperBase::PolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_POLICY_CONVNET_H_
