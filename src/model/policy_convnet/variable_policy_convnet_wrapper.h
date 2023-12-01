// File: variable_policy_convnet_wrapper.h
// Description: Convnet wrapper for policy convnet which groups a collection of observations into a single policy

#ifndef HPTS_WRAPPER_VARIABLE_POLICY_CONVNET_H_
#define HPTS_WRAPPER_VARIABLE_POLICY_CONVNET_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/heuristic_convnet/heuristic_convnet.h"
#include "model/policy_convnet/policy_convnet_wrapper.h"
#include "util/concepts.h"

namespace hpts::model::wrapper {

class VariablePolicyConvNetWrapperBase : public BaseModelWrapper {
public:
    constexpr static std::string ModelType = "policy";
    constexpr static std::string LevinLoss = "levin";
    constexpr static std::string PolicyGradientLoss = "policy_gradient";
    constexpr static std::string PHSLoss = "phs";

    struct InferenceInput {
        std::vector<Observation> observations;
    };

    struct InferenceOutput {
        std::vector<double> logits;
        std::vector<double> policy;
        std::vector<double> log_policy;
    };

    VariablePolicyConvNetWrapperBase(const PolicyConvNetConfig& config, double learning_rate, double l2_weight_decay,
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
    network::HeuristicConvNet model_;    // Batch-wise heuristic net as a polciy net
    torch::optim::Adam model_optimizer_;
    PolicyConvNetConfig config;
    int input_flat_size;
    // NOLINTEND(*-non-private-member-variables-in-classes)
};

// Implementations for various loss types

class VariablePolicyConvNetWrapperLevin : public VariablePolicyConvNetWrapperBase {
public:
    using BaseType = VariablePolicyConvNetWrapperBase;
    struct LearningInput {
        std::vector<Observation> observations;
        int target_action = -1;
        int solution_expanded = 0;
    };

    using VariablePolicyConvNetWrapperBase::VariablePolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class VariablePolicyConvNetWrapperPolicyGradient : public VariablePolicyConvNetWrapperBase {
public:
    using BaseType = VariablePolicyConvNetWrapperBase;
    struct LearningInput {
        std::vector<Observation> observations;
        int target_action = -1;
        double reward = 0;
    };

    using VariablePolicyConvNetWrapperBase::VariablePolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class VariablePolicyConvNetWrapperPHS : public VariablePolicyConvNetWrapperBase {
public:
    using BaseType = VariablePolicyConvNetWrapperBase;
    struct LearningInput {
        std::vector<Observation> observations;
        int target_action = -1;
        double solution_cost = 0;
        int solution_expanded = 0;
        double solution_log_pi = 0;
    };

    using VariablePolicyConvNetWrapperBase::VariablePolicyConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_VARIABLE_POLICY_CONVNET_H_
