// File: twoheaded_convnet_multi_wrapper.h
// Description: Convnet wrapper for multiple policy + heuristic nets

#ifndef HPTS_WRAPPER_TWOHEADED_CONVNET_MULTI_H_
#define HPTS_WRAPPER_TWOHEADED_CONVNET_MULTI_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/twoheaded_convnet/twoheaded_convnet.h"
#include "model/twoheaded_convnet/twoheaded_convnet_wrapper.h"

namespace hpts::model::wrapper {

class TwoHeadedConvNetMultiWrapperBase : public BaseModelWrapper {
public:
    constexpr static std::string ModelType = "twoheaded_multi";
    constexpr static std::string LevinLoss = "levin";
    constexpr static std::string PolicyGradientLoss = "policy_gradient";
    constexpr static std::string PHSLoss = "phs";

    struct InferenceInput {
        Observation observation;
        int subgoal;
    };

    struct InferenceOutput {
        std::vector<double> logits;
        std::vector<double> policy;
        std::vector<double> log_policy;
        double heuristic = 0;
    };

    TwoHeadedConvNetMultiWrapperBase(const TwoHeadedConvNetConfig& config, int num_models, double learning_rate,
                                     double l2_weight_decay, const std::string& device, const std::string& output_path,
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
    std::vector<network::TwoHeadedConvNet> models_;
    std::vector<torch::optim::Adam> model_optimizers_;
    TwoHeadedConvNetConfig config;
    int input_flat_size;
    // NOLINTEND(*-non-private-member-variables-in-classes)
};

// Implementations for various loss types

class TwoHeadedConvNetMultiWrapperLevin : public TwoHeadedConvNetMultiWrapperBase {
public:
    using BaseType = TwoHeadedConvNetMultiWrapperBase;
    struct LearningInput {
        Observation observation;
        int subgoal;
        int target_action = -1;
        double target_cost_to_goal = 0;
        int solution_expanded;
    };

    using TwoHeadedConvNetMultiWrapperBase::TwoHeadedConvNetMultiWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class TwoHeadedConvNetMultiWrapperPolicyGradient : public TwoHeadedConvNetMultiWrapperBase {
public:
    using BaseType = TwoHeadedConvNetMultiWrapperBase;
    struct LearningInput {
        Observation observation;
        int subgoal;
        int target_action = -1;
        double target_cost_to_goal = 0;
        double reward = 0;
    };

    using TwoHeadedConvNetMultiWrapperBase::TwoHeadedConvNetMultiWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_TWOHEADED_CONVNET_MULTI_H_
