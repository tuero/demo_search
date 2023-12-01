// File: twoheaded_convnet_wrapper.h
// Description: Convnet wrapper for policy + heuristic net

#ifndef HPTS_WRAPPER_TWOHEADED_CONVNET_H_
#define HPTS_WRAPPER_TWOHEADED_CONVNET_H_

#include "common/observation.h"
#include "model/base_model_wrapper.h"
#include "model/twoheaded_convnet/twoheaded_convnet.h"
#include "util/concepts.h"

namespace hpts::model::wrapper {

struct TwoHeadedConvNetConfig {
    ObservationShape observation_shape;
    int num_actions;
    int resnet_channels;
    int resnet_blocks;
    int policy_channels;
    int heuristic_channels;
    std::vector<int> policy_mlp_layers;
    std::vector<int> heuristic_mlp_layers;
    bool use_batchnorm;
};

constexpr std::string LevinLoss = "levin";
constexpr std::string PolicyGradientLoss = "policy_gradient";
constexpr std::string PHSLoss = "phs";

// template <TwoHeadedConvNetLossType LossType>
class TwoHeadedConvNetWrapperBase : public BaseModelWrapper {
public:
    constexpr static std::string ModelType = "twoheaded";
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
        double heuristic = 0;
    };

    TwoHeadedConvNetWrapperBase(const TwoHeadedConvNetConfig& config, double learning_rate, double l2_weight_decay,
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
    network::TwoHeadedConvNet model_;
    torch::optim::Adam model_optimizer_;
    TwoHeadedConvNetConfig config;
    int input_flat_size;
    // NOLINTEND(*-non-private-member-variables-in-classes)
};

// Implementations for various loss types

class TwoHeadedConvNetWrapperLevin : public TwoHeadedConvNetWrapperBase {
public:
    using BaseType = TwoHeadedConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        double target_cost_to_goal = 0;
        int solution_expanded;
    };

    using TwoHeadedConvNetWrapperBase::TwoHeadedConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class TwoHeadedConvNetWrapperPolicyGradient : public TwoHeadedConvNetWrapperBase {
public:
    using BaseType = TwoHeadedConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        double target_cost_to_goal = 0;
        double reward = 0;
    };

    using TwoHeadedConvNetWrapperBase::TwoHeadedConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

class TwoHeadedConvNetWrapperPHS : public TwoHeadedConvNetWrapperBase {
public:
    using BaseType = TwoHeadedConvNetWrapperBase;
    struct LearningInput {
        Observation observation;
        int target_action = -1;
        double target_cost_to_goal = 0;
        double solution_cost = 0;
        int solution_expanded = 0;
        double solution_log_pi = 0;
    };

    using TwoHeadedConvNetWrapperBase::TwoHeadedConvNetWrapperBase;
    auto Learn(std::vector<LearningInput>& batch) -> double;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_TWOHEADED_CONVNET_H_
