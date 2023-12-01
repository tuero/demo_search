// File: policy_convent.h
// Description: Convnet for policy predictions

#ifndef HPTS_MODEL_POLICY_CONVNET_H_
#define HPTS_MODEL_POLICY_CONVNET_H_

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include <vector>

#include "common/observation.h"
#include "model/layers.h"

namespace hpts::model::network {

struct PolicyConvNetOutput {
    torch::Tensor logits;
    torch::Tensor policy;
    torch::Tensor log_policy;
};

class PolicyConvNetImpl : public torch::nn::Module {
public:
    /**
     * ResNet style policy convnet
     * @param observation_shape Input observation shape to the network
     * @param num_actions Number of actions for the policy output
     * @param resnet_channels Number of channels for each resenet block
     * @param resnet_blocks Number of resnet blocks
     * @param policy_channels Number of channels in the policy reduce head
     * @param policy_mlp_layers Hidden layer sizes for the policy head MLP
     * @param use_batchnorm Flag to use batchnorm in the resnet layers
     */
    PolicyConvNetImpl(const ObservationShape &observation_shape, int num_actions, int resnet_channels, int resnet_blocks,
                      int policy_channels, const std::vector<int> &policy_mlp_layers, bool use_batchnorm);
    [[nodiscard]] auto forward(torch::Tensor x) -> PolicyConvNetOutput;

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int policy_channels_;
    int policy_mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_policy_;    // Conv pass before passing to policy mlp
    MLP policy_mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(PolicyConvNet);

}    // namespace hpts::model::network

#endif    // HPTS_MODEL_POLICY_CONVNET_H_
