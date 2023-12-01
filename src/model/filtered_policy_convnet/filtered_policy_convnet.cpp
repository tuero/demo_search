// File: policy_convent.cpp
// Description: Convnet for policy predictions

#include "model/filtered_policy_convnet/filtered_policy_convnet.h"

namespace hpts::model::network {

FilteredPolicyConvNetImpl::FilteredPolicyConvNetImpl(const ObservationShape &observation_shape, int num_actions,
                                                     int resnet_channels, int resnet_blocks, int policy_channels,
                                                     const std::vector<int> &policy_mlp_layers, bool use_batchnorm)
    : input_channels_(observation_shape.c),
      input_height_(observation_shape.h),
      input_width_(observation_shape.w),
      num_actions_(num_actions),
      resnet_channels_(resnet_channels),
      policy_channels_(policy_channels),
      policy_mlp_input_size_(policy_channels_ * input_height_ * input_width_),
      resnet_head_(ResidualHead(input_channels_, resnet_channels_, use_batchnorm, "representation_")),
      conv1x1_policy_(conv1x1(resnet_channels_, policy_channels_)),
      policy_mlp_(policy_mlp_input_size_, policy_mlp_layers, 1, "policy_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_policy_);
    register_module("policy_mlp", policy_mlp_);
}

PolicyConvNetOutput FilteredPolicyConvNetImpl::forward(torch::Tensor x) {
    torch::Tensor output = resnet_head_->forward(x);
    // ResNet body
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // Reduce and mlp for policy
    torch::Tensor logits = conv1x1_policy_->forward(output);
    logits = logits.view({-1, policy_mlp_input_size_});
    logits = policy_mlp_->forward(logits);
    logits = logits.reshape({-1, num_actions_});
    const torch::Tensor policy = torch::softmax(logits, 1);
    const torch::Tensor log_policy = torch::log_softmax(logits, 1);
    return {logits, policy, log_policy};
}

}    // namespace hpts::model::network
