// File: heuristic_convnet.cpp
// Description: Convnet for Heuristic predictions

#include "model/heuristic_convnet/heuristic_convnet.h"

namespace hpts::model::network {

HeuristicConvNetImpl::HeuristicConvNetImpl(const ObservationShape &observation_shape, int resnet_channels, int resnet_blocks,
                                           int heuristic_channels, const std::vector<int> &heuristic_mlp_layers,
                                           bool use_batchnorm)
    : input_channels_(observation_shape.c),
      input_height_(observation_shape.h),
      input_width_(observation_shape.w),
      resnet_channels_(resnet_channels),
      heuristic_channels_(heuristic_channels),
      heuristic_mlp_input_size_(heuristic_channels_ * input_height_ * input_width_),
      resnet_head_(ResidualHead(input_channels_, resnet_channels_, use_batchnorm, "representation_")),
      conv1x1_heuristic_(conv1x1(resnet_channels_, heuristic_channels_)),
      heuristic_mlp_(heuristic_mlp_input_size_, heuristic_mlp_layers, 1, "heuristic_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("heuristic_1x1", conv1x1_heuristic_);
    register_module("heuristic_mlp", heuristic_mlp_);
}

torch::Tensor HeuristicConvNetImpl::forward(torch::Tensor x) {
    torch::Tensor output = resnet_head_->forward(x);
    // ResNet body
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // Reduce and mlp
    torch::Tensor heuristic = conv1x1_heuristic_->forward(output);
    heuristic = heuristic.view({-1, heuristic_mlp_input_size_});
    heuristic = heuristic_mlp_->forward(heuristic);
    return heuristic;
}

}    // namespace hpts::model::network
