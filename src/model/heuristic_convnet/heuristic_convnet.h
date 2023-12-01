// File: heuristic_convnet.h
// Description: Convnet for Heuristic predictions

#ifndef HPTS_MODEL_HEURISTIC_CONVNET_H_
#define HPTS_MODEL_HEURISTIC_CONVNET_H_

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include <vector>

#include "common/observation.h"
#include "model/layers.h"

namespace hpts::model::network {

class HeuristicConvNetImpl : public torch::nn::Module {
public:
    /**
     * ResNet style heuristic convnet
     * @param observation_shape Input observation shape to the network
     * @param resnet_channels Number of channels for each resenet block
     * @param resnet_blocks Number of resnet blocks
     * @param heuristic_channels Number of channels in the heuristic reduce head
     * @param heuristic_mlp_layers Hidden layer sizes for the heuristic head MLP
     * @param use_batchnorm Flag to use batchnorm in the resnet layers
     */
    HeuristicConvNetImpl(const ObservationShape &observation_shape, int resnet_channels, int resnet_blocks,
                         int heuristic_channels, const std::vector<int> &heuristic_mlp_layers, bool use_batchnorm);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int heuristic_channels_;
    int heuristic_mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_heuristic_;    // Conv pass before passing to heuristic mlp
    MLP heuristic_mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(HeuristicConvNet);

}    // namespace hpts::model::network

#endif    // HPTS_MODEL_HEURISTIC_CONVNET_H_
