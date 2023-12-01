// File: layers.h
// Description: Model layers/subnets

#ifndef HPTS_MODEL_LAYERS_H_
#define HPTS_MODEL_LAYERS_H_

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts::model {

// Conv and pooling layer defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv1dOptions conv1x1_1d(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride = 1, int padding = 1, bool bias = true,
                                 int groups = 1);
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding);

// MLP
class MLPImpl : public torch::nn::Module {
public:
    /**
     * @param input_size Size of the input layer
     * @param layer_sizes Vector of sizes for each hidden layer
     * @param output_size Size of the output layer
     */
    MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(MLP);

// Main ResNet style residual block
class ResidualBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the resnet block
     * @param layer_num Layer number id, used for pretty printing
     * @param use_batchnorm Flag to use batch normalization
     */
    ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm, int groups = 1);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d batch_norm1;
    torch::nn::BatchNorm2d batch_norm2;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualBlock);

/**
 * Initial input convolutional before ResNet residual blocks
 * Primary use is to take N channels and set to the expected number
 *  of channels for the rest of the resnet body
 */
class ResidualHeadImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of channels the head of the network receives
     * @param output_channels Number of output channels, should match the number of
     *                        channels used for the resnet body
     * @param use_batchnorm Flag to use batch normalization
     * @param name_prefix Used to ID the sub-module for pretty printing
     */
    ResidualHeadImpl(int input_channels, int output_channels, bool use_batchnorm, const std::string &name_prefix = "");
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;
    // Get the observation shape the network outputs given the input
    static ObservationShape encoded_state_shape(ObservationShape observation_shape);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d batch_norm;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualHead);

}    // namespace hpts::model

#endif    // HPTS_MODEL_LAYERS_H_
