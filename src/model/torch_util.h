// torch_util.h
// Utility functions for libtorch c++

#ifndef HPTS_TORCH_UTIL_H_
#define HPTS_TORCH_UTIL_H_

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model {

/**
 * Get tensor from vector
 * @param x The input tensor
 * @return std vector of tensor values
 */
template <typename T>
auto tensor_to_vec(torch::Tensor x) -> std::vector<T> {
    return std::vector<T>(x.data_ptr<T>(),
                          x.data_ptr<T>() + x.numel());    // NOLINT (*-pointer-arithmetic)
}

/**
 * Sum a vector of tensors
 * @param tensors The vector of tensors
 * @return The sum of the vector of tensors
 */
auto tensor_vec_sum(const std::vector<torch::Tensor> &tensors) -> torch::Tensor;

/**
 * Compute the logmeanexp of a tensor
 * @param x The input tensor
 * @param dim The dimension to run the computation on
 * @param keepdim Whether to keep the return dimension the same as the input
 * @return logmeanexp of given tensor x
 */
auto logmeanexp(torch::Tensor x, int dim, bool keepdim = false) -> torch::Tensor;

/**
 * Compute the symlog of a given tensor
 * Symlog is defined as sign(x) * ln(|x| + 1)
 * Symexp is the inverse of symlog
 * https://arxiv.org/pdf/2301.04104.pdf
 */
auto symlog(torch::Tensor x) -> torch::Tensor;

/**
 * Compute the symexp of a given tensor
 * Symexp is defined as sign(x) * (exp(|x|) - 1)
 * Symlog is the inverse of symexp
 * https://arxiv.org/pdf/2301.04104.pdf
 */
auto symexp(torch::Tensor x) -> torch::Tensor;

/**
 * Compute the standard Gaussian CDF at a given value
 * @param x The input tensor -> [batch_size, 1]
 * @returns The Gaussian CDF values -> [batch_size, 1]
 */
auto gaussian_cdf(torch::Tensor x) -> torch::Tensor;

/**
 * Compute the standard Gaussian PDF at a given value
 * @param x The input tensor -> [batch_size, 1]
 * @returns The Gaussian PDF values -> [batch_size, 1]
 */
auto gaussian_pdf(torch::Tensor x) -> torch::Tensor;

/**
 * Compute batch-wise emperical CDF at a given point using Gaussian KDE
 * @param x The point to query the CDF -> [batch_size, 1]
 * @param emperical_observations The emperical observations -> [batch_size, N]
 * @param smoothing_factor The scale to smooth the distribution
 * @return Batch-wise emperical CDF -> [batch_size, 1]
 */
auto gaussian_kde_cdf(torch::Tensor x, torch::Tensor emperical_observations, double smoothing_factor) -> torch::Tensor;

/**
 * Compute batch-wise emperical PDF at a given point using Gaussian KDE
 * @param x The point to query the PDF -> [batch_size, 1]
 * @param emperical_observations The emperical observations -> [batch_size, N]
 * @param smoothing_factor The scale to smooth the distribution
 * @return Batch-wise emperical PDF -> [batch_size, 1]
 */
auto gaussian_kde_pdf(torch::Tensor x, torch::Tensor emperical_observations, double smoothing_factor) -> torch::Tensor;

}    // namespace hpts::model

#endif    // HPTS_TORCH_UTIL_H_
