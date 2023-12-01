// torch_util.cpp
// Utility functions for libtorch c++

#include "model/torch_util.h"

#include <cmath>

namespace hpts::model {

using namespace torch::indexing;

const double SQRT_2 = std::sqrt(2.0);
constexpr double PI = 3.14159265358979;
const double pdf_const = (1.0 / std::sqrt(2.0 * PI));

auto tensor_vec_sum(const std::vector<torch::Tensor> &tensors) -> torch::Tensor {
    assert(tensors.size() > 0);
    torch::Tensor tensor = tensors[0];
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        tensor = tensor + tensors[i];
    }
    return tensor;
}

auto logmeanexp(torch::Tensor x, int dim, bool keepdim) -> torch::Tensor {
    auto res = torch::max(x, dim, true);
    const torch::Tensor x_max = std::get<0>(res);
    x = x_max + torch::log(torch::mean(torch::exp(x - x_max), dim, true));
    return keepdim ? x : x.squeeze(dim);
}

auto symlog(torch::Tensor x) -> torch::Tensor {
    // sign(x) * ln(|x| + 1)
    return torch::sign(x) * torch::log(torch::abs(x) + 1);
}

auto symexp(torch::Tensor x) -> torch::Tensor {
    // sign(x) * (exp(|x|) - 1)
    return torch::sign(x) * (torch::exp(torch::abs(x)) - 1);
}

auto gaussian_cdf(torch::Tensor x) -> torch::Tensor {
    return (torch::erf(x / SQRT_2) + 1) / 2;
}

auto gaussian_pdf(torch::Tensor x) -> torch::Tensor {
    return pdf_const * torch::exp(-torch::pow(x, 2) / 2);
}

auto gaussian_kde_cdf(torch::Tensor x, torch::Tensor emperical_observations, double smoothing_factor) -> torch::Tensor {
    const std::size_t num_samples = emperical_observations.size(1);
    std::vector<torch::Tensor> emperical_cdf;
    for (std::size_t i = 0; i < num_samples; ++i) {
        emperical_cdf.push_back(gaussian_cdf(x - emperical_observations.index({Slice(), Slice(i, i + 1)})) / smoothing_factor);
    }
    return torch::cat(emperical_cdf, 1).sum(1, true) / (smoothing_factor * static_cast<double>(num_samples));
}

auto gaussian_kde_pdf(torch::Tensor x, torch::Tensor emperical_observations, double smoothing_factor) -> torch::Tensor {
    const std::size_t num_samples = emperical_observations.size(1);
    std::vector<torch::Tensor> emperical_pdf;
    for (std::size_t i = 0; i < num_samples; ++i) {
        emperical_pdf.push_back(gaussian_pdf(x - emperical_observations.index({Slice(), Slice(i, i + 1)})) / smoothing_factor);
    }
    return torch::cat(emperical_pdf, 1).sum(1, true) / (smoothing_factor * static_cast<double>(num_samples));
}

}    // namespace hpts::model
