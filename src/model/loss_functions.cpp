// loss_functions.cpp
// Description: Common loss functions

#include "model/loss_functions.h"

#include <iostream>

namespace hpts::model::loss {

auto policy_gradient_loss(torch::Tensor logits, torch::Tensor target_actions, torch::Tensor rewards, bool reduce)
    -> torch::Tensor {
    const torch::Tensor log_prob = torch::log_softmax(logits, 1).gather(1, target_actions);
    const torch::Tensor loss = (-log_prob * rewards);
    return reduce ? loss.mean() : loss;
}

auto cross_entropy_loss(torch::Tensor logits, torch::Tensor target_actions, bool reduce) -> torch::Tensor {
    if (target_actions.dim() > 1) {
        target_actions = target_actions.flatten();
    }
    const torch::Tensor loss = torch::cross_entropy_loss(logits, target_actions, {}, at::Reduction::None);
    return reduce ? loss.mean() : loss;
}

auto mean_squared_error_loss(torch::Tensor output, torch::Tensor target, bool reduce) -> torch::Tensor {
    return torch::mse_loss(output, target, reduce ? at::Reduction::Mean : at::Reduction::None);
}

auto phs_loss(torch::Tensor logits, torch::Tensor target_actions, torch::Tensor depths, torch::Tensor expandeds,
              torch::Tensor log_pis, bool reduce) -> torch::Tensor {
    const torch::Tensor a = torch::log((depths + 1) / (expandeds + 2)) / log_pis;
    const torch::Tensor loss = cross_entropy_loss(logits, target_actions, false) * expandeds * a;
    return reduce ? loss.mean() : loss;
}

}    // namespace hpts::model::loss
