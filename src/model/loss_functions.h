// loss_functions.h
// Description: Common loss functions

#ifndef HPTS_MODEL_LOSS_FUNCTIONS_H_
#define HPTS_MODEL_LOSS_FUNCTIONS_H_

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model::loss {

/**
 * Policy gradient loss
 * @param logits (B, num_actions)
 * @param target_actions (B, 1)
 * @param rewards (B, 1)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
auto policy_gradient_loss(torch::Tensor logits, torch::Tensor target_actions, torch::Tensor rewards, bool reduce = true)
    -> torch::Tensor;

/**
 * Cross entropy loss
 * @param logits (B, num_actions)
 * @param target_actions (B, 1)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
auto cross_entropy_loss(torch::Tensor logits, torch::Tensor target_actions, bool reduce = true) -> torch::Tensor;

/**
 * Mean Squared Error loss
 * @param output (*)
 * @param target (*)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
auto mean_squared_error_loss(torch::Tensor output, torch::Tensor target, bool reduce = true) -> torch::Tensor;

/**
 * PHS loss (CE * ratio of search effort to path probability)
 * @param logits (B, num_actions)
 * @param target_actions (B, 1)
 * @param depths (B, 1)
 * @param expandeds (B, 1)
 * @param log_pis (B, 1)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
auto phs_loss(torch::Tensor logits, torch::Tensor target_actions, torch::Tensor depths, torch::Tensor expandeds,
              torch::Tensor log_pis, bool reduce = true) -> torch::Tensor;

}    // namespace hpts::model::loss

#endif    // HPTS_MODEL_LOSS_FUNCTIONS_H_
