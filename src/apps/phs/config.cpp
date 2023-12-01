#include "apps/phs/config.h"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include "util/utility.h"

constexpr std::size_t INF_SIZE_T = std::numeric_limits<std::size_t>::max();
constexpr double INF_D = std::numeric_limits<double>::max();
constexpr int INF_I = std::numeric_limits<int>::max();
constexpr long long int INF_LLI = std::numeric_limits<long long int>::max();
constexpr double MAX_TIME = 60 * 60 * 24 * 365;

// NOLINTBEGIN
ABSL_FLAG(int, seed, 0, "Seed for all sources of RNG");
ABSL_FLAG(std::string, mode, "train", "Mode to run [train, test]");
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::size_t, max_instances, INF_SIZE_T, "Maximum number of instances from the problem file");
ABSL_FLAG(std::size_t, num_train, INF_SIZE_T, "Number of instances of the max to use for training");
ABSL_FLAG(std::size_t, num_validate, INF_SIZE_T, "Number of instances of the max to use for validation");
ABSL_FLAG(double, validation_solved_ratio, 1, "Percentage of validation set to solve before checkpointing");
ABSL_FLAG(std::string, output_path, "/opt/hpts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, devices, "cpu", "Comma separated list of devices to use to train and run inference (e.g. cuda:0)");
ABSL_FLAG(int, search_budget, -1, "Maximum number of expanded nodes before termination");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating training/testing procedure");
ABSL_FLAG(int, max_iterations, INF_I, "Budget in number of iterations before terminating training/testing procedure");
ABSL_FLAG(long long int, checkpoint_expansions_interval, INF_LLI, "Interval in number of expansions to checkpoint the model");
ABSL_FLAG(long long int, checkpoint_to_load, -1, "Checkpoint number to load, used in testing");
ABSL_FLAG(std::size_t, num_threads_search, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(std::size_t, bootstrap_batch_multiplier, 1, "Multiple of jobs used as a batch to train on");
ABSL_FLAG(std::size_t, inference_batch_size, 32, "Number of search expansions to batch per inference query");
ABSL_FLAG(std::size_t, block_allocation_size, 2000, "Size used for each block for node allocation");
ABSL_FLAG(double, mix_epsilon, 0, "Percentage to mix with uniform policy");
ABSL_FLAG(std::size_t, learning_batch_size, 256, "Batch size used for model updates");
ABSL_FLAG(std::size_t, buffer_capacity, 10000, "Max size for the learning buffer");
// Model and learning flags
ABSL_FLAG(std::size_t, grad_steps, 1, "Number of gradient updates per iteration of the bootstrap algorithm");
ABSL_FLAG(double, learning_rate, 3e-4, "The learning rate for the heuristic net");
ABSL_FLAG(double, weight_decay, 1e-4, "L2 weight decay regularization for the heuristic net");
ABSL_FLAG(int, resnet_channels, 128, "Number of channels per resnet block");
ABSL_FLAG(int, resnet_blocks, 4, "Number of resnet blocks for heuristic net");
ABSL_FLAG(int, policy_reduced_channels, 2, "Number of reduced channels for policy head");
ABSL_FLAG(int, heuristic_reduced_channels, 2, "Number of reduced channels for heuristic head");
ABSL_FLAG(std::vector<std::string>, policy_layers, std::vector<std::string>({"128"}),
          "Comma separated list of layer sizes for policy head");
ABSL_FLAG(std::vector<std::string>, heuristic_layers, std::vector<std::string>({"128"}),
          "Comma separated list of layer sizes for heuristic head");
ABSL_FLAG(std::string, model_type, "twoheaded", "Model type, one of [twoheaded, policy]");
ABSL_FLAG(std::string, loss_type, "policy_gradient", "Loss type, one of [levin, policy_gradient, phs]");
ABSL_FLAG(double, base_reward, 1.0, "Base reward for policy gradient loss function");
ABSL_FLAG(double, discount, 0.997, "Discount factor for policy gradient loss function");
ABSL_FLAG(bool, batch_norm, false, "Whether to use batch norm in the ResNet architecture");
// NOLINTEND

namespace hpts {

std::ostream &operator<<(std::ostream &os, const Config &config) {
    os << "Config:" << std::endl;
    os << absl::StrFormat("\tseed: %d\n", config.seed);
    os << absl::StrFormat("\tmode: %s\n", config.mode);
    os << absl::StrFormat("\tenvironment: %s\n", config.environment);
    os << absl::StrFormat("\tproblems_path: %s\n", config.problems_path);
    os << absl::StrFormat("\tmax_instances: %s\n",
                          config.max_instances == INF_SIZE_T ? "INF" : std::to_string(config.max_instances));
    os << absl::StrFormat("\tnum_train: %s\n", config.num_train == INF_SIZE_T ? "INF" : std::to_string(config.num_train));
    os << absl::StrFormat("\tnum_validate: %s\n",
                          config.num_validate == INF_SIZE_T ? "INF" : std::to_string(config.num_validate));
    os << absl::StrFormat("\tvalidation_solved_ratio: %f\n", config.validation_solved_ratio);                
    os << absl::StrFormat("\toutput_path: %s\n", config.output_path);
    os << absl::StrFormat("\tdevices: %s\n", config.devices);
    os << absl::StrFormat("\tsearch_budget: %s\n", config.search_budget == INF_D || config.search_budget == -1
                                                       ? "INF"
                                                       : std::to_string(config.search_budget));
    os << absl::StrFormat("\ttime_budget: %s\n", config.time_budget == INF_D ? "INF" : std::to_string(config.time_budget));
    os << absl::StrFormat("\tmax_iterations: %s\n", config.max_iterations == INF_I || config.max_iterations == -1
                                                        ? "INF"
                                                        : std::to_string(config.max_iterations));
    os << absl::StrFormat("\tcheckpoint_expansions_interval: %s\n",
                          config.checkpoint_expansions_intervial == INF_LLI || config.checkpoint_expansions_intervial == -1
                              ? "INF"
                              : std::to_string(config.checkpoint_expansions_intervial));
    os << absl::StrFormat("\tcheckpoint_to_load: %d\n", config.checkpoint_to_load);
    os << absl::StrFormat("\tnum_threads_search: %d\n", config.num_threads_search);
    os << absl::StrFormat("\tbootstrap_batch_multiplier: %d\n", config.bootstrap_batch_multiplier);
    os << absl::StrFormat("\tinference_batch_size: %d\n", config.inference_batch_size);
    os << absl::StrFormat("\tblock_allocation_size: %d\n", config.block_allocation_size);
    os << absl::StrFormat("\tmix_epsilon: %f\n", config.mix_epsilon);
    os << absl::StrFormat("\tlearning_batch_size: %d\n", config.learning_batch_size);
    os << absl::StrFormat("\tbuffer_capacity: %d\n", config.buffer_capacity);

    os << absl::StrFormat("\tgrad_steps: %d\n", config.grad_steps);
    os << absl::StrFormat("\tlearning_rate: %f\n", config.learning_rate);
    os << absl::StrFormat("\tweight_decay: %f\n", config.weight_decay);
    os << absl::StrFormat("\tresnet_channels: %d\n", config.resnet_channels);
    os << absl::StrFormat("\tresnet_blocks: %d\n", config.resnet_blocks);
    os << absl::StrFormat("\tpolicy_reduced_channels: %d\n", config.policy_reduced_channels);
    os << absl::StrFormat("\theuristic_reduced_channels: %d\n", config.heuristic_reduced_channels);
    os << absl::StrFormat("\tpolicy_layers: %s\n", vec_to_str(config.policy_layers));
    os << absl::StrFormat("\theuristic_layers: %s\n", vec_to_str(config.heuristic_layers));
    os << absl::StrFormat("\tmodel_type: %s\n", config.model_type);
    os << absl::StrFormat("\tloss_type: %s\n", config.loss_type);
    os << absl::StrFormat("\tbase_reward: %f\n", config.base_reward);
    os << absl::StrFormat("\tdiscount: %f\n", config.discount);
    os << absl::StrFormat("\tbatch_norm: %d\n", config.use_batch_norm);
    return os;
}

Config parse_flags(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    Config config;
    config.seed = absl::GetFlag(FLAGS_seed);
    config.mode = absl::GetFlag(FLAGS_mode);
    config.environment = absl::GetFlag(FLAGS_environment);
    config.problems_path = absl::GetFlag(FLAGS_problems_path);
    config.max_instances = absl::GetFlag(FLAGS_max_instances);
    config.num_train = absl::GetFlag(FLAGS_num_train);
    config.num_validate = absl::GetFlag(FLAGS_num_validate);
    config.validation_solved_ratio = absl::GetFlag(FLAGS_validation_solved_ratio);
    config.output_path = absl::GetFlag(FLAGS_output_path);
    config.devices = absl::GetFlag(FLAGS_devices);
    config.search_budget = absl::GetFlag(FLAGS_search_budget);
    config.max_iterations = std::max(1, absl::GetFlag(FLAGS_max_iterations));
    config.checkpoint_expansions_intervial = absl::GetFlag(FLAGS_checkpoint_expansions_interval);
    config.checkpoint_to_load = absl::GetFlag(FLAGS_checkpoint_to_load);
    config.time_budget = std::min(absl::GetFlag(FLAGS_time_budget), MAX_TIME);
    config.num_threads_search = absl::GetFlag(FLAGS_num_threads_search);
    config.bootstrap_batch_multiplier = absl::GetFlag(FLAGS_bootstrap_batch_multiplier);
    config.inference_batch_size = absl::GetFlag(FLAGS_inference_batch_size);
    config.block_allocation_size = absl::GetFlag(FLAGS_block_allocation_size);
    config.mix_epsilon = absl::GetFlag(FLAGS_mix_epsilon);
    config.learning_batch_size = absl::GetFlag(FLAGS_learning_batch_size);
    config.buffer_capacity = absl::GetFlag(FLAGS_buffer_capacity);

    config.grad_steps = absl::GetFlag(FLAGS_grad_steps);
    config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
    config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
    config.resnet_channels = absl::GetFlag(FLAGS_resnet_channels);
    config.resnet_blocks = absl::GetFlag(FLAGS_resnet_blocks);
    config.policy_reduced_channels = absl::GetFlag(FLAGS_policy_reduced_channels);
    config.heuristic_reduced_channels = absl::GetFlag(FLAGS_heuristic_reduced_channels);
    config.policy_layers.clear();
    for (const auto &r : absl::GetFlag(FLAGS_policy_layers)) {
        config.policy_layers.push_back(std::stoi(r));
    }
    config.heuristic_layers.clear();
    for (const auto &r : absl::GetFlag(FLAGS_heuristic_layers)) {
        config.heuristic_layers.push_back(std::stoi(r));
    }
    config.model_type = absl::GetFlag(FLAGS_model_type);
    config.loss_type = absl::GetFlag(FLAGS_loss_type);
    config.base_reward = absl::GetFlag(FLAGS_base_reward);
    config.discount = absl::GetFlag(FLAGS_discount);
    config.use_batch_norm = absl::GetFlag(FLAGS_batch_norm);
    return config;
}

}    // namespace hpts
