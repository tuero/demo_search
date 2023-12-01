#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "algorithm/phs/phs.h"
#include "algorithm/phs/train.h"
#include "algorithm/test_runner.h"
#include "algorithm/train_bootstrap.h"
#include "apps/phs/config.h"
#include "common/logging.h"
#include "common/signaller.h"
#include "common/state_loader.h"
#include "common/torch_init.h"
#include "env/boxworld/boxworld_base.h"
#include "env/craftworld/craftworld_base.h"
#include "env/rnd/rnd_base.h"
#include "env/rnd/rnd_simple.h"
#include "env/sokoban/sokoban_base.h"
#include "util/utility.h"

using namespace hpts;
using namespace hpts::model;
using namespace hpts::algorithm;
using namespace hpts::env;

// Create inputs to what the search algorithm expects
template <env::SimpleEnv EnvT, typename ModelEvaluatorT>
auto create_problems(const std::vector<EnvT>& problems, int search_budget, std::shared_ptr<StopToken> stop_token,
                     std::shared_ptr<ModelEvaluatorT> model_eval) {
    std::vector<phs::SearchInput<EnvT, ModelEvaluatorT>> search_inputs;
    int problem_number = -1;
    for (const auto& problem : problems) {
        search_inputs.emplace_back(absl::StrFormat("puzzle_%d", ++problem_number), problem, search_budget, stop_token,
                                   model_eval);
    }
    return search_inputs;
}

// Initialize model evaluators
template <typename T>
auto init_model_evaluator(const Config& config, int num_actions, const ObservationShape& observation_shape) = delete;

template <typename T>
    requires std::is_base_of_v<wrapper::PolicyConvNetWrapperBase, T>
auto init_model_evaluator(const Config& config, int num_actions, const ObservationShape& observation_shape) {
    std::unique_ptr<DeviceManager<T>> device_manager = std::make_unique<DeviceManager<T>>();
    const wrapper::PolicyConvNetConfig net_config{
        observation_shape,    num_actions,          config.resnet_channels, config.resnet_blocks, config.policy_reduced_channels,
        config.policy_layers, config.use_batch_norm};
    for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
        device_manager->AddDevice(
            std::make_unique<T>(net_config, config.learning_rate, config.weight_decay, std::string(device), config.output_path));
    }
    // Put this in return type
    return std::make_shared<ModelEvaluator<T>>(std::move(device_manager), 1);
}

template <typename T>
    requires std::is_base_of_v<wrapper::TwoHeadedConvNetWrapperBase, T>
std::shared_ptr<ModelEvaluator<T>> init_model_evaluator(const Config& config, int num_actions,
                                                        const ObservationShape& observation_shape) {
    std::unique_ptr<DeviceManager<T>> device_manager = std::make_unique<DeviceManager<T>>();
    const wrapper::TwoHeadedConvNetConfig net_config{observation_shape,
                                                     num_actions,
                                                     config.resnet_channels,
                                                     config.resnet_blocks,
                                                     config.policy_reduced_channels,
                                                     config.heuristic_reduced_channels,
                                                     config.policy_layers,
                                                     config.heuristic_layers,
                                                     config.use_batch_norm};
    for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
        device_manager->AddDevice(
            std::make_unique<T>(net_config, config.learning_rate, config.weight_decay, std::string(device), config.output_path));
    }
    // Put this in return type
    return std::make_shared<ModelEvaluator<T>>(std::move(device_manager), 1);
}

template <env::SimpleEnv EnvT, typename ModelWrapperT>
void templated_main(const Config& config) {
    using ModelEvaluatorT = model::ModelEvaluator<ModelWrapperT>;
    using SearchInputT = phs::SearchInput<EnvT, ModelEvaluatorT>;
    using SearchOutputT = phs::SearchOutput<EnvT>;
    using LearningHandlerT = phs::LearningHandler<EnvT, ModelEvaluatorT>;

    std::shared_ptr<StopToken> stop_token = signal_installer();

    auto [problems, _] = load_problems<EnvT>(config.problems_path, config.max_instances);
    const auto model_eval = init_model_evaluator<ModelWrapperT>(config, EnvT::num_actions, problems[0].observation_shape());
    model_eval->print();

    phs::INFERENCE_BATCH_SIZE = config.inference_batch_size;
    phs::BLOCK_ALLOCATION_SIZE = config.block_allocation_size;
    phs::MIX_EPSILON = config.mix_epsilon;
    if (config.mode == "train") {
        const auto split_problems = split_train_validate(problems, config.num_train, config.num_validate, config.seed);
        auto problems_train = create_problems(split_problems.first, config.search_budget, stop_token, model_eval);
        auto problems_validate = create_problems(split_problems.second, config.search_budget, stop_token, model_eval);
        LearningHandlerT learning_handler(model_eval, config.buffer_capacity, config.learning_batch_size, config.grad_steps,
                                          config.base_reward, config.discount);
        const TrainingConfig training_config{config.seed,
                                             config.num_threads_search,
                                             config.bootstrap_batch_multiplier,
                                             config.search_budget,
                                             config.time_budget,
                                             config.max_iterations,
                                             config.validation_solved_ratio,
                                             config.checkpoint_expansions_intervial,
                                             config.output_path};
        run_train_levels<SearchInputT, SearchOutputT, LearningHandlerT>(
            problems_train, problems_validate, learning_handler, phs::search<EnvT, ModelEvaluatorT>, training_config, stop_token);
    } else if (config.mode == "test") {
        auto input_problems = create_problems(problems, config.search_budget, stop_token, model_eval);
        model_eval->load_without_optimizer(config.checkpoint_to_load);
        run_test_levels<EnvT, SearchInputT, SearchOutputT>(input_problems, phs::search<EnvT, ModelEvaluatorT>,
                                                           config.num_threads_search, config.search_budget, config.time_budget,
                                                           config.output_path, stop_token, config.max_iterations);
    } else {
        SPDLOG_ERROR("Unknown mode type: {:s}.", config.mode);
        std::exit(1);
    }
}

template <env::SimpleEnv EnvT>
void templated_model_selection(const Config& config) {
    if (config.model_type == wrapper::PolicyConvNetWrapperBase::ModelType &&
        config.loss_type == wrapper::PolicyConvNetWrapperBase::LevinLoss) {
        templated_main<EnvT, model::wrapper::PolicyConvNetWrapperLevin>(config);
    } else if (config.model_type == wrapper::PolicyConvNetWrapperBase::ModelType &&
               config.loss_type == wrapper::PolicyConvNetWrapperBase::PolicyGradientLoss) {
        templated_main<EnvT, model::wrapper::PolicyConvNetWrapperPolicyGradient>(config);
    } else if (config.model_type == wrapper::PolicyConvNetWrapperBase::ModelType &&
               config.loss_type == wrapper::PolicyConvNetWrapperBase::PHSLoss) {
        templated_main<EnvT, model::wrapper::PolicyConvNetWrapperPHS>(config);
    } else if (config.model_type == wrapper::TwoHeadedConvNetWrapperBase::ModelType &&
               config.loss_type == wrapper::TwoHeadedConvNetWrapperBase::LevinLoss) {
        templated_main<EnvT, model::wrapper::TwoHeadedConvNetWrapperLevin>(config);
    } else if (config.model_type == wrapper::TwoHeadedConvNetWrapperBase::ModelType &&
               config.loss_type == wrapper::TwoHeadedConvNetWrapperBase::PolicyGradientLoss) {
        templated_main<EnvT, model::wrapper::TwoHeadedConvNetWrapperPolicyGradient>(config);
    } else if (config.model_type == wrapper::TwoHeadedConvNetWrapperBase::ModelType &&
               config.loss_type == wrapper::TwoHeadedConvNetWrapperBase::PHSLoss) {
        templated_main<EnvT, model::wrapper::TwoHeadedConvNetWrapperPHS>(config);
    } else {
        SPDLOG_ERROR("Unknown loss type: {:s}.", config.loss_type);
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // Get flags -> config
    const Config config = parse_flags(argc, argv);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(config.output_path);

    // Initialize torch and loggers (console + file)
    hpts::init_torch(config.seed);
    hpts::init_loggers(config.output_path, false, absl::StrCat("_", config.mode));

    // Dump invocation of program
    hpts::log_flags(argc, argv);

    SPDLOG_INFO("Configuration used:");
    std::stringstream ss;
    ss << config;
    SPDLOG_INFO("{}", ss.str());

    if (config.environment == rnd::RNDBaseState::name) {
        templated_model_selection<rnd::RNDBaseState>(config);
    } else if (config.environment == rnd::RNDSimpleState::name) {
        templated_model_selection<rnd::RNDSimpleState>(config);
    } else if (config.environment == env::sokoban::SokobanBaseState::name) {
        templated_model_selection<env::sokoban::SokobanBaseState>(config);
    } else if (config.environment == env::cw::CraftWorldBaseState::name) {
        templated_model_selection<env::cw::CraftWorldBaseState>(config);
    } else if (config.environment == env::bw::BoxWorldBaseState::name) {
        templated_model_selection<env::bw::BoxWorldBaseState>(config);
    } else {
        SPDLOG_ERROR("Unknown environment type: {:s}.", config.environment);
        std::exit(1);
    }

    hpts::close_loggers();
}
