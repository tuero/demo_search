#ifndef HPTS_PHS_CONFIG_H_
#define HPTS_PHS_CONFIG_H_

#include <ostream>
#include <string>
#include <vector>

#include "common/observation.h"

namespace hpts {

struct Config {
    int seed;
    std::string mode;
    std::string environment;
    std::string problems_path;
    std::size_t max_instances;
    std::size_t num_train;
    std::size_t num_validate;
    double validation_solved_ratio;
    std::string output_path;
    std::string devices;
    int search_budget;
    double time_budget;
    int max_iterations;
    long long int checkpoint_expansions_intervial;
    long long int checkpoint_to_load;
    std::size_t num_threads_search;
    std::size_t bootstrap_batch_multiplier = 1;
    std::size_t inference_batch_size;
    std::size_t block_allocation_size;
    double mix_epsilon;
    std::size_t learning_batch_size;
    std::size_t buffer_capacity;
    std::size_t grad_steps;
    double learning_rate;
    double weight_decay;
    int resnet_channels;
    int resnet_blocks;
    int policy_reduced_channels;
    int heuristic_reduced_channels;
    std::vector<int> policy_layers;
    std::vector<int> heuristic_layers;
    std::string model_type;
    std::string loss_type;
    double base_reward;
    double discount;
    bool use_batch_norm;
};

std::ostream &operator<<(std::ostream &os, const Config &config);

/**
 * Parse the command line args and store in config
 * @param argc Number of arguments
 * @param argv char array of params
 */
Config parse_flags(int argc, char **argv);

}    // namespace hpts

#endif    // HPTS_PHS_CONFIG_H_
