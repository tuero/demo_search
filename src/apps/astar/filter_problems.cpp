#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "algorithm/astar/astar.h"
#include "common/logging.h"
#include "common/signaller.h"
#include "common/state_loader.h"
#include "env/boxworld/boxworld_base.h"
#include "util/utility.h"


using namespace hpts;
using namespace hpts::algorithm;
using namespace hpts::env;

constexpr std::size_t INF_SIZE_T = std::numeric_limits<std::size_t>::max();

// NOLINTBEGIN
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::size_t, max_instances, INF_SIZE_T, "Maximum number of instances from the problem file");
ABSL_FLAG(std::string, output_path, "/opt/hpts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::size_t, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(int, search_budget, -1, "Maximum number of expanded nodes before termination");
// NOLINTEND

// Create inputs to what the search algorithm expects
template <astar::AStarEnv EnvT>
auto create_problems(const std::vector<EnvT> &problems, int search_budget, std::shared_ptr<StopToken> stop_token) {
    std::vector<astar::SearchInputNoModel<EnvT>> search_inputs;
    int problem_number = -1;
    for (const auto &problem : problems) {
        search_inputs.emplace_back(absl::StrFormat("puzzle_%d", ++problem_number), problem, search_budget, stop_token);
    }
    return search_inputs;
}

template <astar::AStarEnv EnvT>
void templated_main(const std::string &problems_path, const std::string &output_path, std::size_t max_instances,
                    int search_budget, std::size_t num_threads) {
    using SearchInputT = astar::SearchInputNoModel<EnvT>;
    using SearchOutputT = astar::SearchOutput<EnvT>;

    std::shared_ptr<StopToken> stop_token = signal_installer();

    auto [problems, problem_strs] = load_problems<EnvT>(problems_path, max_instances);
    auto input_problems = create_problems(problems, search_budget, stop_token);

    // Run search over problems
    ThreadPool<SearchInputT, SearchOutputT> pool(num_threads);
    auto batched_input = split_to_batch(input_problems, num_threads * 2);
    std::ofstream f(output_path);
    std::size_t counter = 0;
    std::size_t idx = 0;
    
    for (const auto & batch : batched_input) {
        std::vector<SearchOutputT> results = pool.run(astar::search<EnvT>, batch);
        for (auto && res : results) {
            if (res.solution_found) {
                f << problem_strs[idx] << std::endl;
            } else {
                ++counter;
            }
            ++idx;
        }
    }

    f.close();
    SPDLOG_INFO("Filtered out {:d} problems.", counter);
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    const std::string output_path = absl::GetFlag(FLAGS_output_path);
    const std::string problems_path = absl::GetFlag(FLAGS_problems_path);
    const std::string environment = absl::GetFlag(FLAGS_environment);
    std::size_t max_instances = absl::GetFlag(FLAGS_max_instances);
    int search_budget = absl::GetFlag(FLAGS_search_budget);
    std::size_t num_threads = absl::GetFlag(FLAGS_num_threads);

    hpts::init_loggers(output_path, true);

    if (environment == env::bw::BoxWorldBaseState::name) {
        templated_main<env::bw::BoxWorldBaseState>(problems_path, output_path, max_instances, search_budget, num_threads);
    } else {
        SPDLOG_ERROR("Unknown environment type: {:s}.", environment);
        std::exit(1);
    }

    hpts::close_loggers();
}
