// File: phs.h
// Description: PHS* implementation

#ifndef HPTS_ALGORITHM_PHS_H_
#define HPTS_ALGORITHM_PHS_H_

#include <absl/container/flat_hash_set.h>

#include <cmath>
#include <concepts>
#include <exception>
#include <memory>
#include <queue>
#include <string>
#include <variant>
#include <vector>

// Set logging library macro level to remove debug logging out at compile time
// NOLINTBEGIN
#ifdef DEBUG_PRINT
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif
#include <spdlog/spdlog.h>
// NOLINTEND

#include "algorithm/yieldable.h"
#include "common/observation.h"
#include "env/simple_env.h"
#include "model/model_evaluator.h"
#include "model/policy_convnet/policy_convnet_wrapper.h"          // For inference input/output types
#include "model/twoheaded_convnet/twoheaded_convnet_wrapper.h"    // For inference input/output types
#include "util/concepts.h"
#include "util/priority_set.h"
#include "util/utility.h"
#include "util/zip.h"

namespace hpts::algorithm::phs {

// Search properties and various constants
constexpr double WEIGHT = 1.0;
static std::size_t INFERENCE_BATCH_SIZE = 1;         // NOLINT(*-non-const-global-variables,*-avoid-magic-numbers)
static std::size_t BLOCK_ALLOCATION_SIZE = 10000;    // NOLINT(*-non-const-global-variables,*-avoid-magic-numbers)
static double MIX_EPSILON = 0;                       // NOLINT(*-non-const-global-variables,*-avoid-magic-numbers)
constexpr double EPS = 1e-8;                         // NOLINT(*-non-const-global-variables,*-avoid-magic-numbers)

// All states must satisfy constraints
template <typename T>
concept PHSEnv = env::SimpleEnv<T>;

// Model types we can work with
using PHSPolicyNetEvaluator = std::variant<model::ModelEvaluator<model::wrapper::TwoHeadedConvNetWrapperLevin>,
                                           model::ModelEvaluator<model::wrapper::TwoHeadedConvNetWrapperPolicyGradient>,
                                           model::ModelEvaluator<model::wrapper::TwoHeadedConvNetWrapperPHS>,
                                           model::ModelEvaluator<model::wrapper::PolicyConvNetWrapperLevin>,
                                           model::ModelEvaluator<model::wrapper::PolicyConvNetWrapperPolicyGradient>,
                                           model::ModelEvaluator<model::wrapper::PolicyConvNetWrapperPHS>>;

// Input to PHS search algorithm
template <PHSEnv EnvT, model::IsModelEvaluator PHSEvaluatorT>
    requires IsTypeAmongVariant<PHSEvaluatorT, PHSPolicyNetEvaluator>
struct SearchInput {
    std::string puzzle_name;
    EnvT state;
    int search_budget = {};
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<PHSEvaluatorT> model_eval;
};

// Search algorithm output
template <PHSEnv EnvT>
struct SearchOutput {
    std::string puzzle_name;
    bool solution_found = false;
    double solution_cost = -1;
    int num_expanded = 0;
    int num_generated = 0;
    double solution_prob = 1;
    double solution_log_prob = 0;
    std::vector<EnvT> solution_path_states{};
    std::vector<Observation> solution_path_observations{};
    std::vector<int> solution_path_actions{};
    std::vector<double> solution_path_costs{};
};

namespace detail {
// Node used in search
template <PHSEnv EnvT>
struct Node {
    Node() = delete;
    Node(const EnvT &state) : state(state) {}

    void apply_action(Node<EnvT> &current, double cost, int a) {
        state.apply_action(a);
        log_p = current.log_p + current.action_log_prob[a];
        g = current.g + cost;
        action = a;
    }

    struct Hasher {
        using is_transparent = void;
        std::size_t operator()(const Node &node) const {
            return node.state.get_hash();
        }
        std::size_t operator()(const std::unique_ptr<Node> &node) const {
            return node->state.get_hash();
        }
    };
    struct CompareEqual {
        using is_transparent = void;
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.state == rhs.state;
        }
        bool operator()(const std::unique_ptr<Node> &lhs, const std::unique_ptr<Node> &rhs) const {
            return lhs->state == rhs->state;
        }
        bool operator()(const std::unique_ptr<Node> &lhs, const Node &rhs) const {
            return lhs->state == rhs.state;
        }
        bool operator()(const Node &lhs, const std::unique_ptr<Node> &rhs) const {
            return lhs.state == rhs->state;
        }
    };
    struct CompareOrderedLess {
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.cost < rhs.cost;
        }
    };
    struct CompareOrderedGreater {
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.cost > rhs.cost;
        }
    };

    // NOLINTBEGIN (misc-non-private-member-variables-in-classes)
    EnvT state;
    double log_p = 0;
    double g = 0;
    double h = 0;
    double cost = 0;
    Node *parent = nullptr;
    int action = -1;
    std::vector<double> action_log_prob{};
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};

// PHS cost
static constexpr double phs_cost(double log_p, double g, double h) {
    h = (h < 0) ? 0 : h;
    return g == 0 ? 0 : std::log(h + g + EPS) - (log_p * (1.0 + (h / g)));
}
}    // namespace detail

template <PHSEnv EnvT, model::IsModelEvaluator PHSEvaluatorT>
class YieldablePHS {
    using NodeT = detail::Node<EnvT>;
    using InferenceInputT = PHSEvaluatorT::InferenceInput;
    using InferenceOutputT = PHSEvaluatorT::InferenceOutput;
    using OpenListT =
        PrioritySet<NodeT, typename NodeT::CompareOrderedLess, typename NodeT::Hasher, typename NodeT::CompareEqual>;
    using ClosedListT = absl::flat_hash_set<std::unique_ptr<NodeT>, typename NodeT::Hasher, typename NodeT::CompareEqual>;

public:
    YieldablePHS(const SearchInput<EnvT, PHSEvaluatorT> &input) : input(input), status(Status::INIT), model(input.model_eval) {
        reset();
    }

    // Initialize the search with root node inference output
    void init() {
        SPDLOG_DEBUG("Initializing PHS: budget: {:d}", input.search_budget);
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        NodeT root_node(input.state);
        inference_inputs.emplace_back(root_node.state.get_observation());
        inference_nodes.push_back(root_node);
        batch_predict();
        SPDLOG_DEBUG("Initializing open: ");
        status = Status::OK;
    }

    void reset() {
        status = Status::INIT;
        timeout = false;
        search_output = SearchOutput<EnvT>{.puzzle_name = input.puzzle_name};
        inference_nodes.clear();
        inference_inputs.clear();
        open.clear();
        closed.clear();
    }

    void reset(const SearchInput<EnvT, PHSEvaluatorT> &input) {
        this->input = input;
        model = std::get<0>(input.model_evals);
        reset();
    }

    // Single step of the search algorithm
    void step() {
        if (open.empty()) {
            status = Status::ERROR;
            SPDLOG_ERROR("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }

        // Remove top node from open and put into closed
        auto current_u_ptr = std::make_unique<NodeT>(*open.pop_and_move());
        auto &current = static_cast<NodeT &>(*current_u_ptr);
        const auto current_ptr = current_u_ptr.get();
        closed.insert(std::move(current_u_ptr));
        ++search_output.num_expanded;

        SPDLOG_DEBUG("-------------------------------------");
        SPDLOG_DEBUG("Expanding: {:d}, log_p: {:2f}, g: {:.2f}, h: {:.2f}", search_output.num_expanded, current.log_p, current.g,
                     current.h);
        SPDLOG_DEBUG("\n{:s}", current.state.to_str());

        // Timeout
        if (input.search_budget >= 0 && search_output.num_expanded >= input.search_budget) {
            timeout = true;
            SPDLOG_INFO("Buget timeout - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}", input.puzzle_name,
                        search_output.num_expanded, search_output.num_generated, input.search_budget);
            status = Status::TIMEOUT;
            return;
        }

        // Consider all children
        for (const auto &a : current.state.child_actions()) {
            NodeT child_node = current;
            child_node.parent = current_ptr;
            child_node.apply_action(current, 1, a);
            SPDLOG_DEBUG("Generating: {:d}, log_p: {:2f}, g: {:.2f}", a, child_node.log_p, child_node.g);
            SPDLOG_DEBUG("\n{:s}", child_node.state.to_str());

            // Solution found, no optimality guarantees so we return on generation instead of expansion
            if (child_node.state.is_solution()) {
                SPDLOG_INFO("Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}", input.puzzle_name,
                            search_output.num_expanded, search_output.num_generated, input.search_budget, child_node.g);
                set_solution_trajectory(child_node);
                status = Status::SOLVED;
                return;
            }

            // If new state, add to queue for inference
            if (closed.find(child_node) == closed.end() && !open.contains(child_node)) {
                inference_inputs.emplace_back(child_node.state.get_observation());
                inference_nodes.push_back(std::move(child_node));
            }
        }

        SPDLOG_DEBUG("Open size: {:d}, Inference batched size: {:d}", open.size(), inference_inputs.size());

        // Batch inference
        if (open.empty() || inference_inputs.size() >= INFERENCE_BATCH_SIZE) {
            batch_predict();
        }
    }

    [[nodiscard]] Status get_status() const {
        return status;
    }

    [[nodiscard]] SearchOutput<EnvT> get_search_output() const {
        return search_output;
    }

    EnvT get_open_state(std::size_t index = 0) {
        if (index >= open.size()) {
            throw std::invalid_argument("Index out of bounds");
        }
        OpenListT copy = open;
        while (index > 0) {
            open.pop();
        }
        return *open.top()->state;
    }

private:
    // Batch predict inference
    void batch_predict() {
        SPDLOG_DEBUG("Running inference.");
        std::vector<InferenceOutputT> predictions = model->Inference(inference_inputs);
        for (auto &&[child_node, prediction] : zip(inference_nodes, predictions)) {
            // Net output has heuristic data member
            if constexpr (HasHeuristic<InferenceOutputT>) {
                child_node.h = prediction.heuristic;
            }
            child_node.action_log_prob = log_policy_noise(prediction.policy, MIX_EPSILON);
            child_node.cost = detail::phs_cost(child_node.log_p, child_node.g, child_node.h);

            SPDLOG_DEBUG("Adding child to open: logp: {:f}, g: {:.2f}, h: {:.2f}, c: {:.2f}, low: {:s}", child_node.log_p,
                         child_node.g, child_node.h, child_node.cost, vec_to_str(child_node.action_log_prob));
            SPDLOG_DEBUG("\n{:s}", child_node.state.to_str());
            open.push(std::move(child_node));
            ++search_output.num_generated;
        }
        inference_inputs.clear();
        inference_nodes.clear();
    }

    // Walk backwards up until the root, setting data
    void set_solution_trajectory(const NodeT &node) {
        double solution_cost = 0;
        auto current = &node;
        search_output.solution_found = true;
        search_output.solution_cost = current->g;
        search_output.solution_prob = std::exp(current->log_p);
        search_output.solution_log_prob = current->log_p;
        while (current->parent) {
            search_output.solution_path_states.push_back(current->parent->state);
            search_output.solution_path_observations.push_back(current->parent->state.get_observation());
            search_output.solution_path_actions.push_back(current->action);
            SPDLOG_DEBUG("c: {:2f}", solution_cost);
            SPDLOG_DEBUG("\n{:s}", current->state.to_str());
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            current = current->parent;
        }
    }

    SearchInput<EnvT, PHSEvaluatorT> input;           // Search input, contaning problem instance, models, budget, etc.
    Status status{};                                  // Current search status
    bool timeout = false;                             // Timout flag on budget
    std::shared_ptr<PHSEvaluatorT> model;             // Policy network with optional heuristic
    SearchOutput<EnvT> search_output;                 // Output of the search algorithm, containing trajectory + stats
    std::vector<NodeT> inference_nodes;               // Nodes in queue for batch inference
    std::vector<InferenceInputT> inference_inputs;    // Corresponding input structs the network evaluator expects
    OpenListT open;                                   // Open list
    ClosedListT closed;                               // Closed list
};

template <PHSEnv EnvT, model::IsModelEvaluator PHSEvaluatorT>
auto search(const SearchInput<EnvT, PHSEvaluatorT> &input) -> SearchOutput<EnvT> {
    YieldablePHS<EnvT, PHSEvaluatorT> step_phs(input);
    step_phs.init();
    // Iteratively search until status changes (solved or timeout)
    while (step_phs.get_status() == Status::OK && !input.stop_token->stop_requested()) {
        step_phs.step();
    }
    return step_phs.get_search_output();
}

}    // namespace hpts::algorithm::phs

#endif    // HPTS_ALGORITHM_PHS_H_
