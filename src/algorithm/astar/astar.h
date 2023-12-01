// File: astar.h
// Description: A* implementation

#ifndef HPTS_ALGORITHM_ASTAR_H_
#define HPTS_ALGORITHM_ASTAR_H_

// NOLINTBEGIN
#ifdef DEBUG_PRINT
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif
#include <spdlog/spdlog.h>
// NOLINTEND

#include <string>
#include <vector>

#include "algorithm/yieldable.h"
#include "env/simple_env.h"
#include "model/heuristic_convnet/heuristic_convnet_wrapper.h"    // For inference input/output types
#include "model/model_evaluator.h"
#include "util/block_allocator.h"
#include "util/concepts.h"
#include "util/priority_set.h"
#include "util/utility.h"
#include "util/zip.h"

namespace hpts::algorithm::astar {

constexpr double WEIGHT = 1.0;
static std::size_t INFERENCE_BATCH_SIZE = 1;    // NOLINT (*-non-const-global-variables)

// All states must satisfy constraints
template <typename T>
concept AStarEnv = env::SimpleEnv<T>;

// Model types we can work with
using AStarHeuristicNetEvaluator = std::variant<model::ModelEvaluator<model::wrapper::HeuristicConvNetWrapperMSE>>;

// Input to AStar search algorithm
template <AStarEnv EnvT, model::IsModelEvaluator AStarEvaluatorT>
    requires IsTypeAmongVariant<AStarEvaluatorT, AStarHeuristicNetEvaluator>
struct SearchInputModel {
    std::string puzzle_name;
    EnvT state;
    int search_budget = {};
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<AStarEvaluatorT> model_eval;
};
template <AStarEnv EnvT>
struct SearchInputNoModel {
    std::string puzzle_name;
    EnvT state;
    int search_budget = {};
    std::shared_ptr<StopToken> stop_token;
};

// Search algorithm output
template <AStarEnv EnvT>
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
template <AStarEnv EnvT>
struct Node {
    void set(double h, std::vector<double> action_log_prob) {
        this->h = h;
        this->action_log_prob = std::move(action_log_prob);
    }

    void apply_action(Node<EnvT> &current, double cost, int a) {
        state.apply_action(a);
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
            return lhs.cost < rhs.cost || (lhs.cost == rhs.cost && lhs.g > rhs.g);
        }
    };
    struct CompareOrderedGreater {
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.cost > rhs.cost;
        }
    };

    // NOLINTBEGIN (misc-non-private-member-variables-in-classes)
    EnvT state;
    double g = 0;
    double h = 0;
    double cost = 0;
    Node *parent = nullptr;
    int action = -1;
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};
}    // namespace detail

template <AStarEnv EnvT, model::IsModelEvaluator AStarEvaluatorT>
    requires IsTypeAmongVariant<AStarEvaluatorT, AStarHeuristicNetEvaluator> &&
             HasHeuristic<typename AStarEvaluatorT::InferenceOutput>
class YieldableAStarModel {
    using NodeT = detail::Node<EnvT>;
    using InferenceInputT = AStarEvaluatorT::InferenceInput;
    using InferenceOutputT = AStarEvaluatorT::InferenceOutput;
    using OpenListT =
        PrioritySet<NodeT, typename NodeT::CompareOrderedLess, typename NodeT::Hasher, typename NodeT::CompareEqual>;
    using ClosedListT = absl::flat_hash_set<std::unique_ptr<NodeT>, typename NodeT::Hasher, typename NodeT::CompareEqual>;

public:
    YieldableAStarModel(const SearchInputModel<EnvT, AStarEvaluatorT> &input)
        : input(input), status(Status::INIT), model(input.model_eval) {
        reset();
    }

    void init() {
        SPDLOG_DEBUG("Initializing A*: budget: {:d}", input.search_budget);
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        {
            NodeT root_node(input.state);
            inference_inputs.emplace_back(root_node.state.get_observation());
            inference_nodes.push_back(root_node);
            batch_predict();
        }
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

    void step() {
        if (open.empty()) {
            status = Status::ERROR;
            SPDLOG_ERROR("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }
        auto current_u_ptr = std::make_unique<NodeT>(*open.pop_and_move());
        auto &current = static_cast<NodeT &>(*current_u_ptr);
        const auto current_ptr = current_u_ptr.get();
        closed.insert(std::move(current_u_ptr));
        ++search_output.num_expanded;

        SPDLOG_DEBUG("-------------------------------------");
        SPDLOG_DEBUG("Expanding: {:d}, g: {:.2f}, h: {:.2f}, c:{:.2f}", search_output.num_expanded, current.g, current.h,
                     current.cost);
        SPDLOG_DEBUG("\n{:s}", current.state.to_str());

        // Solution found
        if (current.state.is_solution()) {
            SPDLOG_INFO("Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}", input.puzzle_name,
                        search_output.num_expanded, search_output.num_generated, input.search_budget, current.g);
            set_solution_trajectory(current);
            status = Status::SOLVED;
            return;
        }

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
            SPDLOG_DEBUG("Generating: {:d}, g: {:.2f}", a, child_node.g);
            SPDLOG_DEBUG("\n{:s}", child_node.state.to_str());

            if (closed.find(child_node) == closed.end() && !open.contains(child_node)) {
                inference_inputs.emplace_back(child_node.state.get_observation());
                inference_nodes.push_back(child_node);
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
    void batch_predict() {
        SPDLOG_DEBUG("Running inference.");
        std::vector<InferenceOutputT> predictions = model->Inference(inference_inputs);
        for (auto &&[child_node, prediction] : zip(inference_nodes, predictions)) {
            child_node.h = prediction.heuristic;
            child_node.cost = child_node.g + child_node.h;
            open.push(std::move(child_node));
            ++search_output.num_generated;
        }
        inference_inputs.clear();
        inference_nodes.clear();
    }

    void set_solution_trajectory(const NodeT &node) {
        double solution_cost = 0;
        auto current = &node;
        search_output.solution_found = true;
        search_output.solution_cost = current->g;
        search_output.solution_prob = 1;
        search_output.solution_log_prob = 0;
        while (current->parent) {
            search_output.solution_path_states.push_back(current->parent->state);
            search_output.solution_path_observations.push_back(current->parent->state.get_observation());
            search_output.solution_path_actions.push_back(current->action);
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            current = current->parent;
        }
    }

    SearchInputModel<EnvT, AStarEvaluatorT> input;
    Status status{};
    bool timeout = false;
    std::shared_ptr<AStarEvaluatorT> model;
    SearchOutput<EnvT> search_output;
    std::vector<NodeT> inference_nodes;
    std::vector<InferenceInputT> inference_inputs;
    OpenListT open;
    ClosedListT closed;
};

template <AStarEnv EnvT>
class YieldableAStarNoModel {
    using NodeT = detail::Node<EnvT>;
    using OpenListT =
        PrioritySet<NodeT, typename NodeT::CompareOrderedLess, typename NodeT::Hasher, typename NodeT::CompareEqual>;
    using ClosedListT = absl::flat_hash_set<std::unique_ptr<NodeT>, typename NodeT::Hasher, typename NodeT::CompareEqual>;

public:
    YieldableAStarNoModel(const SearchInputNoModel<EnvT> &input) : input(input), status(Status::INIT) {
        reset();
    }

    void init() {
        SPDLOG_DEBUG("Initializing A*: budget: {:d}", input.search_budget);
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        {
            NodeT root_node(input.state);
            root_node.h = root_node.state.get_heuristic();
            root_node.cost = root_node.g + root_node.h;
            open.push(std::move(root_node));
            ++search_output.num_generated;
        }
        SPDLOG_DEBUG("Initializing open: ");
        status = Status::OK;
    }

    void reset() {
        status = Status::INIT;
        timeout = false;
        search_output = SearchOutput<EnvT>{.puzzle_name = input.puzzle_name};
        open.clear();
        closed.clear();
    }

    void step() {
        if (open.empty()) {
            status = Status::ERROR;
            SPDLOG_ERROR("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }
        auto current_u_ptr = std::make_unique<NodeT>(*open.pop_and_move());
        auto &current = static_cast<NodeT &>(*current_u_ptr);
        const auto current_ptr = current_u_ptr.get();
        closed.insert(std::move(current_u_ptr));
        ++search_output.num_expanded;

        SPDLOG_DEBUG("-------------------------------------");
        SPDLOG_DEBUG("Expanding: {:d}, g: {:.2f}, h: {:.2f}, c:{:.2f}", search_output.num_expanded, current.g, current.h,
                     current.cost);
        SPDLOG_DEBUG("\n{:s}", current.state.to_str());

        // Solution found
        if (current.state.is_solution()) {
            SPDLOG_INFO("Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}", input.puzzle_name,
                        search_output.num_expanded, search_output.num_generated, input.search_budget, current.g);
            set_solution_trajectory(current);
            status = Status::SOLVED;
            return;
        }

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

            child_node.h = child_node.state.get_heuristic();
            child_node.cost = child_node.g + child_node.h;
            SPDLOG_DEBUG("Generating: {:d}, g: {:.2f}", a, child_node.g);
            SPDLOG_DEBUG("\n{:s}", child_node.state.to_str());

            // consider
            if (consider_child(std::move(child_node))) {
                ++search_output.num_generated;
            }
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
    bool consider_child(NodeT &&child_node) {
        // Check closed for re-expansion
        const auto closed_iter = closed.find(child_node);
        if (closed_iter != closed.end()) {
            // Technically not needed for consistent heuristic
            if ((*closed_iter)->g > child_node.g) {
                closed.erase(closed_iter);
                open.push(std::move(child_node));
                return true;
            }
        }
        // Check open for better child found
        else if (open.contains(child_node)) {
            double g = open.get(child_node).g;
            if (g > child_node.g) {
                open.update(child_node);
                return true;
            }
        } else {
            open.push(std::move(child_node));
            return true;
        }
        return false;
    }

    void set_solution_trajectory(const NodeT &node) {
        double solution_cost = 0;
        auto current = &node;
        search_output.solution_found = true;
        search_output.solution_cost = current->g;
        search_output.solution_prob = 1;
        search_output.solution_log_prob = 0;
        while (current->parent) {
            search_output.solution_path_states.push_back(current->parent->state);
            search_output.solution_path_observations.push_back(current->parent->state.get_observation());
            search_output.solution_path_actions.push_back(current->action);
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            current = current->parent;
        }
    }

    SearchInputNoModel<EnvT> input;
    Status status{};
    bool timeout = false;
    SearchOutput<EnvT> search_output;
    OpenListT open;
    ClosedListT closed;
};

template <AStarEnv EnvT, model::IsModelEvaluator AStarEvaluatorT>
    requires IsTypeAmongVariant<AStarEvaluatorT, AStarHeuristicNetEvaluator>
auto search(const SearchInputModel<EnvT, AStarEvaluatorT> &input) -> SearchOutput<EnvT> {
    YieldableAStarModel<EnvT, AStarEvaluatorT> step_phs(input);
    step_phs.init();
    while (step_phs.get_status() == Status::OK && !input.stop_token->stop_requested()) {
        step_phs.step();
    }
    return step_phs.get_search_output();
}
template <AStarEnv EnvT>
auto search(const SearchInputNoModel<EnvT> &input) -> SearchOutput<EnvT> {
    YieldableAStarNoModel<EnvT> step_phs(input);
    step_phs.init();
    while (step_phs.get_status() == Status::OK && !input.stop_token->stop_requested()) {
        step_phs.step();
    }
    return step_phs.get_search_output();
}

}    // namespace hpts::algorithm::astar

#endif    // HPTS_ALGORITHM_ASTAR_H_
