// File: metrics_tracker.h
// Description: Holds metrics from search, and saves to file

#ifndef HPTS_UTIL_METRICS_TRACKER_H_
#define HPTS_UTIL_METRICS_TRACKER_H_

#include <absl/strings/str_format.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace hpts {

struct ProblemMetricsItem {
    int bootstrap_iter;
    std::string puzzle_name;
    double solution_cost;
    double solution_prob;
    int expanded_nodes;
    int generated_nodes;
    int budget;
    // Needed to write each item out to file
    friend auto operator<<(std::ostream& os, const ProblemMetricsItem& metrics_item) -> std::ostream&;
};

struct IterationMetricsItem {
    int bootstrap_iter;
    std::size_t outstanding_problems;
    double ellapsed_seconds;
    // Needed to write each item out to file
    friend auto operator<<(std::ostream& os, const IterationMetricsItem& metrics_item) -> std::ostream&;
};

class MetricsTracker {
public:
    MetricsTracker() = delete;
    MetricsTracker(const std::string& export_path, const std::string& file_name);
    void add_problem_row(const ProblemMetricsItem& metrics_item) noexcept;
    void add_iteration_row(const IterationMetricsItem& metrics_item) noexcept;
    void clear() noexcept;
    void save() noexcept;
    void save_problem_metrics() noexcept;
    void save_iteration_metrics() noexcept;

private:
    std::vector<ProblemMetricsItem> rows_problem_metrics;
    std::vector<IterationMetricsItem> rows_iteration_metrics;
    std::string full_path_problem_metrics;
    std::string full_path_iteration_metrics;
};

}    // namespace hpts

#endif    // HPTS_UTIL_METRICS_TRACKER_H_
