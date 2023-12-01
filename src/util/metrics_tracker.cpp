// File: metrics_tracker.cpp
// Description: Holds metrics from search, and saves to file

#include "util/metrics_tracker.h"

#include <absl/strings/str_format.h>
#include <spdlog/spdlog.h>

#include <fstream>

namespace hpts {

MetricsTracker::MetricsTracker(const std::string& export_path, const std::string& file_name)
    : full_path_problem_metrics(absl::StrFormat("%s/metrics/%s_problems.csv", export_path, file_name)),
      full_path_iteration_metrics(absl::StrFormat("%s/metrics/%s_iterations.csv", export_path, file_name)) {
    // create directory for metrics
    if (std::filesystem::exists(full_path_problem_metrics)) {
        std::filesystem::remove(full_path_problem_metrics);
    }
    if (std::filesystem::exists(full_path_iteration_metrics)) {
        std::filesystem::remove(full_path_iteration_metrics);
    }
    std::filesystem::create_directories(absl::StrFormat("%s/metrics/", export_path));
}

auto operator<<(std::ostream& os, const ProblemMetricsItem& metrics_item) -> std::ostream& {
    os << metrics_item.bootstrap_iter << "," << metrics_item.puzzle_name << "," << metrics_item.solution_cost << ","
       << metrics_item.solution_prob << "," << metrics_item.expanded_nodes << "," << metrics_item.generated_nodes << ","
       << metrics_item.budget << "\n";
    return os;
}

auto operator<<(std::ostream& os, const IterationMetricsItem& metrics_item) -> std::ostream& {
    os << metrics_item.bootstrap_iter << "," << metrics_item.outstanding_problems << "," << metrics_item.ellapsed_seconds << "\n";
    return os;
}

void MetricsTracker::add_problem_row(const ProblemMetricsItem& metrics_item) noexcept {
    rows_problem_metrics.push_back(metrics_item);
}

void MetricsTracker::add_iteration_row(const IterationMetricsItem& metrics_item) noexcept {
    rows_iteration_metrics.push_back(metrics_item);
}

void MetricsTracker::clear() noexcept {
    rows_problem_metrics.clear();
    rows_iteration_metrics.clear();
}

void MetricsTracker::save_problem_metrics() noexcept {
    if (rows_problem_metrics.empty()) {
        return;
    }

    // We assume the export parent directory exists and can be written to
    std::ofstream export_file(full_path_problem_metrics, std::ofstream::app | std::ofstream::out);

    SPDLOG_INFO("Exporting metrics to {:s}", full_path_problem_metrics);
    for (auto const& row : rows_problem_metrics) {
        export_file << row;
    }
    export_file.close();
    rows_problem_metrics.clear();
}

void MetricsTracker::save_iteration_metrics() noexcept {
    if (rows_iteration_metrics.empty()) {
        return;
    }

    // We assume the export parent directory exists and can be written to
    std::ofstream export_file(full_path_iteration_metrics, std::ofstream::app | std::ofstream::out);

    SPDLOG_INFO("Exporting metrics to {:s}", full_path_iteration_metrics);
    for (auto const& row : rows_iteration_metrics) {
        export_file << row;
    }
    export_file.close();
    rows_iteration_metrics.clear();
}

void MetricsTracker::save() noexcept {
    save_problem_metrics();
    save_iteration_metrics();
}

}    // namespace hpts
