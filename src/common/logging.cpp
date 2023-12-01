// File: logging.h
// Description: Logging setup

#include "common/logging.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
// NOLINTEND

#include <filesystem>
#include <memory>
#include <vector>

namespace hpts {

void init_loggers(const std::string &path, bool console_only, const std::string &postfix) {
    std::vector<spdlog::sink_ptr> sinks;
    auto log_file_path = absl::StrCat(path, absl::StrFormat("/log%s.log", postfix));
    if (std::filesystem::exists(log_file_path)) {
        std::filesystem::remove(log_file_path);
    }
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    if (!console_only) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path));
    }
    auto combined_logger = std::make_shared<spdlog::logger>("name", begin(sinks), end(sinks));
    spdlog::register_logger(combined_logger);
    spdlog::set_default_logger(combined_logger);
    spdlog::set_pattern("[%Y-%m-%d] [%H:%M:%S] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);
}

void log_flags(int argc, char **argv) {
    std::vector<std::string> all_args_vec(argv, argv + argc);    // NOLINT (*-pointer-arithmetic)
    std::string all_args_str;
    for (const auto &s : all_args_vec) {
        all_args_str += s;
        all_args_str += " ";
    }
    SPDLOG_INFO("Command used: {:s}", all_args_str);
}

void log_flush() {
    spdlog::get("name")->flush();
}

void close_loggers() {
    spdlog::shutdown();
}

}    // namespace hpts
