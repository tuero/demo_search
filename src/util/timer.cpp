// File: timer.cpp
// Description: Measures user space time and signals a timeout

#include "util/timer.h"

namespace hpts {

Timer::Timer(double seconds_limit) : seconds_limit(seconds_limit) {}

void Timer::start() noexcept {
    cpu_start_time = std::clock();
}

auto Timer::is_timeout() const noexcept -> bool {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time) / CLOCKS_PER_SEC;
    return seconds_limit > 0 && current_duration >= seconds_limit;
}

auto Timer::get_duration() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time) / CLOCKS_PER_SEC;
    return current_duration;
}

auto Timer::get_time_remaining() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time) / CLOCKS_PER_SEC;
    return seconds_limit - current_duration;
}

}    // namespace hpts
