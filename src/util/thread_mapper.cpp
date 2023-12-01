// File: thread_mapper.cpp
// Description: Mapping of thread IDs
#include "util/thread_mapper.h"

#include <spdlog/spdlog.h>

#include <mutex>
#include <thread>
#include <unordered_map>

namespace hpts::thread_mapper {

static std::mutex thread_id_mutex;                                        // NOLINT
static std::unordered_map<std::thread::id, std::size_t> thread_id_map;    // NOLINT

void clear() {
    std::lock_guard<std::mutex> lock(thread_id_mutex);
    thread_id_map.clear();
}

void add(std::size_t index) {
    std::lock_guard<std::mutex> lock(thread_id_mutex);
    thread_id_map[std::this_thread::get_id()] = index;
}

std::size_t get() {
    std::lock_guard<std::mutex> lock(thread_id_mutex);
    if (thread_id_map.find(std::this_thread::get_id()) == thread_id_map.end()) {
        SPDLOG_ERROR("Thread not mapped.");
        std::exit(1);
    }
    return thread_id_map[std::this_thread::get_id()];
}

}    // namespace hpts::thread_mapper
