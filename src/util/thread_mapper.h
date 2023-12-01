// File: thread_mapper.h
// Description: Mapping of thread IDs

#ifndef HPTS_UTIL_THREAD_MAPPER_H_
#define HPTS_UTIL_THREAD_MAPPER_H_

#include <utility>

namespace hpts::thread_mapper {

void clear();

void add(std::size_t index);

std::size_t get();

}    // namespace hpts::thread_mapper

#endif    // HPTS_UTIL_THREAD_MAPPER_H_
