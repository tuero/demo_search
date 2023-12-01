add_library(util OBJECT 
    block_allocator.h
    metrics_tracker.cpp 
    metrics_tracker.h
    priority_set.h
    queue.h 
    replay_buffer.h
    stop_token.cpp 
    stop_token.h
    thread_mapper.cpp 
    thread_mapper.h
    thread_pool.h
    timer.cpp
    utility.cpp
    utility.h
    zip.h
)

target_compile_features(util PUBLIC cxx_std_20)
