add_executable(filter_problems filter_problems.cpp  ${HPTS_CORE_OBJECTS}  $<TARGET_OBJECTS:algorithm> $<TARGET_OBJECTS:algorithm_astar>)
target_compile_features(filter_problems PUBLIC cxx_std_20)