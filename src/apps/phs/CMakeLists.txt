add_executable(phs main.cpp config.h config.cpp ${HPTS_CORE_OBJECTS}  $<TARGET_OBJECTS:algorithm> $<TARGET_OBJECTS:algorithm_phs>)
target_compile_features(phs PUBLIC cxx_std_20)
