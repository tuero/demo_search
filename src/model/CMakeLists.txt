set(model_core_files
    base_model_wrapper.cpp 
    base_model_wrapper.h
    device_manager.h
    layers.cpp 
    layers.h
    loss_functions.cpp 
    loss_functions.h
    model_evaluator.h
    torch_util.cpp
    torch_util.h
)

add_library(model OBJECT ${model_core_files})
target_compile_features(model PUBLIC cxx_std_20)

add_subdirectory(heuristic_convnet)
add_subdirectory(policy_convnet)
add_subdirectory(twoheaded_convnet)
# target_link_libraries(model PUBLIC ${TORCH_LIBRARIES})
