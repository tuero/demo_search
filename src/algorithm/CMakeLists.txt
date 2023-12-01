add_library(algorithm OBJECT 
    test_runner.h 
    train_bootstrap.h
)
target_compile_features(algorithm PUBLIC cxx_std_20)

add_subdirectory(astar)
add_subdirectory(phs)
