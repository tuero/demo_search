add_library(algorithm_phs OBJECT 
    phs.h 
    train.h
)
target_compile_features(algorithm_phs PUBLIC cxx_std_20)
