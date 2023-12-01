set(env_core_files 
    simple_env.h
)

add_library(env OBJECT ${env_core_files})
target_compile_features(env PUBLIC cxx_std_20)

add_subdirectory(rnd)
add_subdirectory(sokoban)
add_subdirectory(craftworld)
add_subdirectory(boxworld)
