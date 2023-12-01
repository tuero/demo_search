cmake_minimum_required (VERSION 3.13)

project(hpts)

option(BUILD_TESTS "Build the unit tests" OFF)

include(FetchContent)

# Abseil
message("Configuring absl")
FetchContent_Declare(absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(absl)

# spdlog
message("Configuring spdlog")
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
    add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()

# Pybind11
message("Configuring Pybind11")
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG master
)
FetchContent_MakeAvailable(pybind11)

# Find libtorch
message("Configuring libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Environment dependencies
# Boxworld
message("Configuring BoxWorld")
FetchContent_Declare(boxworld
    GIT_REPOSITORY https://github.com/tuero/boxworld_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(boxworld)

# Craftworld
message("Configuring CraftWorld")
FetchContent_Declare(craftworld
    GIT_REPOSITORY https://github.com/tuero/craftworld_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(craftworld)

# Craftworld
message("Configuring Sokoban")
FetchContent_Declare(sokoban
    GIT_REPOSITORY https://github.com/tuero/sokoban_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(sokoban)

# StonesNGems
message("Configuring Stonesngems")
FetchContent_Declare(stonesngems
    GIT_REPOSITORY https://github.com/tuero/stonesngems_cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stonesngems)


# Project directories
add_subdirectory(src)

