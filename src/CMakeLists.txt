include_directories(.)

link_libraries(${TORCH_LIBRARIES})

link_libraries(spdlog::spdlog_header_only)
include_directories(${CMAKE_SOURCE_DIR}/external/libnop/include)

link_libraries(stonesngems)
link_libraries(sokoban)
link_libraries(craftworld)
link_libraries(boxworld)

link_libraries( 
    absl::base
    absl::flags
    absl::flags_parse
    absl::hash
    absl::strings
    absl::synchronization
    absl::optional
    absl::time
)

# Go through core directories
add_subdirectory(algorithm)
add_subdirectory(common)
add_subdirectory(env)
add_subdirectory(model)
add_subdirectory(util)

set(HPTS_CORE_OBJECTS 
    # $<TARGET_OBJECTS:algorithm>
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:env>
    $<TARGET_OBJECTS:model>
    $<TARGET_OBJECTS:util>
)

# Go through apps
add_subdirectory(apps)

# pybind11
add_subdirectory(python)

if (${BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

