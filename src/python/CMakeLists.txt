# Sources
set(HPTSPY_SOURCES
    common.cpp
    common.h
    env.cpp
    env.h 
    util.cpp
    util.h 
)

# python module
pybind11_add_module(_hptspy EXCLUDE_FROM_ALL hpts_py.cpp ${HPTSPY_SOURCES} ${HPTS_CORE_OBJECTS})
target_compile_options(_hptspy PRIVATE
    -flto=auto 
)
target_compile_features(_hptspy PUBLIC cxx_std_20)

add_subdirectory(algorithm)
add_subdirectory(model)
