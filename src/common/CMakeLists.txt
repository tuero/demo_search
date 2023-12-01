add_library(common OBJECT 
    logging.cpp 
    logging.h
    signaller.cpp 
    signaller.h
    state_loader.h
    torch_init.cpp 
    torch_init.h
    types.h
)
target_compile_features(common PUBLIC cxx_std_20)
