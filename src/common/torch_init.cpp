// File: torch_init.cpp
// Description: Logging setup

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts {

void init_torch(int seed) {
    // Set torch seed
    torch::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);
    torch::globalContext().setDeterministicAlgorithms(true, false);
}

void reset_seed(int seed) {
    torch::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
}

}    // namespace hpts
