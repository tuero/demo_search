// File: torch_init.h
// Description: Initialize torch reproducibility

#ifndef HPTS_COMMON_TORCH_INIT_H_
#define HPTS_COMMON_TORCH_INIT_H_

namespace hpts {

/**
 * Initialize torch reproducibility
 * @param seed The seed to initialize torch rngs
 */
void init_torch(int seed);

/**
 * Reset the state of the torch seed
 * @param seed The seed to initialize torch rngs
 */
void reset_seed(int seed);

}    // namespace hpts

#endif    // HPTS_COMMON_TORCH_INIT_H_
