// File: stop_token.cpp
// Description: std::stop_token like flag class to signal for threads

#include "util/stop_token.h"

namespace hpts {

void StopToken::stop() noexcept {
    flag_ = true;
}

auto StopToken::stop_requested() const noexcept -> bool {
    return flag_;
}

}    // namespace hpts
