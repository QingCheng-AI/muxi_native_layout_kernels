#pragma once

#include <sstream>
#include <stdexcept>

namespace detail {

bool dispatchToStaticIntsInner(int val, auto &&lambda) { return false; }

template <int FIRST_POSSIBLE_VAL, int... OTHER_POSSIBLE_VALS>
bool dispatchToStaticIntsInner(int val, auto &&lambda) {
    if (val == FIRST_POSSIBLE_VAL) {
        lambda.template operator()<FIRST_POSSIBLE_VAL>();
        return true;
    } else {
        return dispatchToStaticIntsInner<OTHER_POSSIBLE_VALS...>(val, lambda);
    }
}

} // namespace detail

/**
 * Dispatch to template integers.
 *
 * Given an run-time integer and a template lambda accepting this integer as a
 * compile-time constant as its first template argument, dispatch to the correct
 * implementation of the lambda.
 */
template <int... POSSIBLE_VALS>
void dispatchToStaticInts(int val, auto &&lambda) {
    auto success =
        detail::dispatchToStaticIntsInner<POSSIBLE_VALS...>(val, lambda);
    if (!success) {
        std::ostringstream oss;
        oss << "Invalid value passed to dispatchToStaticInts: " << val
            << " should be one of: ";
        ((oss << POSSIBLE_VALS << ", "), ...);
        throw std::runtime_error(oss.str());
    }
}
