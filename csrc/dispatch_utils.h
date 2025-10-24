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

template <int ONLY_ORDERED_VAL>
void dispatchToNextIntInStaticOrderedIntsInner(int val, auto &&lambda) {
    if (val <= ONLY_ORDERED_VAL) {
        lambda.template operator()<ONLY_ORDERED_VAL>();
    } else {
        throw std::runtime_error(
            "Invalid value passed to dispatchToStaticInts: " +
            std::to_string(val) +
            ", should be <=" + std::to_string(ONLY_ORDERED_VAL));
    }
}

template <int FIRST_ORDERED_VAL, int... OTHER_ORDERED_VALS>
void dispatchToNextIntInStaticOrderedIntsInner(int val, auto &&lambda) {
    if (val <= FIRST_ORDERED_VAL) {
        lambda.template operator()<FIRST_ORDERED_VAL>();
    } else {
        dispatchToStaticIntsInner<OTHER_ORDERED_VALS...>(val, lambda);
    }
}

} // namespace detail

/**
 * Dispatch to template integers.
 *
 * Given an run-time integer and a template lambda accepting this integer as a
 * compile-time constant as its first template argument, dispatch to the correct
 * implementation of the lambda.
 *
 * Suppose the candidates are 10, 20, 30 and the runtime value is 20, then the
 * lambda will be called with the template argument 20. An exception will be
 * thrown if the runtime value is not one of the candidates.
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

/**
 * Dispatch to the next value in an ordered set of integers as a template
 * argument.
 *
 * Given a run-time integer and a template lambda accepting this integer as a
 * compile-time constant as its first template argument, dispatch to the correct
 * implementation of the lambda.
 *
 * Suppose the candidates are 10, 20, 30 and the runtime value is 15, then the
 * lambda will be called with the template argument 20.
 */
template <int... ORDERED_VALS>
void dispatchToNextIntInStaticOrderedInts(int val, auto &&lambda) {
    detail::dispatchToNextIntInStaticOrderedIntsInner<ORDERED_VALS...>(val,
                                                                       lambda);
}
