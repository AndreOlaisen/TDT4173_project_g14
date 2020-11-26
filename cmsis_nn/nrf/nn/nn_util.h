#ifndef NN_UTIL_H__
#define NN_UTIL_H__

#include <stdint.h>

#include <arm_math.h>
#include <arm_nnsupportfunctions.h>

static inline q7_t preprocess(uint8_t c, int mean, unsigned int shift)
{
    int full_shift = shift + 7;
    int x = c;
    x = (x - mean) << 7;
    x = x + NN_ROUND(full_shift);
    x = x >> full_shift;
    if (x < INT8_MIN) {
        return (q7_t) INT8_MIN;
    } else if (x > INT8_MAX) {
        return (q7_t) INT8_MAX;
    } else {
        return (q7_t) x;
    }
    // return (q7_t) __SSAT(x, 8);
}

#define ARM_STATUS_CHECK(status)


#endif // NN_UTIL_H__