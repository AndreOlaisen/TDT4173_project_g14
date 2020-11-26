#ifndef NN_UTIL_H__
#define NN_UTIL_H__

#include <stdint.h>

#include "arm_math.h"
#include "arm_nnsupportfunctions.h"

#define ARM_STATUS_CHECK(status) nn_arm_status_check(status)

void nn_arm_status_check(arm_status status);

void nn_dump_path_set(const char *path);
void nn_dump_open(void);
void nn_dump_activations(const char *name, const q7_t *act, size_t len);
void nn_dump_close(void);

#endif // NN_UTIL_H__
