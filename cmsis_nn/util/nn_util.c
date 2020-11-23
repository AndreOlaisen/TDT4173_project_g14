#include "nn_util.h"
#include <stdio.h>
#include <stdbool.h>


static const char *file_path;
static FILE *fp;


void nn_arm_status_check(arm_status status)
{
    if (status != ARM_MATH_SUCCESS) {
        printf("Warning: got error %d\n", status);
    }
}

void nn_dump_path_set(const char *path)
{
    file_path = path;
}

void nn_dump_open(void)
{
    if (file_path == NULL) {
        return;
    }

    fp = fopen(file_path, "w");
    fputs("[\n", fp);
}

void nn_dump_activations(const char *name, const q7_t *act, size_t len)
{
    if (fp == NULL) {
        return;
    }

    static bool first = true;
    if (!first) {
        fputs(",\n", fp);
    } else {
        first = false;
    }

    fprintf(fp, "{\n    \"name\": \"%s\",\n", name);
    fputs("    \"activations\": [", fp);
    for (size_t i = 0; i < len - 1; i++) {
        fprintf(fp, "\"0x%02X\", ", ((unsigned int) act[i]) & 0xFF);
    }
    fprintf(fp, "\"0x%02X\"]\n}", ((unsigned int) act[len - 1]) & 0xFF);
}

void nn_dump_close(void)
{
    if (fp == NULL) {
        return;
    }

    fputs("\n]", fp);
    fclose(fp);
    fp = NULL;
}
