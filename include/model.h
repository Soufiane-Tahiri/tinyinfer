#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "layers.h"

typedef struct {
    uint32_t num_layers;
    ti_layer_t *layers;
    ti_dtype_t dtype;
    uint32_t input_size;
    uint32_t output_size;
    float input_scale;
} ti_model_t;

int ti_model_load(ti_model_t *model, const char *path);
int ti_model_run(const ti_model_t *model, ti_tensor_t *input, ti_tensor_t *output);
void ti_model_free(ti_model_t *model);
bool ti_load_model_int8_mem(const uint8_t *buf, size_t size, ti_model_t *model);

#endif