#ifndef LAYERS_H
#define LAYERS_H
#include "tensor.h"

typedef struct TI_ALIGNED(4) {
    uint32_t input_size;
    uint32_t output_size;
    const void *weights;
    const void *bias;
    ti_dtype_t dtype;
} ti_layer_t;


int ti_dense_forward(ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output);

int ti_dense_forward_float32(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output);
int ti_dense_forward_int8(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output);
int ti_dense_validate_config(uint32_t input_size, uint32_t output_size);
int ti_dense_validate_serialized_buffers(
    uint32_t input_size,
    uint32_t output_size,
    uint32_t weights_len_bytes,
    uint32_t bias_len_bytes,
    ti_dtype_t dtype
);

#endif
