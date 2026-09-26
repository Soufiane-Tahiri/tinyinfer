#ifndef LAYERS_H
#define LAYERS_H
#include "tensor.h"

typedef enum {
    TI_ACT_NONE    = 0,
    TI_ACT_RELU    = 1,
    TI_ACT_SOFTMAX = 2,
} ti_activation_type_t;

typedef struct TI_ALIGNED(4) {
    uint32_t input_size;
    uint32_t output_size;
    const void *weights;
    const void *bias;
    ti_dtype_t dtype;
    ti_activation_type_t activation;
    float requant_scale;

} ti_layer_t;

int ti_dense_forward(const ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output);
int ti_dense_forward_float32(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output);
int ti_dense_forward_int8(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output);

#endif