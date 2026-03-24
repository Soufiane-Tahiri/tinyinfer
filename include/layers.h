#ifndef LAYERS_H
#define LAYERS_H
#include "tensor.h"

typedef struct {
    uint32_t input_size;
    uint32_t output_size;
    void *weights;
    void *bias;
    ti_dtype_t dtype;
}ti_layer_t;
int ti_dense_forward(ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output);

#endif
