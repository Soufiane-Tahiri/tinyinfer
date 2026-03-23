#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tensor.h"

int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim, ti_dtype_t dtype) {
    if (tensor == NULL) return -1;

    for (uint8_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->size = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;

    uint32_t elem_size = (dtype == TI_FLOAT32) ? sizeof(float) : sizeof(int8_t);
    tensor->data = malloc(tensor->size * elem_size);
    if (tensor->data == NULL) return -1;

    memset(tensor->data, 0, tensor->size * elem_size);
    return 0;
}

void ti_tensor_free(ti_tensor_t *tensor) {
    if (tensor == NULL) return;
    free(tensor->data);
    tensor->data = NULL;
    tensor->size = 0;
    tensor->ndim = 0;
}