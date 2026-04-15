#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

#define TI_ALIGNED(n) __attribute__((aligned(n)))

typedef enum {
    TI_FLOAT32,
    TI_INT8
} ti_dtype_t;

typedef struct TI_ALIGNED(4) {
    void *data;
    uint32_t shape[4];
    uint8_t ndim;
    uint32_t size;
    ti_dtype_t dtype;
} ti_tensor_t;
int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim, ti_dtype_t dtype);
void ti_tensor_free(ti_tensor_t *tensor);
#endif