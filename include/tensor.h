#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
typedef struct {
    float *data;
    uint32_t shape[4];
    uint8_t ndim;
    uint32_t size;
} ti_tensor_t;
int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim);
void ti_tensor_free(ti_tensor_t *tensor);
#endif