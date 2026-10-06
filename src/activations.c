#include <stdint.h>
#include "activations.h"
int ti_relu_forward_float32(const ti_tensor_t *input, ti_tensor_t *output) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL) return -1;
    if (input->dtype != TI_FLOAT32 || output->dtype != TI_FLOAT32) return -1;
    if (input->size!=output->size) return -1;

    const float *x = (const float *)input->data;
    float *y = (float *)output->data;
    uint32_t i=0;
    for (; i+3<input->size; i+=4) {
        y[i+0] = x[i+0] > 0.0f ? x[i+0] : 0.0f;
        y[i+1] = x[i+1] > 0.0f ? x[i+1] : 0.0f;
        y[i+2] = x[i+2] > 0.0f ? x[i+2] : 0.0f;
        y[i+3] = x[i+3] > 0.0f ? x[i+3] : 0.0f;
    }
    for (; i<input->size; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return 0;
}

int ti_relu_forward_int8(const ti_tensor_t *input, ti_tensor_t *output) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL) return -1;
    if (input->dtype != TI_INT8 || output->dtype != TI_INT8) return -1;
    if (input->size!=output->size) return -1;
    const int8_t *x = (const int8_t *)input->data;
    int8_t *y = (int8_t *)output->data;
    uint32_t i=0;
    for (; i+3<input->size; i+=4) {
        y[i+0] = x[i+0] > 0 ? x[i+0] : 0;
        y[i+1] = x[i+1] > 0 ? x[i+1] : 0;
        y[i+2] = x[i+2] > 0 ? x[i+2] : 0;
        y[i+3] = x[i+3] > 0 ? x[i+3] : 0;
    }
    for (; i<input->size; i++) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
    return 0;
}


int ti_relu_forward(const ti_tensor_t *input, ti_tensor_t *output) {
    if (input == NULL || output == NULL) return -1;
    if (input->dtype != output->dtype) return -1;
    if (input->dtype == TI_FLOAT32) return ti_relu_forward_float32(input, output);
    if (input->dtype == TI_INT8)    return ti_relu_forward_int8(input, output);
    return -1;
}
