#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "activations.h"
int ti_relu_forward_float32(const ti_tensor_t *input, ti_tensor_t *output) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL)
        return -1;
    if (input->dtype != TI_FLOAT32 || output->dtype != TI_FLOAT32)
        return -1;
    if (input->size != output->size)
        return -1;

    const float *x = (const float *)input->data;
    float *y = (float *)output->data;
    uint32_t i = 0;

    for (; i + 3 < input->size; i += 4) {
        y[i+0] = x[i+0] > 0.0f ? x[i+0] : 0.0f;
        y[i+1] = x[i+1] > 0.0f ? x[i+1] : 0.0f;
        y[i+2] = x[i+2] > 0.0f ? x[i+2] : 0.0f;
        y[i+3] = x[i+3] > 0.0f ? x[i+3] : 0.0f;
    }
    for (; i < input->size; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }

    return 0;
}

int ti_relu_forward_int8(const ti_tensor_t *input, ti_tensor_t *output) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL)
        return -1;
    if (input->dtype != TI_INT8 || output->dtype != TI_INT8)
        return -1;
    if (input->size != output->size)
        return -1;

    const int8_t *x = (const int8_t *)input->data;
    int8_t *y = (int8_t *)output->data;
    uint32_t i = 0;

    for (; i + 3 < input->size; i += 4) {
        y[i+0] = x[i+0] > 0 ? x[i+0] : 0;
        y[i+1] = x[i+1] > 0 ? x[i+1] : 0;
        y[i+2] = x[i+2] > 0 ? x[i+2] : 0;
        y[i+3] = x[i+3] > 0 ? x[i+3] : 0;
    }
    for (; i < input->size; i++) {
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
int ti_softmax_forward_float32(const ti_tensor_t *input, ti_tensor_t *output, float *scratch) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL)
        return -1;
    if (scratch == NULL)
        return -1;
    if (input->dtype != TI_FLOAT32 || output->dtype != TI_FLOAT32)
        return -1;
    if (input->size != output->size || input->size == 0)
        return -1;

    const float *x = (const float *)input->data;
    float *y = (float *)output->data;
    uint32_t size = input->size;

    float max_val = x[0];
    uint32_t i = 1;
    for (; i + 3 < size; i += 4) {
        float m0 = x[i+0] > x[i+1] ? x[i+0] : x[i+1];
        float m1 = x[i+2] > x[i+3] ? x[i+2] : x[i+3];
        float m  = m0 > m1 ? m0 : m1;
        max_val  = max_val > m ? max_val : m;
    }
    for (; i < size; i++) {
        max_val = max_val > x[i] ? max_val : x[i];
    }

    float sum = 0.0f;
    i = 0;
    for (; i + 3 < size; i += 4) {
        scratch[i+0] = expf(x[i+0] - max_val);
        scratch[i+1] = expf(x[i+1] - max_val);
        scratch[i+2] = expf(x[i+2] - max_val);
        scratch[i+3] = expf(x[i+3] - max_val);
        sum += scratch[i+0] + scratch[i+1] + scratch[i+2] + scratch[i+3];
    }
    for (; i < size; i++) {
        scratch[i] = expf(x[i] - max_val);
        sum += scratch[i];
    }

    float inv_sum = 1.0f / sum;
    i = 0;
    for (; i + 3 < size; i += 4) {
        y[i+0] = scratch[i+0] * inv_sum;
        y[i+1] = scratch[i+1] * inv_sum;
        y[i+2] = scratch[i+2] * inv_sum;
        y[i+3] = scratch[i+3] * inv_sum;
    }
    for (; i < size; i++) {
        y[i] = scratch[i] * inv_sum;
    }

    return 0;
}

int ti_softmax_forward_int8(const ti_tensor_t *input, ti_tensor_t *output, float *scratch) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL)
        return -1;
    if (scratch == NULL)
        return -1;
    if (input->dtype != TI_INT8 || output->dtype != TI_INT8)
        return -1;
    if (input->size != output->size || input->size == 0)
        return -1;

    const int8_t *x_int8 = (const int8_t *)input->data;
    int8_t *y_int8 = (int8_t *)output->data;
    uint32_t size = input->size;

    float max_val = (float)x_int8[0];
    uint32_t i = 1;
    for (; i + 3 < size; i += 4) {
        float x0 = (float)x_int8[i+0];
        float x1 = (float)x_int8[i+1];
        float x2 = (float)x_int8[i+2];
        float x3 = (float)x_int8[i+3];
        float m0 = x0 > x1 ? x0 : x1;
        float m1 = x2 > x3 ? x2 : x3;
        float m  = m0 > m1 ? m0 : m1;
        max_val  = max_val > m ? max_val : m;
    }
    for (; i < size; i++) {
        float xi = (float)x_int8[i];
        max_val = max_val > xi ? max_val : xi;
    }

    float sum = 0.0f;
    i = 0;
    for (; i + 3 < size; i += 4) {
        scratch[i+0] = expf((float)x_int8[i+0] - max_val);
        scratch[i+1] = expf((float)x_int8[i+1] - max_val);
        scratch[i+2] = expf((float)x_int8[i+2] - max_val);
        scratch[i+3] = expf((float)x_int8[i+3] - max_val);
        sum += scratch[i+0] + scratch[i+1] + scratch[i+2] + scratch[i+3];
    }
    for (; i < size; i++) {
        scratch[i] = expf((float)x_int8[i] - max_val);
        sum += scratch[i];
    }

    float inv_sum = 1.0f / sum;
    i = 0;
    for (; i + 3 < size; i += 4) {
        int32_t q0 = (int32_t)llrintf(scratch[i+0] * inv_sum * 127.0f);
        int32_t q1 = (int32_t)llrintf(scratch[i+1] * inv_sum * 127.0f);
        int32_t q2 = (int32_t)llrintf(scratch[i+2] * inv_sum * 127.0f);
        int32_t q3 = (int32_t)llrintf(scratch[i+3] * inv_sum * 127.0f);

        y_int8[i+0] = (int8_t)(q0 > 127 ? 127 : (q0 < 0 ? 0 : q0));
        y_int8[i+1] = (int8_t)(q1 > 127 ? 127 : (q1 < 0 ? 0 : q1));
        y_int8[i+2] = (int8_t)(q2 > 127 ? 127 : (q2 < 0 ? 0 : q2));
        y_int8[i+3] = (int8_t)(q3 > 127 ? 127 : (q3 < 0 ? 0 : q3));
    }
    for (; i < size; i++) {
        int32_t q = (int32_t)llrintf(scratch[i] * inv_sum * 127.0f);
        y_int8[i] = (int8_t)(q > 127 ? 127 : (q < 0 ? 0 : q));
    }

    return 0;
}


int ti_softmax_forward(const ti_tensor_t *input, ti_tensor_t *output, float *scratch) {
    if (input == NULL || output == NULL || scratch == NULL) return -1;
    if (input->dtype != output->dtype) return -1;

    if (input->dtype == TI_FLOAT32) return ti_softmax_forward_float32(input, output, scratch);
    if (input->dtype == TI_INT8)    return ti_softmax_forward_int8(input, output, scratch);
    return -1;
}