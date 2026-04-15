#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/layers.h"

/* Tile size: process inputs in chunks to maximize cache locality.
 * ESP32 has limited L1 cache, so keep tile small but unroll within it.
 */
#define TI_TILE_SIZE_INPUTS 8
#define TI_TILE_SIZE_OUTPUTS 4


int ti_dense_forward_float32(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->weights == NULL || layer->bias == NULL) return -1;
    if (input->ndim != 2 || input->shape[1] != layer->input_size) return -1;
    if (output->ndim != 2 || output->shape[0] != input->shape[0] || output->shape[1] != layer->output_size) return -1;

    uint32_t batch_size = input->shape[0];
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;

    const float *w = (const float *)layer->weights;
    const float *b = (const float *)layer->bias;
    const float *x_data = (const float *)input->data;
    float *y_data = (float *)output->data;

    for (uint32_t i = 0; i < batch_size; i++) {
        const float *x = x_data + i * input_size;
        float *y = y_data + i * output_size;

        uint32_t j = 0;
        for (; j + 3 < output_size; j += 4) {
            float sum0 = b[j + 0];
            float sum1 = b[j + 1];
            float sum2 = b[j + 2];
            float sum3 = b[j + 3];

            /* Pre-compute weight row offsets for all 4 outputs */
            uint32_t w0 = (j + 0) * input_size;
            uint32_t w1 = (j + 1) * input_size;
            uint32_t w2 = (j + 2) * input_size;
            uint32_t w3 = (j + 3) * input_size;

            uint32_t k = 0;
            for (; k + 7 < input_size; k += 8) {
                float x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
                float x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];

                sum0 += w[w0 + k + 0] * x0 + w[w0 + k + 1] * x1 + w[w0 + k + 2] * x2 + w[w0 + k + 3] * x3
                      + w[w0 + k + 4] * x4 + w[w0 + k + 5] * x5 + w[w0 + k + 6] * x6 + w[w0 + k + 7] * x7;

                sum1 += w[w1 + k + 0] * x0 + w[w1 + k + 1] * x1 + w[w1 + k + 2] * x2 + w[w1 + k + 3] * x3
                      + w[w1 + k + 4] * x4 + w[w1 + k + 5] * x5 + w[w1 + k + 6] * x6 + w[w1 + k + 7] * x7;

                sum2 += w[w2 + k + 0] * x0 + w[w2 + k + 1] * x1 + w[w2 + k + 2] * x2 + w[w2 + k + 3] * x3
                      + w[w2 + k + 4] * x4 + w[w2 + k + 5] * x5 + w[w2 + k + 6] * x6 + w[w2 + k + 7] * x7;

                sum3 += w[w3 + k + 0] * x0 + w[w3 + k + 1] * x1 + w[w3 + k + 2] * x2 + w[w3 + k + 3] * x3
                      + w[w3 + k + 4] * x4 + w[w3 + k + 5] * x5 + w[w3 + k + 6] * x6 + w[w3 + k + 7] * x7;
            }

            for (; k < input_size; k++) {
                float xk = x[k];
                sum0 += w[w0 + k] * xk;
                sum1 += w[w1 + k] * xk;
                sum2 += w[w2 + k] * xk;
                sum3 += w[w3 + k] * xk;
            }

            y[j + 0] = sum0;
            y[j + 1] = sum1;
            y[j + 2] = sum2;
            y[j + 3] = sum3;
        }

        for (; j + 1 < output_size; j += 2) {
            float sum0 = b[j + 0];
            float sum1 = b[j + 1];

            uint32_t w0 = (j + 0) * input_size;
            uint32_t w1 = (j + 1) * input_size;

            for (uint32_t k = 0; k < input_size; k++) {
                sum0 += w[w0 + k] * x[k];
                sum1 += w[w1 + k] * x[k];
            }

            y[j + 0] = sum0;
            y[j + 1] = sum1;
        }

        if (j < output_size) {
            float sum = b[j];
            for (uint32_t k = 0; k < input_size; k++) {
                sum += w[j * input_size + k] * x[k];
            }
            y[j] = sum;
        }
    }

    return 0;
}


int ti_dense_forward_int8(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->weights == NULL || layer->bias == NULL) return -1;
    if (input->ndim != 2 || input->shape[1] != layer->input_size) return -1;
    if (output->ndim != 2 || output->shape[0] != input->shape[0] || output->shape[1] != layer->output_size) return -1;

    uint32_t batch_size = input->shape[0];
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;

    const int8_t *w = (const int8_t *)layer->weights;
    const int32_t *b = (const int32_t *)layer->bias;
    const int8_t *x_data = (const int8_t *)input->data;
    int8_t *y_data = (int8_t *)output->data;

    for (uint32_t i = 0; i < batch_size; i++) {
        const int8_t *x = x_data + i * input_size;
        int8_t *y = y_data + i * output_size;

        uint32_t j = 0;
        for (; j + 3 < output_size; j += 4) {
            int32_t sum0 = b[j+0];
            int32_t sum1 = b[j+1];
            int32_t sum2 = b[j+2];
            int32_t sum3 = b[j+3];

            /* Pre-compute weight row offsets */
            uint32_t w0 = (j + 0) * input_size;
            uint32_t w1 = (j + 1) * input_size;
            uint32_t w2 = (j + 2) * input_size;
            uint32_t w3 = (j + 3) * input_size;

            uint32_t k = 0;
            for (; k + 3 < input_size; k += 4) {
                int8_t x0 = x[k+0], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];
                int32_t xi0 = (int32_t)x0, xi1 = (int32_t)x1, xi2 = (int32_t)x2, xi3 = (int32_t)x3;

                sum0 += (int32_t)w[w0 + k+0] * xi0 + (int32_t)w[w0 + k+1] * xi1
                      + (int32_t)w[w0 + k+2] * xi2 + (int32_t)w[w0 + k+3] * xi3;

                sum1 += (int32_t)w[w1 + k+0] * xi0 + (int32_t)w[w1 + k+1] * xi1
                      + (int32_t)w[w1 + k+2] * xi2 + (int32_t)w[w1 + k+3] * xi3;

                sum2 += (int32_t)w[w2 + k+0] * xi0 + (int32_t)w[w2 + k+1] * xi1
                      + (int32_t)w[w2 + k+2] * xi2 + (int32_t)w[w2 + k+3] * xi3;

                sum3 += (int32_t)w[w3 + k+0] * xi0 + (int32_t)w[w3 + k+1] * xi1
                      + (int32_t)w[w3 + k+2] * xi2 + (int32_t)w[w3 + k+3] * xi3;
            }

            for (; k < input_size; k++) {
                int32_t xk = (int32_t)x[k];
                sum0 += (int32_t)w[w0 + k] * xk;
                sum1 += (int32_t)w[w1 + k] * xk;
                sum2 += (int32_t)w[w2 + k] * xk;
                sum3 += (int32_t)w[w3 + k] * xk;
            }

            /* Saturate using branchless min/max */
            y[j + 0] = (int8_t)(sum0 > 127 ? 127 : (sum0 < -128 ? -128 : sum0));
            y[j + 1] = (int8_t)(sum1 > 127 ? 127 : (sum1 < -128 ? -128 : sum1));
            y[j + 2] = (int8_t)(sum2 > 127 ? 127 : (sum2 < -128 ? -128 : sum2));
            y[j + 3] = (int8_t)(sum3 > 127 ? 127 : (sum3 < -128 ? -128 : sum3));
        }

        for (; j < output_size; j++) {
            int32_t sum = b[j];
            uint32_t w0 = j * input_size;

            for (uint32_t k = 0; k < input_size; k++) {
                sum += (int32_t)w[w0 + k] * (int32_t)x[k];
            }
            y[j] = (int8_t)(sum > 127 ? 127 : (sum < -128 ? -128 : sum));
        }
    }

    return 0;
}


int ti_dense_forward(ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;

    if (layer->dtype == TI_FLOAT32) {
        return ti_dense_forward_float32(layer, input, output);
    } else if (layer->dtype == TI_INT8) {
        return ti_dense_forward_int8(layer, input, output);
    } else {
        return -1;
    }
}