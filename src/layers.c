#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/layers.h"

int ti_dense_forward(ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->weights == NULL || layer->bias == NULL) return -1;
    if (input->ndim != 2 || input->shape[1] != layer->input_size) return -1;
    if (output->ndim != 2 || output->shape[0] != input->shape[0] || output->shape[1] != layer->output_size) return -1;

    uint32_t batch_size = input->shape[0];
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;

    if (layer->dtype == TI_FLOAT32) {
        for (uint32_t i = 0; i < batch_size; i++) {
            for (uint32_t j = 0; j < output_size; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < input_size; k++) {
                    float w = ((float*)layer->weights)[j * input_size + k];
                    float x = ((float*)input->data)[i * input_size + k];
                    sum += w * x;
                }
                sum += ((float*)layer->bias)[j];
                ((float*)output->data)[i * output_size + j] = sum;
            }
        }
    } else if (layer->dtype == TI_INT8) {
        for (uint32_t i = 0; i < batch_size; i++) {
            for (uint32_t j = 0; j < output_size; j++) {
                int32_t sum = 0;
                for (uint32_t k = 0; k < input_size; k++) {
                    int8_t w = ((int8_t*)layer->weights)[j * input_size + k];
                    int8_t x = ((int8_t*)input->data)[i * input_size + k];
                    sum += (int32_t)w * (int32_t)x;
                }
                sum += ((int32_t*)layer->bias)[j];
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                ((int8_t*)output->data)[i * output_size + j] = (int8_t)sum;
            }
        }
    } else {
        return -1;
    }

    return 0;
}