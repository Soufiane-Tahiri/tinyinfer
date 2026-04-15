#include <stdio.h>
#include <stdint.h>
#include "../include/layers.h"
#include "../include/tensor.h"

int main() {
    ti_layer_t layer;
    layer.input_size = 3;
    layer.output_size = 3;
    layer.dtype = TI_INT8;

    /* INT8 weights: const, aligned for DROM + fast loads */
    const int8_t weights[9] __attribute__((aligned(4))) = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    /* INT8 bias MUST be int32_t to prevent overflow during accumulation */
    const int32_t bias[3] __attribute__((aligned(4))) = {0, 0, 0};

    layer.weights = (const void *)weights;
    layer.bias = (const void *)bias;

    ti_tensor_t input;
    ti_tensor_t output;
    uint32_t shape_in[4] __attribute__((aligned(4))) = {1, 3, 0, 0};
    uint32_t shape_out[4] __attribute__((aligned(4))) = {1, 3, 0, 0};

    ti_tensor_create(&input, shape_in, 2, TI_INT8);
    ti_tensor_create(&output, shape_out, 2, TI_INT8);

    ((int8_t*)input.data)[0] = 1;
    ((int8_t*)input.data)[1] = 2;
    ((int8_t*)input.data)[2] = 3;

    /* Call specialized int8 kernel */
    int result = ti_dense_forward_int8(&layer, &input, &output);
    if (result != 0) {
        printf("Dense forward int8 failed: %d\n", result);
        return -1;
    }

    printf("int8 output: %d %d %d\n",
        ((int8_t*)output.data)[0],
        ((int8_t*)output.data)[1],
        ((int8_t*)output.data)[2]);

    ti_tensor_free(&input);
    ti_tensor_free(&output);
    return 0;
}

