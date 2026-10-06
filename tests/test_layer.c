#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/layers.h"

void test_dense_forward_float32_basic(void) {
    uint32_t in_shape[4] = {1, 4, 0, 0};
    uint32_t out_shape[4] = {1, 2, 0, 0};
    ti_tensor_t input, output;

    ti_tensor_create(&input, in_shape, 2, TI_FLOAT32);
    ti_tensor_create(&output, out_shape, 2, TI_FLOAT32);

    float *in_data = (float *)input.data;
    in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f; in_data[3] = 4.0f;

    float weights[8] = {
        0.1f, 0.2f, 0.3f, 0.4f,
       -0.4f, -0.3f, -0.2f, -0.1f
    };
    float bias[2] = { 0.5f, -0.5f };

    ti_layer_t layer;
    layer.input_size = 4;
    layer.output_size = 2;
    layer.dtype = TI_FLOAT32;
    layer.activation = TI_ACT_NONE;
    layer.weights = weights;
    layer.bias = bias;
    layer.requant_scale = 1.0f;

    int status = ti_dense_forward_float32(&layer, &input, &output);
    assert(status == 0);

    float *out_data = (float *)output.data;
    assert(fabsf(out_data[0] - 3.5f) < 1e-5f);
    assert(fabsf(out_data[1] - (-2.5f)) < 1e-5f);

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    printf("[PASS] test_dense_forward_float32_basic\n");
}

int main(void) {
    printf("--- Running Float32 Layer Unit Tests ---\n");
    test_dense_forward_float32_basic();
    printf("--- All Float32 Layer Tests Passed ---\n");
    return 0;
}