#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "../include/tensor.h"
#include "../include/layers.h"

void test_dense_forward_int8_basic(void) {
    uint32_t in_shape[4] = {1, 4, 0, 0};
    uint32_t out_shape[4] = {1, 2, 0, 0};
    ti_tensor_t input, output;

    ti_tensor_create(&input, in_shape, 2, TI_INT8);
    ti_tensor_create(&output, out_shape, 2, TI_INT8);

    int8_t *in_data = (int8_t *)input.data;
    in_data[0] = 10; in_data[1] = 20; in_data[2] = 30; in_data[3] = 40;


    int8_t weights[8] = {
        1,  2,  3,  4,
       -4, -3, -2, -1
   };
    int32_t bias[2] = { 100, -100 };

    ti_layer_t layer;
    layer.input_size = 4;
    layer.output_size = 2;
    layer.dtype = TI_INT8;
    layer.activation = TI_ACT_NONE;
    layer.weights = weights;
    layer.bias = bias;
    layer.requant_scale = 0.1f;

    int status = ti_dense_forward_int8(&layer, &input, &output);
    assert(status == 0);

    int8_t *out_data = (int8_t *)output.data;
    assert(out_data[0] == 40);

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    printf("[PASS] test_dense_forward_int8_basic\n");
}

int main(void) {
    printf("--- Running INT8 Layer Unit Tests ---\n");
    test_dense_forward_int8_basic();
    printf("--- All INT8 Layer Tests Passed ---\n");
    return 0;
}