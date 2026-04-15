#include <stdio.h>
#include <stdint.h>
#include "../include/layers.h"
#include "../include/tensor.h"

int main() {
   ti_layer_t layer;
   layer.input_size = 3;
   layer.output_size = 3;
   layer.dtype = TI_FLOAT32;

   /* Declare weights and bias as const to force DROM (Flash) storage.
    * Aligned for fast ESP32 L32I loads.
    */
   const float weights[9] __attribute__((aligned(4))) = {
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f
   };
   const float bias[3] __attribute__((aligned(4))) = {0.0f, 0.0f, 0.0f};

   layer.weights = (const void *)weights;
   layer.bias = (const void *)bias;

    ti_tensor_t input;
    ti_tensor_t output;
    uint32_t shape_in[4] __attribute__((aligned(4))) = {1, 3, 0, 0};
    uint32_t shape_out[4] __attribute__((aligned(4))) = {1, 3, 0, 0};
    ti_tensor_create(&input, shape_in, 2, TI_FLOAT32);
    ti_tensor_create(&output, shape_out, 2, TI_FLOAT32);
    ((float*)input.data)[0] = 1.0f;
    ((float*)input.data)[1] = 2.0f;
    ((float*)input.data)[2] = 3.0f;

    /* Call specialized float32 kernel */
    int result = ti_dense_forward_float32(&layer, &input, &output);
    if (result != 0) {
        printf("Dense forward failed: %d\n", result);
        return -1;
    }

    printf("output: %f %f %f\n",
    ((float*)output.data)[0],
    ((float*)output.data)[1],
    ((float*)output.data)[2]);

    ti_tensor_free(&input);
    ti_tensor_free(&output);
    return 0;
}