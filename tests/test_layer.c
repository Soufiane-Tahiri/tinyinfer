#include <stdio.h>
#include <stdint.h>
#include "../include/layers.h"
#include "../include/tensor.h"
int main() {
   ti_layer_t layer;
   layer.input_size = 3;
   layer.output_size = 3;
   layer.dtype = TI_FLOAT32;
   float weights[9] = {
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f
  };
   float bias[3] = {0.0f, 0.0f, 0.0f};

   layer.weights = weights;
   layer.bias = bias;

    ti_tensor_t input;
    ti_tensor_t output;
    uint32_t shape_in[4] = {1, 3, 0, 0};
    uint32_t shape_out[4] = {1, 3, 0, 0};
    ti_tensor_create(&input, shape_in, 2, TI_FLOAT32);
    ti_tensor_create(&output, shape_out, 2, TI_FLOAT32);
    ((float*)input.data)[0] = 1.0f;
    ((float*)input.data)[1] = 2.0f;
    ((float*)input.data)[2] = 3.0f;
    ti_dense_forward(&layer, &input, &output);
    printf("output: %f %f %f\n",
    ((float*)output.data)[0],
    ((float*)output.data)[1],
    ((float*)output.data)[2]);


}