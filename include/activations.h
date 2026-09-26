#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

int ti_relu_forward_float32(const ti_tensor_t *input, ti_tensor_t *output);
int ti_relu_forward_int8(const ti_tensor_t *input, ti_tensor_t *output);
int ti_relu_forward(const ti_tensor_t *input, ti_tensor_t *output);

int ti_softmax_forward_float32(const ti_tensor_t *input, ti_tensor_t *output, float *scratch);
int ti_softmax_forward_int8(const ti_tensor_t *input, ti_tensor_t *output, float *scratch);
int ti_softmax_forward(const ti_tensor_t *input, ti_tensor_t *output, float *scratch);

#endif