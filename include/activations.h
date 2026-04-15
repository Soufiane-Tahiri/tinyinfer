#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

/* ===== ReLU ACTIVATION =====
 * Specialized kernels: no dtype checking in element loop
 */
int ti_relu_forward_float32(const ti_tensor_t *input, ti_tensor_t *output);
int ti_relu_forward_int8(const ti_tensor_t *input, ti_tensor_t *output);

/* Generic dispatcher: single dtype check */
int ti_relu_forward(const ti_tensor_t *input, ti_tensor_t *output);

/* ===== SOFTMAX ACTIVATION =====
 * Float32: Direct computation with log-sum-exp stability.
 * Int8: Compute internally in float32, quantize output to [0, 127] representing [0, 1].
 *       On use, dequantize: prob = (float)int8_val / 127.0f
 * Numerically stable via log-sum-exp trick to prevent overflow/underflow.
 */
int ti_softmax_forward_float32(const ti_tensor_t *input, ti_tensor_t *output);
int ti_softmax_forward_int8(const ti_tensor_t *input, ti_tensor_t *output);

/* Generic dispatcher (calls appropriate kernel based on dtype) */
int ti_softmax_forward(const ti_tensor_t *input, ti_tensor_t *output);

#endif
