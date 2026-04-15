#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "activations.h"

int ti_relu_forward_float32(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL || input->data == NULL || output->data == NULL){
    return -1;
  }
  if (input->dtype != TI_FLOAT32 || output->dtype != TI_FLOAT32){
    return -1;
  }
  if (input->size != output->size){
    return -1;
  }

  const float *x = (const float *)input->data;
  float *y = (float *)output->data;
  /* Loop unrolling: 4-way to reduce branch overhead */
  uint32_t i = 0;
  for (; i + 3 < input->size; i += 4){
    y[i+0] = fmaxf(0.0f, x[i+0]);
    y[i+1] = fmaxf(0.0f, x[i+1]);
    y[i+2] = fmaxf(0.0f, x[i+2]);
    y[i+3] = fmaxf(0.0f, x[i+3]);
  }

  /* Handle remainder */
  for (; i < input->size; i++){
    y[i] = fmaxf(0.0f, x[i]);
  }

  return 0;
}
int ti_relu_forward_int8(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL || input->data == NULL || output->data == NULL){
    return -1;
  }
  if (input->dtype != TI_INT8 || output->dtype != TI_INT8){
    return -1;
  }
  if (input->size != output->size){
    return -1;
  }

  const int8_t *x = (const int8_t *)input->data;
  int8_t *y = (int8_t *)output->data;

  /* Loop unrolling: 4-way to reduce branch overhead */
  uint32_t i = 0;
  for (; i + 3 < input->size; i += 4){
    y[i+0] = x[i+0] > 0 ? x[i+0] : 0;
    y[i+1] = x[i+1] > 0 ? x[i+1] : 0;
    y[i+2] = x[i+2] > 0 ? x[i+2] : 0;
    y[i+3] = x[i+3] > 0 ? x[i+3] : 0;
  }

  /* Handle remainder */
  for (; i < input->size; i++){
    y[i] = x[i] > 0 ? x[i] : 0;
  }

  return 0;
}
int ti_relu_forward(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL){
    return -1;
  }
  if (input -> dtype != output->dtype){
    return -1;
  }
  if (input -> dtype == TI_FLOAT32){
    return ti_relu_forward_float32(input, output);
  } else if (input -> dtype == TI_INT8){
    return ti_relu_forward_int8(input, output);
  } else {
    return -1;
  }
}
int ti_softmax_forward_float32(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL || input->data == NULL || output->data == NULL){
    return -1;
  }
  if (input->dtype != TI_FLOAT32 || output->dtype != TI_FLOAT32){
    return -1;
  }
  if (input->size != output->size || input->size == 0){
    return -1;
  }

  const float *x = (const float *)input->data;
  float *y = (float *)output->data;
  uint32_t size = input->size;

  /* PASS 1: Find global maximum (4-way unroll) */
  float max_val = x[0];
  uint32_t i = 1;
  for (; i + 3 < size; i += 4){
    float m0 = fmaxf(x[i+0], x[i+1]);
    float m1 = fmaxf(x[i+2], x[i+3]);
    max_val = fmaxf(fmaxf(max_val, m0), m1);
  }
  /* Handle remainder */
  for (; i < size; i++){
    max_val = fmaxf(max_val, x[i]);
  }

  /* PASS 2: Allocate temp buffer and compute shifted exponentials (4-way unroll) */
  float *exp_vals = (float *)malloc(size * sizeof(float));
  if (exp_vals == NULL){
    return -1;
  }

  float sum = 0.0f;
  i = 0;
  for (; i + 3 < size; i += 4){
    exp_vals[i+0] = expf(x[i+0] - max_val);
    exp_vals[i+1] = expf(x[i+1] - max_val);
    exp_vals[i+2] = expf(x[i+2] - max_val);
    exp_vals[i+3] = expf(x[i+3] - max_val);
    sum += exp_vals[i+0] + exp_vals[i+1] + exp_vals[i+2] + exp_vals[i+3];
  }
  /* Handle remainder */
  for (; i < size; i++){
    exp_vals[i] = expf(x[i] - max_val);
    sum += exp_vals[i];
  }

  /* PASS 3: Normalize and store results (use reciprocal to avoid 4 divisions) */
  float inv_sum = 1.0f / sum;
  i = 0;
  for (; i + 3 < size; i += 4){
    y[i+0] = exp_vals[i+0] * inv_sum;
    y[i+1] = exp_vals[i+1] * inv_sum;
    y[i+2] = exp_vals[i+2] * inv_sum;
    y[i+3] = exp_vals[i+3] * inv_sum;
  }
  /* Handle remainder */
  for (; i < size; i++){
    y[i] = exp_vals[i] * inv_sum;
  }

  free(exp_vals);
  return 0;
}
int ti_softmax_forward(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL){
    return -1;
  }
  if (input->dtype != output->dtype){
    return -1;  /* Input and output must be same dtype */
  }

  if (input->dtype == TI_FLOAT32){
    return ti_softmax_forward_float32(input, output);
  } else if (input->dtype == TI_INT8){
    return ti_softmax_forward_int8(input, output);
  } else {
    return -1;  /* Unsupported dtype */
  }
}

/* ===== INT8 SOFTMAX =====
 * Compute softmax internally in float32 (numerical stability via log-sum-exp),
 * then quantize output to int8 range [0, 127] representing probabilities [0, 1].
 * On use, dequantize: prob = (float)int8_val / 127.0f
 */
int ti_softmax_forward_int8(const ti_tensor_t *input, ti_tensor_t *output){
  if (input == NULL || output == NULL || input->data == NULL || output->data == NULL){
    return -1;
  }
  if (input->dtype != TI_INT8 || output->dtype != TI_INT8){
    return -1;
  }
  if (input->size != output->size || input->size == 0){
    return -1;
  }

  const int8_t *x_int8 = (const int8_t *)input->data;
  int8_t *y_int8 = (int8_t *)output->data;
  uint32_t size = input->size;

  /* PASS 1: Find global maximum (convert int8→float inline, 4-way unroll) */
  float max_val = (float)x_int8[0];
  uint32_t i = 1;
  for (; i + 3 < size; i += 4){
    float x0 = (float)x_int8[i+0];
    float x1 = (float)x_int8[i+1];
    float x2 = (float)x_int8[i+2];
    float x3 = (float)x_int8[i+3];
    float m0 = fmaxf(x0, x1);
    float m1 = fmaxf(x2, x3);
    max_val = fmaxf(fmaxf(max_val, m0), m1);
  }
  for (; i < size; i++){
    max_val = fmaxf(max_val, (float)x_int8[i]);
  }

  /* PASS 2: Allocate temp buffer (SINGLE buffer for exponentials) */
  float *exp_vals = (float *)malloc(size * sizeof(float));
  if (exp_vals == NULL){
    return -1;
  }

  /* Compute shifted exponentials and sum (4-way unroll, convert inline) */
  float sum = 0.0f;
  i = 0;
  for (; i + 3 < size; i += 4){
    float x0 = (float)x_int8[i+0] - max_val;
    float x1 = (float)x_int8[i+1] - max_val;
    float x2 = (float)x_int8[i+2] - max_val;
    float x3 = (float)x_int8[i+3] - max_val;
    exp_vals[i+0] = expf(x0);
    exp_vals[i+1] = expf(x1);
    exp_vals[i+2] = expf(x2);
    exp_vals[i+3] = expf(x3);
    sum += exp_vals[i+0] + exp_vals[i+1] + exp_vals[i+2] + exp_vals[i+3];
  }
  for (; i < size; i++){
    exp_vals[i] = expf((float)x_int8[i] - max_val);
    sum += exp_vals[i];
  }

  /* PASS 3: Compute reciprocal (1 division) then multiply (4-way unroll) */
  /* Range: 0.0 → 0, 1.0 → 127 */
  float inv_sum = 1.0f / sum;  /* Single division, not 4 per iteration */
  i = 0;
  for (; i + 3 < size; i += 4){
    float prob0 = exp_vals[i+0] * inv_sum;
    float prob1 = exp_vals[i+1] * inv_sum;
    float prob2 = exp_vals[i+2] * inv_sum;
    float prob3 = exp_vals[i+3] * inv_sum;

    /* Use llrintf for faster rounding on most architectures */
    int32_t q0 = (int32_t)llrintf(prob0 * 127.0f);
    int32_t q1 = (int32_t)llrintf(prob1 * 127.0f);
    int32_t q2 = (int32_t)llrintf(prob2 * 127.0f);
    int32_t q3 = (int32_t)llrintf(prob3 * 127.0f);

    y_int8[i+0] = (int8_t)(q0 > 127 ? 127 : (q0 < 0 ? 0 : q0));
    y_int8[i+1] = (int8_t)(q1 > 127 ? 127 : (q1 < 0 ? 0 : q1));
    y_int8[i+2] = (int8_t)(q2 > 127 ? 127 : (q2 < 0 ? 0 : q2));
    y_int8[i+3] = (int8_t)(q3 > 127 ? 127 : (q3 < 0 ? 0 : q3));
  }
  for (; i < size; i++){
    float prob = exp_vals[i] * inv_sum;
    int32_t q = (int32_t)llrintf(prob * 127.0f);
    y_int8[i] = (int8_t)(q > 127 ? 127 : (q < 0 ? 0 : q));
  }

  free(exp_vals);
  return 0;
}