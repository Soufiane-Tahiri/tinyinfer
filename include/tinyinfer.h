
#ifndef TINYINFER_H
#define TINYINFER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define TI_VERSION 1

typedef struct {
    uint8_t  magic[4];
    int32_t  version;
    int32_t  dtype;
    int32_t  num_layers;
    int32_t  in_dim;
    int32_t  out_dim;
} ti_model_header_t;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint8_t  act;
    const float *weights;
    const float *biases;
} ti_layer_f32_t;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint8_t  act;
    const int8_t *weights;
    const int32_t *biases;
    float requant_scale;
} ti_layer_int8_t;

typedef struct {
    ti_model_header_t header;
    ti_layer_f32_t *layers;
    uint8_t *raw_buffer;
} ti_model_f32_t;

typedef struct {
    ti_model_header_t header;
    ti_layer_int8_t *layers;
    float input_scale;
    uint8_t *raw_buffer;
} ti_model_int8_t;

bool ti_load_model_f32(const char *filepath,ti_model_f32_t *model);
bool ti_load_model_int8(const char *filepath,ti_model_int8_t *model);


void ti_run_f32(const ti_model_f32_t *model, const float *input, float *output);
void ti_run_int8(const ti_model_int8_t *model, const float *input, float *output);

bool ti_load_model_int8_mem(const uint8_t *buf, size_t size, ti_model_int8_t *model);

void ti_free_model_f32(ti_model_f32_t *model);
void ti_free_model_int8(ti_model_int8_t *model);

#endif
