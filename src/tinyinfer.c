#include "tinyinfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>


bool ti_load_model_f32(const char *filepath, ti_model_f32_t *model) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    return false;
}
fseek(file, 0, SEEK_END);
size_t file_size = ftell(file);
fseek(file, 0, SEEK_SET);
model->raw_buffer = (uint8_t *)malloc(file_size);
if (model->raw_buffer== NULL) {
    fclose(file);
    return false;
}
if (fread(model->raw_buffer, 1, file_size, file) != file_size) {
    free(model->raw_buffer);
    fclose(file);
    return false;
}
ti_model_header_t *header = (ti_model_header_t *)model->raw_buffer;
if (memcmp(header->magic, "TINF", 4) != 0 || header->version != TI_VERSION) {
        free(model->raw_buffer);
        fclose(file);
        return false;
    }
model->header = *header;
model->layers = malloc(header->num_layers * sizeof(ti_layer_f32_t));
if (model->layers == NULL) {
    free(model->raw_buffer);
    fclose(file);
    return false;
}
uint8_t *ptr = model->raw_buffer + sizeof(ti_model_header_t);
for (int i = 0; i < header->num_layers; i++) {
int32_t in_dim = *(const int32_t *)ptr; ptr += sizeof(int32_t);
int32_t out_dim = *(const int32_t *)ptr; ptr += sizeof(int32_t);
uint8_t act = *(const uint8_t *)ptr; ptr += sizeof(uint8_t);
ptr += 3;

model->layers[i].in_dim = in_dim;
model->layers[i].out_dim = out_dim;
model->layers[i].act = act;

model->layers[i].weights = (const float *)ptr;
ptr += in_dim * out_dim * sizeof(float);
model->layers[i].biases = (const float *)ptr;
ptr += out_dim * sizeof(float);

}
fclose(file);
return true;
}

bool ti_load_model_int8(const char *filepath, ti_model_int8_t *model) {
    FILE *file = fopen(filepath, "rb");
    if (!file) return false;

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    model->raw_buffer = (uint8_t *)malloc(file_size);
    if (model->raw_buffer == NULL) {
        fclose(file);
        return false;
    }

    if (fread(model->raw_buffer, 1, file_size, file) != file_size) {
        free(model->raw_buffer);
        fclose(file);
        return false;
    }

    ti_model_header_t *header = (ti_model_header_t *)model->raw_buffer;
    if (memcmp(header->magic, "TINF", 4) != 0 || header->version != TI_VERSION) {
        free(model->raw_buffer);
        fclose(file);
        return false;
    }

    model->header = *header;
    model->layers = malloc(header->num_layers * sizeof(ti_layer_int8_t));
    if (model->layers == NULL) {
        free(model->raw_buffer);
        fclose(file);
        return false;
    }

    uint8_t *ptr = model->raw_buffer + sizeof(ti_model_header_t);

    model->input_scale = *(const float *)ptr;
    ptr += sizeof(float);

    for (int i = 0; i < header->num_layers; i++) {
        int32_t in_dim  = *(const int32_t *)ptr; ptr += sizeof(int32_t);
        int32_t out_dim = *(const int32_t *)ptr; ptr += sizeof(int32_t);
        uint8_t act     = *(const uint8_t *)ptr; ptr += sizeof(uint8_t);
        ptr += 3;

        float requant_scale = *(const float *)ptr; ptr += sizeof(float);

        model->layers[i].in_dim        = in_dim;
        model->layers[i].out_dim       = out_dim;
        model->layers[i].act           = act;
        model->layers[i].requant_scale = requant_scale;

        model->layers[i].weights = (const int8_t *)ptr;
        ptr += in_dim * out_dim * sizeof(int8_t);

        model->layers[i].biases = (const int32_t *)ptr;
        ptr += out_dim * sizeof(int32_t);
    }

    fclose(file);
    return true;
}

void ti_run_f32(const ti_model_f32_t *model, const float *input, float *output){
  assert(model->header.in_dim <= 128);
  for (int i = 0; i < model->header.num_layers; i++) {
    assert(model->layers[i].in_dim <= 128);
  }
  float buff_in [128];
  float buff_out [128];
  memcpy(buff_in, input, model->header.in_dim * sizeof(float));
  uint8_t i = 0;
  for (;i < model->header.num_layers; i++) {
    const float * weight = model->layers[i].weights;
    const float * bias = model->layers[i].biases;
    int32_t out_dim = model->layers[i].out_dim;
    int32_t in_dim = model->layers[i].in_dim;
    int8_t act = model->layers[i].act;
    for (int j = 0; j < out_dim; j++) {
      float sum = bias[j];
      const float *w_row = &weight[j * in_dim];
      int k = 0;
      for (; k+3 < in_dim; k+=4) {
        sum+=buff_in[k]*w_row[k+0]+buff_in[k+1]*w_row[k+1]+buff_in[k+2]*w_row[k+2]+buff_in[k+3]*w_row[k+3];
    }
    for(;k < in_dim; k++) {
      sum+=buff_in[k]*w_row[k];
    }
    if (act == 1 && sum < 0.0f) {
    sum = 0.0f;
	}
    buff_out[j] = sum;
  }
  int32_t j = 0;
  for (; j < out_dim; j++) {
    buff_in[j] = buff_out[j];
  }
  }

  for (int32_t k = 0; k < model->header.out_dim; k++) {
        output[k] = buff_in[k];
    }
}
void ti_run_int8(const ti_model_int8_t *model, const float *input, float *output) {
    assert(model->header.in_dim <= 128);
  	for (int i = 0; i < model->header.num_layers; i++) {
    assert(model->layers[i].in_dim <= 128);
  }
    int8_t buff_in[128];
    int8_t buff_out[128];
    for (int i = 0; i < model->header.in_dim; i++) {
        float val = roundf(input[i] / model->input_scale);

        if (val > 127.0f)  val = 127.0f;
        if (val < -128.0f) val = -128.0f;
        buff_in[i] = (int8_t)val;
    }

    for (int i = 0; i < model->header.num_layers; i++) {
        const int8_t *weight = model->layers[i].weights;
        const int32_t *bias  = model->layers[i].biases;
        const int8_t act     = model->layers[i].act;
        uint32_t in_dim      = model->layers[i].in_dim;
        uint32_t out_dim     = model->layers[i].out_dim;

        for (int32_t j = 0; j < out_dim; j++) {
            int32_t sum = bias[j];
            const int8_t *w_row = &weight[j * in_dim];

            int32_t k = 0;
            for (; k + 3 < in_dim; k += 4) {
                sum += buff_in[k + 0] * w_row[k + 0]
                     + buff_in[k + 1] * w_row[k + 1]
                     + buff_in[k + 2] * w_row[k + 2]
                     + buff_in[k + 3] * w_row[k + 3];
            }
            for (; k < in_dim; k++) {
                sum += buff_in[k] * w_row[k];
            }

            float scaled = (float)sum * model->layers[i].requant_scale;
            if (act == 1 && scaled < 0.0f) {
                scaled = 0.0f;
            }
            if (i == model->header.num_layers - 1) {
   			 output[j] = scaled;
			}
            else {
			if (scaled > 127.0f) scaled = 127.0f;
    		if (scaled < -128.0f) scaled = -128.0f;
    		buff_out[j] = (int8_t)scaled;
}
        }


  if (i < model->header.num_layers - 1) {
    for (int32_t j = 0; j < (int32_t)out_dim; j++) {
        buff_in[j] = buff_out[j];
    }
}
    }
}
void ti_free_model_f32(ti_model_f32_t *model){
  if (model->layers) {
    free(model->layers);
  }
  if(model->raw_buffer){
    free(model->raw_buffer);
  }
  model->layers = NULL;
  model->raw_buffer = NULL;
}
void ti_free_model_int8(ti_model_int8_t *model) {
    if (model->layers)     free(model->layers);
    if (model->raw_buffer) free(model->raw_buffer);
    model->layers = NULL;
    model->raw_buffer = NULL;
}

