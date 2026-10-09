#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "model.h"

#define TI_MAGIC   "TINF"
#define TI_VERSION 1


int ti_model_load(ti_model_t *model, const char *path) {
    FILE* f;
    f = fopen(path, "rb");
    if (f == NULL) {
        return -1;
    }
    char magic[4];
    int32_t version;
    int32_t dtype_tag;
    int32_t n_layers;
    int32_t input_dim;
    int32_t output_dim;
  if (fread(magic, sizeof(char), 4, f) != 4) { printf("Failed reading magic\n"); fclose(f); return -1; }
  if (fread(&version, sizeof(int32_t), 1, f) != 1) { printf("Failed reading version\n"); fclose(f); return -1; }
  if (fread(&dtype_tag, sizeof(int32_t), 1, f) != 1) { printf("Failed reading dtype_tag\n"); fclose(f); return -1; }
    if (fread(&n_layers, sizeof(int32_t), 1, f) != 1) { printf("Failed reading n_layers\n"); fclose(f); return -1; }
    if (fread(&input_dim, sizeof(int32_t), 1, f) != 1) { printf("Failed reading input_dim\n"); fclose(f); return -1; }
    if (fread(&output_dim, sizeof(int32_t), 1, f) != 1) { printf("Failed reading output_dim\n"); fclose(f); return -1; }

    if (strncmp(magic, TI_MAGIC, 4) != 0) {
      printf("Magic mismatch (%.4s != %s)\n", magic, TI_MAGIC);
      fclose(f);
      return -1;
    }
    if (version != TI_VERSION) {
      printf("Version mismatch (%d != %d)\n", version, TI_VERSION);
      fclose(f);
      return -1;
    }
    if (dtype_tag != TI_FLOAT32 && dtype_tag != TI_INT8) {
      printf("Unknown dtype_tag (%d)\n", dtype_tag);
      fclose(f);
      return -1;
    }

 	if (dtype_tag == TI_INT8) {
      if (fread(&model->input_scale, sizeof(float), 1, f) != 1) {
        printf("Failed reading input_scale for INT8\n");
        fclose(f);
        return -1;
      }
    }
    else {
      model->input_scale = 1.0f;
    }

    if (n_layers <= 0 || n_layers > 32) {
      printf("Invalid n_layers (%d)\n", n_layers);
      fclose(f);
      return -1;
    }

    model->layers = calloc(n_layers , sizeof(ti_layer_t));
    if (model->layers == NULL) {
      printf("calloc failed for layers\n");
      fclose(f);
      return -1;
    }

    if (input_dim <= 0 || input_dim > 4096) {
      printf("Invalid input_dim (%d)\n", input_dim);
      fclose(f);
      return -1;
    }
    if (output_dim <= 0 || output_dim > 4096) { fclose(f); return -1; }

    model->num_layers = n_layers;
    model->dtype = dtype_tag;
    model->input_size = input_dim;
    uint32_t i=0;
    for (i = 0; i < n_layers; i++) {

      if (fread(&model->layers[i].input_size, sizeof(int32_t), 1, f) != 1) goto error;
      if (fread(&model->layers[i].output_size, sizeof(int32_t), 1, f) != 1) goto error;

      if (model->layers[i].input_size == 0 || model->layers[i].input_size > TI_MAX_LAYER_DIM) goto error;
      if (model->layers[i].output_size == 0 || model->layers[i].output_size > TI_MAX_LAYER_DIM) goto error;

      model->layers[i].dtype = dtype_tag;
      uint8_t act_byte;
      if (fread(&act_byte, 1, 1, f) != 1) goto error;
      char padding[3];
      if (fread(padding, 1, 3, f) != 3) goto error;
      model->layers[i].activation = (ti_activation_type_t)act_byte;

      if (model->layers[i].output_size > UINT32_MAX / model->layers[i].input_size) goto error;
      uint32_t weight_count = model->layers[i].input_size * model->layers[i].output_size;

      if (dtype_tag == TI_INT8) {
       	if (fread(&model->layers[i].requant_scale, sizeof(float), 1, f) != 1) goto error;
        model->layers[i].weights = (const void *)malloc(weight_count * sizeof(int8_t));
        if (model->layers[i].weights == NULL) goto error;

		if (fread((void*)model->layers[i].weights, sizeof(int8_t), weight_count, f) != (weight_count)) goto error;
        model->layers[i].bias = malloc(model->layers[i].output_size * sizeof(int32_t));
        if (model->layers[i].bias == NULL) goto error;
		if (fread(model->layers[i].bias, sizeof(int32_t), model->layers[i].output_size, f) != model->layers[i].output_size) goto error;      }
      if (dtype_tag == TI_FLOAT32) {
        model->layers[i].weights = (const void *)malloc(weight_count * sizeof(float));
        if (model->layers[i].weights == NULL) goto error;
        if (fread((void*)model->layers[i].weights, sizeof(float), weight_count, f)!= (weight_count)) goto error;
        model->layers[i].bias = malloc(model->layers[i].output_size * sizeof(float));
        if (model->layers[i].bias == NULL) goto error;
        if (fread(model->layers[i].bias, sizeof(float), model->layers[i].output_size, f)!=model->layers[i].output_size) goto error;
      }

    }
  fclose(f);
  return 0;

  error:
    if (model->layers != NULL) {
      for (uint32_t j = 0; j < i; j++) {
        if (model->layers[j].weights != NULL) {
          free(model->layers[j].weights);
        }
        if (model->layers[j].bias != NULL) {
          free(model->layers[j].bias);
        }
      }
      if (model->layers[i].weights != NULL) free((void*)model->layers[i].weights);
      if (model->layers[i].bias != NULL) free(model->layers[i].bias);
      free(model->layers);
      model->layers = NULL;
    }
    if (f != NULL) {
      fclose(f);
    }
    return -1;
}


int ti_model_run(const ti_model_t *model, ti_tensor_t *input, ti_tensor_t *output) {
  if (model == NULL || input == NULL || output == NULL) return -1;
  if (model->layers == NULL) return -1;
  if (input->ndim != 2) return -1;
  if (input->shape[1] != model->input_size) return -1;
  if (model->dtype != TI_INT8 && model->dtype != TI_FLOAT32) return -1;
  if (model->num_layers <= 0) return -1;
  if (input->dtype != model->dtype) return -1;
  if (model->dtype == TI_INT8 && output->dtype != TI_INT32) return -1;
  if (model->dtype == TI_FLOAT32 && output->dtype != TI_FLOAT32) return -1;

  ti_tensor_t *curr_in = input;
  ti_tensor_t *curr_out;
  ti_tensor_t scratch[2];
  uint32_t i=0;
  for (i = 0; i < model->num_layers; i++) {
    if (i == model->num_layers-1) curr_out = output;
    else {
      curr_out=&scratch[i%2];
      uint32_t shape[4] = {
          curr_in->shape[0],
          model->layers[i].output_size,
          1,
          1
      };
      if (ti_tensor_create(curr_out, shape, 2, model->layers[i].dtype) != 0) {
        if (curr_in != input)
          ti_tensor_free(curr_in);
        return -1;
      }
    }
    int status = ti_dense_forward(&model->layers[i], curr_in, curr_out);
    if (status != 0) {
      if (curr_in != input) ti_tensor_free(curr_in);
      if (curr_out != output) ti_tensor_free(curr_out);
      return -1;
    }
    if (curr_in != input) {
      ti_tensor_free(curr_in);
    }
    curr_in = curr_out;
  }
  return 0;
}


void ti_model_free(ti_model_t *model) {
  if (model == NULL || model->layers == NULL) return;

  for (uint32_t j = 0; j < model->num_layers; j++) {
    if (model->layers[j].weights != NULL) {
      free(model->layers[j].weights);
      model->layers[j].weights = NULL;
    }
    if (model->layers[j].bias != NULL) {
      free(model->layers[j].bias);
      model->layers[j].bias = NULL;
    }
  }
  free(model->layers);
  model->layers = NULL;
  model->num_layers = 0;
  model->input_size = 0;
  model->output_size = 0;
  model->input_scale = 0;
}