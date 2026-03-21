#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tensor.h"
int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim){
  if (tensor == NULL){
    return -1;
  }
  for (uint8_t i = 0; i < ndim; i++){
    tensor -> shape[i] = shape[i];
  }
  tensor->size = 1;
  for (uint8_t i = 0; i < ndim; i++){
    tensor -> size = tensor -> size * shape[i];
  }
  tensor -> ndim = ndim;
  tensor -> data = (float *)malloc(tensor -> size * sizeof(float));
  if (tensor -> data == NULL){
    return -1;
  }
  memset(tensor -> data, 0, tensor -> size * sizeof(float));
  return 0;
}
void ti_tensor_free(ti_tensor_t *tensor){
  if (tensor == NULL){
    return;
  }
  free(tensor -> data);
  tensor -> size = 0;
  tensor -> ndim = 0;
  tensor -> data = NULL;
}
