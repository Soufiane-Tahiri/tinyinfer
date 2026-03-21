#include <stdio.h>
#include <stdint.h>
#include "../include/tensor.h"
int main() {
    ti_tensor_t tensor;
    uint32_t shape[4] = {2, 3, 4, 5};
    uint8_t ndim = 4;

    if (ti_tensor_create(&tensor, shape, ndim) != 0) {
        printf("Failed to create tensor\n");
        return -1;
    }

    printf("Tensor created with shape: [%u, %u, %u, %u]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
    printf("Tensor size: %u\n", tensor.size);
    printf("Tensor ndim: %u\n", tensor.ndim);

    ti_tensor_free(&tensor);
    printf("Tensor freed\n");

    return 0;
}