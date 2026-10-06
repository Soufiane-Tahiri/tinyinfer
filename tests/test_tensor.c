#include <stdio.h>
#include <assert.h>
#include "../include/tensor.h"

void test_tensor_creation_and_free(void) {
    uint32_t shape[4] = {2, 40, 0, 0};
    ti_tensor_t tensor;

    int res = ti_tensor_create(&tensor, shape, 2, TI_FLOAT32);
    assert(res == 0);
    assert(tensor.ndim == 2);
    assert(tensor.dtype == TI_FLOAT32);
    assert(tensor.size == 80);
    assert(tensor.data != NULL);

    ti_tensor_free(&tensor);
    assert(tensor.data == NULL);
    assert(tensor.size == 0);
    assert(tensor.ndim == 0);

    printf("[PASS] test_tensor_creation_and_free\n");
}

void test_tensor_invalid_params(void) {
    uint32_t shape[4] = {1, 1, 0, 0};
    ti_tensor_t tensor;

    assert(ti_tensor_create(&tensor, shape, 0, TI_FLOAT32) != 0);
    assert(ti_tensor_create(&tensor, shape, 5, TI_FLOAT32) != 0);
    assert(ti_tensor_create(NULL, shape, 2, TI_FLOAT32) != 0);

    printf("[PASS] test_tensor_invalid_params\n");
}

int main(void) {
    printf("--- Running Tensor Unit Tests ---\n");
    test_tensor_creation_and_free();
    test_tensor_invalid_params();
    printf("--- All Tensor Tests Passed ---\n");
    return 0;
}