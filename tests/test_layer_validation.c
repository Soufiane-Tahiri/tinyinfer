#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include "../include/layers.h"
#include "../include/tensor.h"

static int expect_equal(int actual, int expected, const char *name) {
    if (actual != expected) {
        printf("%s failed: got %d expected %d\n", name, actual, expected);
        return -1;
    }
    return 0;
}

int main() {
    if (expect_equal(ti_dense_validate_config(3, 4), 0, "valid config") != 0) return -1;
    if (expect_equal(ti_dense_validate_config(0, 4), -1, "zero input rejected") != 0) return -1;
    if (expect_equal(ti_dense_validate_config(4, 0), -1, "zero output rejected") != 0) return -1;
    if (expect_equal(ti_dense_validate_config(65536U, 2), -1, "oversized dim rejected") != 0) return -1;
    if (expect_equal(ti_dense_validate_config(UINT32_MAX, 2), -1, "overflowing dim rejected") != 0) return -1;

    if (expect_equal(ti_dense_validate_serialized_buffers(3, 4, 48, 16, TI_FLOAT32), 0, "float32 lengths accepted") != 0) return -1;
    if (expect_equal(ti_dense_validate_serialized_buffers(3, 4, 12, 16, TI_INT8), 0, "int8 lengths accepted") != 0) return -1;
    if (expect_equal(ti_dense_validate_serialized_buffers(3, 4, 47, 16, TI_FLOAT32), -1, "weights mismatch rejected") != 0) return -1;
    if (expect_equal(ti_dense_validate_serialized_buffers(3, 4, 12, 15, TI_INT8), -1, "bias mismatch rejected") != 0) return -1;

    {
        const float weights[1] = {0.0f};
        const float bias[1] = {0.0f};
        ti_layer_t layer = {
            .input_size = UINT32_MAX,
            .output_size = 2,
            .weights = weights,
            .bias = bias,
            .dtype = TI_FLOAT32
        };

        ti_tensor_t input = {0};
        ti_tensor_t output = {0};
        input.ndim = 2;
        input.shape[0] = 1;
        input.shape[1] = UINT32_MAX;
        output.ndim = 2;
        output.shape[0] = 1;
        output.shape[1] = 2;

        if (expect_equal(ti_dense_forward_float32(&layer, &input, &output), -1, "forward rejects invalid config") != 0) return -1;
    }

    return 0;
}
