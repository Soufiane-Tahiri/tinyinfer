#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/activations.h"

#define EPSILON 1e-5f
#define INT8_EPSILON 2

int close_enough_float(float got, float expected) {
    float diff = fabs(got - expected);
    if (diff < EPSILON) return 1;
    float rel_error = diff / (fabs(expected) + EPSILON);
    return rel_error < 0.01f;
}

int close_enough_int8(int8_t got, int8_t expected) {
    return (got >= expected - INT8_EPSILON) && (got <= expected + INT8_EPSILON);
}

int test_relu_float32() {
    printf("\n=== TEST: ReLU Float32 ===\n");

    ti_tensor_t input, output;
    uint32_t shape[4] = {1, 5, 0, 0};

    ti_tensor_create(&input, shape, 2, TI_FLOAT32);
    ti_tensor_create(&output, shape, 2, TI_FLOAT32);

    float *x = (float *)input.data;
    float *y = (float *)output.data;


    x[0] = -2.0f;
    x[1] = -0.5f;
    x[2] = 0.0f;
    x[3] = 1.5f;
    x[4] = 3.0f;

    int result = ti_relu_forward(&input, &output);
    if (result != 0) {
        printf("FAIL: ti_relu_forward returned %d\n", result);
        return 0;
    }

    float expected[5] = {0.0f, 0.0f, 0.0f, 1.5f, 3.0f};

    printf("Input:    [");
    for (int i = 0; i < 5; i++) printf("%.2f ", x[i]);
    printf("]\n");

    printf("Output:   [");
    for (int i = 0; i < 5; i++) printf("%.2f ", y[i]);
    printf("]\n");

    printf("Expected: [");
    for (int i = 0; i < 5; i++) printf("%.2f ", expected[i]);
    printf("]\n");

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (!close_enough_float(y[i], expected[i])) {
            printf("FAIL at index %d: got %.6f, expected %.6f\n", i, y[i], expected[i]);
            pass = 0;
        }
    }

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    if (pass) printf("PASS\n");
    return pass;
}

int test_relu_int8() {
    printf("\n=== TEST: ReLU Int8 ===\n");

    ti_tensor_t input, output;
    uint32_t shape[4] = {1, 5, 0, 0};

    ti_tensor_create(&input, shape, 2, TI_INT8);
    ti_tensor_create(&output, shape, 2, TI_INT8);

    int8_t *x = (int8_t *)input.data;
    int8_t *y = (int8_t *)output.data;

    x[0] = -10;
    x[1] = -1;
    x[2] = 0;
    x[3] = 5;
    x[4] = 20;

    int result = ti_relu_forward(&input, &output);
    if (result != 0) {
        printf("FAIL: ti_relu_forward returned %d\n", result);
        return 0;
    }

    int8_t expected[5] = {0, 0, 0, 5, 20};

    printf("Input:    [");
    for (int i = 0; i < 5; i++) printf("%d ", x[i]);
    printf("]\n");

    printf("Output:   [");
    for (int i = 0; i < 5; i++) printf("%d ", y[i]);
    printf("]\n");

    printf("Expected: [");
    for (int i = 0; i < 5; i++) printf("%d ", expected[i]);
    printf("]\n");

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (y[i] != expected[i]) {
            printf("FAIL at index %d: got %d, expected %d\n", i, y[i], expected[i]);
            pass = 0;
        }
    }

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    if (pass) printf("PASS\n");
    return pass;
}

int test_softmax_float32() {
    printf("\n=== TEST: Softmax Float32 ===\n");

    ti_tensor_t input, output;
    uint32_t shape[4] = {1, 3, 0, 0};

    ti_tensor_create(&input, shape, 2, TI_FLOAT32);
    ti_tensor_create(&output, shape, 2, TI_FLOAT32);

    float *x = (float *)input.data;
    float *y = (float *)output.data;

    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    int result = ti_softmax_forward(&input, &output);
    if (result != 0) {
        printf("FAIL: ti_softmax_forward returned %d\n", result);
        return 0;
    }

    printf("Input:    [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);
    printf("Output:   [%.6f, %.6f, %.6f]\n", y[0], y[1], y[2]);


    float sum = y[0] + y[1] + y[2];
    printf("Sum:      %.6f (should be ~1.0)\n", sum);

    int pass = 1;

    for (int i = 0; i < 3; i++) {
        if (y[i] < 0.0f || y[i] > 1.0f) {
            printf("FAIL at index %d: value %.6f outside [0, 1]\n", i, y[i]);
            pass = 0;
        }
    }

    if (!close_enough_float(sum, 1.0f)) {
        printf("FAIL: sum %.6f not close to 1.0\n", sum);
        pass = 0;
    }

    if (!(y[2] > y[1] && y[1] > y[0])) {
        printf("FAIL: ordering wrong. Expected y[2] > y[1] > y[0]\n");
        pass = 0;
    }

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    if (pass) printf("PASS\n");
    return pass;
}

int test_softmax_int8() {
    printf("\n=== TEST: Softmax Int8 ===\n");

    ti_tensor_t input, output;
    uint32_t shape[4] = {1, 3, 0, 0};

    ti_tensor_create(&input, shape, 2, TI_INT8);
    ti_tensor_create(&output, shape, 2, TI_INT8);

    int8_t *x = (int8_t *)input.data;
    int8_t *y = (int8_t *)output.data;

    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    int result = ti_softmax_forward(&input, &output);
    if (result != 0) {
        printf("FAIL: ti_softmax_forward returned %d\n", result);
        return 0;
    }

    printf("Input:    [%d, %d, %d]\n", x[0], x[1], x[2]);
    printf("Output:   [%d, %d, %d] (quantized to [0, 127])\n", y[0], y[1], y[2]);

    float y0 = (float)y[0] / 127.0f;
    float y1 = (float)y[1] / 127.0f;
    float y2 = (float)y[2] / 127.0f;

    printf("Dequant:  [%.6f, %.6f, %.6f]\n", y0, y1, y2);

    float sum = y0 + y1 + y2;
    printf("Sum:      %.6f (should be ~1.0, allow ~2%% quant error)\n", sum);

    int pass = 1;

    for (int i = 0; i < 3; i++) {
        if (y[i] < 0 || y[i] > 127) {
            printf("FAIL at index %d: value %d outside [0, 127]\n", i, y[i]);
            pass = 0;
        }
    }

    if (sum < 0.95f || sum > 1.05f) {
        printf("FAIL: sum %.6f too far from 1.0 (quantization error?)\n", sum);
        pass = 0;
    }

    if (!(y[2] > y[1] && y[1] > y[0])) {
        printf("FAIL: ordering wrong. Expected y[2] > y[1] > y[0]\n");
        pass = 0;
    }

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    if (pass) printf("PASS\n");
    return pass;
}

int test_softmax_float32_large_values() {
    printf("\n=== TEST: Softmax Float32 (Large Values - Log-Sum-Exp Stability) ===\n");

    ti_tensor_t input, output;
    uint32_t shape[4] = {1, 3, 0, 0};

    ti_tensor_create(&input, shape, 2, TI_FLOAT32);
    ti_tensor_create(&output, shape, 2, TI_FLOAT32);

    float *x = (float *)input.data;
    float *y = (float *)output.data;

    x[0] = 1000.0f;
    x[1] = 1001.0f;
    x[2] = 999.0f;

    int result = ti_softmax_forward(&input, &output);
    if (result != 0) {
        printf("FAIL: ti_softmax_forward returned %d\n", result);
        return 0;
    }

    printf("Input:    [%.0f, %.0f, %.0f] (large values)\n", x[0], x[1], x[2]);
    printf("Output:   [%.6f, %.6f, %.6f]\n", y[0], y[1], y[2]);

    float sum = y[0] + y[1] + y[2];
    printf("Sum:      %.6f (should be ~1.0)\n", sum);

    int pass = 1;

    for (int i = 0; i < 3; i++) {
        if (isnan(y[i]) || isinf(y[i])) {
            printf("FAIL at index %d: got NaN or Inf\n", i);
            pass = 0;
        }
    }

    if (!close_enough_float(sum, 1.0f)) {
        printf("FAIL: sum %.6f not close to 1.0\n", sum);
        pass = 0;
    }

    ti_tensor_free(&input);
    ti_tensor_free(&output);

    if (pass) printf("PASS\n");
    return pass;
}

int main() {
    printf("========================================\n");
    printf("TINYINFER ACTIVATION LAYER TEST SUITE\n");
    printf("========================================\n");

    int passed = 0;
    int total = 0;

    /* Run all tests */
    total++; if (test_relu_float32()) passed++;
    total++; if (test_relu_int8()) passed++;
    total++; if (test_softmax_float32()) passed++;
    total++; if (test_softmax_int8()) passed++;
    total++; if (test_softmax_float32_large_values()) passed++;

    printf("\n========================================\n");
    printf("RESULTS: %d / %d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}

