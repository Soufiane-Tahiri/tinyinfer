#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/activations.h"

#define EPSILON     1e-5f
#define SOFTMAX_SCRATCH_SIZE 64

static int close_enough(float got, float expected) {
    float diff = fabsf(got - expected);
    if (diff < EPSILON) return 1;
    return (diff / (fabsf(expected) + EPSILON)) < 0.01f;
}

static int test_relu_float32(void) {
    printf("  [relu-f32] basic: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 5, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_FLOAT32);
    ti_tensor_create(&out, shape, 2, TI_FLOAT32);

    float *x = (float *)in.data;
    float *y = (float *)out.data;

    x[0] = -2.0f; x[1] = -0.5f; x[2] = 0.0f; x[3] = 1.5f; x[4] = 3.0f;
    float expected[5] = {0.0f, 0.0f, 0.0f, 1.5f, 3.0f};

    if (ti_relu_forward(&in, &out) != 0) { puts("FAIL: returned error"); return 0; }

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (!close_enough(y[i], expected[i])) {
            printf("FAIL at [%d]: got %.4f expected %.4f\n", i, y[i], expected[i]);
            pass = 0;
        }
    }

    ti_tensor_free(&in); ti_tensor_free(&out);
    if (pass) puts("PASS");
    return pass;
}

static int test_relu_float32_null(void) {
    printf("  [relu-f32] null input rejected: ");
    if (ti_relu_forward(NULL, NULL) != -1) { puts("FAIL"); return 0; }
    puts("PASS"); return 1;
}

static int test_relu_float32_dtype_mismatch(void) {
    printf("  [relu-f32] dtype mismatch rejected: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 4, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_FLOAT32);
    ti_tensor_create(&out, shape, 2, TI_INT8);

    int result = ti_relu_forward(&in, &out);
    ti_tensor_free(&in); ti_tensor_free(&out);

    if (result != -1) { puts("FAIL: accepted mismatched dtypes"); return 0; }
    puts("PASS"); return 1;
}

static int test_relu_int8(void) {
    printf("  [relu-i8]  basic: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 5, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_INT8);
    ti_tensor_create(&out, shape, 2, TI_INT8);

    int8_t *x = (int8_t *)in.data;
    int8_t *y = (int8_t *)out.data;

    x[0] = -10; x[1] = -1; x[2] = 0; x[3] = 5; x[4] = 20;
    int8_t expected[5] = {0, 0, 0, 5, 20};

    if (ti_relu_forward(&in, &out) != 0) { puts("FAIL: returned error"); return 0; }

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (y[i] != expected[i]) {
            printf("FAIL at [%d]: got %d expected %d\n", i, y[i], expected[i]);
            pass = 0;
        }
    }

    ti_tensor_free(&in); ti_tensor_free(&out);
    if (pass) puts("PASS");
    return pass;
}

static int test_softmax_float32_basic(void) {
    printf("  [smx-f32]  basic [1,2,3]: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 3, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_FLOAT32);
    ti_tensor_create(&out, shape, 2, TI_FLOAT32);

    float scratch[SOFTMAX_SCRATCH_SIZE];
    float *x = (float *)in.data;
    float *y = (float *)out.data;

    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;

    if (ti_softmax_forward(&in, &out, scratch) != 0) { puts("FAIL: returned error"); return 0; }

    float sum = y[0] + y[1] + y[2];
    int pass = 1;

    for (int i = 0; i < 3; i++) {
        if (y[i] < 0.0f || y[i] > 1.0f) {
            printf("FAIL: y[%d]=%.6f outside [0,1]\n", i, y[i]);
            pass = 0;
        }
    }
    if (!close_enough(sum, 1.0f)) {
        printf("FAIL: sum=%.6f not ~1.0\n", sum); pass = 0;
    }
    if (!(y[2] > y[1] && y[1] > y[0])) {
        puts("FAIL: ordering wrong, expected y[2]>y[1]>y[0]"); pass = 0;
    }

    ti_tensor_free(&in); ti_tensor_free(&out);
    if (pass) puts("PASS");
    return pass;
}

static int test_softmax_float32_stability(void) {
    printf("  [smx-f32]  large values (log-sum-exp stability): ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 3, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_FLOAT32);
    ti_tensor_create(&out, shape, 2, TI_FLOAT32);

    float scratch[SOFTMAX_SCRATCH_SIZE];
    float *x = (float *)in.data;
    float *y = (float *)out.data;

    x[0] = 1000.0f; x[1] = 1001.0f; x[2] = 999.0f;

    if (ti_softmax_forward(&in, &out, scratch) != 0) { puts("FAIL: returned error"); return 0; }

    int pass = 1;
    for (int i = 0; i < 3; i++) {
        if (isnan(y[i]) || isinf(y[i])) {
            printf("FAIL: y[%d] is NaN or Inf\n", i); pass = 0;
        }
    }
    float sum = y[0] + y[1] + y[2];
    if (!close_enough(sum, 1.0f)) {
        printf("FAIL: sum=%.6f\n", sum); pass = 0;
    }

    ti_tensor_free(&in); ti_tensor_free(&out);
    if (pass) puts("PASS");
    return pass;
}

static int test_softmax_float32_null_scratch(void) {
    printf("  [smx-f32]  NULL scratch rejected: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 3, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_FLOAT32);
    ti_tensor_create(&out, shape, 2, TI_FLOAT32);

    int result = ti_softmax_forward(&in, &out, NULL);
    ti_tensor_free(&in); ti_tensor_free(&out);

    if (result != -1) { puts("FAIL: NULL scratch accepted"); return 0; }
    puts("PASS"); return 1;
}

static int test_softmax_int8(void) {
    printf("  [smx-i8]   basic [1,2,3]: ");

    ti_tensor_t in, out;
    uint32_t shape[4] = {1, 3, 0, 0};
    ti_tensor_create(&in,  shape, 2, TI_INT8);
    ti_tensor_create(&out, shape, 2, TI_INT8);

    float scratch[SOFTMAX_SCRATCH_SIZE];
    int8_t *x = (int8_t *)in.data;
    int8_t *y = (int8_t *)out.data;

    x[0] = 1; x[1] = 2; x[2] = 3;

    if (ti_softmax_forward(&in, &out, scratch) != 0) { puts("FAIL: returned error"); return 0; }

    int pass = 1;
    for (int i = 0; i < 3; i++) {
        int val = (int)y[i];
        if (val < 0 || val > 127) {
            printf("FAIL: y[%d]=%d outside [0,127]\n", i, val); pass = 0;
        }
    }

    float y0 = (float)y[0] / 127.0f;
    float y1 = (float)y[1] / 127.0f;
    float y2 = (float)y[2] / 127.0f;
    float sum = y0 + y1 + y2;

    if (sum < 0.95f || sum > 1.05f) {
        printf("FAIL: dequant sum=%.4f outside [0.95,1.05]\n", sum); pass = 0;
    }
    if (!(y[2] > y[1] && y[1] > y[0])) {
        puts("FAIL: ordering wrong, expected y[2]>y[1]>y[0]"); pass = 0;
    }

    ti_tensor_free(&in); ti_tensor_free(&out);
    if (pass) puts("PASS");
    return pass;
}

int main(void) {
    printf("========================================\n");
    printf("TINYINFER ACTIVATION TEST SUITE\n");
    printf("========================================\n");

    int passed = 0, total = 0;

    total++; passed += test_relu_float32();
    total++; passed += test_relu_float32_null();
    total++; passed += test_relu_float32_dtype_mismatch();
    total++; passed += test_relu_int8();
    total++; passed += test_softmax_float32_basic();
    total++; passed += test_softmax_float32_stability();
    total++; passed += test_softmax_float32_null_scratch();
    total++; passed += test_softmax_int8();

    printf("========================================\n");
    printf("RESULTS: %d / %d passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}