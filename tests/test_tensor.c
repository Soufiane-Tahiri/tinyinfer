#include <stdio.h>
#include <stdint.h>
#include "../include/tensor.h"

/* ------------------------------------------------------------------ */
static int test_basic_create_free(void) {
    printf("  [tensor] basic create/free float32: ");

    ti_tensor_t t;
    uint32_t shape[4] = {2, 3, 4, 5};

    if (ti_tensor_create(&t, shape, 4, TI_FLOAT32) != 0) { puts("FAIL"); return 0; }
    if (t.size != 120)  { printf("FAIL: size=%u expected 120\n", t.size); return 0; }
    if (t.ndim != 4)    { printf("FAIL: ndim=%u\n", t.ndim); return 0; }
    if (t.data == NULL) { puts("FAIL: data is NULL"); return 0; }

    ti_tensor_free(&t);
    if (t.data != NULL) { puts("FAIL: data not NULLed after free"); return 0; }

    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_int8_create(void) {
    printf("  [tensor] create int8: ");

    ti_tensor_t t;
    uint32_t shape[4] = {1, 8, 0, 0};

    if (ti_tensor_create(&t, shape, 2, TI_INT8) != 0) { puts("FAIL"); return 0; }
    if (t.size != 8)        { printf("FAIL: size=%u\n", t.size); return 0; }
    if (t.dtype != TI_INT8) { puts("FAIL: dtype wrong"); return 0; }

    ti_tensor_free(&t);
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_null_rejected(void) {
    printf("  [tensor] NULL pointer rejected: ");

    if (ti_tensor_create(NULL, NULL, 2, TI_FLOAT32) != -1) {
        puts("FAIL: accepted NULL tensor pointer"); return 0;
    }
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_zero_dim_rejected(void) {
    printf("  [tensor] zero dimension rejected: ");

    ti_tensor_t t;
    uint32_t shape[4] = {4, 0, 0, 0};

    if (ti_tensor_create(&t, shape, 2, TI_FLOAT32) != -1) {
        puts("FAIL: accepted zero-size dimension"); return 0;
    }
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_overflow_rejected(void) {
    printf("  [tensor] shape overflow rejected (security): ");

    ti_tensor_t t;
    /* 65536 * 65537 = 4,295,032,832 > UINT32_MAX (4,294,967,295).
     * Without the overflow guard, size wraps to 65536, ti_aligned_alloc
     * allocates 256KB, but all kernels trust ->size and write 4GB+.
     * On ESP32 (no MMU) this silently corrupts the heap. */
    uint32_t shape[4] = {65536u, 65537u, 0, 0};

    if (ti_tensor_create(&t, shape, 2, TI_FLOAT32) != -1) {
        puts("FAIL: overflow not caught — heap-overflow possible on weight load");
        return 0;
    }
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_ndim_bounds(void) {
    printf("  [tensor] ndim=0 and ndim>4 rejected: ");

    ti_tensor_t t;
    uint32_t shape[4] = {4, 4, 4, 4};

    if (ti_tensor_create(&t, shape, 0, TI_FLOAT32) != -1) {
        puts("FAIL: ndim=0 accepted"); return 0;
    }
    if (ti_tensor_create(&t, shape, 5, TI_FLOAT32) != -1) {
        puts("FAIL: ndim=5 accepted"); return 0;
    }
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
static int test_alignment(void) {
    printf("  [tensor] data pointer 4-byte aligned: ");

    ti_tensor_t t;
    uint32_t shape[4] = {1, 32, 0, 0};

    if (ti_tensor_create(&t, shape, 2, TI_FLOAT32) != 0) { puts("FAIL"); return 0; }

    uintptr_t addr = (uintptr_t)t.data;
    if (addr % 4 != 0) {
        printf("FAIL: addr %% 4 = %u (not 4-byte aligned)\n",
               (unsigned)(addr % 4));
        ti_tensor_free(&t);
        return 0;
    }

    ti_tensor_free(&t);
    puts("PASS"); return 1;
}

/* ------------------------------------------------------------------ */
int main(void) {
    printf("========================================\n");
    printf("TINYINFER TENSOR TEST SUITE\n");
    printf("========================================\n");

    int passed = 0, total = 0;

    total++; passed += test_basic_create_free();
    total++; passed += test_int8_create();
    total++; passed += test_null_rejected();
    total++; passed += test_zero_dim_rejected();
    total++; passed += test_overflow_rejected();
    total++; passed += test_ndim_bounds();
    total++; passed += test_alignment();

    printf("========================================\n");
    printf("RESULTS: %d / %d passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}