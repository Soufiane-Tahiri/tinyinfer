#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "include/tinyinfer.h"
#include "include/model.h"
#include "include/tensor.h"

#define LOGIT_DIFF_THRESHOLD  (-3.2191344738006595f)

#define INT8_OUTPUT_SCALE     (0.02546551823616028f)

#define TOLERANCE             (0.05f)

#define MODEL_F32_PATH   "../models/model_f32.bin"
#define MODEL_INT8_PATH  "../models/model_int8.bin"

#define INPUT_DIM  40
#define OUTPUT_DIM  2

static const float test_input[INPUT_DIM] = {
    -0.221617f,  0.710031f,  0.785513f, -0.014089f,
    -0.089486f, -0.007736f, -0.127866f, -0.027023f,
     1.235694f, -0.070058f, -0.036652f, -0.024437f,
    -0.055255f, -0.039599f, -0.018610f, -0.041221f,
    -0.002817f, -0.097531f, -0.104898f,  0.711294f,
    -0.637209f, -0.631929f, -0.374362f, -0.374432f,
     0.771283f, -0.349683f, -0.028179f,  0.590697f,
     1.026739f,  1.066401f, -0.439078f, -0.480197f,
    -0.289103f, -0.639532f, -0.624871f, -0.387635f,
    -0.376387f,  0.476175f, -0.367555f, -0.265429f
};

static const float ref_f32[OUTPUT_DIM]  = { 1.739524f, -1.948402f };

static const float ref_int8[OUTPUT_DIM] = { 1.757121f, -1.986310f };

static int check(const char *label, const float *got, const float *expected, int n, float tol) {
    int pass = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - expected[i]) >= tol) {
            printf("  [FAIL] %s logit[%d]: got %.6f, expected %.6f (err %.6f)\n",
                label, i, got[i], expected[i], fabsf(got[i] - expected[i]));
            pass = 0;
        }
    }
    if (pass)
        printf("  [PASS] %s  logits [%.6f, %.6f]\n",
               label, got[0], got[1]);
    return pass;
}

int main(void) {
    printf("==============================================\n");
    printf("  tinyinfer  end-to-end validation harness\n");
    printf("==============================================\n\n");

    int all_pass = 1;

    printf("[1] Framework float32 (host engine)\n");

    ti_model_t fw_f32;
    if (ti_model_load(&fw_f32, MODEL_F32_PATH) != 0) {
        printf("  ERROR: failed to load %s\n", MODEL_F32_PATH);
        return 1;
    }

    ti_tensor_t in_tensor, out_tensor_f32;
    uint32_t in_shape[4]  = {1, INPUT_DIM,  0, 0};
    uint32_t out_shape[4] = {1, OUTPUT_DIM, 0, 0};

    if (ti_tensor_create(&in_tensor, in_shape, 2, TI_FLOAT32) != 0) {
        printf("  ERROR: failed to create input tensor\n");
        ti_model_free(&fw_f32);
        return 1;
    }
    memcpy(in_tensor.data, test_input, INPUT_DIM * sizeof(float));

    if (ti_tensor_create(&out_tensor_f32, out_shape, 2, TI_FLOAT32) != 0) {
        printf("  ERROR: failed to create f32 output tensor\n");
        ti_tensor_free(&in_tensor);
        ti_model_free(&fw_f32);
        return 1;
    }

    clock_t t0 = clock();
    if (ti_model_run(&fw_f32, &in_tensor, &out_tensor_f32) != 0) {
        printf("  ERROR: ti_model_run f32 failed\n");
        ti_tensor_free(&in_tensor);
        ti_tensor_free(&out_tensor_f32);
        ti_model_free(&fw_f32);
        return 1;
    }
    clock_t t1 = clock();

    float *fw_f32_out = (float *)out_tensor_f32.data;
    all_pass &= check("fw-f32", fw_f32_out, ref_f32, OUTPUT_DIM, TOLERANCE);
    printf("  latency (host): %.3f ms\n\n", (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0);

    ti_tensor_free(&out_tensor_f32);

    printf("[2] Framework int8 (host engine)\n");

    ti_tensor_t in_tensor_i8;
    if (ti_tensor_create(&in_tensor_i8, in_shape, 2, TI_INT8) != 0) {
        printf("  ERROR: failed to create int8 input tensor\n");
        ti_tensor_free(&in_tensor);
        ti_model_free(&fw_f32);
        return 1;
    }

    ti_model_t fw_int8;
    if (ti_model_load(&fw_int8, MODEL_INT8_PATH) != 0) {
        printf("  ERROR: failed to load %s\n", MODEL_INT8_PATH);
        ti_tensor_free(&in_tensor);
        ti_tensor_free(&in_tensor_i8);
        ti_model_free(&fw_f32);
        return 1;
    }

    float in_scale = fw_int8.input_scale;
    int8_t *qi = (int8_t *)in_tensor_i8.data;
    for (int i = 0; i < INPUT_DIM; i++) {
        float v = roundf(test_input[i] / in_scale);
        if (v >  127.0f) v =  127.0f;
        if (v < -128.0f) v = -128.0f;
        qi[i] = (int8_t)v;
    }

    ti_tensor_t out_tensor_i32;
    if (ti_tensor_create(&out_tensor_i32, out_shape, 2, TI_INT32) != 0) {
        printf("  ERROR: failed to create int32 output tensor\n");
        ti_tensor_free(&in_tensor);
        ti_tensor_free(&in_tensor_i8);
        ti_model_free(&fw_f32);
        ti_model_free(&fw_int8);
        return 1;
    }

    clock_t t2 = clock();
    if (ti_model_run(&fw_int8, &in_tensor_i8, &out_tensor_i32) != 0) {
        printf("  ERROR: ti_model_run int8 failed\n");
        ti_tensor_free(&in_tensor);
        ti_tensor_free(&in_tensor_i8);
        ti_tensor_free(&out_tensor_i32);
        ti_model_free(&fw_f32);
        ti_model_free(&fw_int8);
        return 1;
    }
    clock_t t3 = clock();

    int32_t *raw_i32 = (int32_t *)out_tensor_i32.data;
    float fw_int8_dequant[OUTPUT_DIM];

    float final_requant_scale = fw_int8.layers[fw_int8.num_layers - 1].requant_scale;

    for (int i = 0; i < OUTPUT_DIM; i++) {
        float q = roundf((float)raw_i32[i] * final_requant_scale);

        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;

        fw_int8_dequant[i] = q * INT8_OUTPUT_SCALE;
    }


    all_pass &= check("fw-int8", fw_int8_dequant, ref_int8, OUTPUT_DIM, TOLERANCE);
    printf("  latency (host): %.3f ms\n\n", (double)(t3 - t2) / CLOCKS_PER_SEC * 1000.0);

    ti_tensor_free(&out_tensor_i32);
    ti_tensor_free(&in_tensor_i8);

    printf("[3] Micro-kernel float32 (ESP32 runtime)\n");

    ti_model_f32_t tk_f32;
    if (!ti_load_model_f32(MODEL_F32_PATH, &tk_f32)) {
        printf("  ERROR: failed to load kernel f32 model\n");
        ti_tensor_free(&in_tensor);
        ti_model_free(&fw_f32);
        ti_model_free(&fw_int8);
        return 1;
    }

    float tk_f32_out[OUTPUT_DIM] = {0};
    clock_t t4 = clock();
    ti_run_f32(&tk_f32, test_input, tk_f32_out);
    clock_t t5 = clock();

    all_pass &= check("tk-f32", tk_f32_out, ref_f32, OUTPUT_DIM, TOLERANCE);
    printf("  latency (host): %.3f ms\n\n", (double)(t5 - t4) / CLOCKS_PER_SEC * 1000.0);

    printf("[4] Micro-kernel int8 (ESP32 runtime)\n");

    ti_model_int8_t tk_int8;
    if (!ti_load_model_int8(MODEL_INT8_PATH, &tk_int8)) {
        printf("  ERROR: failed to load kernel int8 model\n");
        ti_tensor_free(&in_tensor);
        ti_model_free(&fw_f32);
        ti_model_free(&fw_int8);
        ti_free_model_f32(&tk_f32);
        return 1;
    }

    float tk_int8_out[OUTPUT_DIM] = {0};
    clock_t t6 = clock();
    ti_run_int8(&tk_int8, test_input, tk_int8_out);
    clock_t t7 = clock();

    // APPLY MISSING OUTPUT SCALE
    for (int i = 0; i < OUTPUT_DIM; i++) {
        tk_int8_out[i] = tk_int8_out[i] * INT8_OUTPUT_SCALE;
    }

    all_pass &= check("tk-int8", tk_int8_out, ref_int8, OUTPUT_DIM, TOLERANCE);
    printf("  latency (host): %.3f ms\n\n", (double)(t7 - t6) / CLOCKS_PER_SEC * 1000.0);

    float margin = tk_int8_out[1] - tk_int8_out[0];
    const char *decision = (margin > LOGIT_DIFF_THRESHOLD) ? "ATTACK" : "NORMAL";

    printf("==============================================\n");
    printf("  Embedded Decision (kernel int8)\n");
    printf("  Logits:   [%.6f, %.6f]\n", tk_int8_out[0], tk_int8_out[1]);
    printf("  Margin:   %.6f (threshold %.6f)\n", margin, LOGIT_DIFF_THRESHOLD);
    printf("  Decision: %s\n", decision);
    printf("  Validation: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");
    printf("==============================================\n");

    ti_tensor_free(&in_tensor);
    ti_model_free(&fw_f32);
    ti_model_free(&fw_int8);
    ti_free_model_f32(&tk_f32);
    ti_free_model_int8(&tk_int8);

    return all_pass ? 0 : 1;
}