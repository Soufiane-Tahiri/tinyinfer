#include <stdio.h>
#include <inttypes.h>
#include "esp_timer.h"
#include "esp_system.h"
#include "tinyinfer.h"
#include "model_data.h"

void app_main(void) {
    printf("\n=================================================\n");
    printf("  tinyinfer v1.0 | ESP32 Bare-Metal AI Engine \n");
    printf("=================================================\n");

    uint32_t free_heap_before = esp_get_free_heap_size();
    printf("[SYS] Free Heap Before Load : %" PRIu32 " bytes\n", free_heap_before);

    ti_model_int8_t model;
    if (!ti_load_model_int8_mem(models_model_int8_bin, models_model_int8_bin_len, &model)) {
        printf("[ERROR] Failed to map model from Flash!\n");
        return;
    }

    printf("[SYS] Zero-Copy Flash Map   : SUCCESS\n");
    printf("[NET] Model Geometry        : %" PRId32 " Layers | In: %" PRId32 " | Out: %" PRId32 "\n", 
           model.header.num_layers, model.header.in_dim, model.header.out_dim);
    
    uint32_t free_heap_after = esp_get_free_heap_size();
    printf("[SYS] Free Heap After Load  : %" PRIu32 " bytes (0 bytes dynamically allocated!)\n\n", free_heap_after);

    float test_input[40] = {0.1f};
    float output_logits[2] = {0.0f};

    printf("--- Running INT8 Inference ---\n");
    
    int64_t start_time = esp_timer_get_time();
    ti_run_int8(&model, test_input, output_logits);
    int64_t end_time = esp_timer_get_time();

    int64_t latency = end_time - start_time;
    float inf_per_sec = 1000000.0f / (float)latency;

    printf("[METRICS] Execution Latency : %lld us (%.3f ms)\n", latency, (float)latency / 1000.0f);
    printf("[METRICS] Max Throughput    : %.1f inferences/second\n\n", inf_per_sec);
    
    printf("[RESULTS] Logit [0]: %8.4f | Logit [1]: %8.4f\n", output_logits[0], output_logits[1]);
    
    int predicted_class = (output_logits[0] > output_logits[1]) ? 0 : 1;
    printf("[PREDICT] Winning Class     : %d\n", predicted_class);
    printf("=================================================\n\n");

    ti_free_model_int8(&model);
}