#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
static inline int ti_argmax_float32(const float *logits, uint32_t n) {
    int best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = (int)i;
    return best;
}

static inline int ti_argmax_int32(const int32_t *logits, uint32_t n) {
    int best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = (int)i;
    return best;
}

static inline float ti_margin_float32(const float *logits) {
    return logits[1] - logits[0];
}

static inline int32_t ti_margin_int32(const int32_t *logits) {
    return logits[1] - logits[0];
}
#endif
