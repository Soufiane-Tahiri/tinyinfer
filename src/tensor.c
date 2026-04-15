#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tensor.h"

/* Aligned allocation wrapper: prefer posix_memalign for 4-byte alignment on ESP32 */
static void* ti_aligned_alloc(size_t alignment, size_t size) {
    #ifdef __GLIBC__
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;
    #else
    /* Fallback: regular malloc (relies on compiler alignment) */
    return malloc(size);
    #endif
}

int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim, ti_dtype_t dtype) {
    if (tensor == NULL) return -1;

    for (uint8_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->size = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;

    uint32_t elem_size = (dtype == TI_FLOAT32) ? sizeof(float) : sizeof(int8_t);
    /* Allocate with 4-byte alignment for L32I efficiency on ESP32 */
    tensor->data = ti_aligned_alloc(4, tensor->size * elem_size);
    if (tensor->data == NULL) return -1;

    /* NOTE: User is responsible for initialization if needed.
     * Skipping memset saves one full memory pass, critical on ESP32.
     * Inference kernels overwrite all outputs anyway. */
    return 0;
}

void ti_tensor_free(ti_tensor_t *tensor) {
    if (tensor == NULL) return;
    free(tensor->data);
    tensor->data = NULL;
    tensor->size = 0;
    tensor->ndim = 0;
}