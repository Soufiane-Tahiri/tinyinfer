#define _POSIX_C_SOURCE 200112L


#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tensor.h"

static void *ti_aligned_alloc(size_t alignment, size_t size) {
#if defined(ESP_PLATFORM)
    #include "esp_heap_caps.h"
    (void)alignment;
    return heap_caps_malloc(size, MALLOC_CAP_32BIT | MALLOC_CAP_INTERNAL);

#elif defined(__GLIBC__)
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;

#else
    #warning "ti_aligned_alloc: no aligned allocator available for this platform. \
              Falling back to malloc(). 4-byte alignment is NOT guaranteed. \
              L32I efficiency on Xtensa LX6 will be lost."
    (void)alignment;
    return malloc(size);
#endif
}

int ti_tensor_create(ti_tensor_t *tensor, uint32_t shape[4], uint8_t ndim, ti_dtype_t dtype) {
    if (tensor == NULL) return -1;
    if (ndim == 0 || ndim > 4) return -1;

    for (uint8_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) return -1;
        tensor->shape[i] = shape[i];
    }

    for (uint8_t i = ndim; i < 4; i++) {
        tensor->shape[i] = 0;
    }

    tensor->size = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        if (shape[i] == 0 || tensor->size > UINT32_MAX / shape[i]) return -1;
        tensor->size *= shape[i];
    }

    tensor->ndim = ndim;
    tensor->dtype = dtype;

    size_t elem_size = (dtype == TI_FLOAT32) ? sizeof(float) : sizeof(int8_t);

    tensor->data = ti_aligned_alloc(sizeof(void *), tensor->size * elem_size);
    if (tensor->data == NULL) return -1;
    return 0;
}

void ti_tensor_free(ti_tensor_t *tensor) {
    if (tensor == NULL) return;
    free(tensor->data);
    tensor->data = NULL;
    tensor->size = 0;
    tensor->ndim = 0;
}