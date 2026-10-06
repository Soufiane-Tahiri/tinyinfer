#define _POSIX_C_SOURCE 200112L
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "layers.h"

#define TI_TILE_FLOAT32_OUTPUTS 4
#define TI_TILE_FLOAT32_INPUTS  8
#define TI_TILE_INT8_OUTPUTS    8
#define TI_TILE_INT8_INPUTS     4


static inline int8_t ti_requant(int32_t acc, float rscale, int relu) {
    int32_t q = (int32_t)lrintf((float) acc * rscale);
    if (relu) {
        return (int8_t) (q>127 ? 127 : (q<0 ? 0 : q));
    }
    else {
        return (int8_t) (q>127 ? 127 : (q<-128 ? -128 : q));
    }
}
int ti_dense_forward_float32(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->weights == NULL || layer->bias == NULL) return -1;
    if (input->ndim != 2 || input->shape[1]!=layer->input_size) return -1;
    if (output->ndim != 2 ||output->shape[0] != input->shape[0] || output->shape[1] != layer->output_size) return -1;
    if (layer->input_size == 0 || layer->input_size > TI_MAX_LAYER_DIM) return -1;
    if (layer->output_size == 0 || layer->output_size > TI_MAX_LAYER_DIM) return -1;
    uint32_t batch_size = input->shape[0];
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;
    int relu = (layer->activation == TI_ACT_RELU);
    const float *w = (const float *)layer->weights;
    const float *b = (const float *)layer->bias;
    const float *x_data = (const float *)input->data;
    float *y_data = (float *)output->data;
    for (uint32_t i = 0; i < batch_size; i++) {
        const float *x = x_data + i * input_size;
        float *y = y_data + i * output_size;
        uint32_t j= 0;
        for (; j+3 < output_size; j+=4) {
            float sum0 = b[j+0] , sum1=b[j+1], sum2=b[j+2], sum3=b[j+3];
            uint32_t w0= (j+0)*input_size , w1=(j+1)*input_size, w2=(j+2)*input_size, w3=(j+3)*input_size;
            uint32_t k=0;
            for (; k+7<input_size; k+=8) {
                float x0=x[k+0], x1=x[k+1], x2=x[k+2], x3=x[k+3];
                float x4=x[k+4], x5=x[k+5], x6=x[k+6], x7=x[k+7];
                sum0 += w[w0+k+0]*x0 + w[w0+k+1]*x1 + w[w0+k+2]*x2 + w[w0+k+3]*x3 + w[w0+k+4]*x4 + w[w0+k+5]*x5 + w[w0+k+6]*x6 + w[w0+k+7]*x7;
                sum1 += w[w1+k+0]*x0 + w[w1+k+1]*x1 + w[w1+k+2]*x2 + w[w1+k+3]*x3 + w[w1+k+4]*x4 + w[w1+k+5]*x5 + w[w1+k+6]*x6 + w[w1+k+7]*x7;
                sum2 += w[w2+k+0]*x0 + w[w2+k+1]*x1 + w[w2+k+2]*x2 + w[w2+k+3]*x3 + w[w2+k+4]*x4 + w[w2+k+5]*x5 + w[w2+k+6]*x6 + w[w2+k+7]*x7;
                sum3 += w[w3+k+0]*x0 + w[w3+k+1]*x1 + w[w3+k+2]*x2 + w[w3+k+3]*x3 + w[w3+k+4]*x4 + w[w3+k+5]*x5 + w[w3+k+6]*x6 + w[w3+k+7]*x7;
            }
            for (; k<input_size; k++) {
                float xk = x[k];
                sum0 += w[w0+k]*xk;
                sum1 += w[w1+k]*xk;
                sum2 += w[w2+k]*xk;
                sum3 += w[w3+k]*xk;
            }
            if (relu) {
                y[j+0] = sum0>0.0f ? sum0 : 0.0f;
                y[j+1] = sum1>0.0f ? sum1 : 0.0f;
                y[j+2] = sum2>0.0f ? sum2 : 0.0f;
                y[j+3] = sum3>0.0f ? sum3 : 0.0f;
            }
            else {
                y[j+0] = sum0;
                y[j+1] = sum1;
                y[j+2] = sum2;
                y[j+3] = sum3;
            }
        }
        for (; j+1<output_size; j+=2) {
            float sum0=b[j+0],sum1=b[j+1];
            uint32_t w0=(j+0)*input_size, w1=(j+1)*input_size;
            for (uint32_t k=0; k<input_size; k++) {
                sum0 += w[w0+k]*x[k];
                sum1 += w[w1+k]*x[k];
            }
            if (relu) {
                y[j+0] = sum0>0.0f ? sum0 : 0.0f;
                y[j+1] = sum1>0.0f ? sum1 : 0.0f;
            }
            else {
                y[j+0] = sum0;
                y[j+1] = sum1;
            }

        }
        if (j<output_size) {
            float sum = b[j];
            uint32_t w0=j*input_size;
            for (uint32_t k=0; k<input_size; k++) {
                sum += w[w0+k]*x[k];
            }
            if (relu) {
                y[j] = sum>0.0f ? sum : 0.0f;
            }
            else {
                y[j] = sum;
            }

        }
    }
        return 0 ;

}

int ti_dense_forward_int8(const ti_layer_t *layer, const ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->weights == NULL || layer->bias == NULL) return -1;
    if (input->ndim != 2 || input->shape[1]!=layer->input_size) return -1;
    if (output->ndim != 2 || output->shape[0] != input->shape[0] ||output->shape[1]!=layer->output_size) return -1;
    if (layer->input_size == 0 || layer->input_size > TI_MAX_LAYER_DIM) return -1;
    if (layer->output_size == 0 || layer->output_size > TI_MAX_LAYER_DIM) return -1;
    if (output->dtype == TI_INT8 && layer->requant_scale <= 0.0f) return -1;
    uint32_t batch_size = input->shape[0];
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;
    int relu = (layer->activation == TI_ACT_RELU);
    float rscale = layer->requant_scale;

    const int8_t *w = (const int8_t *)layer->weights;
    const int32_t *b = (const int32_t *)layer->bias;
    const int8_t *x_data = (const int8_t *)input->data;
    int8_t *y_data_int8 = NULL;
    int32_t *y_data_int32 = NULL;
    if (output->dtype == TI_INT8) {
        y_data_int8 = (int8_t *)output->data;
    } else if (output->dtype == TI_INT32) {
        y_data_int32 = (int32_t *)output->data;
    } else {
        return -1;
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        const int8_t *x = x_data + i*input_size;
        uint32_t j=0;
        for (; j+7 < output_size; j+=8) {
            int32_t sum0 = b[j+0], sum1 = b[j+1], sum2 = b[j+2], sum3 = b[j+3];
            int32_t sum4 = b[j+4], sum5 = b[j+5], sum6 = b[j+6], sum7 = b[j+7];

            uint32_t w0 = (j+0)*input_size, w1 = (j+1)*input_size, w2 = (j+2)*input_size, w3 = (j+3)*input_size;
            uint32_t w4 = (j+4)*input_size, w5 = (j+5)*input_size, w6 = (j+6)*input_size, w7 = (j+7)*input_size;

            uint32_t k = 0;
            for (; k+3 <input_size; k+=4) {
                int32_t x0 = (int32_t)x[k+0], x1 = (int32_t)x[k+1], x2 = (int32_t)x[k+2], x3 = (int32_t)x[k+3];
                sum0 += (int32_t)w[w0+k+0]*x0 + (int32_t)w[w0+k+1]*x1 + (int32_t)w[w0+k+2]*x2 + (int32_t)w[w0+k+3]*x3;
                sum1 += (int32_t)w[w1+k+0]*x0 + (int32_t)w[w1+k+1]*x1 + (int32_t)w[w1+k+2]*x2 + (int32_t)w[w1+k+3]*x3;
                sum2 += (int32_t)w[w2+k+0]*x0 + (int32_t)w[w2+k+1]*x1 + (int32_t)w[w2+k+2]*x2 + (int32_t)w[w2+k+3]*x3;
                sum3 += (int32_t)w[w3+k+0]*x0 + (int32_t)w[w3+k+1]*x1 + (int32_t)w[w3+k+2]*x2 + (int32_t)w[w3+k+3]*x3;
                sum4 += (int32_t)w[w4+k+0]*x0 + (int32_t)w[w4+k+1]*x1 + (int32_t)w[w4+k+2]*x2 + (int32_t)w[w4+k+3]*x3;
                sum5 += (int32_t)w[w5+k+0]*x0 + (int32_t)w[w5+k+1]*x1 + (int32_t)w[w5+k+2]*x2 + (int32_t)w[w5+k+3]*x3;
                sum6 += (int32_t)w[w6+k+0]*x0 + (int32_t)w[w6+k+1]*x1 + (int32_t)w[w6+k+2]*x2 + (int32_t)w[w6+k+3]*x3;
                sum7 += (int32_t)w[w7+k+0]*x0 + (int32_t)w[w7+k+1]*x1 + (int32_t)w[w7+k+2]*x2 + (int32_t)w[w7+k+3]*x3;
            }
            for (;k < input_size; k++) {
                int32_t xk=(int32_t)x[k];
                sum0 += (int32_t)w[w0+k]*xk;
                sum1 += (int32_t)w[w1+k]*xk;
                sum2 += (int32_t)w[w2+k]*xk;
                sum3 += (int32_t)w[w3+k]*xk;
                sum4 += (int32_t)w[w4+k]*xk;
                sum5 += (int32_t)w[w5+k]*xk;
                sum6 += (int32_t)w[w6+k]*xk;
                sum7 += (int32_t)w[w7+k]*xk;
            }
            if (y_data_int8) {
                y_data_int8[j+0] = ti_requant(sum0, rscale, relu);
                y_data_int8[j+1] = ti_requant(sum1, rscale, relu);
                y_data_int8[j+2] = ti_requant(sum2, rscale, relu);
                y_data_int8[j+3] = ti_requant(sum3, rscale, relu);
                y_data_int8[j+4] = ti_requant(sum4, rscale, relu);
                y_data_int8[j+5] = ti_requant(sum5, rscale, relu);
                y_data_int8[j+6] = ti_requant(sum6, rscale, relu);
                y_data_int8[j+7] = ti_requant(sum7, rscale, relu);
            }
            else {
                y_data_int32[j+0] = sum0;
                y_data_int32[j+1] = sum1;
                y_data_int32[j+2] = sum2;
                y_data_int32[j+3] = sum3;
                y_data_int32[j+4] = sum4;
                y_data_int32[j+5] = sum5;
                y_data_int32[j+6] = sum6;
                y_data_int32[j+7] = sum7;
            }
        }
        for (; j < output_size; j++) {
            int32_t sum = b[j];
            uint32_t w0 = j*input_size;
            for (uint32_t k = 0; k < input_size; k++) {
                sum += (int32_t)w[w0+k]*x[k];
            }
            if (y_data_int8) {
                y_data_int8[j]=ti_requant(sum, rscale, relu);
            }
            else {
                y_data_int32[j]=sum;
            }


        }
    }
return 0;
}

int ti_dense_forward(const ti_layer_t *layer, ti_tensor_t *input, ti_tensor_t *output) {
    if (layer == NULL || input == NULL || output == NULL) return -1;
    if (layer->dtype == TI_FLOAT32)
        return ti_dense_forward_float32(layer, input, output);
    if (layer->dtype == TI_INT8)
        return ti_dense_forward_int8(layer, input, output);
    if (layer->dtype == TI_INT32)
        return -1;
    return -1;
}