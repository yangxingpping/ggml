
#include "ggml.h"
#include "ggml-cuda.h"
#include "spdlog/spdlog.h"

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void* wdata;

    struct ggml_threadpool* threadpool;
};

extern void ggml_compute_forward(struct ggml_compute_params* params, struct ggml_tensor* tensor);

int main(void)
{
    struct ggml_init_params params;
    params.mem_size = 1024 * 1024;
    params.mem_buffer = NULL;
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);

    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;

    // A = [[1,2,3],[4,5,6]]
    for (int i = 0; i < 6; i++) a_data[i] = i + 1;

    // B = [[7,8],[9,10],[11,12]]
    for (int i = 0; i < 6; i++) b_data[i] = i + 7;

    struct ggml_tensor* C = ggml_mul_mat(ctx, A, B);

    struct ggml_compute_params cparams;
    cparams.ith = 0;
    cparams.nth = 1;

    ggml_compute_forward(&cparams, C);

    float* c_data = (float*)C->data;


    ggml_free(ctx);
    SPDLOG_INFO("hello,world");
    return 0;
}
