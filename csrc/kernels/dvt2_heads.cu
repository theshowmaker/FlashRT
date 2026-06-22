#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float sums[kBlock];
    sums[threadIdx.x] = v;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return sums[0];
}

__global__ void dvt2_pool_two_slices_kernel(
    const __half* __restrict__ enc,
    float* __restrict__ base_pool,
    float* __restrict__ prompt_pool,
    unsigned int* __restrict__ nonfinite,
    int de,
    int base_start,
    int base_rows,
    int prompt_start,
    int prompt_rows) {
    int d = static_cast<int>(blockIdx.x);
    int which = static_cast<int>(blockIdx.y);
    if (d >= de) {
        return;
    }

    int row_start = which == 0 ? base_start : prompt_start;
    int rows = which == 0 ? base_rows : prompt_rows;
    float local_sum = 0.0f;
    unsigned int local_bad = 0;
    for (int r = static_cast<int>(threadIdx.x); r < rows; r += kBlock) {
        float v = __half2float(enc[(row_start + r) * de + d]);
        if (isfinite(v)) {
            local_sum += v;
        } else {
            local_bad += 1;
        }
    }

    __shared__ float sums[kBlock];
    __shared__ unsigned int bad[kBlock];
    sums[threadIdx.x] = local_sum;
    bad[threadIdx.x] = local_bad;
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            bad[threadIdx.x] += bad[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = rows > 0 ? (sums[0] / static_cast<float>(rows)) : 0.0f;
        if (which == 0) {
            base_pool[d] = mean;
        } else {
            prompt_pool[d] = mean;
        }
        if (bad[0] != 0) {
            atomicAdd(nonfinite, bad[0]);
        }
    }
}

__global__ void dvt2_stage_hidden_kernel(
    const float* __restrict__ prompt_pool,
    const float* __restrict__ stage_task_embed,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ hidden,
    int de,
    int task_category,
    int task_dim,
    int hidden_dim) {
    int h = static_cast<int>(blockIdx.x);
    if (h >= hidden_dim) {
        return;
    }
    const float* task = stage_task_embed + task_category * task_dim;
    int input_dim = de + task_dim;
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < input_dim; i += kBlock) {
        float v = i < de ? prompt_pool[i] : task[i - de];
        local += v * w[i * hidden_dim + h];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        hidden[h] = gelu_exact(sum + b[h]);
    }
}

__global__ void dvt2_stage_logits_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ logits,
    int hidden_dim,
    int num_logits,
    int valid_logits) {
    int n = static_cast<int>(blockIdx.x);
    if (n >= num_logits) {
        return;
    }
    if (n >= valid_logits) {
        if (threadIdx.x == 0) {
            logits[n] = -CUDART_INF_F;
        }
        return;
    }
    float local = 0.0f;
    for (int h = static_cast<int>(threadIdx.x); h < hidden_dim; h += kBlock) {
        local += hidden[h] * w[h * num_logits + n];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        logits[n] = sum + b[n];
    }
}

__global__ void dvt2_exist_hidden_kernel(
    const float* __restrict__ base_pool,
    const float* __restrict__ prompt_pool,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ hidden,
    int de,
    int hidden_dim) {
    int h = static_cast<int>(blockIdx.x);
    if (h >= hidden_dim) {
        return;
    }
    int input_dim = de * 2;
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < input_dim; i += kBlock) {
        float v = i < de ? base_pool[i] : prompt_pool[i - de];
        local += v * w[i * hidden_dim + h];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        hidden[h] = gelu_exact(sum + b[h]);
    }
}

__global__ void dvt2_exist_output_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const unsigned int* __restrict__ nonfinite,
    float* __restrict__ result_out,
    int hidden_dim,
    int num_logits) {
    float local = 0.0f;
    for (int h = static_cast<int>(threadIdx.x); h < hidden_dim; h += kBlock) {
        local += hidden[h] * w[h];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        float prob = 1.0f / (1.0f + expf(-(sum + b[0])));
        result_out[num_logits] = prob;
        result_out[num_logits + 1] = prob > 0.5f ? 1.0f : 0.0f;
        result_out[num_logits + 2] = static_cast<float>(*nonfinite);
    }
}

__device__ __forceinline__ float dvt2_all_input_value(
    int idx,
    const float* __restrict__ task_mean,
    const float* __restrict__ stage_enc,
    const float* __restrict__ task_stage,
    int de,
    int sub_dim) {
    if (idx < de) {
        return task_mean[idx];
    }
    idx -= de;
    if (idx < sub_dim) {
        return stage_enc[idx];
    }
    return task_stage[idx - sub_dim];
}

__global__ void dvt2_prompt_mean_kernel(
    const __half* __restrict__ lang_emb,
    float* __restrict__ task_mean,
    int de,
    int prompt_len) {
    int d = static_cast<int>(blockIdx.x);
    if (d >= de) {
        return;
    }
    float local_sum = 0.0f;
    for (int r = static_cast<int>(threadIdx.x); r < prompt_len; r += kBlock) {
        float v = __half2float(lang_emb[r * de + d]);
        local_sum += isfinite(v) ? v : 0.0f;
    }
    float sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        task_mean[d] = prompt_len > 0 ? sum / static_cast<float>(prompt_len) : 0.0f;
    }
}

__global__ void dvt2_all_input_linear_kernel(
    const float* __restrict__ task_mean,
    const float* __restrict__ stage_enc,
    const float* __restrict__ task_stage,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int de,
    int sub_dim,
    int input_dim,
    int out_dim,
    int activation) {
    int o = static_cast<int>(blockIdx.x);
    if (o >= out_dim) {
        return;
    }
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < input_dim; i += kBlock) {
        float v = dvt2_all_input_value(i, task_mean, stage_enc, task_stage, de, sub_dim);
        local += v * w[i * out_dim + o];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        sum += b[o];
        if (activation == 1) {
            sum = 1.0f / (1.0f + expf(-sum));
        } else if (activation == 2) {
            sum = fmaxf(sum, 0.0f);
        }
        out[o] = sum;
    }
}

__global__ void dvt2_task_gated_kernel(
    __half* __restrict__ lang_emb,
    const float* __restrict__ task_mean,
    const float* __restrict__ gate_task,
    int de,
    int fusion_start) {
    int d = static_cast<int>(blockIdx.x);
    if (d < de) {
        lang_emb[fusion_start * de + d] = __float2half(task_mean[d] * gate_task[d]);
    }
}

__global__ void dvt2_fusion_layer2_kernel(
    __half* __restrict__ lang_emb,
    const float* __restrict__ hidden,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int de,
    int hidden_dim,
    int fusion_start) {
    int d = static_cast<int>(blockIdx.x);
    if (d >= de) {
        return;
    }
    float local = 0.0f;
    for (int h = static_cast<int>(threadIdx.x); h < hidden_dim; h += kBlock) {
        local += hidden[h] * w[h * de + d];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        lang_emb[(fusion_start + 1) * de + d] = __float2half(sum + b[d]);
    }
}

__global__ void dvt2_stage_dominant_kernel(
    __half* __restrict__ lang_emb,
    const float* __restrict__ stage_enc,
    const float* __restrict__ task_stage,
    const float* __restrict__ gate_sincos,
    const float* __restrict__ gate_task_stage,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int de,
    int sub_dim,
    int fusion_start) {
    int d = static_cast<int>(blockIdx.x);
    if (d >= de) {
        return;
    }
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < sub_dim; i += kBlock) {
        local += stage_enc[i] * gate_sincos[i] * w[i * de + d];
        local += task_stage[i] * gate_task_stage[i] * w[(sub_dim + i) * de + d];
    }
    float sum = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        lang_emb[(fusion_start + 2) * de + d] = __float2half(sum + b[d]);
    }
}

__global__ void dvt2_pure_stage_kernel(
    __half* __restrict__ lang_emb,
    const float* __restrict__ stage_enc,
    const float* __restrict__ task_stage,
    int de,
    int sub_dim,
    int fusion_start) {
    int d = static_cast<int>(blockIdx.x);
    if (d >= de) {
        return;
    }
    float v = d < sub_dim ? stage_enc[d] : task_stage[d - sub_dim];
    lang_emb[(fusion_start + 3) * de + d] = __float2half(v);
}

}  // namespace

extern "C" int dvt2_heads_fp16_f32_launch(
    const void* enc_final_fp16,
    const float* stage_task_embed,
    const float* stage_mlp_1_w,
    const float* stage_mlp_1_b,
    const float* stage_mlp_2_w,
    const float* stage_mlp_2_b,
    const float* exist_mlp_1_w,
    const float* exist_mlp_1_b,
    const float* exist_mlp_2_w,
    const float* exist_mlp_2_b,
    float* base_pool,
    float* prompt_pool,
    float* stage_hidden,
    float* exist_hidden,
    float* result_out,
    unsigned int* nonfinite_out,
    int de,
    int base_start,
    int base_rows,
    int prompt_start,
    int prompt_rows,
    int task_category,
    int task_dim,
    int stage_hidden_dim,
    int stage_num_logits,
    int valid_stage_logits,
    int exist_hidden_dim,
    cudaStream_t stream) {
    if (de <= 0 || base_rows <= 0 || prompt_rows <= 0 ||
        task_dim <= 0 || stage_hidden_dim <= 0 ||
        stage_num_logits <= 0 || exist_hidden_dim <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaMemsetAsync(nonfinite_out, 0, sizeof(unsigned int), stream);

    dim3 pool_grid(static_cast<unsigned int>(de), 2u, 1u);
    dvt2_pool_two_slices_kernel<<<pool_grid, kBlock, 0, stream>>>(
        static_cast<const __half*>(enc_final_fp16),
        base_pool,
        prompt_pool,
        nonfinite_out,
        de,
        base_start,
        base_rows,
        prompt_start,
        prompt_rows);

    dvt2_stage_hidden_kernel<<<stage_hidden_dim, kBlock, 0, stream>>>(
        prompt_pool,
        stage_task_embed,
        stage_mlp_1_w,
        stage_mlp_1_b,
        stage_hidden,
        de,
        task_category,
        task_dim,
        stage_hidden_dim);

    dvt2_stage_logits_kernel<<<stage_num_logits, kBlock, 0, stream>>>(
        stage_hidden,
        stage_mlp_2_w,
        stage_mlp_2_b,
        result_out,
        stage_hidden_dim,
        stage_num_logits,
        valid_stage_logits);

    dvt2_exist_hidden_kernel<<<exist_hidden_dim, kBlock, 0, stream>>>(
        base_pool,
        prompt_pool,
        exist_mlp_1_w,
        exist_mlp_1_b,
        exist_hidden,
        de,
        exist_hidden_dim);

    dvt2_exist_output_kernel<<<1, kBlock, 0, stream>>>(
        exist_hidden,
        exist_mlp_2_w,
        exist_mlp_2_b,
        nonfinite_out,
        result_out,
        exist_hidden_dim,
        stage_num_logits);

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dvt2_fusion_tokens_fp16_f32_launch(
    void* lang_emb_fp16,
    const float* stage_class_embeddings,
    const float* task_stage_embeddings,
    const float* gate_sincos_w,
    const float* gate_sincos_b,
    const float* gate_task_stage_w,
    const float* gate_task_stage_b,
    const float* gate_task_w,
    const float* gate_task_b,
    const float* fusion_layer1_w,
    const float* fusion_layer1_b,
    const float* fusion_layer2_w,
    const float* fusion_layer2_b,
    const float* stage_projection_w,
    const float* stage_projection_b,
    float* task_mean,
    float* gate_sincos,
    float* gate_task_stage,
    float* gate_task,
    float* fusion_hidden,
    int de,
    int prompt_len,
    int prompt_capacity,
    int fusion_start,
    int class_state,
    int task_stage_idx,
    int sub_dim,
    int fusion_hidden_dim,
    cudaStream_t stream) {
    if (de <= 0 || prompt_len <= 0 || prompt_capacity <= 0 ||
        fusion_start < 0 || sub_dim <= 0 || fusion_hidden_dim <= 0 ||
        de != sub_dim * 2) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    auto* lang = static_cast<__half*>(lang_emb_fp16);
    const float* stage_enc = stage_class_embeddings + class_state * sub_dim;
    const float* task_stage = task_stage_embeddings + task_stage_idx * sub_dim;
    int input_dim = de + sub_dim * 2;

    dvt2_prompt_mean_kernel<<<de, kBlock, 0, stream>>>(
        lang, task_mean, de, prompt_len);
    dvt2_all_input_linear_kernel<<<sub_dim, kBlock, 0, stream>>>(
        task_mean, stage_enc, task_stage, gate_sincos_w, gate_sincos_b,
        gate_sincos, de, sub_dim, input_dim, sub_dim, 1);
    dvt2_all_input_linear_kernel<<<sub_dim, kBlock, 0, stream>>>(
        task_mean, stage_enc, task_stage, gate_task_stage_w, gate_task_stage_b,
        gate_task_stage, de, sub_dim, input_dim, sub_dim, 1);
    dvt2_all_input_linear_kernel<<<de, kBlock, 0, stream>>>(
        task_mean, stage_enc, task_stage, gate_task_w, gate_task_b,
        gate_task, de, sub_dim, input_dim, de, 1);
    dvt2_all_input_linear_kernel<<<fusion_hidden_dim, kBlock, 0, stream>>>(
        task_mean, stage_enc, task_stage, fusion_layer1_w, fusion_layer1_b,
        fusion_hidden, de, sub_dim, input_dim, fusion_hidden_dim, 2);

    dvt2_task_gated_kernel<<<de, 1, 0, stream>>>(
        lang, task_mean, gate_task, de, fusion_start);
    dvt2_fusion_layer2_kernel<<<de, kBlock, 0, stream>>>(
        lang, fusion_hidden, fusion_layer2_w, fusion_layer2_b,
        de, fusion_hidden_dim, fusion_start);
    dvt2_stage_dominant_kernel<<<de, kBlock, 0, stream>>>(
        lang, stage_enc, task_stage, gate_sincos, gate_task_stage,
        stage_projection_w, stage_projection_b, de, sub_dim, fusion_start);
    dvt2_pure_stage_kernel<<<de, 1, 0, stream>>>(
        lang, stage_enc, task_stage, de, sub_dim, fusion_start);

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int>(err);
}
