#define PANGULU_PLATFORM_ENV
#include "../../../../pangulu_common.h"
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUDA_LAST_ERROR() check_cuda_last_error(__FILE__, __LINE__)

#define PANGULU_GPU_OPDST 0
#define PANGULU_GPU_OP1 1
#define PANGULU_GPU_OP2 2

#define PANGULU_SSSSM_BATCHED_THREADPERBLOCK 256
// #define PANGULU_SSSSM_BATCHED_SHAREDMEM_LEN 256
#define PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM 1
#define PANGULU_SSSSM_DATAMOV_THREADPERBLOCK 128
#define PANGULU_GETRF_THREAD_PER_BLOCK 256
#define PANGULU_TSTRF_SHARED_MEM_LEN 1024
#define PANGULU_GESSM_SHARED_MEM_LEN 1024

#define TROJAN_HORSE_THREAD_PER_BLOCK 128
#define TROJAN_HORSE_SHARED_MEM_LEN 1024

#define PANGULU_WARP_SIZE 32

// pangulu_int32_t **hd_getrf_nnzu = NULL;

FILE *result_file = NULL;
char mtx_name_glo[100];

// pangulu_inblock_ptr* d_opdst_colptr = NULL;
// pangulu_inblock_idx* d_opdst_rowidx = NULL;
// calculate_type* d_opdst_value = NULL;
// pangulu_inblock_ptr* d_op1_colptr = NULL;
// pangulu_inblock_idx* d_op1_rowidx = NULL;
// calculate_type* d_op1_value = NULL;
// pangulu_inblock_ptr* d_op2_colptr = NULL;
// pangulu_inblock_idx* d_op2_rowidx = NULL;
// calculate_type* d_op2_value = NULL;

// pangulu_inblock_ptr* h_csr0_rowptr = NULL;
// pangulu_inblock_idx* h_csr0_colidx = NULL;
// pangulu_inblock_ptr* h_csr0_valueidx = NULL;
// pangulu_inblock_ptr* h_trans_aidptr = NULL;
// pangulu_inblock_ptr* h_csr1_rowptr = NULL;
// pangulu_inblock_idx* h_csr1_colidx = NULL;
// pangulu_inblock_ptr* h_csr1_valueidx = NULL;

// pangulu_inblock_ptr* d_csr0_rowptr = NULL;
// pangulu_inblock_idx* d_csr0_colidx = NULL;
// pangulu_inblock_ptr* d_csr0_valueidx = NULL;

calculate_type *getrf_dense_buf_d;

void check_cuda_last_error(const char *const file, int const line)
{
    cudaError_t result = cudaGetLastError();
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void pangulu_platform_0201000_malloc(void **platform_address, size_t size)
{
    cudaError_t err = cudaMalloc(platform_address, size);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memset(void *s, int c, size_t n)
{
    cudaError_t err = cudaMemset(s, c, n);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_create_stream(void **stream)
{
    cudaError_t err = cudaStreamCreate((cudaStream_t *)stream);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy(void *dst, const void *src, size_t count, unsigned int kind)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    }
    else if (kind == 1)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    }
    else if (kind == 2)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    // CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void *stream)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    }
    else if (kind == 1)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    }
    else if (kind == 2)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_free(void *devptr)
{
    cudaError_t err = cudaFree(devptr);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_get_device_num(int *device_num)
{
    cudaError_t err = cudaGetDeviceCount(device_num);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_set_default_device(int device_num)
{
    cudaError_t err = cudaSetDevice(device_num);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_get_device_name(char *name, int device_num)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_num);
    CHECK_CUDA_ERROR(err);
    strcpy(name, prop.name);
}

#ifdef PANGULU_NONSHAREDMEM

__device__ pangulu_inblock_ptr
binarysearch_inblk_cuda(
    pangulu_inblock_idx *ridx,
    pangulu_int32_t left,
    pangulu_int32_t right,
    pangulu_inblock_idx target)
{
    pangulu_int32_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (ridx[mid] == target)
        {
            return mid;
        }
        else if (ridx[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return 0xffffffff;
}

// void pangulu_cuda_upload_block(
//     pangulu_inblock_idx nb,
//     pangulu_uint16_t dst,
//     pangulu_storage_slot_t* src
// ){
//     switch(dst){
//         case PANGULU_GPU_OPDST:
//             cudaMemcpy(d_opdst_colptr, src->columnpointer, (nb + 1) * sizeof(pangulu_inblock_ptr), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_opdst_rowidx, src->rowindex, src->columnpointer[nb] * sizeof(pangulu_inblock_idx), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_opdst_value, src->value, src->columnpointer[nb] * sizeof(calculate_type), cudaMemcpyHostToDevice);
//             break;
//         case PANGULU_GPU_OP1:
//             cudaMemcpy(d_op1_colptr, src->columnpointer, (nb + 1) * sizeof(pangulu_inblock_ptr), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_op1_rowidx, src->rowindex, src->columnpointer[nb] * sizeof(pangulu_inblock_idx), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_op1_value, src->value, src->columnpointer[nb] * sizeof(calculate_type), cudaMemcpyHostToDevice);
//             break;
//         case PANGULU_GPU_OP2:
//             cudaMemcpy(d_op2_colptr, src->columnpointer, (nb + 1) * sizeof(pangulu_inblock_ptr), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_op2_rowidx, src->rowindex, src->columnpointer[nb] * sizeof(pangulu_inblock_idx), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_op2_value, src->value, src->columnpointer[nb] * sizeof(calculate_type), cudaMemcpyHostToDevice);
//             break;
//     }
// }

void pangulu_cuda_download_block(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *slot)
{
    pangulu_platform_0201000_memcpy(slot->value, slot->d_value, sizeof(calculate_type) * slot->columnpointer[nb], 1);
}

// __global__ void
// pangulu_cuda_transpose_onlyval(
//     const pangulu_inblock_idx nb,
//     const pangulu_inblock_ptr* in_ptr,
//     const pangulu_inblock_idx* in_idx,
//     const calculate_type* in_val,
//     const pangulu_inblock_ptr* out_ptr,
//     pangulu_inblock_idx* out_idx,
//     calculate_type* out_val
// ){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx >= in_ptr[nb]){return;}
//     pangulu_int16_t left = 0, right = nb;
//     pangulu_int16_t row = 0;
//     while(right - left > 1){
//         row = left + (right - left) / 2;
//         if(in_ptr[row] > idx){
//             right = row;
//         }else{
//             left = row;
//         }
//     }
//     row = left;
//     pangulu_inblock_ptr target_idx = binarysearch_inblk_cuda(out_idx, out_ptr[in_idx[idx]], out_ptr[in_idx[idx]+1] - 1, row);
//     out_val[target_idx] = in_val[idx];
// }

__global__ void getrf_get_nnzu(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *opdst_colptr,
    pangulu_inblock_idx *opdst_rowidx,
    pangulu_int32_t *getrf_nnzu)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nb)
    {
        return;
    }
    // if(tid == 0){
    opdst_colptr[0] = 0;
    // }
    getrf_nnzu[tid] = binarysearch_inblk_cuda(opdst_rowidx, opdst_colptr[tid], opdst_colptr[tid + 1] - 1, tid) - opdst_colptr[tid];
    if (getrf_nnzu[tid] == 0xffffffff)
    {
        printf("[PanguLU Kernel Error] No diagonal value : (%d, %d)\n", tid, tid);
    }
}

__global__ void getrf_cuda(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *opdst_colptr,
    pangulu_inblock_idx *opdst_rowidx,
    calculate_type *opdst_value,
    pangulu_int32_t *getrf_nnzu)
{
    opdst_colptr[0] = 0;
    const pangulu_inblock_idx col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    __shared__ int s_nnzu[1];
    pangulu_inblock_ptr idx_a_lb = opdst_colptr[col];
    pangulu_inblock_ptr idx_a_ub = opdst_colptr[col + 1];
    pangulu_inblock_ptr smemlen = 256;
    pangulu_inblock_ptr collen_a = idx_a_ub - idx_a_lb;

    if (collen_a == 0)
    {
        return;
    }

    // use shared memory
    if (collen_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxa[256];
        __shared__ calculate_type s_valuea[256];

        for (pangulu_inblock_ptr i = threadIdx.x; i < collen_a; i += blockDim.x)
        {
            s_rowidxa[i] = opdst_rowidx[i + idx_a_lb];
            s_valuea[i] = opdst_value[i + idx_a_lb];
        }

        if (!threadIdx.x)
        {
            s_nnzu[0] = getrf_nnzu[col];
        }
        __syncthreads();

        // step one
        pangulu_inblock_ptr idx_a_diag = binarysearch_inblk_cuda(opdst_rowidx, idx_a_lb, idx_a_ub - 1, col);
        for (pangulu_inblock_ptr idx_a = idx_a_lb; idx_a < idx_a_diag; idx_a++)
        {
            const pangulu_inblock_idx rowidx = opdst_rowidx[idx_a];
            calculate_type value3 = s_valuea[idx_a - idx_a_lb];

            pangulu_inblock_ptr idx_rowa_lb = opdst_colptr[rowidx];
            pangulu_inblock_ptr idx_rowa_ub = opdst_colptr[rowidx + 1];
            pangulu_inblock_ptr idx_rowa_diag = binarysearch_inblk_cuda(opdst_rowidx, idx_rowa_lb, idx_rowa_ub - 1, rowidx);

            // busy-wait until nnzu[rowidx]==0
            // if (!threadIdx.x)
            // {
            do
            {
                __threadfence();
                // __threadfence_block();
            } while (getrf_nnzu[rowidx] != -1);
            // }
            __syncthreads();

            for (pangulu_inblock_ptr j = idx_rowa_diag + 1 + threadIdx.x; j < idx_rowa_ub; j += blockDim.x)
            {
                const pangulu_inblock_idx lrowindex = opdst_rowidx[j];
                const pangulu_inblock_idx thecolidx = rowidx;

                pangulu_inblock_ptr flag1 = binarysearch_inblk_cuda(s_rowidxa, 0, collen_a - 1, lrowindex);
                pangulu_inblock_ptr flag2 = binarysearch_inblk_cuda(opdst_rowidx, opdst_colptr[thecolidx], opdst_colptr[thecolidx + 1] - 1, lrowindex);
                // pangulu_inblock_ptr flag2 = binarysearch_inblk_cuda(opdst_rowidx, idx_rowa_lb, idx_rowa_ub - 1, lrowindex);
                // s_valuea[flag1] -= opdst_value[flag2] * value3;
                s_valuea[flag1] -= opdst_value[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        // if (fabs(s_valuea[idx_a_diag - idx_a_lb]) < PANGULU_TOL)
        // {
        //     s_valuea[idx_a_diag - idx_a_lb] = PANGULU_TOL;
        // }
        calculate_type diag_val = s_valuea[idx_a_diag - idx_a_lb];

        for (pangulu_int64_t i = idx_a_diag + 1 + threadIdx.x; i < idx_a_ub; i += blockDim.x)
        {
            s_valuea[i - idx_a_lb] = s_valuea[i - idx_a_lb] / diag_val;
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < collen_a; i += blockDim.x)
        {
            opdst_value[i + idx_a_lb] = s_valuea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            getrf_nnzu[col] = s_nnzu[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzu[0] = getrf_nnzu[col];
        }
        __syncthreads();

        // step one
        pangulu_inblock_ptr idx_a_diag = binarysearch_inblk_cuda(opdst_rowidx, idx_a_lb, idx_a_ub - 1, col);
        for (pangulu_inblock_ptr idx_a = idx_a_lb; idx_a < idx_a_diag; idx_a++)
        {
            const pangulu_inblock_idx rowidx = opdst_rowidx[idx_a];
            calculate_type value3 = opdst_value[idx_a];

            pangulu_inblock_ptr idx_rowa_lb = opdst_colptr[rowidx];
            pangulu_inblock_ptr idx_rowa_ub = opdst_colptr[rowidx + 1];
            pangulu_inblock_ptr idx_rowa_diag = binarysearch_inblk_cuda(opdst_rowidx, idx_rowa_lb, idx_rowa_ub - 1, rowidx);

            // busy-wait until nnzu[rowidx]==0
            // if (!threadIdx.x)
            // {
            do
            {
                // __threadfence_block();
                __threadfence();
            } while (getrf_nnzu[rowidx] != -1);
            // }
            __syncthreads();

            for (pangulu_inblock_ptr j = idx_rowa_diag + 1 + threadIdx.x; j < idx_rowa_ub; j += blockDim.x)
            {
                const pangulu_inblock_idx lrowindex = opdst_rowidx[j];
                const pangulu_inblock_idx thecolidx = rowidx;

                pangulu_inblock_ptr flag1 = binarysearch_inblk_cuda(opdst_rowidx, idx_a_lb, idx_a_ub - 1, lrowindex);
                pangulu_inblock_ptr flag2 = binarysearch_inblk_cuda(opdst_rowidx, opdst_colptr[thecolidx], opdst_colptr[thecolidx + 1] - 1, lrowindex);

                opdst_value[flag1] -= opdst_value[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        // if (fabs(opdst_value[idx_a_diag]) < PANGULU_TOL)
        // {
        //     opdst_value[idx_a_diag] = PANGULU_TOL;
        // }
        calculate_type diag_val = opdst_value[idx_a_diag];

        for (pangulu_inblock_ptr i = idx_a_diag + 1 + threadIdx.x; i < idx_a_ub; i += blockDim.x)
        {
            opdst_value[i] = opdst_value[i] / diag_val;
        }
        __syncthreads();

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            getrf_nnzu[col] = s_nnzu[0];
        }
    }
}

// __global__ void tstrf_cuda_block(
//     pangulu_inblock_idx n,
//     pangulu_inblock_ptr *a_columnpointer,
//     pangulu_inblock_idx *a_rowindex,
//     pangulu_inblock_ptr *a_valueidx,
//     calculate_type *a_value,
//     pangulu_inblock_ptr *l_columnpointer,
//     pangulu_inblock_idx *l_rowindex,
//     pangulu_inblock_ptr *l_valueidx,
//     calculate_type *l_value)
// {
// #define PANGULU_SHARED_MEM_LEN 1024
//     pangulu_inblock_idx colidx = blockIdx.x;
//     if (colidx >= n)
//     {
//         return;
//     }
//     // a_columnpointer[0] = 0;
//     // l_columnpointer[0] = 0;
//     pangulu_inblock_ptr cola1 = (colidx == 0) ? 0 : a_columnpointer[colidx];
//     pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
//     if (cola2 == cola1)
//     {
//         return;
//     }

//     if (cola2 - cola1 >= PANGULU_SHARED_MEM_LEN)
//     {
//         for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
//         {
//             pangulu_int64_t rowa = a_rowindex[i];
//             pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
//             pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
//             calculate_type vala = a_value[a_valueidx[i]];
//             vala /= l_value[l_valueidx[coll1]];
//             if (threadIdx.x == 0)
//             {
//                 a_value[a_valueidx[i]] = vala;
//             }
//             for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
//             {
//                 // update a's value;
//                 pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - 1, l_rowindex[j]);
//                 // pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1, cola2 - 1, l_rowindex[j]);
//                 if (f != 0xffffffff)
//                 {
//                     a_value[a_valueidx[f]] -= vala * l_value[l_valueidx[j]];
//                 }
//             }
//             __syncthreads();
//         }
//     }
//     else
//     {
//         __shared__ pangulu_inblock_idx s_idxa[PANGULU_SHARED_MEM_LEN];
//         __shared__ calculate_type s_vala[PANGULU_SHARED_MEM_LEN];

//         for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
//         {
//             s_idxa[i] = a_rowindex[cola1 + i];
//             s_vala[i] = a_value[a_valueidx[cola1 + i]];
//         }
//         __syncthreads();

//         for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
//         {
//             pangulu_int64_t rowa = s_idxa[t];
//             pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
//             pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

//             // calculate_type vala = s_vala[t];
//             // vala /= l_value[l_valueidx[coll1]];
//             // if(threadIdx.x == 0){
//             //     s_vala[t] = vala;
//             // }

//             calculate_type vala;
//             if ((threadIdx.x / 32) == 0)
//             {
//                 vala = s_vala[t];
//                 vala /= l_value[l_valueidx[coll1]];
//                 s_vala[t] = vala;
//                 __syncthreads();
//             }
//             else
//             {
//                 __syncthreads();
//                 __threadfence_block();
//                 vala = s_vala[t];
//             }

//             for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
//             {
//                 // update a's value;
//                 pangulu_inblock_ptr f = binarysearch_inblk_cuda(s_idxa, 1 + t + p, cola2 - cola1 - coll2 + j, l_rowindex[j]);
//                 if (f != 0xffffffff)
//                 {
//                     s_vala[f] -= vala * l_value[l_valueidx[j]];
//                 }
//             }
//             __syncthreads();
//         }

//         for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
//         {
//             a_value[a_valueidx[cola1 + i]] = s_vala[i];
//         }
//     }
// #undef PANGULU_SHARED_MEM_LEN
// }


__global__ void tstrf_cuda_block(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *a_columnpointer,
    pangulu_inblock_idx *a_rowindex,
    pangulu_inblock_ptr *a_valueidx,
    calculate_type *a_value,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    pangulu_inblock_ptr *l_valueidx,
    calculate_type *l_value)
{
#define PANGULU_SHARED_MEM_LEN 1024
    pangulu_inblock_idx colidx = blockIdx.x;
    if (colidx >= n)
    {
        return;
    }
    // a_columnpointer[0] = 0;
    // l_columnpointer[0] = 0;
    pangulu_inblock_ptr cola1 = (colidx == 0) ? 0 : a_columnpointer[colidx];
    pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
    if (cola2 == cola1)
    {
        return;
    }

    if (n > PANGULU_SHARED_MEM_LEN)
    {
        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = a_rowindex[i];
            pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
            calculate_type vala = a_value[a_valueidx[i]];
            vala /= l_value[l_valueidx[coll1]];
            if (threadIdx.x == 0)
            {
                a_value[a_valueidx[i]] = vala;
            }
            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - 1, l_rowindex[j]);
                // pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1, cola2 - 1, l_rowindex[j]);
                if (f != 0xffffffff)
                {
                    a_value[a_valueidx[f]] -= vala * l_value[l_valueidx[j]];
                }
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_idxa[PANGULU_SHARED_MEM_LEN];
        __shared__ calculate_type s_dense[PANGULU_SHARED_MEM_LEN];

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_dense[s_idxa[i]] = a_value[a_valueidx[cola1 + i]];
        }
        __syncthreads();

        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = s_idxa[t];
            pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

            calculate_type vala;
            if ((threadIdx.x / 32) == 0)
            {
                vala = s_dense[s_idxa[t]];
                vala /= l_value[l_valueidx[coll1]];
                s_dense[s_idxa[t]] = vala;
                __syncthreads();
            }
            else
            {
                __syncthreads();
                __threadfence_block();
                vala = s_dense[s_idxa[t]];
            }

            // s_dense[colidx] = s_dense[colidx] / l_value[l_valueidx[coll1]];
            // calculate_type vala = s_dense[colidx];

            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                s_dense[l_rowindex[j]] -= vala * l_value[l_valueidx[j]];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[a_valueidx[cola1 + i]] = s_dense[s_idxa[i]];
        }
    }
#undef PANGULU_SHARED_MEM_LEN
}



__global__ void tstrf_cuda_warp(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *a_columnpointer,
    pangulu_inblock_idx *a_rowindex,
    pangulu_inblock_ptr *a_valueidx,
    calculate_type *a_value,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    pangulu_inblock_ptr *l_valueidx,
    calculate_type *l_value)
{
#define PANGULU_SHARED_MEM_LEN 64
#define PANGULU_WARP_PER_BLOCK 2
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int warpid = tid / PANGULU_WARP_SIZE;
    const int warp_tid = tid % PANGULU_WARP_SIZE;

    pangulu_inblock_idx colidx = warpid;
    if (colidx >= n)
    {
        return;
    }
    // a_columnpointer[0] = 0;
    // l_columnpointer[0] = 0;
    pangulu_inblock_ptr cola1 = (colidx == 0) ? 0 : a_columnpointer[colidx];
    pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
    if (cola2 == cola1)
    {
        return;
    }

    if (cola2 - cola1 >= PANGULU_SHARED_MEM_LEN)
    {
        for (int i = cola1, t = 0; i < cola2; i++, t++)
        {
            int rowa = a_rowindex[i];
            int coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
            int coll2 = l_columnpointer[rowa + 1];
            calculate_type vala = a_value[a_valueidx[i]];
            vala /= l_value[l_valueidx[coll1]];
            // if (warp_tid == 0)
            // {
                a_value[a_valueidx[i]] = vala;
            // }
            for (int j = coll1 + 1 + warp_tid, p = warp_tid; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
            {
                // update a's value;
                pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - coll2 + j, l_rowindex[j]);
                if (f != 0xffffffff)
                {
                    a_value[a_valueidx[f]] -= vala * l_value[l_valueidx[j]];
                }
            }
            // __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_idxa_block[PANGULU_SHARED_MEM_LEN*PANGULU_WARP_PER_BLOCK];
        __shared__ calculate_type s_vala_block[PANGULU_SHARED_MEM_LEN*PANGULU_WARP_PER_BLOCK];
        pangulu_inblock_idx* s_idxa = s_idxa_block + (warpid % PANGULU_WARP_PER_BLOCK) * PANGULU_SHARED_MEM_LEN;
        calculate_type* s_vala = s_vala_block + (warpid % PANGULU_WARP_PER_BLOCK) * PANGULU_SHARED_MEM_LEN;

        for (int i = warp_tid; i < cola2 - cola1; i += PANGULU_WARP_SIZE)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_vala[i] = a_value[a_valueidx[cola1 + i]];
        }
        // __syncthreads();

        for (int i = cola1, t = 0; i < cola2; i++, t++)
        {
            int rowa = s_idxa[t];
            int coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
            int coll2 = l_columnpointer[rowa + 1];

            calculate_type vala = s_vala[t];
            vala /= l_value[l_valueidx[coll1]];
            s_vala[t] = vala;

            for (int j = coll1 + 1 + warp_tid, p = warp_tid; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
            {
                // update a's value;
                pangulu_inblock_ptr f = binarysearch_inblk_cuda(s_idxa, 1 + t + p, cola2 - cola1 - coll2 + j, l_rowindex[j]);
                if (f != 0xffffffff)
                {
                    s_vala[f] -= vala * l_value[l_valueidx[j]];
                }
            }
            // __syncthreads();
        }

        for (int i = warp_tid; i < cola2 - cola1; i += PANGULU_WARP_SIZE)
        {
            a_value[a_valueidx[cola1 + i]] = s_vala[i];
        }
    }
#undef PANGULU_SHARED_MEM_LEN
}

// __global__ void gessm_cuda(
//     pangulu_inblock_idx n,
//     pangulu_inblock_ptr *a_columnpointer,
//     pangulu_inblock_idx *a_rowindex,
//     calculate_type *a_value,
//     pangulu_inblock_ptr *l_columnpointer,
//     pangulu_inblock_idx *l_rowindex,
//     calculate_type *l_value)
// {
// #define PANGULU_SHARED_MEM_LEN 1024
//     pangulu_inblock_idx colidx = blockIdx.x;
//     if (colidx >= n)
//     {
//         return;
//     }
//     a_columnpointer[0] = 0;
//     l_columnpointer[0] = 0;
//     pangulu_inblock_ptr cola1 = a_columnpointer[colidx];
//     pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
//     if (cola2 == cola1)
//     {
//         return;
//     }

//     if (cola2 - cola1 >= PANGULU_SHARED_MEM_LEN)
//     {
//         for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
//         {
//             pangulu_int64_t rowa = a_rowindex[i];
//             calculate_type vala = a_value[i];
//             pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
//             pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
//             for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
//             {
//                 // update a's value;
//                 pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - coll2 + j, l_rowindex[j]);
//                 if (f != 0xffffffff)
//                 {
//                     a_value[f] -= vala * l_value[j];
//                 }
//             }
//             __syncthreads();
//         }
//     }
//     else
//     {
//         __shared__ pangulu_inblock_idx s_idxa[PANGULU_SHARED_MEM_LEN];
//         __shared__ calculate_type s_vala[PANGULU_SHARED_MEM_LEN];

//         for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
//         {
//             s_idxa[i] = a_rowindex[cola1 + i];
//             s_vala[i] = a_value[cola1 + i];
//         }
//         __syncthreads();

//         for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
//         {
//             pangulu_int64_t rowa = s_idxa[t];
//             calculate_type vala = s_vala[t];

//             pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
//             pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

//             for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
//             {
//                 // update a's value;
//                 pangulu_inblock_ptr f = binarysearch_inblk_cuda(s_idxa, 1 + t + p, cola2 - cola1 - coll2 + j, l_rowindex[j]);
//                 if (f != 0xffffffff)
//                 {
//                     s_vala[f] -= vala * l_value[j];
//                 }
//             }
//             __syncthreads();
//         }

//         for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
//         {
//             a_value[cola1 + i] = s_vala[i];
//         }
//         //__syncthreads();
//     }
// #undef PANGULU_SHARED_MEM_LEN
// }

__global__ void gessm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *a_columnpointer,
    pangulu_inblock_idx *a_rowindex,
    calculate_type *a_value,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_type *l_value)
{
#define PANGULU_SHARED_MEM_LEN 1024
    pangulu_inblock_idx colidx = blockIdx.x;
    if (colidx >= n)
    {
        return;
    }
    a_columnpointer[0] = 0;
    l_columnpointer[0] = 0;
    pangulu_inblock_ptr cola1 = a_columnpointer[colidx];
    pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
    if (cola2 == cola1)
    {
        return;
    }

    if (n > PANGULU_SHARED_MEM_LEN)
    {
        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = a_rowindex[i];
            calculate_type vala = a_value[i];
            pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - 1, l_rowindex[j]);
                if (f != 0xffffffff)
                {
                    a_value[f] -= vala * l_value[j];
                }
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_idxa[PANGULU_SHARED_MEM_LEN];
        __shared__ calculate_type s_dense[PANGULU_SHARED_MEM_LEN];

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_dense[s_idxa[i]] = a_value[cola1 + i];
        }
        __syncthreads();

        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = s_idxa[t];
            calculate_type vala = s_dense[s_idxa[t]];

            pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                s_dense[l_rowindex[j]] -= vala * l_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[cola1 + i] = s_dense[s_idxa[i]];
        }
        //__syncthreads();
    }
#undef PANGULU_SHARED_MEM_LEN
}

__global__ void ssssm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *d_rowptrc,
    pangulu_inblock_idx *d_colidxc,
    calculate_type *d_valuec,
    pangulu_inblock_ptr *d_rowptrb,
    pangulu_inblock_idx *d_colidxb,
    calculate_type *d_valueb,
    pangulu_inblock_ptr *d_rowptra,
    pangulu_inblock_idx *d_colidxa,
    calculate_type *d_valuea)
{
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
    {
        d_rowptrc[0] = 0;
        d_rowptrb[0] = 0;
        d_rowptra[0] = 0;
    }

    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = blockIdx.x;

    if (rowidx >= n)
    {
        return;
    }

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    if (nextrowc - therowc <= 128)
    {
        __shared__ pangulu_inblock_idx s_idxc[128];
        __shared__ calculate_type s_valc[128];
        for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
        {
            s_idxc[i] = d_colidxc[therowc + i];
            s_valc[i] = 0;
        }
        __syncthreads();

        pangulu_int64_t therow = d_rowptra[rowidx];
        pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

        for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
        {
            pangulu_int64_t cola = d_colidxa[i];
            calculate_type vala = d_valuea[i];

            pangulu_int64_t therowb = d_rowptrb[cola];
            pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

            for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += 32)
            {
                pangulu_int64_t colb = d_colidxb[j];
                pangulu_int64_t flag = binarysearch_inblk_cuda(s_idxc, 0, nextrowc - therowc - 1, colb);
                if (flag != 0xffffffff)
                {
                    atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
                }
            }
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
        {
            d_valuec[therowc + i] += s_valc[i];
        }
    }
    else
    {
        pangulu_int64_t therow = d_rowptra[rowidx];
        pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

        for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
        {
            pangulu_int64_t cola = d_colidxa[i];
            calculate_type vala = d_valuea[i];

            pangulu_int64_t therowb = d_rowptrb[cola];
            pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

            for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += 32)
            {
                pangulu_int64_t colb = d_colidxb[j];
                pangulu_int64_t flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
                if (flag != 0xffffffff)
                {
                    atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                }
            }
        }
        __syncthreads();
    }
}

// void pangulu_platform_0201000_getrf(
//     pangulu_inblock_idx nb,
//     pangulu_storage_slot_t *opdst,
//     int tid)
// {
//     if (!hd_getrf_nnzu[tid])
//     {
//         pangulu_platform_0201000_malloc((void **)(&(hd_getrf_nnzu[tid])), sizeof(pangulu_int32_t) * (nb + 1));
//     }

//     getrf_get_nnzu<<<PANGULU_ICEIL(nb, 128), 128>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, hd_getrf_nnzu[tid]);
//     getrf_cuda<<<nb, 32>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, opdst->d_value, hd_getrf_nnzu[tid]);
//     pangulu_cuda_download_block(nb, opdst);
// }

__global__ void pangulu_load_dense(
    int nb,
    pangulu_inblock_ptr* columnpointer,
    pangulu_inblock_idx* rowindex,
    calculate_type* value,
    calculate_type* dense_buf
){
    int col = blockIdx.x;
    if(col >= nb){
        return;
    }
    for(pangulu_inblock_idx row = threadIdx.x; row < nb; row+=blockDim.x){
        dense_buf[col * nb + row] = 0;
    }
    __syncthreads();
    columnpointer[0] = 0;
    for(int idx = columnpointer[col] + threadIdx.x; idx < columnpointer[col+1]; idx+=blockDim.x){
        int row = rowindex[idx];
        dense_buf[col * nb + row] = value[idx];
    }
}

__global__ void pangulu_store_dense(
    int nb,
    pangulu_inblock_ptr* columnpointer,
    pangulu_inblock_idx* rowindex,
    calculate_type* value,
    calculate_type* dense_buf
){
    int col = blockIdx.x;
    if(col >= nb){
        return;
    }
    for(int idx = columnpointer[col] + threadIdx.x; idx < columnpointer[col+1]; idx+=blockDim.x){
        int row = rowindex[idx];
        value[idx] = dense_buf[col  * nb + row];
    }
}


__global__ void lunumeric_cuda_kernel_v2(
    pangulu_inblock_idx n,
    pangulu_int32_t *d_nnzu,
    calculate_type *d_dense_tag_double,
    pangulu_inblock_ptr *d_csccolptrl_upperbound,
    pangulu_inblock_idx *d_cscrowidxl_upperbound,
    pangulu_inblock_ptr *d_csccolptru_upperbound,
    pangulu_inblock_idx *d_cscrowidxu_upperbound
){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int warpid = tid / PANGULU_WARP_SIZE;
    const int warp_tid = tid % PANGULU_WARP_SIZE;

    const int colidx = warpid;
    if(colidx >= n){
        return;
    }

    // if(!warp_tid){
    //     printf("colidx=%d\n", colidx);
    // }

    const pangulu_inblock_ptr baseu_colidx = d_csccolptru_upperbound[colidx];
    const pangulu_inblock_ptr baseu_colidx1 = d_csccolptru_upperbound[colidx + 1];
    const pangulu_inblock_ptr basel_colidx = d_csccolptrl_upperbound[colidx];
    const pangulu_inblock_ptr basel_colidx1 = d_csccolptrl_upperbound[colidx + 1];

    // step one
    for (pangulu_inblock_ptr j = baseu_colidx; j < baseu_colidx1 - 1; j++)
    {
        const pangulu_inblock_idx rowidx = d_cscrowidxu_upperbound[j];
        // busy-wait until nnzu[rowidx] == 0
        do
        {
            __threadfence();
        } while (d_nnzu[rowidx] != 0);

        calculate_type bcast_value = d_dense_tag_double[colidx * n + rowidx];
        for (pangulu_inblock_ptr i = d_csccolptrl_upperbound[rowidx] + 1 + warp_tid; i < d_csccolptrl_upperbound[rowidx + 1]; i += PANGULU_WARP_SIZE)
        {
            const int lrowindex = d_cscrowidxl_upperbound[i];
            const int lcolindex = rowidx;
            d_dense_tag_double[colidx * n + lrowindex] -= d_dense_tag_double[lcolindex * n + lrowindex] * bcast_value;
            // atomicAdd(&d_dense_tag_double[colidx * n + lrowindex], -d_dense_tag_double[lcolindex * n + lrowindex] * bcast_value);
        }
    }

    // __threadfence();
    //  step two
    calculate_type diag_value_inv = 1.0 / d_dense_tag_double[colidx * n + colidx];
    for (pangulu_inblock_ptr i = basel_colidx + warp_tid + 1; i < d_csccolptrl_upperbound[colidx + 1]; i += PANGULU_WARP_SIZE)
    {
        const int lrowindex = d_cscrowidxl_upperbound[i];
        d_dense_tag_double[colidx * n + lrowindex] = d_dense_tag_double[colidx * n + lrowindex] * diag_value_inv;
    }

    if (!warp_tid)
    {
        d_nnzu[colidx] = 0;
    }
}

// void pangulu_platform_0201000_getrf(
//     pangulu_inblock_idx nb,
//     pangulu_storage_slot_t *opdst,
//     int tid)
// {
//     if (!hd_getrf_nnzu[tid])
//     {
//         pangulu_platform_0201000_malloc((void **)(&(hd_getrf_nnzu[tid])), sizeof(pangulu_int32_t) * (nb + 1));
//     }

//     getrf_get_nnzu<<<PANGULU_ICEIL(nb, 128), 128>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, hd_getrf_nnzu[tid]);
//     getrf_cuda<<<nb, 32>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, opdst->d_value, hd_getrf_nnzu[tid]);
//     pangulu_load_dense<<<nb, 256>>>(
//         nb, 
//         opdst->d_columnpointer, 
//         opdst->d_rowindex, 
//         opdst->d_value, 
//         getrf_dense_buf_d
//     );
//     cudaMemset(opdst->d_value, 0, sizeof(calculate_type) * opdst->columnpointer[nb]);
//     pangulu_store_dense<<<nb, 256>>>(
//         nb, 
//         opdst->d_columnpointer, 
//         opdst->d_rowindex, 
//         opdst->d_value, 
//         getrf_dense_buf_d
//     );
//     pangulu_cuda_download_block(nb, opdst);
// }

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{

    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();
    // printf("000\n");

    pangulu_load_dense<<<nb, PANGULU_GETRF_THREAD_PER_BLOCK>>>(
        nb, 
        opdst->d_columnpointer, 
        opdst->d_rowindex, 
        opdst->d_value, 
        getrf_dense_buf_d
    );
    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();

    const int nblock = PANGULU_ICEIL(nb, PANGULU_GETRF_THREAD_PER_BLOCK/PANGULU_WARP_SIZE);
    lunumeric_cuda_kernel_v2<<<nblock, PANGULU_GETRF_THREAD_PER_BLOCK>>>(
        nb,
        opdst->d_nnzu,
        getrf_dense_buf_d,
        opdst->d_csccolptrl_upperbound,
        opdst->d_cscrowidxl_upperbound,
        opdst->d_csccolptru_upperbound,
        opdst->d_cscrowidxu_upperbound);
    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();

    pangulu_store_dense<<<nb, PANGULU_GETRF_THREAD_PER_BLOCK>>>(
        nb, 
        opdst->d_columnpointer, 
        opdst->d_rowindex, 
        opdst->d_value, 
        getrf_dense_buf_d
    );
    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();

    pangulu_cuda_download_block(nb, opdst);
    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();

}

void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    // pangulu_int32_t nnz_diag_avg = opdiag->columnpointer[nb] / nb;
    // if(nnz_diag_avg < 64){
    //     const int nblock = PANGULU_ICEIL(nb, 128/PANGULU_WARP_SIZE);
    //     tstrf_cuda_warp<<<nblock, 128>>>(nb,
    //         opdst->d_rowpointer, opdst->d_columnindex, opdst->d_idx_of_csc_value_for_csr, opdst->d_value,
    //         opdiag->d_rowpointer, opdiag->d_columnindex, opdiag->d_idx_of_csc_value_for_csr, opdiag->d_value);
    // }else{
        tstrf_cuda_block<<<nb, 128>>>(nb,
            opdst->d_rowpointer, opdst->d_columnindex, opdst->d_idx_of_csc_value_for_csr, opdst->d_value,
            opdiag->d_rowpointer, opdiag->d_columnindex, opdiag->d_idx_of_csc_value_for_csr, opdiag->d_value);
    // }
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    gessm_cuda<<<nb, 128>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, opdst->d_value, opdiag->d_columnpointer, opdiag->d_rowindex, opdiag->d_value);
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
    struct timeval time_start;
    pangulu_time_start(&time_start);
    ssssm_cuda<<<nb, 128>>>(nb, opdst->d_columnpointer, opdst->d_rowindex, opdst->d_value, op1->d_columnpointer, op1->d_rowindex, op1->d_value, op2->d_columnpointer, op2->d_rowindex, op2->d_value);
    // pangulu_cuda_download_block(nb, opdst);
    pangulu_platform_0201000_synchronize();
    double elapsed_time = pangulu_time_stop(&time_start);

    pangulu_int64_t cub = 0;
    pangulu_int64_t memsize = 0;
    for (int col = 0; col < nb; col++)
    {
        for (int idx = op2->columnpointer[col]; idx < op2->columnpointer[col + 1]; idx++)
        {
            int row = op2->rowindex[idx];
            cub += op1->columnpointer[row + 1] - op1->columnpointer[row];
            // memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (op1->columnpointer[row + 1] - op1->columnpointer[row]);
            // memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (opdst->columnpointer[row + 1] - opdst->columnpointer[row]);
        }
        // for (int idx = op2->columnpointer[col]; idx < op2->columnpointer[col + 1]; idx++)
        // {
        //     int row = op2->rowindex[idx];
        //     cub += op1->columnpointer[row + 1] - op1->columnpointer[row];
        //     memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (op1->columnpointer[row + 1] - op1->columnpointer[row]);
        //     if (opdst->columnpointer[col + 1] - opdst->columnpointer[col] > 128)
        //     {
        //         memsize += (op1->columnpointer[row + 1] - op1->columnpointer[row]) * (sizeof(pangulu_inblock_idx) * log2(opdst->columnpointer[col + 1] - opdst->columnpointer[col]) + sizeof(calculate_type));
        //     }
        // }
        // if (opdst->columnpointer[col + 1] - opdst->columnpointer[col] <= 128)
        // {
        //     memsize += sizeof(pangulu_inblock_ptr) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (opdst->columnpointer[col + 1] - opdst->columnpointer[col]);
        // }
    }
    memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + op2->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));
    memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + op1->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));
    memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + opdst->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));

    double gflops = cub / elapsed_time / 1e9;
    double gBps = memsize / elapsed_time / 1e9;
    struct timeval timestamp;
    gettimeofday(&timestamp, NULL);
    // timestamp, gflops, GB/s, cub, elapsed_time
    fprintf(result_file, "%lld, %lf, %lf, %lld, %le\n",
            (long long)(timestamp.tv_sec * 1000000 + timestamp.tv_usec),
            gflops,
            gBps,
            cub,
            elapsed_time);
}

// __global__ void ssssm_batched_cuda(
//     pangulu_inblock_idx n,
//     pangulu_uint64_t ntasks,
//     pangulu_inblock_ptr **dd_rowptrc,
//     pangulu_inblock_idx **dd_colidxc,
//     calculate_type **dd_valuec,
//     pangulu_inblock_ptr **dd_rowptrb,
//     pangulu_inblock_idx **dd_colidxb,
//     calculate_type **dd_valueb,
//     pangulu_inblock_ptr **dd_rowptra,
//     pangulu_inblock_idx **dd_colidxa,
//     calculate_type **dd_valuea)
// {

//     pangulu_uint64_t task_id = blockIdx.x / n;
//     pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//     pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//     calculate_type *d_valuec = dd_valuec[task_id];
//     pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//     pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//     calculate_type *d_valueb = dd_valueb[task_id];
//     pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//     pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//     calculate_type *d_valuea = dd_valuea[task_id];

//     pangulu_uint32_t blockidx_intask = blockIdx.x % n;

//     if ((blockidx_intask == 0) && (threadIdx.x == 0))
//     {
//         d_rowptrc[0] = 0;
//         d_rowptrb[0] = 0;
//         d_rowptra[0] = 0;
//     }

//     pangulu_int64_t warp_local_id = threadIdx.x / 32;
//     pangulu_int64_t warp_num = blockDim.x / 32;
//     pangulu_int64_t lane_id = threadIdx.x % 32;

//     const pangulu_inblock_idx rowidx = blockidx_intask;

//     if (rowidx >= n)
//     {
//         return;
//     }

//     pangulu_int64_t therowc = d_rowptrc[rowidx];
//     pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

//     if (nextrowc - therowc <= 128)
//     {
//         __shared__ pangulu_inblock_idx s_idxc[128];
//         __shared__ calculate_type s_valc[128];
//         for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
//         {
//             s_idxc[i] = d_colidxc[therowc + i];
//             s_valc[i] = 0;
//         }
//         __syncthreads();

//         pangulu_int64_t therow = d_rowptra[rowidx];
//         pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

//         for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
//         {
//             pangulu_int64_t cola = d_colidxa[i];
//             calculate_type vala = d_valuea[i];

//             pangulu_int64_t therowb = d_rowptrb[cola];
//             pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

//             for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += 32)
//             {
//                 pangulu_int64_t colb = d_colidxb[j];
//                 pangulu_int64_t flag = binarysearch_inblk_cuda(s_idxc, 0, nextrowc - therowc - 1, colb);
//                 if (flag != 0xffffffff)
//                 {
//                     atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
//                 }
//             }
//         }
//         __syncthreads();

//         for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
//         {
//             // d_valuec[therowc + i] += s_valc[i];
//             atomicAdd(&d_valuec[therowc + i], s_valc[i]);
//         }
//     }
//     else
//     {
//         pangulu_int64_t therow = d_rowptra[rowidx];
//         pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

//         for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
//         {
//             pangulu_int64_t cola = d_colidxa[i];
//             calculate_type vala = d_valuea[i];

//             pangulu_int64_t therowb = d_rowptrb[cola];
//             pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

//             for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += 32)
//             {
//                 pangulu_int64_t colb = d_colidxb[j];
//                 pangulu_int64_t flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//                 if (flag != 0xffffffff)
//                 {
//                     atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//                 }
//             }
//         }
//         __syncthreads();
//     }
// }

// __global__ void pangulu_cuda_display_block(
//     pangulu_inblock_idx nb,
//     pangulu_inblock_ptr* pointer,
//     pangulu_inblock_idx* index,
//     calculate_type* value
// ){
//     if(blockDim.x * blockIdx.x + threadIdx.x == 0){
//         for(int i=0;i<nb;i++){
//             printf("(%d): ", i);
//             for(int j=pointer[i]; j<pointer[i+1]; j++){
//                 printf("%le(%d,%d) ", value[j], index[j], j);
//             }
//             printf("\n");
//         }
//     }
// }

__device__ pangulu_int32_t
get_task_id(
    pangulu_int32_t *blockmap,
    pangulu_int32_t left,
    pangulu_int32_t right,
    pangulu_int32_t target)
{
    pangulu_int32_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (blockmap[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

// __global__ void ssssm_batched_cuda_threadcol(
//     pangulu_inblock_idx n,
//     pangulu_uint64_t ntasks,
//     pangulu_inblock_ptr **dd_rowptrc,
//     pangulu_inblock_idx **dd_colidxc,
//     calculate_type **dd_valuec,
//     pangulu_inblock_ptr **dd_rowptrb,
//     pangulu_inblock_idx **dd_colidxb,
//     calculate_type **dd_valueb,
//     pangulu_inblock_ptr **dd_rowptra,
//     pangulu_inblock_idx **dd_colidxa,
//     calculate_type **dd_valuea,
//     pangulu_int32_t *d_task_block_ptr)
// {
//     pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//     pangulu_int32_t block_offset;
//     if (task_id == 0)
//     {
//         block_offset = blockIdx.x;
//     }
//     else
//     {
//         block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//     }
//     // if (threadIdx.x == 0)
//     // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d\n", blockIdx.x, task_id, ntasks, block_offset);
//     // return;

//     pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//     pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//     calculate_type *d_valuec = dd_valuec[task_id];
//     pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//     pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//     calculate_type *d_valueb = dd_valueb[task_id];
//     pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//     pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//     calculate_type *d_valuea = dd_valuea[task_id];

//     if ((block_offset == 0) && (threadIdx.x == 0))
//     {
//         d_rowptrc[0] = 0;
//         d_rowptrb[0] = 0;
//         d_rowptra[0] = 0;
//     }

//     const pangulu_inblock_idx row = block_offset * blockDim.x + threadIdx.x;

//     if (row >= n)
//     {
//         return;
//     }

//     pangulu_int64_t therowc = d_rowptrc[row];
//     pangulu_int64_t nextrowc = d_rowptrc[row + 1];

//     pangulu_int64_t therow = d_rowptra[row];
//     pangulu_int64_t nextrow = d_rowptra[row + 1];

//     for (pangulu_int64_t i = therow; i < nextrow; i++)
//     {
//         pangulu_int64_t cola = d_colidxa[i];
//         calculate_type vala = d_valuea[i];

//         pangulu_int64_t therowb = d_rowptrb[cola];
//         pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

//         for (pangulu_int64_t j = therowb; j < nextrowb; j++)
//         {
//             pangulu_int64_t colb = d_colidxb[j];
//             pangulu_int64_t flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//             if (flag != 0xffffffff)
//             {
//                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//             }
//         }
//     }
//     // __syncthreads();
// }

// __global__ void ssssm_batched_cuda_threadcol(
//     pangulu_inblock_idx n,
//     pangulu_uint64_t ntasks,
//     pangulu_inblock_ptr **dd_rowptrc,
//     pangulu_inblock_idx **dd_colidxc,
//     calculate_type **dd_valuec,
//     pangulu_inblock_ptr **dd_rowptrb,
//     pangulu_inblock_idx **dd_colidxb,
//     calculate_type **dd_valueb,
//     pangulu_inblock_ptr **dd_rowptra,
//     pangulu_inblock_idx **dd_colidxa,
//     calculate_type **dd_valuea,
//     pangulu_int32_t *d_task_block_ptr)
// {
//     pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//     pangulu_int32_t block_offset;
//     if (task_id == 0)
//     {
//         block_offset = blockIdx.x;
//     }
//     else
//     {
//         block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//     }
//     // if (threadIdx.x == 0)
//     // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d\n", blockIdx.x, task_id, ntasks, block_offset);
//     // return;

//     pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//     pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//     calculate_type *d_valuec = dd_valuec[task_id];
//     pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//     pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//     calculate_type *d_valueb = dd_valueb[task_id];
//     pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//     pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//     calculate_type *d_valuea = dd_valuea[task_id];

//     if ((block_offset == 0) && (threadIdx.x == 0))
//     {
//         d_rowptrc[0] = 0;
//         d_rowptrb[0] = 0;
//         d_rowptra[0] = 0;
//     }

//     const pangulu_inblock_idx row = block_offset * blockDim.x + threadIdx.x;

//     if (row >= n)
//     {
//         return;
//     }

//     pangulu_inblock_ptr therowc = d_rowptrc[row];
//     pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

//     pangulu_inblock_ptr therow = d_rowptra[row];
//     pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

//     for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//     {
//         pangulu_inblock_idx cola = d_colidxa[i];
//         calculate_type vala = d_valuea[i];

//         pangulu_inblock_ptr therowb = d_rowptrb[cola];
//         pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

//         for (pangulu_inblock_ptr j = therowb; j < nextrowb; j++)
//         {
//             pangulu_inblock_idx colb = d_colidxb[j];
//             pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//             if (flag != 0xffffffff)
//             {
//                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//             }
//         }
//     }
//     // __syncthreads();
// }

// __global__ void ssssm_batched_cuda_dynamic(
//     pangulu_inblock_idx n,
//     pangulu_uint64_t ntasks,
//     pangulu_inblock_ptr **dd_rowptrc,
//     pangulu_inblock_idx **dd_colidxc,
//     calculate_type **dd_valuec,
//     pangulu_inblock_ptr **dd_rowptrb,
//     pangulu_inblock_idx **dd_colidxb,
//     calculate_type **dd_valueb,
//     pangulu_inblock_ptr **dd_rowptra,
//     pangulu_inblock_idx **dd_colidxa,
//     calculate_type **dd_valuea,
//     pangulu_int32_t *d_task_block_ptr)
// {
//     pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//     pangulu_int32_t block_offset = 0;
//     pangulu_int32_t nblock_for_task = 0;
//     if (task_id == 0)
//     {
//         block_offset = blockIdx.x;
//         nblock_for_task = d_task_block_ptr[0];
//     }
//     else
//     {
//         block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//         nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
//     }

//     pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n));
//     pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//     pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

//     // if (threadIdx.x == 0)
//     //     printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d\n", blockIdx.x, task_id, ntasks, block_offset, nblock_for_task);
//     // return;

//     // if (threadIdx.x == 0 && blockIdx.x == 0)
//     // {
//     //     printf("ntasks=%lld total_block=%d\n", ntasks, d_task_block_ptr[ntasks - 1]);
//     // }

//     pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//     pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//     calculate_type *d_valuec = dd_valuec[task_id];
//     pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//     pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//     calculate_type *d_valueb = dd_valueb[task_id];
//     pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//     pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//     calculate_type *d_valuea = dd_valuea[task_id];

//     if ((block_offset == 0) && (threadIdx.x == 0))
//     {
//         d_rowptrc[0] = 0;
//         d_rowptrb[0] = 0;
//         d_rowptra[0] = 0;
//     }

//     // const pangulu_inblock_idx row =  block_offset * blockDim.x + threadIdx.x;
//     const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
//     const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

//     if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
//     {
//         return;
//     }

//     if (row > n)
//     {
//         return;
//     }

//     // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
//     //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);

//     pangulu_inblock_ptr therowc = d_rowptrc[row];
//     pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

//     pangulu_inblock_ptr therow = d_rowptra[row];
//     pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

//     for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//     {
//         pangulu_inblock_idx cola = d_colidxa[i];
//         calculate_type vala = d_valuea[i];

//         pangulu_inblock_ptr therowb = d_rowptrb[cola];
//         pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

//         for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//         {
//             pangulu_inblock_idx colb = d_colidxb[j];
//             pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//             if (flag != 0xffffffff)
//             {
//                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//             }
//         }
//     }
//     // __syncthreads();
// }

//__global__ void ssssm_batched_cuda_dynamic(
//    pangulu_inblock_idx n,
//    pangulu_uint64_t ntasks,
//    pangulu_inblock_ptr **dd_rowptrc,
//    pangulu_inblock_idx **dd_colidxc,
//    calculate_type **dd_valuec,
//    pangulu_inblock_ptr **dd_rowptrb,
//    pangulu_inblock_idx **dd_colidxb,
//    calculate_type **dd_valueb,
//    pangulu_inblock_ptr **dd_rowptra,
//    pangulu_inblock_idx **dd_colidxa,
//    calculate_type **dd_valuea,
//    pangulu_int32_t *d_task_block_ptr)
//{
//    pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//    pangulu_int32_t block_offset = 0;
//    pangulu_int32_t nblock_for_task = 0;
//    if (task_id == 0)
//    {
//        block_offset = blockIdx.x;
//        nblock_for_task = d_task_block_ptr[0];
//    }
//    else
//    {
//        block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//        nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
//    }
//
//    pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//    pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//    pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;
//
//    // if (threadIdx.x == 0)
//    //     printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d\n", blockIdx.x, task_id, ntasks, block_offset, nblock_for_task);
//    // return;
//
//    // if (threadIdx.x == 0 && blockIdx.x == 0)
//    // {
//    //     printf("ntasks=%lld total_block=%d\n", ntasks, d_task_block_ptr[ntasks - 1]);
//    // }
//
//    pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//    pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//    calculate_type *d_valuec = dd_valuec[task_id];
//    pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//    pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//    calculate_type *d_valueb = dd_valueb[task_id];
//    pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//    pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//    calculate_type *d_valuea = dd_valuea[task_id];
//
//    if ((block_offset == 0) && (threadIdx.x == 0))
//    {
//        d_rowptrc[0] = 0;
//        d_rowptrb[0] = 0;
//        d_rowptra[0] = 0;
//    }
//
//    if (d_rowptrc[n] < PANGULU_SSSSM_BATCHED_SHAREDMEM_LEN)
//    {
//        __shared__ pangulu_inblock_idx s_colidxc[PANGULU_SSSSM_BATCHED_SHAREDMEM_LEN];
//        for (pangulu_inblock_ptr idx = threadIdx.x; idx < d_rowptrc[n]; idx += blockDim.x)
//        {
//            s_colidxc[idx] = d_colidxc[idx];
//        }
//        __syncthreads();
//
//        // const pangulu_inblock_idx row =  block_offset * blockDim.x + threadIdx.x;
//        const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
//        const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;
//        if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
//        {
//            return;
//        }
//        if (row > n)
//        {
//            return;
//        }
//
//        // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
//        //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);
//
//        pangulu_inblock_ptr therowc = d_rowptrc[row];
//        pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];
//
//        pangulu_inblock_ptr therow = d_rowptra[row];
//        pangulu_inblock_ptr nextrow = d_rowptra[row + 1];
//        for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//        {
//            pangulu_inblock_idx cola = d_colidxa[i];
//            calculate_type vala = d_valuea[i];
//
//            pangulu_inblock_ptr therowb = d_rowptrb[cola];
//            pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];
//
//            for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//            {
//                pangulu_inblock_idx colb = d_colidxb[j];
//                pangulu_inblock_ptr flag = binarysearch_inblk_cuda(s_colidxc, therowc, nextrowc - 1, colb);
//                if (flag != 0xffffffff)
//                {
//                    atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//                }
//            }
//        }
//    }
//    else
//    {
//        // const pangulu_inblock_idx row =  block_offset * blockDim.x + threadIdx.x;
//        const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
//        const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;
//
//        if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
//        {
//            return;
//        }
//        if (row > n)
//        {
//            return;
//        }
//
//        // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
//        //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);
//
//        pangulu_inblock_ptr therowc = d_rowptrc[row];
//        pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];
//
//        pangulu_inblock_ptr therow = d_rowptra[row];
//        pangulu_inblock_ptr nextrow = d_rowptra[row + 1];
//        for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//        {
//            pangulu_inblock_idx cola = d_colidxa[i];
//            calculate_type vala = d_valuea[i];
//
//            pangulu_inblock_ptr therowb = d_rowptrb[cola];
//            pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];
//
//            for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//            {
//                pangulu_inblock_idx colb = d_colidxb[j];
//                pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//                if (flag != 0xffffffff)
//                {
//                    atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//                }
//            }
//        }
//    }
//
//    // __syncthreads();
//}

//__global__ void ssssm_batched_cuda_dynamic(
//    pangulu_inblock_idx n,
//    pangulu_uint64_t ntasks,
//    pangulu_inblock_ptr **dd_rowptrc,
//    pangulu_inblock_idx **dd_colidxc,
//    calculate_type **dd_valuec,
//    pangulu_inblock_ptr **dd_rowptrb,
//    pangulu_inblock_idx **dd_colidxb,
//    calculate_type **dd_valueb,
//    pangulu_inblock_ptr **dd_rowptra,
//    pangulu_inblock_idx **dd_colidxa,
//    calculate_type **dd_valuea,
//    pangulu_int32_t *d_task_block_ptr)
//{
//    pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//    pangulu_int32_t block_offset = 0;
//    pangulu_int32_t nblock_for_task = 0;
//    if (task_id == 0)
//    {
//        block_offset = blockIdx.x;
//        nblock_for_task = d_task_block_ptr[0];
//    }
//    else
//    {
//        block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//        nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
//    }
//
//    pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//    pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//    pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;
//
//    // if (threadIdx.x == 0)
//    //     printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d\n", blockIdx.x, task_id, ntasks, block_offset, nblock_for_task);
//    // return;
//
//    // if (threadIdx.x == 0 && blockIdx.x == 0)
//    // {
//    //     printf("ntasks=%lld total_block=%d\n", ntasks, d_task_block_ptr[ntasks - 1]);
//    // }
//
//    pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//    pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//    calculate_type *d_valuec = dd_valuec[task_id];
//    pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//    pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//    calculate_type *d_valueb = dd_valueb[task_id];
//    pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//    pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//    calculate_type *d_valuea = dd_valuea[task_id];
//
//    if ((block_offset == 0) && (threadIdx.x == 0))
//    {
//        d_rowptrc[0] = 0;
//        d_rowptrb[0] = 0;
//        d_rowptra[0] = 0;
//    }
//
//    // const pangulu_inblock_idx row =  block_offset * blockDim.x + threadIdx.x;
//    const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
//    const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;
//
//    if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
//    {
//        return;
//    }
//    if (row > n)
//    {
//        return;
//    }
//
//    // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
//    //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);
//
//    if((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n)){
//        for(pangulu_inblock_idx rowb = 0; rowb < n; rowb++){
//            calculate_type a_val = d_valuea[row * n + rowb];
//            for(pangulu_inblock_idx colb = thread_offset; colb < n; colb+=how_many_thread_a_col_need){
//                atomicAdd(&d_valuec[row * n + colb], - a_val * d_valueb[rowb * n + colb]);
//            }
//        }
//    }else{
//        pangulu_inblock_ptr therowc = d_rowptrc[row];
//        pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];
//
//        pangulu_inblock_ptr therow = d_rowptra[row];
//        pangulu_inblock_ptr nextrow = d_rowptra[row + 1];
//
//        for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//        {
//            pangulu_inblock_idx cola = d_colidxa[i];
//            calculate_type vala = d_valuea[i];
//
//            pangulu_inblock_ptr therowb = d_rowptrb[cola];
//            pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];
//
//            for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//            {
//                pangulu_inblock_idx colb = d_colidxb[j];
//                pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//                if (flag != 0xffffffff)
//                {
//                    atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//                }
//            }
//        }
//    }
//}

__global__ void ssssm_batched_cuda_dynamic(
    pangulu_inblock_idx n,
    pangulu_uint64_t ntasks,
    pangulu_inblock_ptr **dd_rowptrc,
    pangulu_inblock_idx **dd_colidxc,
    calculate_type **dd_valuec,
    pangulu_inblock_ptr **dd_rowptrb,
    pangulu_inblock_idx **dd_colidxb,
    calculate_type **dd_valueb,
    pangulu_inblock_ptr **dd_rowptra,
    pangulu_inblock_idx **dd_colidxa,
    calculate_type **dd_valuea,
    pangulu_int32_t *d_task_block_ptr)
{

    pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
    pangulu_int32_t block_offset = 0;
    pangulu_int32_t nblock_for_task = 0;
    if (task_id == 0)
    {
        block_offset = blockIdx.x;
        nblock_for_task = d_task_block_ptr[0];
    }
    else
    {
        block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
        nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
    }

    pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
    pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
    pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

    // if (threadIdx.x == 0)
    //     printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d\n", blockIdx.x, task_id, ntasks, block_offset, nblock_for_task);
    // return;

    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //     printf("ntasks=%lld total_block=%d\n", ntasks, d_task_block_ptr[ntasks - 1]);
    // }

    pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
    pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
    calculate_type *d_valuec = dd_valuec[task_id];
    pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
    pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
    calculate_type *d_valueb = dd_valueb[task_id];
    pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
    pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
    calculate_type *d_valuea = dd_valuea[task_id];

    if ((block_offset == 0) && (threadIdx.x == 0))
    {
        d_rowptrc[0] = 0;
        d_rowptrb[0] = 0;
        d_rowptra[0] = 0;
    }

    const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
    const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

    if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
    {
        return;
    }
    if (row >= n)
    {
        return;
    }

    // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
    //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);

    if ((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n))
    {
        for (pangulu_inblock_idx rowb = 0; rowb < n; rowb++)
        {
            calculate_type a_val = d_valuea[row * n + rowb];
            for (pangulu_inblock_idx colb = thread_offset; colb < n; colb += how_many_thread_a_col_need)
            {
                atomicAdd(&d_valuec[row * n + colb], -a_val * d_valueb[rowb * n + colb]);
            }
        }
    }
    else
    {
        pangulu_inblock_ptr therowc = d_rowptrc[row];
        pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

        pangulu_inblock_ptr therow = d_rowptra[row];
        pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

        for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
        {
            pangulu_inblock_idx cola = d_colidxa[i];
            calculate_type vala = d_valuea[i];

            pangulu_inblock_ptr therowb = d_rowptrb[cola];
            pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

            for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
            {
                pangulu_inblock_idx colb = d_colidxb[j];
                pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
                if (flag != 0xffffffff)
                {
                    atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                }
            }
        }
    }
}

__global__ void trojan_horse_batched_kernel_cuda(
    pangulu_inblock_idx n,
    pangulu_uint64_t ntasks,
    pangulu_int32_t* d_task_types,
    pangulu_int32_t* d_task_block_ptr,
    pangulu_inblock_ptr **dd_rowptrc,
    pangulu_inblock_idx **dd_colidxc,
    calculate_type **dd_valuec,
    pangulu_inblock_ptr **dd_rowptrb,
    pangulu_inblock_idx **dd_colidxb,
    calculate_type **dd_valueb,
    pangulu_inblock_ptr **dd_rowptra,
    pangulu_inblock_idx **dd_colidxa,
    calculate_type **dd_valuea,
    calculate_type **dd_getrf_tag_double,
    pangulu_int32_t **dd_getrf_nnzu,
    pangulu_inblock_ptr **dd_getrf_csccolptrl_upperbound,
    pangulu_inblock_idx **dd_getrf_cscrowidxl_upperbound,
    pangulu_inblock_ptr **dd_getrf_csccolptru_upperbound,
    pangulu_inblock_idx **dd_getrf_cscrowidxu_upperbound,
    pangulu_inblock_ptr **dd_tstrf_a_valueidx,
    pangulu_inblock_ptr **dd_tstrf_l_valueidx
)
{
    __shared__ pangulu_inblock_idx s_idxa[TROJAN_HORSE_SHARED_MEM_LEN];
    __shared__ calculate_type s_dense[TROJAN_HORSE_SHARED_MEM_LEN];

    pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
    pangulu_int32_t task_type = d_task_types[task_id];
    pangulu_int32_t block_offset = 0;
    pangulu_int32_t nblock_for_task = 0;
    if (task_id == 0)
    {
        block_offset = blockIdx.x;
        nblock_for_task = d_task_block_ptr[0];
    }
    else
    {
        block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
        nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
    }

    // if(threadIdx.x == 0){
    //     printf("%d::%d\n", task_id, task_type);
    // }

    pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
    pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
    calculate_type *d_valuec = dd_valuec[task_id];
    pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
    pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
    calculate_type *d_valueb = dd_valueb[task_id];
    pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
    pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
    calculate_type *d_valuea = dd_valuea[task_id];

    if(task_type == PANGULU_TASK_SSSSM){
        pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(TROJAN_HORSE_THREAD_PER_BLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
        pangulu_int32_t how_many_col_each_block_can_process = TROJAN_HORSE_THREAD_PER_BLOCK / how_many_thread_a_col_need;
        pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

        if ((block_offset == 0) && (threadIdx.x == 0))
        {
            d_rowptrc[0] = 0;
            d_rowptrb[0] = 0;
            d_rowptra[0] = 0;
        }


        const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
        const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

        if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
        {
            return;
        }
        if (row >= n)
        {
            return;
        }


        if ((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n))
        {
            for (pangulu_inblock_idx rowb = 0; rowb < n; rowb++)
            {
                calculate_type a_val = d_valuea[row * n + rowb];
                for (pangulu_inblock_idx colb = thread_offset; colb < n; colb += how_many_thread_a_col_need)
                {
                    atomicAdd(&d_valuec[row * n + colb], -a_val * d_valueb[rowb * n + colb]);
                }
            }
        }
        else
        {
            pangulu_inblock_ptr therowc = d_rowptrc[row];
            pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

            pangulu_inblock_ptr therow = d_rowptra[row];
            pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

            for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
            {
                pangulu_inblock_idx cola = d_colidxa[i];
                calculate_type vala = d_valuea[i];

                pangulu_inblock_ptr therowb = d_rowptrb[cola];
                pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

                for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
                {
                    pangulu_inblock_idx colb = d_colidxb[j];
                    pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
                    if (flag != 0xffffffff)
                    {
                        atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                    }
                }
            }
        }

    }else if(task_type == PANGULU_TASK_GETRF){
        const int tid = blockDim.x * block_offset + threadIdx.x;
        const int warpid = tid / PANGULU_WARP_SIZE;
        const int warp_tid = tid % PANGULU_WARP_SIZE;

        const int colidx = warpid;
        if(colidx >= n){
            return;
        }

        const pangulu_inblock_ptr baseu_colidx = dd_getrf_csccolptru_upperbound[task_id][colidx];
        const pangulu_inblock_ptr baseu_colidx1 = dd_getrf_csccolptru_upperbound[task_id][colidx + 1];
        const pangulu_inblock_ptr basel_colidx = dd_getrf_csccolptrl_upperbound[task_id][colidx];
        const pangulu_inblock_ptr basel_colidx1 = dd_getrf_csccolptrl_upperbound[task_id][colidx + 1];

        // step one
        for (pangulu_inblock_ptr j = baseu_colidx; j < baseu_colidx1 - 1; j++)
        {
            const pangulu_inblock_idx rowidx = dd_getrf_cscrowidxu_upperbound[task_id][j];
            // busy-wait until nnzu[rowidx] == 0
            do
            {
                __threadfence();
            } while (dd_getrf_nnzu[task_id][rowidx] != 0);

            calculate_type bcast_value = dd_getrf_tag_double[task_id][colidx * n + rowidx];
            for (pangulu_inblock_ptr i = dd_getrf_csccolptrl_upperbound[task_id][rowidx] + 1 + warp_tid; i < dd_getrf_csccolptrl_upperbound[task_id][rowidx + 1]; i += PANGULU_WARP_SIZE)
            {
                const int lrowindex = dd_getrf_cscrowidxl_upperbound[task_id][i];
                const int lcolindex = rowidx;
                dd_getrf_tag_double[task_id][colidx * n + lrowindex] -= dd_getrf_tag_double[task_id][lcolindex * n + lrowindex] * bcast_value;
                // atomicAdd(&d_dense_tag_double[colidx * n + lrowindex], -d_dense_tag_double[lcolindex * n + lrowindex] * bcast_value);
            }
        }

        // __threadfence();
        //  step two
        calculate_type diag_value_inv = 1.0 / dd_getrf_tag_double[task_id][colidx * n + colidx];
        for (pangulu_inblock_ptr i = basel_colidx + warp_tid + 1; i < dd_getrf_csccolptrl_upperbound[task_id][colidx + 1]; i += PANGULU_WARP_SIZE)
        {
            const int lrowindex = dd_getrf_cscrowidxl_upperbound[task_id][i];
            dd_getrf_tag_double[task_id][colidx * n + lrowindex] = dd_getrf_tag_double[task_id][colidx * n + lrowindex] * diag_value_inv;
        }

        if (!warp_tid)
        {
            dd_getrf_nnzu[task_id][colidx] = 0;
        }
    }else if(task_type == PANGULU_TASK_GESSM){
        pangulu_inblock_idx colidx = block_offset;
        if (colidx >= n)
        {
            return;
        }

        pangulu_inblock_ptr *a_columnpointer = d_rowptrc;
        pangulu_inblock_idx *a_rowindex = d_colidxc;
        calculate_type *a_value = d_valuec;
        pangulu_inblock_ptr *l_columnpointer = d_rowptrb;
        pangulu_inblock_idx *l_rowindex = d_colidxb;
        calculate_type *l_value = d_valueb;

        a_columnpointer[0] = 0;
        l_columnpointer[0] = 0;
        pangulu_inblock_ptr cola1 = a_columnpointer[colidx];
        pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
        if (cola2 == cola1)
        {
            return;
        }

        if (n > PANGULU_GESSM_SHARED_MEM_LEN)
        {
            for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
            {
                pangulu_int64_t rowa = a_rowindex[i];
                calculate_type vala = a_value[i];
                pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
                pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
                for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
                {
                    // update a's value;
                    pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - 1, l_rowindex[j]);
                    if (f != 0xffffffff)
                    {
                        a_value[f] -= vala * l_value[j];
                    }
                }
                __syncthreads();
            }
        }
        else
        {
            for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
            {
                s_idxa[i] = a_rowindex[cola1 + i];
                s_dense[s_idxa[i]] = a_value[cola1 + i];
            }
            __syncthreads();

            for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
            {
                pangulu_int64_t rowa = s_idxa[t];
                calculate_type vala = s_dense[s_idxa[t]];

                pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1], rowa);
                pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

                for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
                {
                    s_dense[l_rowindex[j]] -= vala * l_value[j];
                }
                __syncthreads();
            }

            for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
            {
                a_value[cola1 + i] = s_dense[s_idxa[i]];
            }
            //__syncthreads();
        }
    }else if(task_type == PANGULU_TASK_TSTRF){

        // if(threadIdx.x == 0){
        //     printf("nblock_for_task = %d\n", nblock_for_task);
        // }

        pangulu_inblock_ptr *a_columnpointer = d_rowptrc;
        pangulu_inblock_idx *a_rowindex = d_colidxc;
        calculate_type *a_value = d_valuec;
        pangulu_inblock_ptr* a_valueidx = dd_tstrf_a_valueidx[task_id];

        pangulu_inblock_ptr *l_columnpointer = d_rowptrb;
        pangulu_inblock_idx *l_rowindex = d_colidxb;
        calculate_type *l_value = d_valueb;
        pangulu_inblock_ptr* l_valueidx = dd_tstrf_l_valueidx[task_id];

        pangulu_inblock_idx colidx = block_offset;
        if (colidx >= n)
        {
            return;
        }

        // a_columnpointer[0] = 0;
        // l_columnpointer[0] = 0;
        pangulu_inblock_ptr cola1 = (colidx == 0) ? 0 : a_columnpointer[colidx];
        pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
        if (cola2 == cola1)
        {
            return;
        }

        if (n > PANGULU_TSTRF_SHARED_MEM_LEN)
        {
            for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
            {
                pangulu_int64_t rowa = a_rowindex[i];
                pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
                pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
                calculate_type vala = a_value[a_valueidx[i]];
                vala /= l_value[l_valueidx[coll1]];
                if (threadIdx.x == 0)
                {
                    a_value[a_valueidx[i]] = vala;
                }
                for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
                {
                    // update a's value;
                    pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1 + 1 + t + p, cola2 - 1, l_rowindex[j]);
                    // pangulu_inblock_ptr f = binarysearch_inblk_cuda(a_rowindex, cola1, cola2 - 1, l_rowindex[j]);
                    if (f != 0xffffffff)
                    {
                        a_value[a_valueidx[f]] -= vala * l_value[l_valueidx[j]];
                    }
                }
                __syncthreads();
            }
        }
        else
        {
            

            for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
            {
                s_idxa[i] = a_rowindex[cola1 + i];
                s_dense[s_idxa[i]] = a_value[a_valueidx[cola1 + i]];
            }
            __syncthreads();

            for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
            {
                pangulu_int64_t rowa = s_idxa[t];
                pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, ((rowa == 0) ? 0 : l_columnpointer[rowa]), l_columnpointer[rowa + 1] - 1, rowa);
                pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

                calculate_type vala;
                if ((threadIdx.x / 32) == 0)
                {
                    vala = s_dense[s_idxa[t]];
                    vala /= l_value[l_valueidx[coll1]];
                    s_dense[s_idxa[t]] = vala;
                    __syncthreads();
                }
                else
                {
                    __syncthreads();
                    __threadfence_block();
                    vala = s_dense[s_idxa[t]];
                }

                // s_dense[colidx] = s_dense[colidx] / l_value[l_valueidx[coll1]];
                // calculate_type vala = s_dense[colidx];

                for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
                {
                    // update a's value;
                    s_dense[l_rowindex[j]] -= vala * l_value[l_valueidx[j]];
                }
                __syncthreads();
            }

            for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
            {
                a_value[a_valueidx[cola1 + i]] = s_dense[s_idxa[i]];
            }
        }
    }
}

// __global__ void ssssm_batched_cuda_dynamic(
//     pangulu_inblock_idx n,
//     pangulu_uint64_t ntasks,
//     pangulu_inblock_ptr **dd_rowptrc,
//     pangulu_inblock_idx **dd_colidxc,
//     calculate_type **dd_valuec,
//     pangulu_inblock_ptr **dd_rowptrb,
//     pangulu_inblock_idx **dd_colidxb,
//     calculate_type **dd_valueb,
//     pangulu_inblock_ptr **dd_rowptra,
//     pangulu_inblock_idx **dd_colidxa,
//     calculate_type **dd_valuea,
//     pangulu_int32_t *d_task_block_ptr)
// {
//     pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
//     pangulu_int32_t block_offset = 0;
//     pangulu_int32_t nblock_for_task = 0;
//     if (task_id == 0)
//     {
//         block_offset = blockIdx.x;
//         nblock_for_task = d_task_block_ptr[0];
//     }
//     else
//     {
//         block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
//         nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
//     }

//     pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//     pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//     pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

//     // if (threadIdx.x == 0)
//     //     printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d\n", blockIdx.x, task_id, ntasks, block_offset, nblock_for_task);
//     // return;

//     // if (threadIdx.x == 0 && blockIdx.x == 0)
//     // {
//     //     printf("ntasks=%lld total_block=%d\n", ntasks, d_task_block_ptr[ntasks - 1]);
//     // }

//     pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
//     pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
//     calculate_type *d_valuec = dd_valuec[task_id];
//     pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
//     pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
//     calculate_type *d_valueb = dd_valueb[task_id];
//     pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
//     pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
//     calculate_type *d_valuea = dd_valuea[task_id];

//     if ((block_offset == 0) && (threadIdx.x == 0))
//     {
//         d_rowptrc[0] = 0;
//         d_rowptrb[0] = 0;
//         d_rowptra[0] = 0;
//     }

//     // const pangulu_inblock_idx row =  block_offset * blockDim.x + threadIdx.x;
//     const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
//     const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

//     if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
//     {
//         return;
//     }
//     if (row > n)
//     {
//         return;
//     }

//     // printf("bid=%d task_id=%d ntasks=%lld block_offset=%d nblock_for_task=%d row=%d thread_offset=%d thrprow=%d\n",
//     //        blockIdx.x, task_id, ntasks, block_offset, nblock_for_task, row, thread_offset, how_many_thread_a_col_need);

//     if ((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n))
//     {
//         for (pangulu_inblock_idx rowb = 0; rowb < n; rowb++)
//         {
//             calculate_type a_val = d_valuea[row * n + rowb];
//             for (pangulu_inblock_idx colb = thread_offset; colb < n; colb += how_many_thread_a_col_need)
//             {
//                 atomicAdd(&d_valuec[row * n + colb], -a_val * d_valueb[rowb * n + colb]);
//             }
//         }
//     }
//     // else if (how_many_thread_a_col_need == 1)
//     // {
//     //     pangulu_inblock_ptr therowc = d_rowptrc[row];
//     //     pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

//     //     pangulu_inblock_ptr therow = d_rowptra[row];
//     //     pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

//     //     for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//     //     {
//     //         pangulu_inblock_idx cola = d_colidxa[i];
//     //         calculate_type vala = d_valuea[i];

//     //         pangulu_inblock_ptr therowb = d_rowptrb[cola];
//     //         pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

//     //         for (pangulu_inblock_ptr j = therowb; j < nextrowb; j++)
//     //         {
//     //             pangulu_inblock_idx colb = d_colidxb[j];
//     //             pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//     //             if (flag != 0xffffffff)
//     //             {
//     //                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//     //             }
//     //         }
//     //     }
//     // }
//     // else if (d_rowptrc[n] < 3 * n)
//     // {
//     //     pangulu_inblock_ptr therowc = d_rowptrc[row];
//     //     pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

//     //     pangulu_inblock_ptr therow = d_rowptra[row];
//     //     pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

//     //     for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//     //     {
//     //         pangulu_inblock_idx cola = d_colidxa[i];
//     //         calculate_type vala = d_valuea[i];

//     //         pangulu_inblock_ptr therowb = d_rowptrb[cola];
//     //         pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

//     //         for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//     //         {
//     //             pangulu_inblock_idx colb = d_colidxb[j];
//     //             pangulu_inblock_ptr flag = sequentialsearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//     //             if (flag != 0xffffffff)
//     //             {
//     //                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//     //             }
//     //         }
//     //     }
//     // }
//     else
//     {
//         pangulu_inblock_ptr therowc = d_rowptrc[row];
//         pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

//         pangulu_inblock_ptr therow = d_rowptra[row];
//         pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

//         for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
//         {
//             pangulu_inblock_idx cola = d_colidxa[i];
//             calculate_type vala = d_valuea[i];

//             pangulu_inblock_ptr therowb = d_rowptrb[cola];
//             pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

//             for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
//             {
//                 pangulu_inblock_idx colb = d_colidxb[j];
//                 pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
//                 if (flag != 0xffffffff)
//                 {
//                     atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
//                 }
//             }
//         }
//     }
// }

__global__ void store_csc_to_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = col * nb + threadIdx.x; idx < (col + 1) * nb; idx += blockDim.x)
    {
        dense[idx] = 0.0;
    }
    // __syncthreads();
    for (int idx = (col == 0 ? 0 : d_colptr[col]) + threadIdx.x; idx < d_colptr[col + 1]; idx += blockDim.x)
    {
        dense[col * nb + d_rowidx[idx]] = d_value[idx];
    }

    //__syncthreads();
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //    printf("D:\n");
    //    for(int i=0;i<nb;i++){
    //        printf("(%d) ", i);
    //        for(int j=0;j<nb;j++){
    //            printf("%.1le ", dense[j * nb + i]);
    //        }
    //        printf("\n");
    //    }
    //}

    //__syncthreads();
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //    printf("R:\n");
    //    for(int i=0;i<nb;i++){
    //        printf("(%d) ", i);
    //        for(int j=0;j<nb;j++){
    //            printf("%.1le ", dense[j * nb + i]);
    //        }
    //        printf("\n");
    //    }
    //    for(int i=0;i<nb;i++){
    //        printf("(%d) ", i);
    //        for(int idx = d_colptr[i]; idx < d_colptr[i+1]; idx++){
    //            printf("%hu(%.1le) ", d_rowidx[idx], d_value[idx]);
    //        }
    //        printf("\n");
    //    }
    //}
}

__global__ void clear_dense(
    pangulu_inblock_idx nb,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = col * nb + threadIdx.x; idx < (col + 1) * nb; idx += blockDim.x)
    {
        dense[idx] = 0.0;
    }
}

__global__ void csc_add_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = (col == 0 ? 0 : d_colptr[col]) + threadIdx.x; idx < d_colptr[col + 1]; idx += blockDim.x)
    {
        d_value[idx] += dense[col * nb + d_rowidx[idx]];
    }

    //__syncthreads();
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //    printf("R:\n");
    //    for(int i=0;i<nb;i++){
    //        printf("(%d) ", i);
    //        for(int j=0;j<nb;j++){
    //            printf("%.1le ", dense[j * nb + i]);
    //        }
    //        printf("\n");
    //    }
    //    for(int i=0;i<nb;i++){
    //        printf("(%d) ", i);
    //        for(int idx = d_colptr[i]; idx < d_colptr[i+1]; idx++){
    //            printf("%hu(%.1le) ", d_rowidx[idx], d_value[idx]);
    //        }
    //        printf("\n");
    //    }
    //}
}

pangulu_uint64_t task_pointer_buf_capacity = 0;

pangulu_inblock_ptr **hd_rowptrc;
pangulu_inblock_idx **hd_colidxc;
calculate_type **hd_valuec;
pangulu_inblock_ptr **hd_rowptrb;
pangulu_inblock_idx **hd_colidxb;
calculate_type **hd_valueb;
pangulu_inblock_ptr **hd_rowptra;
pangulu_inblock_idx **hd_colidxa;
calculate_type **hd_valuea;

pangulu_inblock_ptr **dd_rowptrc;
pangulu_inblock_idx **dd_colidxc;
calculate_type **dd_valuec;
pangulu_inblock_ptr **dd_rowptrb;
pangulu_inblock_idx **dd_colidxb;
calculate_type **dd_valueb;
pangulu_inblock_ptr **dd_rowptra;
pangulu_inblock_idx **dd_colidxa;
calculate_type **dd_valuea;

pangulu_int32_t* h_task_types;
pangulu_int32_t* d_task_types;

pangulu_int32_t *h_task_block_ptr;
pangulu_int32_t *d_task_block_ptr;

calculate_type **hd_getrf_tag_double;
pangulu_int32_t** hd_getrf_nnzu;
pangulu_inblock_ptr **hd_getrf_csccolptrl_upperbound;
pangulu_inblock_idx **hd_getrf_cscrowidxl_upperbound;
pangulu_inblock_ptr **hd_getrf_csccolptru_upperbound;
pangulu_inblock_idx **hd_getrf_cscrowidxu_upperbound;
pangulu_inblock_ptr **hd_tstrf_a_valueidx;
pangulu_inblock_ptr **hd_tstrf_l_valueidx;

calculate_type **dd_getrf_tag_double;
pangulu_int32_t** dd_getrf_nnzu;
pangulu_inblock_ptr **dd_getrf_csccolptrl_upperbound;
pangulu_inblock_idx **dd_getrf_cscrowidxl_upperbound;
pangulu_inblock_ptr **dd_getrf_csccolptru_upperbound;
pangulu_inblock_idx **dd_getrf_cscrowidxu_upperbound;
pangulu_inblock_ptr **dd_tstrf_a_valueidx;
pangulu_inblock_ptr **dd_tstrf_l_valueidx;

// void pangulu_platform_0201000_ssssm_batched(
//     pangulu_inblock_idx nb,
//     pangulu_uint64_t ntask,
//     pangulu_task_t *tasks)
//{
// #define PANGULU_REMALLOC_HOST(ptr, type) ptr = (type)pangulu_realloc(__FILE__, __LINE__, ptr, sizeof(ptr) * ntask)
// #def ine PANGULU_REMALLOC_DEVICE(ptr, type) \
//    pangulu_platform_0201000_free(ptr);    \
//    pangulu_platform_0201000_malloc((void **)&(ptr), sizeof(ptr) * ntask)
// #define PANGULU_PTR_UPLOAD(dptr, hptr) pangulu_platform_0201000_memcpy(dptr, hptr, sizeof(hptr) * ntask, 0)
//     if (task_pointer_buf_capacity < ntask)
//     {
//         PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
//         PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);
//
//         PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);
//
//         task_pointer_buf_capacity = ntask;
//     }
//
//     for (pangulu_uint64_t i = 0; i < ntask; i++)
//     {
//         hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
//         hd_colidxc[i] = tasks[i].opdst->d_rowindex;
//         hd_valuec[i] = tasks[i].opdst->d_value;
//
//         hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
//         hd_colidxb[i] = tasks[i].op1->d_rowindex;
//         hd_valueb[i] = tasks[i].op1->d_value;
//
//         hd_rowptra[i] = tasks[i].op2->d_columnpointer;
//         hd_colidxa[i] = tasks[i].op2->d_rowindex;
//         hd_valuea[i] = tasks[i].op2->d_value;
//
//         // pangulu_int32_t need_block = PANGULU_ICEIL(PANGULU_ICEIL(hd_rowptrb[i][nb], nb) * nb, PANGULU_SSSSM_BATCHED_THREADPERBLOCK);
//         pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//         pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//         pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
//         if (i == 0)
//         {
//             h_task_block_ptr[0] = need_block;
//         }
//         else
//         {
//             h_task_block_ptr[i] = h_task_block_ptr[i - 1] + need_block;
//         }
//     }
//
//     PANGULU_PTR_UPLOAD(dd_rowptrc, hd_rowptrc);
//     PANGULU_PTR_UPLOAD(dd_colidxc, hd_colidxc);
//     PANGULU_PTR_UPLOAD(dd_valuec, hd_valuec);
//     PANGULU_PTR_UPLOAD(dd_rowptrb, hd_rowptrb);
//     PANGULU_PTR_UPLOAD(dd_colidxb, hd_colidxb);
//     PANGULU_PTR_UPLOAD(dd_valueb, hd_valueb);
//     PANGULU_PTR_UPLOAD(dd_rowptra, hd_rowptra);
//     PANGULU_PTR_UPLOAD(dd_colidxa, hd_colidxa);
//     PANGULU_PTR_UPLOAD(dd_valuea, hd_valuea);
//     PANGULU_PTR_UPLOAD(d_task_block_ptr, h_task_block_ptr);
//
//     // struct timeval time_start;
//     // pangulu_time_start(&time_start);
//     ssssm_batched_cuda_dynamic<<<h_task_block_ptr[ntask - 1], PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(
//         nb,
//         ntask,
//         dd_rowptrc,
//         dd_colidxc,
//         dd_valuec,
//         dd_rowptrb,
//         dd_colidxb,
//         dd_valueb,
//         dd_rowptra,
//         dd_colidxa,
//         dd_valuea,
//         d_task_block_ptr);
//     pangulu_platform_0201000_synchronize();
//     // sleep(1);
//     // printf("\n");
//     // double elapsed_time = pangulu_time_stop(&time_start);
//
//     // exit(0);
//
//     // pangulu_int64_t cub = 0;
//     // pangulu_int64_t memsize = 0;
//     // for (int itask = 0; itask < ntask; itask++)
//     //{
//     //     pangulu_storage_slot_t *op1 = tasks[itask].op1;
//     //     pangulu_storage_slot_t *op2 = tasks[itask].op2;
//     //     pangulu_storage_slot_t *opdst = tasks[itask].opdst;
//     //     for (int col = 0; col < nb; col++)
//     //     {
//     //         for (int idx = op2->columnpointer[col]; idx < op2->columnpointer[col + 1]; idx++)
//     //         {
//     //             int row = op2->rowindex[idx];
//     //             cub += op1->columnpointer[row + 1] - op1->columnpointer[row];
//     //             // memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (op1->columnpointer[row + 1] - op1->columnpointer[row]);
//     //             // memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (opdst->columnpointer[row + 1] - opdst->columnpointer[row]);
//     //         }
//     //         // for (int idx = op2->columnpointer[col]; idx < op2->columnpointer[col + 1]; idx++)
//     //         // {
//     //         //     int row = op2->rowindex[idx];
//     //         //     cub += op1->columnpointer[row + 1] - op1->columnpointer[row];
//     //         //     memsize += (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (op1->columnpointer[row + 1] - op1->columnpointer[row]);
//     //         //     if (opdst->columnpointer[col + 1] - opdst->columnpointer[col] > 128)
//     //         //     {
//     //         //         memsize += (op1->columnpointer[row + 1] - op1->columnpointer[row]) * (sizeof(pangulu_inblock_idx) * log2(opdst->columnpointer[col + 1] - opdst->columnpointer[col]) + sizeof(calculate_type));
//     //         //     }
//     //         // }
//     //         // if (opdst->columnpointer[col + 1] - opdst->columnpointer[col] <= 128)
//     //         // {
//     //         //     memsize += sizeof(pangulu_inblock_ptr) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (opdst->columnpointer[col + 1] - opdst->columnpointer[col]);
//     //         // }
//     //     }
//     //     memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + op2->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));
//     //     memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + op1->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));
//     //     memsize += ((nb + 1) * sizeof(pangulu_inblock_ptr) + opdst->columnpointer[nb] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)));
//     // }
//
//     // double gflops = cub / elapsed_time / 1e9;
//     // double gBps = memsize / elapsed_time / 1e9;
//     // struct timeval timestamp;
//     // gettimeofday(&timestamp, NULL);
//     //// timestamp, gflops, GB/s, cub, elapsed_time
//     // fprintf(result_file, "%lld, %lf, %lf, %lld, %le\n",
//     //         (long long)(timestamp.tv_sec * 1000000 + timestamp.tv_usec),
//     //         gflops,
//     //         gBps,
//     //         cub,
//     //         elapsed_time);
//
// #undef PANGULU_REALLOC_HOST
// #undef PANGULU_REALLOC_DEVICE
// #undef PANGULU_PTR_UPLOAD
// }



pangulu_inblock_ptr *d_general_dense_columnpointer = NULL;
pangulu_inblock_idx *d_general_dense_rowindex = NULL;
calculate_type *d_dense_buffer = NULL;
pangulu_uint64_t dense_buffer_block_cap = 0;
pangulu_uint64_t *dense_task_indeces = NULL;
pangulu_uint64_t dense_task_indeces_cap = 0;

calculate_type *d_getrf_tag_buffer = NULL;
pangulu_uint64_t getrf_buffer_cap = 0;
pangulu_uint64_t *getrf_indeces = NULL;
pangulu_uint64_t getrf_indeces_cap = 0;


// void pangulu_platform_0201000_ssssm_batched(
//     pangulu_inblock_idx nb,
//     pangulu_uint64_t ntask,
//     pangulu_task_t *tasks)
// {
// #define PANGULU_REMALLOC_HOST(ptr, type) ptr = (type)pangulu_realloc(__FILE__, __LINE__, ptr, sizeof(ptr) * ntask)
// #define PANGULU_REMALLOC_DEVICE(ptr, type) \
//     pangulu_platform_0201000_free(ptr);    \
//     pangulu_platform_0201000_malloc((void **)&(ptr), sizeof(ptr) * ntask)
// #define PANGULU_PTR_UPLOAD(dptr, hptr) pangulu_platform_0201000_memcpy(dptr, hptr, sizeof(hptr) * ntask, 0)
//     if (task_pointer_buf_capacity < ntask)
//     {
//         PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
//         PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);

//         PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);

//         task_pointer_buf_capacity = ntask;
//     }

//     double dense_threshold = 0.65;

//     pangulu_uint64_t dense_task_idx = 0;
//     for (pangulu_uint64_t i = 0; i < ntask; i++)
//     {
//         if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
//         {
//             dense_task_idx += 1;
//         }
//     }
//     if (dense_task_idx > dense_task_indeces_cap)
//     {
//         dense_task_indeces_cap = dense_task_idx;
//         dense_task_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, dense_task_indeces, sizeof(pangulu_uint64_t) * dense_task_indeces_cap);

//         dense_buffer_block_cap = 3 * dense_task_indeces_cap;
//         if (d_dense_buffer)
//         {
//             cudaFree(d_dense_buffer);
//         }
//         cudaMalloc(&d_dense_buffer, sizeof(calculate_type) * nb * nb * dense_buffer_block_cap);
//     }

//     // printf("dense_task_idx=%lld\n", dense_task_idx);

//     dense_task_idx = 0;
//     for (pangulu_uint64_t i = 0; i < ntask; i++)
//     {
//         if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
//         {
//             if (d_general_dense_columnpointer == NULL)
//             {
//                 pangulu_inblock_ptr *h_general_dense_columnpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
//                 pangulu_inblock_idx *h_general_dense_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nb * nb));
//                 for (int col = 0; col <= nb; col++)
//                 {
//                     h_general_dense_columnpointer[col] = col * nb;
//                 }
//                 for (int idx = 0; idx < nb * nb; idx++)
//                 {
//                     h_general_dense_rowindex[idx] = idx % nb;
//                 }
//                 cudaMalloc(&d_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1));
//                 cudaMalloc(&d_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb));
//                 cudaMemcpy(d_general_dense_columnpointer, h_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1), cudaMemcpyHostToDevice);
//                 cudaMemcpy(d_general_dense_rowindex, h_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb), cudaMemcpyHostToDevice);
//                 pangulu_free(__FILE__, __LINE__, h_general_dense_columnpointer);
//                 pangulu_free(__FILE__, __LINE__, h_general_dense_rowindex);
//             }

//             store_csc_to_dense<<<nb, PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(nb, tasks[i].op1->d_columnpointer, tasks[i].op1->d_rowindex, tasks[i].op1->d_value, d_dense_buffer + ((3 * dense_task_idx) * nb * nb));
//             store_csc_to_dense<<<nb, PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(nb, tasks[i].op2->d_columnpointer, tasks[i].op2->d_rowindex, tasks[i].op2->d_value, d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb));
//             clear_dense<<<nb, PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(nb, d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb));
//             // pangulu_platform_0201000_synchronize();
//             cudaError_t err = cudaGetLastError();
//             if (err)
//             {
//                 printf("error : %s\n", cudaGetErrorString(err));
//             }

//             hd_rowptrb[i] = d_general_dense_columnpointer;
//             hd_colidxb[i] = d_general_dense_rowindex;
//             hd_valueb[i] = d_dense_buffer + ((3 * dense_task_idx) * nb * nb);

//             hd_rowptra[i] = d_general_dense_columnpointer;
//             hd_colidxa[i] = d_general_dense_rowindex;
//             hd_valuea[i] = d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb);

//             hd_rowptrc[i] = d_general_dense_columnpointer;
//             hd_colidxc[i] = d_general_dense_rowindex;
//             hd_valuec[i] = d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb);

//             pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, (nb * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM));
//             pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//             pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);

//             if (i == 0)
//             {
//                 h_task_block_ptr[0] = need_block;
//             }
//             else
//             {
//                 h_task_block_ptr[i] = h_task_block_ptr[i - 1] + need_block;
//             }

//             dense_task_indeces[dense_task_idx] = i;
//             dense_task_idx++;
//         }
//         else
//         {
//             hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
//             hd_colidxc[i] = tasks[i].opdst->d_rowindex;
//             hd_valuec[i] = tasks[i].opdst->d_value;

//             hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
//             hd_colidxb[i] = tasks[i].op1->d_rowindex;
//             hd_valueb[i] = tasks[i].op1->d_value;

//             hd_rowptra[i] = tasks[i].op2->d_columnpointer;
//             hd_colidxa[i] = tasks[i].op2->d_rowindex;
//             hd_valuea[i] = tasks[i].op2->d_value;

//             pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//             pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//             pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
//             if (i == 0)
//             {
//                 h_task_block_ptr[0] = need_block;
//             }
//             else
//             {
//                 h_task_block_ptr[i] = h_task_block_ptr[i - 1] + need_block;
//             }
//         }
//     }

//     PANGULU_PTR_UPLOAD(dd_rowptrc, hd_rowptrc);
//     PANGULU_PTR_UPLOAD(dd_colidxc, hd_colidxc);
//     PANGULU_PTR_UPLOAD(dd_valuec, hd_valuec);
//     PANGULU_PTR_UPLOAD(dd_rowptrb, hd_rowptrb);
//     PANGULU_PTR_UPLOAD(dd_colidxb, hd_colidxb);
//     PANGULU_PTR_UPLOAD(dd_valueb, hd_valueb);
//     PANGULU_PTR_UPLOAD(dd_rowptra, hd_rowptra);
//     PANGULU_PTR_UPLOAD(dd_colidxa, hd_colidxa);
//     PANGULU_PTR_UPLOAD(dd_valuea, hd_valuea);
//     PANGULU_PTR_UPLOAD(d_task_block_ptr, h_task_block_ptr);

//     // pangulu_platform_0201000_synchronize();
//     ssssm_batched_cuda_dynamic<<<h_task_block_ptr[ntask - 1], PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(
//         nb,
//         ntask,
//         dd_rowptrc,
//         dd_colidxc,
//         dd_valuec,
//         dd_rowptrb,
//         dd_colidxb,
//         dd_valueb,
//         dd_rowptra,
//         dd_colidxa,
//         dd_valuea,
//         d_task_block_ptr);
//     // pangulu_platform_0201000_synchronize();
//     for (int idense = 0; idense < dense_task_idx; idense++)
//     {
//         int itask = dense_task_indeces[idense];
//         // printf("itask=%d\n", itask);
//         csc_add_dense<<<nb, PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(
//             nb,
//             tasks[itask].opdst->d_columnpointer,
//             tasks[itask].opdst->d_rowindex,
//             tasks[itask].opdst->d_value,
//             hd_valuec[itask]);
//     }
//     pangulu_platform_0201000_synchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err)
//     {
//         printf("error2 : %s\n", cudaGetErrorString(err));
//     }

// #undef PANGULU_REALLOC_HOST
// #undef PANGULU_REALLOC_DEVICE
// #undef PANGULU_PTR_UPLOAD
// }

char *info_pool_h = NULL;
char *info_pool_d = NULL;

// void pangulu_platform_0201000_ssssm_batched(
//     pangulu_inblock_idx nb,
//     pangulu_uint64_t ntask,
//     pangulu_task_t *tasks)
// {
// #define PANGULU_REMALLOC_HOST(ptr, type)     \
//     ptr = (type)(info_pool_h + pool_offset); \
//     pool_offset += sizeof(*ptr) * ntask
// #define PANGULU_REMALLOC_DEVICE(ptr, type)   \
//     ptr = (type)(info_pool_d + pool_offset); \
//     pool_offset += sizeof(*ptr) * ntask
//     if (task_pointer_buf_capacity < ntask)
//     {
//         if (info_pool_h)
//         {
//             pangulu_free(__FILE__, __LINE__, info_pool_h);
//         }
//         info_pool_h = (char *)pangulu_malloc(__FILE__, __LINE__,
//                                              ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)));
//         if (info_pool_d)
//         {
//             pangulu_platform_0201000_free(info_pool_d);
//         }
//         pangulu_platform_0201000_malloc((void **)&(info_pool_d),
//                                         ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)));

//         unsigned long long pool_offset = 0;
//         PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
//         PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
//         PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);

//         pool_offset = 0;
//         PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
//         PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
//         PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
//         PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);

//         printf("%llu %llu %llu\n", ntask, pool_offset, ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)));

//         task_pointer_buf_capacity = ntask;
//     }

//     double dense_threshold = 0.65;

//     pangulu_uint64_t dense_task_idx = 0;
//     for (pangulu_uint64_t i = 0; i < ntask; i++)
//     {
//         if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
//         {
//             dense_task_idx += 1;
//         }
//     }
//     if (dense_task_idx > dense_task_indeces_cap)
//     {
//         dense_task_indeces_cap = dense_task_idx;
//         dense_task_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, dense_task_indeces, sizeof(pangulu_uint64_t) * dense_task_indeces_cap);

//         dense_buffer_block_cap = 3 * dense_task_indeces_cap;
//         if (d_dense_buffer)
//         {
//             cudaFree(d_dense_buffer);
//         }
//         cudaMalloc(&d_dense_buffer, sizeof(calculate_type) * nb * nb * dense_buffer_block_cap);
//     }

//     printf("dense_task_idx=%lld\n", dense_task_idx);

//     dense_task_idx = 0;
//     for (pangulu_uint64_t i = 0; i < ntask; i++)
//     {
//         if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
//         {
//             if (d_general_dense_columnpointer == NULL)
//             {
//                 pangulu_inblock_ptr *h_general_dense_columnpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
//                 pangulu_inblock_idx *h_general_dense_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nb * nb));
//                 for (int col = 0; col <= nb; col++)
//                 {
//                     h_general_dense_columnpointer[col] = col * nb;
//                 }
//                 for (int idx = 0; idx < nb * nb; idx++)
//                 {
//                     h_general_dense_rowindex[idx] = idx % nb;
//                 }
//                 cudaMalloc(&d_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1));
//                 cudaMalloc(&d_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb));
//                 cudaMemcpy(d_general_dense_columnpointer, h_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1), cudaMemcpyHostToDevice);
//                 cudaMemcpy(d_general_dense_rowindex, h_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb), cudaMemcpyHostToDevice);
//                 pangulu_free(__FILE__, __LINE__, h_general_dense_columnpointer);
//                 pangulu_free(__FILE__, __LINE__, h_general_dense_rowindex);
//             }

//             store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op1->d_columnpointer, tasks[i].op1->d_rowindex, tasks[i].op1->d_value, d_dense_buffer + ((3 * dense_task_idx) * nb * nb));
//             store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op2->d_columnpointer, tasks[i].op2->d_rowindex, tasks[i].op2->d_value, d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb));
//             clear_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb));
//             pangulu_platform_0201000_synchronize();
//             cudaError_t err = cudaGetLastError();
//             if (err)
//             {
//                 printf("error : %s\n", cudaGetErrorString(err));
//             }

//             hd_rowptrb[i] = d_general_dense_columnpointer;
//             hd_colidxb[i] = d_general_dense_rowindex;
//             hd_valueb[i] = d_dense_buffer + ((3 * dense_task_idx) * nb * nb);

//             hd_rowptra[i] = d_general_dense_columnpointer;
//             hd_colidxa[i] = d_general_dense_rowindex;
//             hd_valuea[i] = d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb);

//             hd_rowptrc[i] = d_general_dense_columnpointer;
//             hd_colidxc[i] = d_general_dense_rowindex;
//             hd_valuec[i] = d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb);

//             pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, (nb * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM));
//             pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//             pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);

//             // if (i == 0)
//             // {
//             //     h_task_block_ptr[0] = need_block;
//             // }
//             // else
//             // {
//             //     h_task_block_ptr[i] = h_task_block_ptr[i - 1] + need_block;
//             // }
//             h_task_block_ptr[i] = need_block;

//             dense_task_indeces[dense_task_idx] = i;
//             dense_task_idx++;
//         }
//         else
//         {
//             hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
//             hd_colidxc[i] = tasks[i].opdst->d_rowindex;
//             hd_valuec[i] = tasks[i].opdst->d_value;

//             hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
//             hd_colidxb[i] = tasks[i].op1->d_rowindex;
//             hd_valueb[i] = tasks[i].op1->d_value;

//             hd_rowptra[i] = tasks[i].op2->d_columnpointer;
//             hd_colidxa[i] = tasks[i].op2->d_rowindex;
//             hd_valuea[i] = tasks[i].op2->d_value;

//             pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
//             pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
//             pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
//             // if (i == 0)
//             // {
//             //     h_task_block_ptr[0] = need_block;
//             // }
//             // else
//             // {
//             //     h_task_block_ptr[i] = h_task_block_ptr[i - 1] + need_block;
//             // }
//             h_task_block_ptr[i] = need_block;
//         }
//     }

//     for (int i = 1; i < ntask; i++)
//     {
//         h_task_block_ptr[i] += h_task_block_ptr[i - 1];
//     }

//     pangulu_platform_0201000_memcpy(info_pool_d, info_pool_h,
//                                     ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)), 0);

//     // pangulu_platform_0201000_synchronize();
//     // struct timeval start;
//     // pangulu_time_start(&start);
//     ssssm_batched_cuda_dynamic<<<h_task_block_ptr[ntask - 1], PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(
//         nb,
//         ntask,
//         dd_rowptrc,
//         dd_colidxc,
//         dd_valuec,
//         dd_rowptrb,
//         dd_colidxb,
//         dd_valueb,
//         dd_rowptra,
//         dd_colidxa,
//         dd_valuea,
//         d_task_block_ptr);
//     pangulu_platform_0201000_synchronize();
//     // inner_kernel_time += pangulu_time_stop(&start);
//     for (int idense = 0; idense < dense_task_idx; idense++)
//     {
//         int itask = dense_task_indeces[idense];
//         csc_add_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(
//             nb,
//             tasks[itask].opdst->d_columnpointer,
//             tasks[itask].opdst->d_rowindex,
//             tasks[itask].opdst->d_value,
//             hd_valuec[itask]);
//     }
//     pangulu_platform_0201000_synchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err)
//     {
//         printf("error2 : %s\n", cudaGetErrorString(err));
//     }

// #undef PANGULU_REMALLOC_HOST
// #undef PANGULU_REMALLOC_DEVICE
// }

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
#define PANGULU_REMALLOC_HOST(ptr, type)     \
    ptr = (type)(info_pool_h + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;
#define PANGULU_REMALLOC_DEVICE(ptr, type)   \
    ptr = (type)(info_pool_d + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;

    if (task_pointer_buf_capacity < ntask)
    {
        if (info_pool_h)
        {
            pangulu_free(__FILE__, __LINE__, info_pool_h);
        }
        info_pool_h = (char *)pangulu_malloc(__FILE__, __LINE__,
                                             ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)));
        if (info_pool_d)
        {
            pangulu_platform_0201000_free(info_pool_d);
        }
        pangulu_platform_0201000_malloc((void **)&(info_pool_d),
                                        ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)));

        task_pointer_buf_capacity = ntask;
    }

    unsigned long long pool_offset = 0;
    PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
    PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);

    pool_offset = 0;
    PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
    PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);

    double dense_threshold = 0.65;

    pangulu_uint64_t dense_task_idx = 0;
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
        {
            dense_task_idx += 1;
        }
    }
    if (dense_task_idx > dense_task_indeces_cap)
    {
        dense_task_indeces_cap = dense_task_idx;
        dense_task_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, dense_task_indeces, sizeof(pangulu_uint64_t) * dense_task_indeces_cap);

        dense_buffer_block_cap = 3 * dense_task_indeces_cap;
        if (d_dense_buffer)
        {
            cudaFree(d_dense_buffer);
        }
        cudaMalloc(&d_dense_buffer, sizeof(calculate_type) * nb * nb * dense_buffer_block_cap);
        if (!d_dense_buffer)
        {
            printf("cudaMalloc error NULL, allocationg %lld B\n", nb * nb * dense_buffer_block_cap);
        }
    }

    // printf("dense_task_idx=%lld\n", dense_task_idx);

    dense_task_idx = 0;
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        if (tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
        {
            if (d_general_dense_columnpointer == NULL)
            {
                pangulu_inblock_ptr *h_general_dense_columnpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
                pangulu_inblock_idx *h_general_dense_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nb * nb));
                for (int col = 0; col <= nb; col++)
                {
                    h_general_dense_columnpointer[col] = col * nb;
                }
                for (int idx = 0; idx < nb * nb; idx++)
                {
                    h_general_dense_rowindex[idx] = idx % nb;
                }
                cudaMalloc(&d_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1));
                cudaMalloc(&d_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb));
                cudaMemcpy(d_general_dense_columnpointer, h_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1), cudaMemcpyHostToDevice);
                cudaMemcpy(d_general_dense_rowindex, h_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb), cudaMemcpyHostToDevice);
                pangulu_free(__FILE__, __LINE__, h_general_dense_columnpointer);
                pangulu_free(__FILE__, __LINE__, h_general_dense_rowindex);
            }

            store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op1->d_columnpointer, tasks[i].op1->d_rowindex, tasks[i].op1->d_value, d_dense_buffer + ((3 * dense_task_idx) * nb * nb));
            store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op2->d_columnpointer, tasks[i].op2->d_rowindex, tasks[i].op2->d_value, d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb));
            clear_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb));
            pangulu_platform_0201000_synchronize();
            cudaError_t err = cudaGetLastError();
            if (err)
            {
                printf("error : %s\n", cudaGetErrorString(err));
            }

            hd_rowptrb[i] = d_general_dense_columnpointer;
            hd_colidxb[i] = d_general_dense_rowindex;
            hd_valueb[i] = d_dense_buffer + ((3 * dense_task_idx) * nb * nb);

            hd_rowptra[i] = d_general_dense_columnpointer;
            hd_colidxa[i] = d_general_dense_rowindex;
            hd_valuea[i] = d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb);

            hd_rowptrc[i] = d_general_dense_columnpointer;
            hd_colidxc[i] = d_general_dense_rowindex;
            hd_valuec[i] = d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb);

            pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, (nb * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM));
            pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
            pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);

            h_task_block_ptr[i] = need_block;

            dense_task_indeces[dense_task_idx] = i;
            dense_task_idx++;
        }
        else
        {
            hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
            hd_colidxc[i] = tasks[i].opdst->d_rowindex;
            hd_valuec[i] = tasks[i].opdst->d_value;

            hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
            hd_colidxb[i] = tasks[i].op1->d_rowindex;
            hd_valueb[i] = tasks[i].op1->d_value;

            hd_rowptra[i] = tasks[i].op2->d_columnpointer;
            hd_colidxa[i] = tasks[i].op2->d_rowindex;
            hd_valuea[i] = tasks[i].op2->d_value;

            pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_SSSSM_BATCHED_THREADPERBLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
            pangulu_int32_t how_many_col_each_block_can_process = PANGULU_SSSSM_BATCHED_THREADPERBLOCK / how_many_thread_a_col_need;
            pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
            h_task_block_ptr[i] = need_block;
        }
    }

    for (int i = 1; i < ntask; i++)
    {
        h_task_block_ptr[i] += h_task_block_ptr[i - 1];
    }

    pangulu_platform_0201000_memcpy(
        info_pool_d, info_pool_h,
        ntask * (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t)),
        0);

    // pangulu_platform_0201000_synchronize();
    // struct timeval start;
    // pangulu_time_start(&start);
    ssssm_batched_cuda_dynamic<<<h_task_block_ptr[ntask - 1], PANGULU_SSSSM_BATCHED_THREADPERBLOCK>>>(
        nb,
        ntask,
        dd_rowptrc,
        dd_colidxc,
        dd_valuec,
        dd_rowptrb,
        dd_colidxb,
        dd_valueb,
        dd_rowptra,
        dd_colidxa,
        dd_valuea,
        d_task_block_ptr);
    pangulu_platform_0201000_synchronize();
    // inner_kernel_time += pangulu_time_stop(&start);
    for (int idense = 0; idense < dense_task_idx; idense++)
    {
        int itask = dense_task_indeces[idense];
        csc_add_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(
            nb,
            tasks[itask].opdst->d_columnpointer,
            tasks[itask].opdst->d_rowindex,
            tasks[itask].opdst->d_value,
            hd_valuec[itask]);
    }
    pangulu_platform_0201000_synchronize();
    cudaError_t err = cudaGetLastError();
    if (err)
    {
        printf("error2 : %s\n", cudaGetErrorString(err));
    }

#undef PANGULU_REMALLOC_HOST
#undef PANGULU_REMALLOC_DEVICE
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
#define PANGULU_REMALLOC_HOST(ptr, type)     \
    ptr = (type)(info_pool_h + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;
#define PANGULU_REMALLOC_DEVICE(ptr, type)   \
    ptr = (type)(info_pool_d + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;
#define PANGULU_HYBRID_PARAM_SIZE \
    (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) \
        + sizeof(pangulu_int32_t) \
        + sizeof(pangulu_int32_t) \
        + sizeof(calculate_type *) \
        + sizeof(pangulu_int32_t *) \
        + sizeof(pangulu_inblock_ptr *) * 4 \
        + sizeof(pangulu_inblock_idx *) * 2)

    if (task_pointer_buf_capacity < ntask)
    {
        if (info_pool_h)
        {
            pangulu_free(__FILE__, __LINE__, info_pool_h);
        }
        info_pool_h = (char *)pangulu_malloc(__FILE__, __LINE__, ntask * PANGULU_HYBRID_PARAM_SIZE);
        if (info_pool_d)
        {
            pangulu_platform_0201000_free(info_pool_d);
        }
        pangulu_platform_0201000_malloc((void **)&(info_pool_d), ntask * PANGULU_HYBRID_PARAM_SIZE);

        task_pointer_buf_capacity = ntask;
    }

    unsigned long long pool_offset = 0;
    PANGULU_REMALLOC_HOST(h_task_types, pangulu_int32_t *);
    PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);
    PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_getrf_tag_double, calculate_type **);
    PANGULU_REMALLOC_HOST(hd_getrf_nnzu, pangulu_int32_t**);
    PANGULU_REMALLOC_HOST(hd_getrf_csccolptrl_upperbound, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_getrf_cscrowidxl_upperbound, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_getrf_csccolptru_upperbound, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_getrf_cscrowidxu_upperbound, pangulu_inblock_idx **);
    PANGULU_REMALLOC_HOST(hd_tstrf_a_valueidx, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_HOST(hd_tstrf_l_valueidx, pangulu_inblock_ptr **);

    pool_offset = 0;
    PANGULU_REMALLOC_DEVICE(d_task_types, pangulu_int32_t *);
    PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);
    PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_getrf_tag_double, calculate_type **);
    PANGULU_REMALLOC_DEVICE(dd_getrf_nnzu, pangulu_int32_t**);
    PANGULU_REMALLOC_DEVICE(dd_getrf_csccolptrl_upperbound, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_getrf_cscrowidxl_upperbound, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_getrf_csccolptru_upperbound, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_getrf_cscrowidxu_upperbound, pangulu_inblock_idx **);
    PANGULU_REMALLOC_DEVICE(dd_tstrf_a_valueidx, pangulu_inblock_ptr **);
    PANGULU_REMALLOC_DEVICE(dd_tstrf_l_valueidx, pangulu_inblock_ptr **);

    double dense_threshold = 0.8;

    pangulu_uint64_t dense_task_idx = 0;
    pangulu_uint64_t getrf_idx = 0;
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        h_task_types[i] = tasks[i].kernel_id;
        // printf("%lld : %hd\n", i, h_task_types[i]);
        if (tasks[i].kernel_id == PANGULU_TASK_SSSSM && tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
        {
            dense_task_idx += 1;
        }
        if(tasks[i].kernel_id == PANGULU_TASK_GETRF){
            getrf_idx++;
        }
    }

    if (dense_task_idx > dense_task_indeces_cap)
    {
        dense_task_indeces_cap = dense_task_idx;
        dense_task_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, dense_task_indeces, sizeof(pangulu_uint64_t) * dense_task_indeces_cap);

        dense_buffer_block_cap = 3 * dense_task_indeces_cap;
        if (d_dense_buffer)
        {
            cudaFree(d_dense_buffer);
        }
        cudaMalloc(&d_dense_buffer, sizeof(calculate_type) * nb * nb * dense_buffer_block_cap);
        if (!d_dense_buffer)
        {
            printf("cudaMalloc error NULL (1), allocationg %lld B\n", sizeof(calculate_type) * nb * nb * dense_buffer_block_cap);
        }
    }
    if(getrf_idx > getrf_buffer_cap){
        getrf_indeces_cap = getrf_idx;
        getrf_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, getrf_indeces, sizeof(pangulu_uint64_t) * getrf_indeces_cap);

        getrf_buffer_cap = getrf_indeces_cap;
        if (d_getrf_tag_buffer)
        {
            cudaFree(d_getrf_tag_buffer);
        }
        cudaMalloc(&d_getrf_tag_buffer, sizeof(calculate_type) * nb * nb * getrf_buffer_cap);
        if (!d_getrf_tag_buffer)
        {
            printf("cudaMalloc error NULL (2), allocationg %lld B\n", sizeof(calculate_type) * nb * nb * getrf_buffer_cap);
        }
    }

    // printf("dense_task_idx=%lld\n", dense_task_idx);

    dense_task_idx = 0;
    getrf_idx = 0;
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        if (tasks[i].kernel_id == PANGULU_TASK_SSSSM && tasks[i].op1->columnpointer[nb] >= dense_threshold * nb * nb)
        {
            if (d_general_dense_columnpointer == NULL)
            {
                pangulu_inblock_ptr *h_general_dense_columnpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
                pangulu_inblock_idx *h_general_dense_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nb * nb));
                for (int col = 0; col <= nb; col++)
                {
                    h_general_dense_columnpointer[col] = col * nb;
                }
                for (int idx = 0; idx < nb * nb; idx++)
                {
                    h_general_dense_rowindex[idx] = idx % nb;
                }
                cudaMalloc(&d_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1));
                cudaMalloc(&d_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb));
                cudaMemcpy(d_general_dense_columnpointer, h_general_dense_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1), cudaMemcpyHostToDevice);
                cudaMemcpy(d_general_dense_rowindex, h_general_dense_rowindex, sizeof(pangulu_inblock_idx) * (nb * nb), cudaMemcpyHostToDevice);
                pangulu_free(__FILE__, __LINE__, h_general_dense_columnpointer);
                pangulu_free(__FILE__, __LINE__, h_general_dense_rowindex);
            }

            store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op1->d_columnpointer, tasks[i].op1->d_rowindex, tasks[i].op1->d_value, d_dense_buffer + ((3 * dense_task_idx) * nb * nb));
            store_csc_to_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op2->d_columnpointer, tasks[i].op2->d_rowindex, tasks[i].op2->d_value, d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb));
            clear_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(nb, d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb));
            pangulu_platform_0201000_synchronize();
            cudaError_t err = cudaGetLastError();
            if (err)
            {
                printf("error : %s\n", cudaGetErrorString(err));
            }

            hd_rowptrb[i] = d_general_dense_columnpointer;
            hd_colidxb[i] = d_general_dense_rowindex;
            hd_valueb[i] = d_dense_buffer + ((3 * dense_task_idx) * nb * nb);

            hd_rowptra[i] = d_general_dense_columnpointer;
            hd_colidxa[i] = d_general_dense_rowindex;
            hd_valuea[i] = d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb);

            hd_rowptrc[i] = d_general_dense_columnpointer;
            hd_colidxc[i] = d_general_dense_rowindex;
            hd_valuec[i] = d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb);

            pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(TROJAN_HORSE_THREAD_PER_BLOCK, (nb * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM));
            pangulu_int32_t how_many_col_each_block_can_process = TROJAN_HORSE_THREAD_PER_BLOCK / how_many_thread_a_col_need;
            pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);

            h_task_block_ptr[i] = need_block;

            dense_task_indeces[dense_task_idx] = i;
            dense_task_idx++;
        }
        else
        {
            hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
            hd_colidxc[i] = tasks[i].opdst->d_rowindex;
            hd_valuec[i] = tasks[i].opdst->d_value;

            if(tasks[i].kernel_id == PANGULU_TASK_TSTRF){
                hd_rowptrc[i] = tasks[i].opdst->d_rowpointer;
                hd_colidxc[i] = tasks[i].opdst->d_columnindex;
                hd_valuec[i] = tasks[i].opdst->d_value;

                hd_rowptrb[i] = tasks[i].op1->d_rowpointer;
                hd_colidxb[i] = tasks[i].op1->d_columnindex;
                hd_valueb[i] = tasks[i].op1->d_value;

                hd_tstrf_a_valueidx[i] = tasks[i].opdst->d_idx_of_csc_value_for_csr;
                hd_tstrf_l_valueidx[i] = tasks[i].op1->d_idx_of_csc_value_for_csr;
            }else{
                if(tasks[i].kernel_id != PANGULU_TASK_GETRF){
                    hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
                    hd_colidxb[i] = tasks[i].op1->d_rowindex;
                    hd_valueb[i] = tasks[i].op1->d_value;
                }
                if(tasks[i].kernel_id == PANGULU_TASK_SSSSM){
                    hd_rowptra[i] = tasks[i].op2->d_columnpointer;
                    hd_colidxa[i] = tasks[i].op2->d_rowindex;
                    hd_valuea[i] = tasks[i].op2->d_value;
                }

                if(tasks[i].kernel_id == PANGULU_TASK_GETRF){
                    hd_getrf_nnzu[i] = tasks[i].opdst->d_nnzu;
                    hd_getrf_csccolptrl_upperbound[i] = tasks[i].opdst->d_csccolptrl_upperbound;
                    hd_getrf_cscrowidxl_upperbound[i] = tasks[i].opdst->d_cscrowidxl_upperbound;
                    hd_getrf_csccolptru_upperbound[i] = tasks[i].opdst->d_csccolptru_upperbound;
                    hd_getrf_cscrowidxu_upperbound[i] = tasks[i].opdst->d_cscrowidxu_upperbound;
    
                    hd_getrf_tag_double[i] = d_getrf_tag_buffer + getrf_idx * nb * nb;
                    pangulu_load_dense<<<nb, TROJAN_HORSE_THREAD_PER_BLOCK>>>(
                        nb, 
                        tasks[i].opdst->d_columnpointer, 
                        tasks[i].opdst->d_rowindex, 
                        tasks[i].opdst->d_value, 
                        hd_getrf_tag_double[i]
                    );
                    getrf_indeces[getrf_idx] = i;
                    getrf_idx++;
                }
            }
            


            if(tasks[i].kernel_id == PANGULU_TASK_GETRF){
                h_task_block_ptr[i] = PANGULU_ICEIL(nb, TROJAN_HORSE_THREAD_PER_BLOCK/PANGULU_WARP_SIZE);
            }else if(tasks[i].kernel_id == PANGULU_TASK_TSTRF){
                h_task_block_ptr[i] = nb;
            }else if(tasks[i].kernel_id == PANGULU_TASK_GESSM){
                h_task_block_ptr[i] = nb;
            }else if(tasks[i].kernel_id == PANGULU_TASK_SSSSM){
                pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(TROJAN_HORSE_THREAD_PER_BLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM);
                pangulu_int32_t how_many_col_each_block_can_process = TROJAN_HORSE_THREAD_PER_BLOCK / how_many_thread_a_col_need;
                pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
                h_task_block_ptr[i] = need_block;
            }
        }
    }

    for (int i = 1; i < ntask; i++)
    {
        h_task_block_ptr[i] += h_task_block_ptr[i - 1];
    }

    pangulu_platform_0201000_memcpy(
        info_pool_d, info_pool_h,
        ntask * PANGULU_HYBRID_PARAM_SIZE,
        0);

    // pangulu_platform_0201000_synchronize();
    // struct timeval start;
    // pangulu_time_start(&start);
    trojan_horse_batched_kernel_cuda<<<h_task_block_ptr[ntask - 1], TROJAN_HORSE_THREAD_PER_BLOCK>>>(
        nb,
        ntask,
        d_task_types,
        d_task_block_ptr,
        dd_rowptrc,
        dd_colidxc,
        dd_valuec,
        dd_rowptrb,
        dd_colidxb,
        dd_valueb,
        dd_rowptra,
        dd_colidxa,
        dd_valuea,
        dd_getrf_tag_double,
        dd_getrf_nnzu,
        dd_getrf_csccolptrl_upperbound,
        dd_getrf_cscrowidxl_upperbound,
        dd_getrf_csccolptru_upperbound,
        dd_getrf_cscrowidxu_upperbound,
        dd_tstrf_a_valueidx,
        dd_tstrf_l_valueidx);
    
    pangulu_platform_0201000_synchronize();
    // inner_kernel_time += pangulu_time_stop(&start);
    for (int idense = 0; idense < dense_task_idx; idense++)
    {
        int itask = dense_task_indeces[idense];
        csc_add_dense<<<nb, PANGULU_SSSSM_DATAMOV_THREADPERBLOCK>>>(
            nb,
            tasks[itask].opdst->d_columnpointer,
            tasks[itask].opdst->d_rowindex,
            tasks[itask].opdst->d_value,
            hd_valuec[itask]);
    }
    for(int i_getrf = 0; i_getrf < getrf_idx; i_getrf++){
        int itask = getrf_indeces[i_getrf];
        pangulu_store_dense<<<nb, TROJAN_HORSE_THREAD_PER_BLOCK>>>(
            nb, 
            tasks[itask].opdst->d_columnpointer, 
            tasks[itask].opdst->d_rowindex, 
            tasks[itask].opdst->d_value, 
            d_getrf_tag_buffer + i_getrf * nb * nb
        );
    }
    for(int i=0;i<ntask;i++){
        if(tasks[i].kernel_id != PANGULU_TASK_SSSSM){
            pangulu_cuda_download_block(nb, tasks[i].opdst);
        }
    }
    pangulu_platform_0201000_synchronize();
    cudaError_t err = cudaGetLastError();
    if (err)
    {
        printf("error2 : %s\n", cudaGetErrorString(err));
    }

#undef PANGULU_REMALLOC_HOST
#undef PANGULU_REMALLOC_DEVICE
}

#else

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
}
void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
}

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

#endif

void pangulu_platform_0201000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *a,
    calculate_type *x,
    calculate_type *y)
{
    // for(int i=0;i<x->row;i++){
    //     for(int j=((i==0)?0:a->columnpointer[i]);j<a->columnpointer[i+1];j++){
    //         printf("%6.2lf ", a->value[j]);
    //     }
    //     printf("\n");
    // }
    // for(int i=0;i<x->row; i++){
    //     printf("%6.2lf ", x->value[i]);
    // }
    // printf("\n");
    // for(int i=0;i<y->row; i++){
    //     printf("%6.2lf ", y->value[i]);
    // }
    // printf("\n");

    // printf("spmv\n");
    if (nb > 0)
    {
        for (int idx = 0; idx < a->columnpointer[1]; idx++)
        {
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[0];
        }
    }
    for (int col = 1; col < nb; col++)
    {
        for (int idx = a->columnpointer[col]; idx < a->columnpointer[col + 1]; idx++)
        {
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[col];
        }
    }

    // for(int i=0;i<y->row; i++){
    //     printf("%6.2lf ", y->value[i]);
    // }
    // printf("\n");
}

void pangulu_platform_0201000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval,
    calculate_type *xval)
{
    // printf("vecadd\n");
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        bval[i] += xval[i];
    }
}

void pangulu_platform_0201000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type *xval,
    pangulu_int64_t uplo)
{
    pangulu_int64_t col = nb;
    pangulu_inblock_ptr *csc_column_ptr_tmp = s->columnpointer;
    pangulu_inblock_idx *csc_row_idx_tmp = s->rowindex;
    volatile calculate_type *cscVal_tmp = s->value;

    // for(int i=0;i<nb;i++){
    //     for(int j=s->columnpointer[i];j<s->columnpointer[i+1];j++){
    //         printf("%6.2lf ", s->value[j]);
    //     }
    //     printf("\n");
    // }
    // for(int i=0; i<nb; i++){
    //     printf("%f ", b->value[i]);
    // }
    // printf("\n");

    if (uplo == 0)
    {
        // printf("sptrsv_lower kernel\n");
        for (pangulu_int64_t i = 0; i < col; i++)
        {
            pangulu_int32_t diag_idx = binarysearch_inblk(
                csc_row_idx_tmp,
                (i == 0) ? 0 : csc_column_ptr_tmp[i],
                csc_column_ptr_tmp[i + 1],
                i);
            if (diag_idx == -1)
            {
                xval[i] = 0.0;
                continue;
            }
            for (pangulu_int64_t j = diag_idx + 1; j < csc_column_ptr_tmp[i + 1]; j++)
            {
                pangulu_inblock_idx row = csc_row_idx_tmp[j];
                xval[row] -= cscVal_tmp[j] * xval[i];
            }
        }
    }
    else
    {
        // printf("sptrsv_upper kernel\n");
        for (pangulu_int64_t i = col - 1; i >= 0; i--)
        {
            pangulu_int32_t diag_idx = binarysearch_inblk(
                csc_row_idx_tmp,
                (i == 0) ? 0 : csc_column_ptr_tmp[i],
                csc_column_ptr_tmp[i + 1],
                i);
            if (diag_idx != -1)
            {
                if (fabs(cscVal_tmp[diag_idx]) > PANGULU_SPTRSV_TOL)
                    xval[i] = xval[i] / cscVal_tmp[diag_idx];
                else
                    xval[i] = xval[i] / PANGULU_SPTRSV_TOL;
            }
            else
            {
                xval[i] = 0.0;
                continue;
            }
            for (pangulu_int64_t j = diag_idx - 1; j >= ((i == 0) ? 0 : csc_column_ptr_tmp[i]); j--)
            {
                pangulu_inblock_idx row = csc_row_idx_tmp[j];
                xval[row] -= cscVal_tmp[j] * xval[i];
            }
        }
    }
    // for(int i=0; i<nb; i++){
    //     printf("%f ", x->value[i]);
    // }
    // printf("\n");
}

#undef PANGULU_GPU_OPDST
#undef PANGULU_GPU_OP1
#undef PANGULU_GPU_OP2
