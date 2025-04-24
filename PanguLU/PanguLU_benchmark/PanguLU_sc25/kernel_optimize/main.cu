#define PANGULU_PLATFORM_ENV
#define GPU_OPEN
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <random>
#include <algorithm>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUDA_LAST_ERROR() check_cuda_last_error(__FILE__, __LINE__)

#define PANGULU_GPU_OPDST 0
#define PANGULU_GPU_OP1 1
#define PANGULU_GPU_OP2 2

#define PANGULU_SSSSM_BATCHED_THREADPERBLOCK 256
#define PANGULU_SSSSM_BATCHED_SHAREDMEM_LEN 256
#define PANGULU_SSSSM_BATCHED_THREAD_PER_ELEM 1
#define PANGULU_SSSSM_DATAMOV_THREADPERBLOCK 128

#include "../src/platforms/02_NONSHAREDMEM/01_GPU/000_CUDA/pangulu_platform_0201000.h"

#define NB 1000         // Block size
#define BATCH_SIZE 50 // Number of tasks in batch

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

char *info_pool_h = NULL;
char *info_pool_d = NULL;
pangulu_uint64_t dense_task_indeces_cap = 0;
pangulu_uint64_t *dense_task_indeces = NULL;
pangulu_int32_t dense_buffer_block_cap = 0;
calculate_type *d_dense_buffer = NULL;
pangulu_inblock_ptr *d_general_dense_columnpointer = NULL;
pangulu_inblock_idx *d_general_dense_rowindex = NULL;

// pangulu_storage_slot_t *h_matrices_A[BATCH_SIZE];
// pangulu_storage_slot_t *h_matrices_B[BATCH_SIZE];
// pangulu_storage_slot_t *h_matrices_C[BATCH_SIZE];
// pangulu_storage_slot_t *d_matrices_A[BATCH_SIZE];
// pangulu_storage_slot_t *d_matrices_B[BATCH_SIZE];
// pangulu_storage_slot_t *d_matrices_C[BATCH_SIZE];

pangulu_int32_t *h_task_block_ptr;
pangulu_int32_t *d_task_block_ptr;

void *pangulu_malloc(const char* file, pangulu_int64_t line, pangulu_int64_t size)
{
    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    if (malloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        exit(1);
    }
    // memset(malloc_address, 0, size);
    return malloc_address;
}

void *pangulu_realloc(const char* file, pangulu_int64_t line, void* oldptr, pangulu_int64_t size)
{
    void *realloc_address = NULL;
    realloc_address = (void *)realloc(oldptr, size);
    if (realloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        exit(1);
    }
    return realloc_address;
}

void pangulu_free(const char* file, pangulu_int64_t line, void* ptr){
    if(ptr==NULL){
        return;
    }
    free(ptr);
}

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
}

__device__ pangulu_int32_t get_task_id(
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

__device__ pangulu_inblock_ptr binarysearch_inblk_cuda(
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
}


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
                    // atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                    d_valuec[flag] -= vala * d_valueb[j];
                }
            }
        }
    }
}

// Generate random sparse matrix in CSC format on host
pangulu_storage_slot_t *generate_random_sparse_matrix(pangulu_inblock_idx nb, float sparsity)
{
    pangulu_storage_slot_t *matrix = (pangulu_storage_slot_t *)malloc(sizeof(pangulu_storage_slot_t));
    int nnz_max = nb * nb * sparsity;
    matrix->columnpointer = (pangulu_inblock_ptr *)malloc(sizeof(pangulu_inblock_ptr) * (nb + 1));
    matrix->rowindex = (pangulu_inblock_idx *)malloc(sizeof(pangulu_inblock_idx) * nnz_max);
    matrix->value = (calculate_type *)malloc(sizeof(calculate_type) * nnz_max);
    if (!matrix->columnpointer || !matrix->rowindex || !matrix->value)
    {
        fprintf(stderr, "Failed to allocate sparse matrix\n");
        exit(1);
    }

    int current_col = 0;
    int nnz = 0;
    char break_flag = 0;
    for (int col = 0; col < nb; ++col)
    {
        matrix->columnpointer[col] = nnz;
        for (int row = 0; row < nb; ++row)
        {
            if (rand() % 100 < sparsity * 100)
            {
                matrix->rowindex[nnz] = row;
                matrix->value[nnz] = (calculate_type)((rand() % 1000) / 500.0 - 1.0);
                nnz++;
                if(nnz >= nnz_max){
                    break_flag = 1;
                    break;
                }
            }
        }
        if(break_flag){
            break;
        }
    }
    matrix->columnpointer[nb] = nnz;
    return matrix;
}

void free_sparse_matrix_host(pangulu_storage_slot_t *matrix)
{
    free(matrix->columnpointer);
    free(matrix->rowindex);
    free(matrix->value);
    free(matrix);
}

void free_sparse_matrix_device(pangulu_storage_slot_t d_matrix)
{
    cudaFree(d_matrix.d_columnpointer);
    cudaFree(d_matrix.d_rowindex);
    cudaFree(d_matrix.d_value);
}

// CPU batched sparse gemm (naive for correctness check)
void cpu_batched_gemm(pangulu_inblock_idx nb, pangulu_uint64_t ntask, pangulu_task_t *tasks, pangulu_storage_slot_t *h_matrices_A[], pangulu_storage_slot_t *h_matrices_B[], pangulu_storage_slot_t *h_matrices_C[])
{
    for (int task_id = 0; task_id < ntask; ++task_id)
    {
        pangulu_storage_slot_t *matA = h_matrices_A[task_id];
        pangulu_storage_slot_t *matB = h_matrices_B[task_id];
        pangulu_storage_slot_t *matC = h_matrices_C[task_id];

        // Initialize C to zero (dense matrix)
        for (int i = 0; i < nb * nb; ++i)
        {
            matC->value[i] = 0.0;
        }

        // Perform batched gemm on CPU (naive implementation)
        for (int row = 0; row < nb; ++row)
        {
            for (int k = 0; k < nb; ++k)
            {
                // Get value from A(row, k)
                calculate_type a_val = 0.0;
                for (int idx_a = matA->columnpointer[k]; idx_a < matA->columnpointer[k + 1]; ++idx_a)
                {
                    if (matA->rowindex[idx_a] == row)
                    {
                        a_val = matA->value[idx_a];
                        break;
                    }
                }

                if (a_val != 0.0)
                {
                    for (int col = 0; col < nb; ++col)
                    {
                        // Get value from B(k, col)
                        calculate_type b_val = 0.0;
                        for (int idx_b = matB->columnpointer[col]; idx_b < matB->columnpointer[col + 1]; ++idx_b)
                        {
                            if (matB->rowindex[idx_b] == k)
                            {
                                b_val = matB->value[idx_b];
                                break;
                            }
                        }
                        matC->value[row * nb + col] -= a_val * b_val; // C = C - A * B
                    }
                }
            }
        }
    }
}

// Copy sparse matrix from host to device
void copy_sparse_matrix_to_device(pangulu_inblock_idx nb, pangulu_storage_slot_t *h_matrix)
{
    pangulu_inblock_ptr nnz = h_matrix->columnpointer[nb];
    cudaMalloc(&h_matrix->d_columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1));
    cudaMalloc(&h_matrix->d_rowindex, sizeof(pangulu_inblock_idx) * nnz);
    cudaMalloc(&h_matrix->d_value, sizeof(calculate_type) * nnz);
    if (!h_matrix->d_columnpointer || !h_matrix->d_rowindex || !h_matrix->d_value)
    {
        fprintf(stderr, "CUDA malloc failed for sparse matrix\n");
        exit(1);
    }
    cudaMemcpy(h_matrix->d_columnpointer, h_matrix->columnpointer, sizeof(pangulu_inblock_ptr) * (nb + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(h_matrix->d_rowindex, h_matrix->rowindex, sizeof(pangulu_inblock_idx) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(h_matrix->d_value, h_matrix->value, sizeof(calculate_type) * nnz, cudaMemcpyHostToDevice);
}


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

    // double dense_threshold = 0.8;
    double dense_threshold = 1.2;

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
        if(!d_dense_buffer){
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

int main()
{
    srand(time(NULL));
    // 1. generate a group of tasks, uploading to GPU
    pangulu_inblock_idx nb = NB;
    pangulu_uint64_t ntask = BATCH_SIZE;
    pangulu_task_t *tasks = (pangulu_task_t *)malloc(sizeof(pangulu_task_t) * ntask);
    if (!tasks)
    {
        fprintf(stderr, "Failed to allocate tasks\n");
        return 1;
    }

    pangulu_storage_slot_t *h_matrices_A[BATCH_SIZE];
    pangulu_storage_slot_t *h_matrices_B[BATCH_SIZE];
    pangulu_storage_slot_t *h_matrices_C[BATCH_SIZE];

    float sparsity_A = 0.1;
    float sparsity_B = 0.1;

    for (int i = 0; i < ntask; ++i)
    {
        h_matrices_A[i] = generate_random_sparse_matrix(nb, sparsity_A);
        h_matrices_B[i] = generate_random_sparse_matrix(nb, sparsity_B);
        h_matrices_C[i] = generate_random_sparse_matrix(nb, 1.0); // Dense C for simplicity, will be cleared in kernel
        copy_sparse_matrix_to_device(nb, h_matrices_A[i]);
        copy_sparse_matrix_to_device(nb, h_matrices_B[i]);
        copy_sparse_matrix_to_device(nb, h_matrices_C[i]);

        tasks[i].op1 = h_matrices_B[i];   // op1 = B
        tasks[i].op2 = h_matrices_A[i];   // op2 = A
        tasks[i].opdst = h_matrices_C[i]; // opdst = C
    }

    // 2. call kernel pangulu_platform_0201000_ssssm_batched() and record the time
    cudaEvent_t start, stop;
    float elapsed_time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    pangulu_platform_0201000_ssssm_batched(nb, ntask, tasks);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", elapsed_time_ms);

    // // 3. check the correctness on CPU.
    // pangulu_storage_slot_t h_matrices_C_cpu[BATCH_SIZE]; // Create a copy for CPU results
    // for (int i = 0; i < ntask; ++i)
    //     h_matrices_C_cpu[i] = h_matrices_C[i];
    // cpu_batched_gemm(nb, ntask, tasks, h_matrices_A, h_matrices_B, h_matrices_C_cpu);

    // // Download GPU results to host for comparison
    // pangulu_storage_slot_t h_matrices_C_gpu[BATCH_SIZE];
    // for (int i = 0; i < ntask; ++i)
    //     h_matrices_C_gpu[i] = h_matrices_C[i];
    // for (int i = 0; i < ntask; ++i)
    // {
    //     cudaMemcpy(h_matrices_C_gpu[i].value, d_matrices_C[i].d_value, sizeof(calculate_type) * nb * nb, cudaMemcpyDeviceToHost);
    // }

    // // Compare CPU and GPU results (simple norm check for demonstration)
    // double norm_diff = 0.0;
    // for (int i = 0; i < ntask; ++i)
    // {
    //     for (int j = 0; j < nb * nb; ++j)
    //     {
    //         norm_diff += pow(h_matrices_C_cpu[i].value[j] - h_matrices_C_gpu[i].value[j], 2);
    //     }
    // }
    // norm_diff = sqrt(norm_diff);
    // printf("Frobenius norm of difference between CPU and GPU results: %.3f\n", norm_diff);
    // if (norm_diff < 1e-6)
    // {
    //     printf("Correctness check passed!\n");
    // }
    // else
    // {
    //     printf("Correctness check failed!\n");
    // }

    // // 4. calculate the GFLOPs of the GPU kernel.
    // long long effective_flops = 0;
    // for (int i = 0; i < ntask; ++i)
    // {
    //     effective_flops += 2LL * h_matrices_A[i].nnz * nb;
    // }
    // double gflops = (double)effective_flops / (elapsed_time_ms * 1e-6 * 1e9);
    // printf("GFLOPs: %.3f\n", gflops);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // free(tasks);
    // for (int i = 0; i < ntask; ++i)
    // {
    //     free_sparse_matrix_host(h_matrices_A[i]);
    //     free_sparse_matrix_host(h_matrices_B[i]);
    //     free_sparse_matrix_host(h_matrices_C[i]);
    //     free_sparse_matrix_device(d_matrices_A[i]);
    //     free_sparse_matrix_device(d_matrices_B[i]);
    //     free_sparse_matrix_device(d_matrices_C[i]);
    // }

    return 0;
}
