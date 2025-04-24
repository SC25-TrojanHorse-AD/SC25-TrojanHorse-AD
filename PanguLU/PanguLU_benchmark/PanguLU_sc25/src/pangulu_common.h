#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

#ifdef GPU_OPEN
#define PANGULU_DEFAULT_PLATFORM PANGULU_PLATFORM_GPU_CUDA
#define PANGULU_NONSHAREDMEM
#else
#define PANGULU_DEFAULT_PLATFORM PANGULU_PLATFORM_CPU_NAIVE
#endif

#if defined(CALCULATE_TYPE_CR64)
#define calculate_type double _Complex
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE_COMPLEX
#define PANGULU_COMPLEX
#elif defined(CALCULATE_TYPE_R64)
#define calculate_type double
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE
#elif defined(CALCULATE_TYPE_CR32)
#define calculate_type float _Complex
#define calculate_real_type float
#define MPI_VAL_TYPE MPI_C_FLOAT_COMPLEX
#define PANGULU_COMPLEX
#elif defined(CALCULATE_TYPE_R32)
#define calculate_type float
#define calculate_real_type float
#define MPI_VAL_TYPE MPI_FLOAT
#else
#define calculate_type double
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE
#endif

typedef long long int pangulu_int64_t;
#define MPI_PANGULU_INT64_T MPI_LONG_LONG_INT
#define FMT_PANGULU_INT64_T "%lld"
typedef unsigned long long int pangulu_uint64_t;
#define MPI_PANGULU_UINT64_T MPI_UNSIGNED_LONG_LONG
#define FMT_PANGULU_UINT64_T "%llu"
typedef int pangulu_int32_t;
#define MPI_PANGULU_INT32_T MPI_INT
#define FMT_PANGULU_INT32_T "%d"
typedef unsigned int pangulu_uint32_t;
#define MPI_PANGULU_UINT32_T MPI_UNSIGNED
#define FMT_PANGULU_UINT32_T "%u"
typedef short int pangulu_int16_t;
#define MPI_PANGULU_INT16_T MPI_SHORT
#define FMT_PANGULU_INT16_T "%hd"
typedef unsigned short int pangulu_uint16_t;
#define MPI_PANGULU_UINT16_T MPI_UNSIGNED_SHORT
#define FMT_PANGULU_UINT16_T "%hu"

typedef pangulu_uint64_t pangulu_exblock_ptr;
#define MPI_PANGULU_EXBLOCK_PTR MPI_PANGULU_UINT64_T
#define FMT_PANGULU_EXBLOCK_PTR FMT_PANGULU_UINT64_T
typedef pangulu_uint32_t pangulu_exblock_idx;
#define MPI_PANGULU_EXBLOCK_IDX MPI_PANGULU_UINT32_T
#define FMT_PANGULU_EXBLOCK_IDX FMT_PANGULU_UINT32_T
typedef pangulu_uint32_t pangulu_inblock_ptr;
#define MPI_PANGULU_INBLOCK_PTR MPI_PANGULU_UINT32_T
#define FMT_PANGULU_INBLOCK_PTR FMT_PANGULU_UINT32_T
typedef pangulu_uint16_t pangulu_inblock_idx;
#define MPI_PANGULU_INBLOCK_IDX MPI_PANGULU_UINT16_T
#define FMT_PANGULU_INBLOCK_IDX FMT_PANGULU_UINT16_T

typedef pangulu_exblock_ptr sparse_pointer_t;
typedef pangulu_exblock_idx sparse_index_t;
typedef calculate_type sparse_value_t;
typedef calculate_real_type sparse_value_real_t;

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef PANGULU_PLATFORM_ENV
#include <mpi.h>
#endif
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#ifndef PANGULU_PLATFORM_ENV
#include <cblas.h>
#endif
#include <getopt.h>
#include <omp.h>
#include <sys/resource.h>
#include "../include/pangulu.h"
#include "./languages/pangulu_en_us.h"

#ifndef PANGULU_PLATFORM_ENV
#include "sc25.h"
#endif

#ifndef PANGULU_PLATFORM_ENV
#ifdef METIS
#include <metis.h>
#else
typedef int idx_t;
#endif
#endif

#define PANGULU_ICEIL(a, b) (((a) + (b) - 1) / (b))
#define PANGULU_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PANGULU_MAX(a, b) ((a) > (b) ? (a) : (b))
#define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x
#define PANGULU_DIGINFO_OFFSET_STOREIDX (0)
#define PANGULU_DIGINFO_OFFSET_NNZ (39)
#define PANGULU_DIGINFO_OFFSET_BINID (61)
#define PANGULU_DIGINFO_MASK_STOREIDX (0x7FFFFFFFFF)
#define PANGULU_DIGINFO_MASK_NNZ (0x3FFFFF)
#define PANGULU_DIGINFO_MASK_BINID (0x7)
#define PANGULU_DIGINFO_SET_STOREIDX(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_STOREIDX) << PANGULU_DIGINFO_OFFSET_STOREIDX)
#define PANGULU_DIGINFO_SET_NNZ(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_NNZ) << PANGULU_DIGINFO_OFFSET_NNZ)
#define PANGULU_DIGINFO_SET_BINID(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_BINID) << PANGULU_DIGINFO_OFFSET_BINID)
#define PANGULU_DIGINFO_GET_STOREIDX(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_STOREIDX) & PANGULU_DIGINFO_MASK_STOREIDX)
#define PANGULU_DIGINFO_GET_NNZ(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_NNZ) & PANGULU_DIGINFO_MASK_NNZ)
#define PANGULU_DIGINFO_GET_BINID(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_BINID) & PANGULU_DIGINFO_MASK_BINID)
#define PANGULU_TASK_GETRF 1
#define PANGULU_TASK_TSTRF 2
#define PANGULU_TASK_GESSM 3
#define PANGULU_TASK_SSSSM 4
#define PANGULU_TASK_SPTRSV_L 11
#define PANGULU_TASK_SPTRSV_U 12
#define PANGULU_TASK_SPMV_L 13
#define PANGULU_TASK_SPMV_U 14
#define PANGULU_LOWER 0
#define PANGULU_UPPER 1
#define PANGULU_DATA_INVALID 0
#define PANGULU_DATA_PREPARING 1 // 同进程在更新是PREPARING，不同进程在接收是PREPARING
#define PANGULU_DATA_READY 2 // 块已经是最终数据
#define PANGULU_TOL 1e-16
#define PANGULU_SPTRSV_TOL 1e-16
#define PANGULU_CALC_RANK(row, col, p, q) (((row) % (p)) * (q) + ((col) % (q)))
#define PANGULU_CALC_OFFSET(offset_init, now_level, PQ_length) \
    (((offset_init) - (now_level) % (PQ_length)) < 0) ? ((offset_init) - (now_level) % (PQ_length) + (PQ_length)) : ((offset_init) - (now_level) % (PQ_length))

typedef struct pangulu_stat_t{
    double time_getrf;
    double time_tstrf;
    double time_gessm;
    double time_ssssm;
}pangulu_stat_t;
extern pangulu_stat_t global_stat;

typedef struct pangulu_common
{
    pangulu_int32_t rank;
    pangulu_int32_t size;
    pangulu_exblock_idx n;
    pangulu_inblock_idx nb;
    pangulu_int32_t sum_rank_size;
    pangulu_int32_t omp_thread;
    pangulu_int32_t p;
    pangulu_int32_t q;
} pangulu_common;

typedef struct pangulu_origin_smatrix
{
    pangulu_exblock_idx column;
    pangulu_exblock_idx row;
    pangulu_exblock_ptr nnz;
    pangulu_exblock_ptr *rowpointer;
    pangulu_exblock_idx *columnindex;
    calculate_type *value;
    pangulu_exblock_ptr *columnpointer;
    pangulu_exblock_idx *rowindex;
    calculate_type *value_csc;
    pangulu_exblock_ptr *csc_to_csr_index;
} pangulu_origin_smatrix;

typedef struct pangulu_bsem_t
{
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    pangulu_int32_t v;
} pangulu_bsem_t;

typedef struct pangulu_storage_slot_t{
    // const void* raw_data;
    // pangulu_int64_t data_capacity;

    pangulu_exblock_idx brow_pos;
    pangulu_exblock_idx bcol_pos;
    pangulu_inblock_ptr* columnpointer;
    pangulu_inblock_idx* rowindex;
    calculate_type* value;
    // pangulu_uint64_t slot_addr;
    volatile char data_status; // PANGULU_DATA_INVALID, PANGULU_DATA_PREPARING, PANGULU_DATA_READY 
    // char data_status; // PANGULU_DATA_INVALID, PANGULU_DATA_PREPARING, PANGULU_DATA_READY 
#ifdef PANGULU_NONSHAREDMEM
    pangulu_inblock_ptr* d_columnpointer;
    pangulu_inblock_idx* d_rowindex;
    calculate_type* d_value;
    pangulu_inblock_ptr* d_rowpointer;
    pangulu_inblock_idx* d_columnindex;
    pangulu_inblock_ptr* d_idx_of_csc_value_for_csr;
    char have_csr_data;

    pangulu_inblock_ptr *d_csccolptrl_upperbound;
    pangulu_inblock_idx *d_cscrowidxl_upperbound;
    pangulu_inblock_ptr *d_csccolptru_upperbound;
    pangulu_inblock_idx *d_cscrowidxu_upperbound;
    pangulu_int32_t *d_nnzu;
#endif
}pangulu_storage_slot_t;

typedef struct pangulu_task_t
{
    pangulu_exblock_idx row;
    pangulu_exblock_idx col;
    pangulu_int16_t kernel_id;
    pangulu_exblock_idx task_level;
    pangulu_int64_t compare_flag;
    pangulu_storage_slot_t* opdst;
    pangulu_storage_slot_t* op1;
    pangulu_storage_slot_t* op2;
} pangulu_task_t;

typedef struct pangulu_task_queue_t
{
    pangulu_int64_t length;
    pangulu_int64_t capacity;
    pangulu_int64_t *task_index_heap;
    pangulu_int64_t *task_storage_avail_queue;
    pangulu_int64_t task_storage_avail_queue_head;
    pangulu_int64_t task_storage_avail_queue_tail;
    pangulu_task_t *task_storage;
    pangulu_int32_t cmp_strategy;
    
    // pangulu_int64_t nnz_flag;
    pangulu_bsem_t *heap_bsem;
} pangulu_task_queue_t;

// typedef struct pangulu_storage_queue_t{

// }pangulu_storage_queue_t;

typedef struct pangulu_storage_bin_t{
    pangulu_storage_slot_t* slots;
    pangulu_int64_t slot_capacity;
    pangulu_int32_t slot_count;

    pangulu_int32_t* avail_slot_queue;
    pangulu_int32_t queue_head;
    pangulu_int32_t queue_tail;
}pangulu_storage_bin_t;

typedef struct pangulu_storage_t{
    pangulu_int32_t n_bin;
    pangulu_storage_bin_t* bins;
    pthread_mutex_t* mutex;
}pangulu_storage_t;

typedef struct pangulu_vector
{
    calculate_type *value;
    pangulu_int64_t row;
} pangulu_vector;

typedef struct pangulu_block_smatrix
{
    pangulu_exblock_idx *row_perm;
    pangulu_exblock_idx *col_perm;
    pangulu_exblock_idx *metis_perm;
    calculate_type *row_scale;
    calculate_type *col_scale;
    pangulu_exblock_ptr* symbolic_rowpointer;
    pangulu_exblock_idx* symbolic_columnindex;
    pangulu_int64_t rank_remain_task_count; // reuse with SpTRSV
    pangulu_int64_t rank_remain_recv_block_count; // reuse with SpTRSV
    pangulu_bsem_t *run_bsem1;
    pangulu_task_queue_t *heap;
    pangulu_exblock_ptr symbolic_nnz;
    pangulu_storage_t* storage;
    pthread_mutex_t* info_mutex;
    char* sent_rank_flag; // 记录当前块已经发过的进程，避免重复发送

    pangulu_exblock_ptr* bcsr_related_pointer;
    pangulu_exblock_idx* bcsr_related_index;
    pangulu_exblock_ptr* bcsr_index_bcsc;
    
    pangulu_exblock_ptr* bcsc_related_pointer;
    pangulu_exblock_idx* bcsc_related_index;
    pangulu_uint64_t* bcsc_related_draft_info;
    pangulu_int32_t* bcsc_remain_task_count;

    pangulu_exblock_ptr* bcsc_pointer;
    pangulu_exblock_idx* bcsc_index;
    pangulu_exblock_ptr *bcsc_blknnzptr;

    pangulu_int64_t sc25_batch_tileid_capacity;
    unsigned long long* sc25_batch_tileid;

    calculate_type* A_rowsum_reordered;

    // sptrsv
    pangulu_vector **big_row_vector;
    pangulu_vector **big_col_vector;
    char *diagonal_flag;
    pangulu_exblock_ptr *l_row_task_nnz;
    pangulu_exblock_ptr *l_col_task_nnz;
    pangulu_exblock_ptr *u_row_task_nnz;
    pangulu_exblock_ptr *u_col_task_nnz;
    pangulu_task_queue_t *sptrsv_heap;
    pangulu_vector *save_vector;
    char *l_send_flag;
    char *u_send_flag;
    pangulu_exblock_ptr *l_sptrsv_task_columnpointer;
    pangulu_exblock_idx *l_sptrsv_task_rowindex;
    pangulu_exblock_ptr *u_sptrsv_task_columnpointer;
    pangulu_exblock_idx *u_sptrsv_task_rowindex;

    // new_sptrsv
    volatile pangulu_int32_t* rhs_remain_task_count;
    volatile pangulu_int32_t* rhs_remain_recv_count;
    calculate_type* rhs;
    calculate_type* recv_buffer;
    // pangulu_uint16_t* sptrsv_lower_rank_prio;
    // pangulu_uint16_t* sptrsv_lower_rank_index;
    // pangulu_uint16_t* sptrsv_upper_rank_prio;
    // pangulu_uint16_t* sptrsv_upper_rank_index;

} pangulu_block_smatrix;

typedef struct pangulu_block_common
{
    pangulu_int32_t rank;
    pangulu_int32_t p;
    pangulu_int32_t q;
    pangulu_inblock_idx nb;
    pangulu_exblock_idx n;
    pangulu_exblock_idx block_length;
    pangulu_int32_t sum_rank_size;

    // sptrsv
    pangulu_int32_t rank_row_length;
    pangulu_int32_t rank_col_length;
} pangulu_block_common;

typedef struct pangulu_numeric_thread_param
{
    pangulu_common* pangulu_common;
    pangulu_block_common *block_common;
    pangulu_block_smatrix *block_smatrix;
} pangulu_numeric_thread_param;


typedef struct pangulu_digest_coo_t{
    pangulu_exblock_idx row;
    pangulu_exblock_idx col;
    pangulu_inblock_ptr nnz;
}pangulu_digest_coo_t;

typedef struct pangulu_handle_t
{
    pangulu_block_common *block_common;
    pangulu_block_smatrix *block_smatrix;
    pangulu_common *commmon;
} pangulu_handle_t;

typedef struct pangulu_symbolic_node_t
{
    pangulu_int64_t value;
    struct pangulu_symbolic_node_t *next;
} pangulu_symbolic_node_t;

#ifdef __cplusplus
extern "C" {
#endif

// pangulu_communication.c
#ifndef PANGULU_PLATFORM_ENV
void pangulu_cm_rank(pangulu_int32_t* rank);
void pangulu_cm_size(pangulu_int32_t* size);
void pangulu_cm_sync();
void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root);
void pangulu_cm_distribute_csc_to_distcsc(
    pangulu_int32_t root_rank,
    int rootproc_free_originmatrix,
    pangulu_exblock_idx* n,
    pangulu_inblock_idx rowchunk_align,
    pangulu_int32_t* distcsc_nproc,
    pangulu_exblock_idx* n_loc,
    
    pangulu_exblock_ptr** distcsc_proc_nnzptr,
    pangulu_exblock_ptr** distcsc_pointer,
    pangulu_exblock_idx** distcsc_index,
    calculate_type** distcsc_value
);
void pangulu_cm_distribute_distcsc_to_distbcsc(
    int rootproc_free_originmatrix,
    int malloc_distbcsc_value,
    pangulu_exblock_idx n_glo,
    pangulu_exblock_idx n_loc,
    pangulu_inblock_idx block_order,
    
    pangulu_exblock_ptr* distcsc_proc_nnzptr,
    pangulu_exblock_ptr* distcsc_pointer,
    pangulu_exblock_idx* distcsc_index,
    calculate_type* distcsc_value,

    pangulu_exblock_ptr** bcsc_struct_pointer,
    pangulu_exblock_idx** bcsc_struct_index,
    pangulu_exblock_ptr** bcsc_struct_nnzptr,
    pangulu_inblock_ptr*** bcsc_inblock_pointers,
    pangulu_inblock_idx*** bcsc_inblock_indeces,
    calculate_type*** bcsc_values
);
void pangulu_cm_recv_block(
    MPI_Status* msg_stat,
    pangulu_storage_t* storage,
    pangulu_uint64_t slot_addr,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx* bcol_pos,
    pangulu_exblock_idx* brow_pos,
    pangulu_exblock_ptr* bcsc_related_pointer,
    pangulu_exblock_idx* bcsc_related_index,
    pangulu_uint64_t* bcsc_related_draft_info
);
void pangulu_cm_isend_block(
    pangulu_storage_slot_t* slot,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx brow_pos,
    pangulu_exblock_idx bcol_pos,
    pangulu_int32_t target_rank
);
void pangulu_cm_probe(MPI_Status *status);
void pangulu_cm_sync();
void pangulu_cm_isend(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
void pangulu_cm_recv(char* buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
#endif
// pangulu_communication.c end

// pangulu_conversion.c
void pangulu_convert_csr_to_csc(
    int free_csrmatrix,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** csr_pointer,
    pangulu_exblock_idx** csr_index,
    calculate_type** csr_value,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index,
    calculate_type** csc_value
);
void pangulu_convert_halfsymcsc_to_csc_struct(
    int free_halfmatrix,
    int if_colsort,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** half_csc_pointer,
    pangulu_exblock_idx** half_csc_index,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index
);
void pangulu_convert_bcsc_fill_value_to_struct(
    int free_valuebcsc,
    pangulu_exblock_idx n,
    pangulu_inblock_idx nb,

    pangulu_exblock_ptr* value_bcsc_struct_pointer,
    pangulu_exblock_idx* value_bcsc_struct_index,
    pangulu_inblock_ptr* value_bcsc_struct_nnzptr,
    pangulu_inblock_ptr** value_bcsc_inblock_pointers,
    pangulu_inblock_idx** value_bcsc_inblock_indeces,
    calculate_type** value_bcsc_values,

    pangulu_exblock_ptr* struct_bcsc_struct_pointer,
    pangulu_exblock_idx* struct_bcsc_struct_index,
    pangulu_inblock_ptr* struct_bcsc_struct_nnzptr,
    pangulu_inblock_ptr** struct_bcsc_inblock_pointers,
    pangulu_inblock_idx** struct_bcsc_inblock_indeces,
    calculate_type** struct_bcsc_values
);
void pangulu_convert_bcsc_to_digestcoo(
    pangulu_exblock_idx block_length,
    const pangulu_exblock_ptr* bcsc_struct_pointer,
    const pangulu_exblock_idx* bcsc_struct_index,
    const pangulu_exblock_ptr* bcsc_struct_nnzptr,
    pangulu_digest_coo_t* digest_info
);
// pangulu_conversion.c end

// pangulu_lifecycle.c
void pangulu_init_pangulu_vector(pangulu_vector *b, pangulu_int64_t n);
pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *v);
// pangulu_lifecycle.c end

// pangulu_memory.c
void *pangulu_malloc(const char* file, pangulu_int64_t line, pangulu_int64_t size);
void *pangulu_realloc(const char* file, pangulu_int64_t line, void* oldptr, pangulu_int64_t size);
void pangulu_free(const char* file, pangulu_int64_t line, void* ptr);
void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a);
// pangulu_memory.c end

// pangulu_numeric.c
void* pangulu_numeric_compute_thread(void *param);
void pangulu_numeric(
    pangulu_common* common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
);
// pangulu_numeric.c end

// pangulu_preprocessing.c
void pangulu_preprocessing(
    pangulu_common* common,
    pangulu_block_common* bcommon,
    pangulu_block_smatrix* bsmatrix,
    pangulu_origin_smatrix* reorder_matrix,
    pangulu_int32_t nthread
);
// pangulu_preprocessing.c end

// pangulu_reordering.c
#ifdef PANGULU_MC64
void pangulu_mc64dd(
    pangulu_int64_t col, 
    pangulu_int64_t n, 
    pangulu_int64_t *queue, 
    const calculate_type *row_scale_value, 
    pangulu_int64_t *save_tmp
);
void pangulu_mc64ed(
    pangulu_int64_t *queue_length, 
    pangulu_int64_t n, 
    pangulu_int64_t *queue, 
    const calculate_type *row_scale_value, 
    pangulu_int64_t *save_tmp
);
void pangulu_mc64fd(
    pangulu_int64_t loc_origin, 
    pangulu_int64_t *queue_length, 
    pangulu_int64_t n, 
    pangulu_int64_t *queue, 
    const calculate_type *row_scale_value, 
    pangulu_int64_t *save_tmp
);
void pangulu_mc64(
    pangulu_origin_smatrix *s, 
    pangulu_exblock_idx **perm, 
    pangulu_exblock_idx **iperm,
    calculate_type **row_scale, 
    calculate_type **col_scale
);
#endif
void pangulu_reorder_vector_b_tran(
    pangulu_exblock_idx *row_perm,
    pangulu_exblock_idx *metis_perm,
    calculate_type *row_scale,
    pangulu_vector *B_origin,
    pangulu_vector *B_trans
);
#ifndef PANGULU_PLATFORM_ENV
#ifdef METIS
void pangulu_get_graph_struct(pangulu_origin_smatrix *s, idx_t **xadj_address, idx_t **adjincy_address);
void pangulu_metis(pangulu_origin_smatrix *a, idx_t **metis_perm);
#endif
#endif
void pangulu_origin_smatrix_transport_transport_iperm(
    pangulu_origin_smatrix *s, 
    pangulu_origin_smatrix *new_S, 
    const pangulu_exblock_idx *metis_perm
);
void pangulu_reordering(
    pangulu_block_smatrix *block_smatrix,
    pangulu_origin_smatrix *origin_matrix,
    pangulu_origin_smatrix *reorder_matrix
);
// pangulu_reordering.c end

// pangulu_thread.c
void bind_to_core(int core);
void pangulu_mutex_init(pthread_mutex_t *mutex);
void pangulu_bsem_init(pangulu_bsem_t *bsem_p, pangulu_int64_t value);
pangulu_bsem_t *pangulu_bsem_destory(pangulu_bsem_t *bsem_p);
void pangulu_bsem_post(pangulu_task_queue_t *heap);
pangulu_int64_t pangulu_bsem_wait(pangulu_task_queue_t *heap);
void pangulu_bsem_stop(pangulu_task_queue_t *heap);
void pangulu_bsem_synchronize(pangulu_bsem_t *bsem_p);
// pangulu_thread.c end

// pangulu_utils.c
void pangulu_init_pangulu_origin_smatrix(pangulu_origin_smatrix* s);
void pangulu_init_pangulu_block_smatrix(pangulu_block_smatrix* bs);
void pangulu_time_start(struct timeval* start);
double pangulu_time_stop(struct timeval* start);
void pangulu_add_diagonal_element(pangulu_origin_smatrix *s);
void pangulu_kvsort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end);
void pangulu_sort_pangulu_origin_smatrix(pangulu_origin_smatrix *s);
void swap_index_1(pangulu_exblock_idx *a, pangulu_exblock_idx *b);
void swap_value(calculate_type *a, calculate_type *b);
pangulu_int64_t binarysearch(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);
// pangulu_utils.c end

void pangulu_task_queue_push(
    pangulu_task_queue_t *heap, 
    pangulu_int64_t row, 
    pangulu_int64_t col, 
    pangulu_int64_t task_level, 
    pangulu_int64_t kernel_id, 
    pangulu_int64_t compare_flag,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* op1,
    pangulu_storage_slot_t* op2,
    pangulu_int64_t block_length,
    const char* file,
    int line
);
pangulu_storage_slot_t*
pangulu_storage_get_slot(
    pangulu_storage_t* storage,
    pangulu_uint64_t slot_addr
);
pangulu_int32_t
pangulu_storage_slot_queue_alloc(
    pangulu_storage_bin_t* bin
);
pangulu_uint64_t
pangulu_storage_allocate_slot(
    pangulu_storage_t* storage,
    pangulu_int64_t size
);

pangulu_task_t pangulu_task_queue_pop(pangulu_task_queue_t *heap);
pangulu_task_t pangulu_task_queue_delete(pangulu_task_queue_t *heap);

#ifndef PANGULU_PLATFORM_ENV
void pangulu_cm_recv_block(
    MPI_Status* msg_stat,
    pangulu_storage_t* storage,
    pangulu_uint64_t slot_addr,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx* bcol_pos,
    pangulu_exblock_idx* brow_pos,
    pangulu_exblock_ptr* bcsc_related_pointer,
    pangulu_exblock_idx* bcsc_related_index,
    pangulu_uint64_t* bcsc_related_draft_info
);
#endif

void pangulu_cm_isend_block(
    pangulu_storage_slot_t* slot,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx brow_pos,
    pangulu_exblock_idx bcol_pos,
    pangulu_int32_t target_rank
);

void pangulu_storage_slot_queue_recycle(
    pangulu_storage_t* storage,
    pangulu_uint64_t* slot_addr
);

void pangulu_kvsort2(
    pangulu_exblock_idx *key, 
    pangulu_uint64_t *val, 
    pangulu_int64_t start, 
    pangulu_int64_t end
);

#ifndef PANGULU_PLATFORM_ENV
void pangulu_cm_probe(MPI_Status *status);
void pangulu_cm_sync();
void pangulu_cm_isend(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
void pangulu_cm_recv(char* buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
#endif
pangulu_int32_t binarysearch_inblk(const pangulu_inblock_idx *arr, pangulu_int32_t left, pangulu_int32_t right, pangulu_int32_t target);

void pangulu_sptrsv(
    pangulu_block_common* block_common,
    pangulu_block_smatrix* block_smatrix
);
void pangulu_sptrsv_vector_gather(
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix,
    pangulu_vector *vector
);
void pangulu_reorder_vector_x_tran(
    pangulu_block_smatrix *block_smatrix,
    pangulu_vector *X_origin,
    pangulu_vector *X_trans
);
void pangulu_sptrsv_preprocessing(
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix,
    pangulu_vector *vector
);

void pangulu_storage_init(
    pangulu_storage_t* storage,
    pangulu_int64_t* slot_capacity,
    pangulu_int32_t* slot_count,

    pangulu_exblock_idx block_length,
    pangulu_exblock_ptr* bcsc_pointer,
    pangulu_exblock_idx* bcsc_index,
    pangulu_exblock_ptr* bcsc_blknnzptr,
    pangulu_inblock_ptr** bcsc_inblk_pointers,
    pangulu_inblock_idx** bcsc_inblk_indeces,
    calculate_type** bcsc_inblk_values,
    pangulu_inblock_idx nb
);
void pangulu_task_queue_init(pangulu_task_queue_t *heap, pangulu_int64_t capacity);

void pangulu_task_queue_cmp_strategy(
    pangulu_task_queue_t* tq,
    pangulu_int32_t cmp_strategy
);

pangulu_int64_t binarysearch_first_ge(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);
pangulu_int64_t binarysearch_last_le(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);

void pangulu_sptrsv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type* xval,
    pangulu_int64_t uplo
);
void pangulu_spmv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* a,
    calculate_type* x,
    calculate_type* y
);
#ifndef PANGULU_PLATFORM_ENV
void pangulu_sptrsv_receive_message(
    MPI_Status status,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
);
#endif
void pangulu_vecadd_interface(
    pangulu_int64_t length,
    calculate_type *bval, 
    calculate_type *xval
);
void pangulu_task_queue_clear(pangulu_task_queue_t *heap);
void pangulu_numeric_work_batched(
    pangulu_int64_t ntask,
    pangulu_task_t* tasks,
    pangulu_common* common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
);
void pangulu_numeric_work(
    pangulu_task_t *task,
    pangulu_common* common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
);
char pangulu_task_queue_empty(pangulu_task_queue_t *heap);

void pangulu_numeric_check(
    pangulu_common* common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
);

// void pangulu_transpose_struct_inblock(
//     const pangulu_inblock_idx nb,
//     const pangulu_inblock_ptr* in_ptr,
//     const pangulu_inblock_idx* in_idx,
//     pangulu_inblock_ptr* out_ptr,
//     pangulu_inblock_idx* out_idx,
//     pangulu_inblock_ptr* aid_ptr
// );

void pangulu_transpose_struct_with_valueidx_inblock(
    const pangulu_inblock_idx nb,
    const pangulu_inblock_ptr* in_ptr,
    const pangulu_inblock_idx* in_idx,
    pangulu_inblock_ptr* out_ptr,
    pangulu_inblock_idx* out_idx,
    pangulu_inblock_ptr* out_valueidx,
    pangulu_inblock_ptr* aid_ptr
);

#ifdef __cplusplus
} // extern "C"
#endif

#include "./platforms/pangulu_platform_common.h"
#endif
