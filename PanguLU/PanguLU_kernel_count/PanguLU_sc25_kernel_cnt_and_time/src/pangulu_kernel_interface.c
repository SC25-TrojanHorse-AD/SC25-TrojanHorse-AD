#include "pangulu_common.h"

extern unsigned long long kernel_count;

void pangulu_getrf_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_platform_getrf(nb, opdst, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_getrf += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_tstrf_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_platform_tstrf(nb, opdst, opdiag, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_tstrf += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_gessm_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_platform_gessm(nb, opdst, opdiag, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_gessm += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_ssssm_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_platform_ssssm(nb, opdst, op1, op2, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_ssssm += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_hybrid_batched_interface(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_platform_hybrid_batched(nb, ntask, tasks, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_ssssm += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_ssssm_batched_interface(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
    struct timeval start;
    pangulu_time_start(&start);
    // pangulu_platform_ssssm_batched(nb, ntask, tasks, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_hybrid_batched(nb, ntask, tasks, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
    global_stat.time_ssssm += pangulu_time_stop(&start);
    kernel_count++;
}

void pangulu_spmv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *a,
    calculate_type *x,
    calculate_type *y)
{
    pangulu_platform_spmv(nb, a, x, y, PANGULU_DEFAULT_PLATFORM);
}

void pangulu_vecadd_interface(
    pangulu_int64_t length,
    calculate_type *bval,
    calculate_type *xval)
{
    pangulu_platform_vecadd(length, bval, xval, PANGULU_DEFAULT_PLATFORM);
}

void pangulu_sptrsv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type *xval,
    pangulu_int64_t uplo)
{
    pangulu_platform_sptrsv(nb, s, xval, uplo, PANGULU_DEFAULT_PLATFORM);
}
