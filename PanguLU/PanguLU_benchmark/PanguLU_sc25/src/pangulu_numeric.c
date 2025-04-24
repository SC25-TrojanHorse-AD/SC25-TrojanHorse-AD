#include "pangulu_common.h"

pangulu_task_t *working_task_buf = NULL;
pangulu_int64_t working_task_buf_capacity = 0;

void pangulu_numerical_receive_message(MPI_Status status,
                                       pangulu_block_common *block_common,
                                       pangulu_block_smatrix *block_smatrix)
{

    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    pangulu_exblock_idx bcol_pos = 0;
    pangulu_exblock_idx brow_pos = 0;

    int fetch_size;
    MPI_Get_count(&status, MPI_CHAR, &fetch_size);
    pangulu_uint64_t slot_addr = pangulu_storage_allocate_slot(storage, fetch_size);
    pangulu_cm_recv_block(&status, storage, slot_addr, nb, &bcol_pos, &brow_pos, bcsc_related_pointer, bcsc_related_index, bcsc_related_draft_info);

    pthread_mutex_lock(block_smatrix->info_mutex);
    pangulu_storage_get_slot(storage, slot_addr)->data_status = PANGULU_DATA_READY;
    pangulu_storage_slot_t *slot_recv = pangulu_storage_get_slot(storage, slot_addr);
    pangulu_exblock_idx level = PANGULU_MIN(bcol_pos, brow_pos);

    if (bcol_pos == brow_pos)
    {
        for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], level) + 1;
             bidx < bcsc_related_pointer[level + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (level % q);
            if (target_rank == rank)
            {
                if (bcsc_remain_task_count[bidx] == 1)
                {
                    bcsc_remain_task_count[bidx]--;
                    pangulu_task_queue_push(heap, brow, level, level, PANGULU_TASK_TSTRF, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), slot_recv, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
        for (pangulu_exblock_ptr bidx = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], level) + 1;
             bidx < bcsr_related_pointer[level + 1]; bidx++)
        {
            pangulu_exblock_idx bcol = bcsr_related_index[bidx];
            pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                if (bcsc_remain_task_count[bcsr_index_bcsc[bidx]] == 1)
                {
                    bcsc_remain_task_count[bcsr_index_bcsc[bidx]]--;
                    pangulu_task_queue_push(heap, level, bcol, level, PANGULU_TASK_GESSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx]]), slot_recv, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
    }
    else if (brow_pos > bcol_pos)
    {
        pangulu_exblock_ptr bidx_csr_diag = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], bcol_pos);
        if (bidx_csr_diag == 0xFFFFFFFFFFFFFFFF)
        {
            bidx_csr_diag = bcsr_related_pointer[level];
        }
        for (pangulu_exblock_ptr bidx_csr = binarysearch(bcsr_related_index, bcsr_related_pointer[brow_pos], bcsr_related_pointer[brow_pos + 1], bcol_pos) + 1;
             bidx_csr < bcsr_related_pointer[brow_pos + 1]; bidx_csr++)
        {
            pangulu_exblock_idx bcol = bcsr_related_index[bidx_csr];
            pangulu_int32_t target_rank = (brow_pos % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                while ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] < bcol))
                {
                    bidx_csr_diag++;
                }
                if ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] == bcol))
                {
                    pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
                    if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr]]--;
                        pangulu_task_queue_push(heap, brow_pos, bcol, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr]]), slot_recv, ssssm_op2, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
    }
    else
    {
        pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], brow_pos);
        if (bidx_diag == 0xFFFFFFFFFFFFFFFF)
        {
            bidx_diag = bcsc_related_pointer[level];
        }
        for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_pos], bcsc_related_pointer[bcol_pos + 1], brow_pos) + 1;
             bidx < bcsc_related_pointer[bcol_pos + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (bcol_pos % q);
            if (target_rank == rank)
            {
                while ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] < brow))
                {
                    bidx_diag++;
                }
                if ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] == brow))
                {
                    pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                    if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bidx]--;
                        pangulu_task_queue_push(heap, brow, bcol_pos, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), ssssm_op1, slot_recv, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);
}

int pangulu_sc25_batch_ssssm_callback(
    unsigned long long ntask,
    void *_task_descriptors,
    void *_extra_params)
{
    pangulu_task_t *tasks = (pangulu_task_t *)_task_descriptors;
    pangulu_numeric_thread_param *extra_params = (pangulu_numeric_thread_param *)_extra_params;
    pangulu_common *common = extra_params->pangulu_common;
    pangulu_block_common *block_common = extra_params->block_common;
    pangulu_block_smatrix *block_smatrix = extra_params->block_smatrix;

    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    // #pragma omp parallel for num_threads(common->omp_thread) schedule(guided)
    // for(pangulu_uint64_t i = 0; i < ntask; i++){
    //     int tid = omp_get_thread_num();
    //     // int tid = i%common->omp_thread;
    //     pangulu_ssssm_interface(nb, tasks[i].opdst, tasks[i].op1, tasks[i].op2, tid);
    // }

    // pangulu_ssssm_batched_interface(nb, ntask, tasks);
    pangulu_hybrid_batched_interface(nb, ntask, tasks);

    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        pangulu_task_t *task = &tasks[i];
        pangulu_int16_t kernel_id = task->kernel_id;
        pangulu_int64_t brow_task = task->row;
        pangulu_int64_t bcol_task = task->col;
        pangulu_int64_t level = task->task_level;
        if ((task->op1->brow_pos % p) * q + (task->op1->bcol_pos % q) != rank)
        {
            pangulu_exblock_ptr bidx_op1 = binarysearch(bcsc_related_index, bcsc_related_pointer[task->op1->bcol_pos], bcsc_related_pointer[task->op1->bcol_pos + 1], task->op1->brow_pos);
            bcsc_remain_task_count[bidx_op1]--;
            if (bcsc_remain_task_count[bidx_op1] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_op1]);
            }
        }
        if ((task->op2->brow_pos % p) * q + (task->op2->bcol_pos % q) != rank)
        {
            pangulu_exblock_ptr bidx_op2 = binarysearch(bcsc_related_index, bcsc_related_pointer[task->op2->bcol_pos], bcsc_related_pointer[task->op2->bcol_pos + 1], task->op2->brow_pos);
            bcsc_remain_task_count[bidx_op2]--;
            if (bcsc_remain_task_count[bidx_op2] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_op2]);
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);

    return 0;
}

void pangulu_numeric_work_batched(
    pangulu_int64_t ntask,
    pangulu_task_t *tasks,
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    pangulu_int64_t *sc25_batch_tileid_capacity = &block_smatrix->sc25_batch_tileid_capacity;
    unsigned long long **sc25_batch_tileid = &block_smatrix->sc25_batch_tileid;

    if (ntask > *sc25_batch_tileid_capacity)
    {
        *sc25_batch_tileid_capacity = ntask;
        *sc25_batch_tileid = pangulu_realloc(__FILE__, __LINE__, *sc25_batch_tileid, sizeof(unsigned long long) * *sc25_batch_tileid_capacity);
    }
    pangulu_int64_t ntask_no_ssssm = 0;
    for (pangulu_int64_t i = 0; i < ntask; i++)
    {
        if (tasks[i].kernel_id != PANGULU_TASK_SSSSM)
        {
            (*sc25_batch_tileid)[ntask_no_ssssm] = tasks[i].row * block_length + tasks[i].col;
            if (i != ntask_no_ssssm)
            {
                pangulu_task_t tmp = tasks[i];
                tasks[i] = tasks[ntask_no_ssssm];
                tasks[ntask_no_ssssm] = tmp;
            }
            ntask_no_ssssm++;
        }
    }

    pangulu_numeric_thread_param param;
    param.pangulu_common = common;
    param.block_common = block_common;
    param.block_smatrix = block_smatrix;
    sc25_task_compute_multi_tile(NULL, ntask_no_ssssm, *sc25_batch_tileid, pangulu_sc25_batch_ssssm_callback, &param);

    pangulu_task_t* tasks_fetched = NULL;
    // int ntasks_fetched = sc25_get_idle_tasks(NULL, 10, &tasks_fetched);
    int ntasks_fetched = 0;
    // printf("ntask = %d, ntasks_no_ssssm = %d, ntasks_fetched = %d\n", ntask, ntask_no_ssssm, ntasks_fetched);
    while (ntask + ntasks_fetched > working_task_buf_capacity)
    {
        working_task_buf_capacity = (working_task_buf_capacity + ntask + ntasks_fetched) * 2;
        working_task_buf = pangulu_realloc(__FILE__, __LINE__, working_task_buf, sizeof(pangulu_task_t) * working_task_buf_capacity);
        tasks = working_task_buf;
    }
    // memcpy(tasks+ntask_no_ssssm+ntasks_fetched, tasks+ntask_no_ssssm, sizeof(pangulu_task_t)*(ntask - ntask_no_ssssm));
    for(int i = ntask - ntask_no_ssssm; i>=0; i--){
        tasks[ntask_no_ssssm+ntasks_fetched+i] = tasks[ntask_no_ssssm+i];
    }
    memcpy(tasks+ntask_no_ssssm, tasks_fetched, sizeof(pangulu_task_t) * ntasks_fetched);
    ntask_no_ssssm = ntask_no_ssssm + ntasks_fetched; // have ssssm from taskpool
    ntask = ntask + ntasks_fetched;

    // free(tasks_fetched);

    if(ntask_no_ssssm > 0){
        pangulu_hybrid_batched_interface(nb, ntask_no_ssssm, tasks);
    }
    // for(int i=0;i<5;i++){
    //     sc25_idle_work(NULL, pangulu_sc25_batch_ssssm_callback, &param);
    // }

    pangulu_int64_t ntask_fact = 0;
    pangulu_int64_t ntask_trsm = 0;
    for (pangulu_int64_t itask = 0; itask < ntask_no_ssssm; itask++)
    {
        pangulu_task_t *task = &tasks[itask];
        pangulu_int16_t kernel_id = task->kernel_id;
        if (kernel_id == PANGULU_TASK_GETRF)
        {
            ntask_fact++;
        }
        else if (kernel_id == PANGULU_TASK_TSTRF)
        {
            ntask_trsm++;
        }
        else if (kernel_id == PANGULU_TASK_GESSM)
        {
            ntask_trsm++;
        }
    }

    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_int64_t itask = 0; itask < ntask; itask++)
    {
        
        pangulu_task_t *task = &tasks[itask];
        pangulu_int16_t kernel_id = task->kernel_id;
        pangulu_int64_t level = task->task_level;
        pangulu_int64_t brow_task = task->row;
        pangulu_int64_t bcol_task = task->col;
        if((itask < ntask_no_ssssm) && (kernel_id == PANGULU_TASK_SSSSM)){
            continue;
        }

        memset(sent_rank_flag, 0, sizeof(char) * nproc);
        if (kernel_id == PANGULU_TASK_GETRF)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], level) + 1;
                 bidx < bcsc_related_pointer[level + 1]; bidx++)
            {
                pangulu_exblock_idx brow = bcsc_related_index[bidx];
                pangulu_int32_t target_rank = (brow % p) * q + (level % q);
                if (target_rank == rank)
                {
                    if (bcsc_remain_task_count[bidx] == 1)
                    {
                        bcsc_remain_task_count[bidx]--;
                        pangulu_task_queue_push(heap, brow, level, level, PANGULU_TASK_TSTRF, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), task->opdst, NULL, block_length, __FILE__, __LINE__);
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
            for (pangulu_exblock_ptr bidx = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], level) + 1;
                 bidx < bcsr_related_pointer[level + 1]; bidx++)
            {
                pangulu_exblock_idx bcol = bcsr_related_index[bidx];
                pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
                if (target_rank == rank)
                {
                    if (bcsc_remain_task_count[bcsr_index_bcsc[bidx]] == 1)
                    {
                        bcsc_remain_task_count[bcsr_index_bcsc[bidx]]--;
                        pangulu_task_queue_push(heap, level, bcol, level, PANGULU_TASK_GESSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx]]), task->opdst, NULL, block_length, __FILE__, __LINE__);
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_TSTRF)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            pangulu_exblock_ptr bidx_csr_diag = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], bcol_task);
            if ((level % p) * q + (level % q) != rank)
            {
                bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]]--;
                if (bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]] == 0)
                {
                    pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
                }
            }
            for (pangulu_exblock_ptr bidx_csr = binarysearch(bcsr_related_index, bcsr_related_pointer[brow_task], bcsr_related_pointer[brow_task + 1], bcol_task) + 1;
                 bidx_csr < bcsr_related_pointer[brow_task + 1]; bidx_csr++)
            {
                pangulu_exblock_idx bcol = bcsr_related_index[bidx_csr];
                pangulu_int32_t target_rank = (brow_task % p) * q + (bcol % q);
                if (target_rank == rank)
                {
                    while ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] < bcol))
                    {
                        bidx_csr_diag++;
                    }
                    if ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] == bcol))
                    {
                        pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
                        if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                        {
                            bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr]]--;
                            pangulu_task_queue_push(heap, brow_task, bcol, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr]]), task->opdst, ssssm_op2, block_length, __FILE__, __LINE__);
                        }
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_GESSM)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], brow_task);
            if ((level % p) * q + (level % q) != rank)
            {
                bcsc_remain_task_count[bidx_diag]--;
                if (bcsc_remain_task_count[bidx_diag] == 0)
                {
                    pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_diag]);
                }
            }
            for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task) + 1;
                 bidx < bcsc_related_pointer[bcol_task + 1]; bidx++)
            {
                pangulu_exblock_idx brow = bcsc_related_index[bidx];
                pangulu_int32_t target_rank = (brow % p) * q + (bcol_task % q);
                if (target_rank == rank)
                {
                    while ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] < brow))
                    {
                        bidx_diag++;
                    }
                    if ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] == brow))
                    {
                        pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                        if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                        {
                            bcsc_remain_task_count[bidx]--;
                            pangulu_task_queue_push(heap, brow, bcol_task, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), ssssm_op1, task->opdst, block_length, __FILE__, __LINE__);
                        }
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_SSSSM)
        {
            pangulu_exblock_ptr bidx_task = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task);
            pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task)], bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task) + 1], PANGULU_MIN(brow_task, bcol_task));
            if (brow_task == bcol_task)
            {
                if (bcsc_remain_task_count[bidx_task] == 1)
                {
                    bcsc_remain_task_count[bidx_task]--;
                    pangulu_task_queue_push(heap, brow_task, bcol_task, brow_task, PANGULU_TASK_GETRF, brow_task, task->opdst, NULL, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else if (brow_task < bcol_task)
            { // GESSM
                if (bcsc_remain_task_count[bidx_task] == 1)
                {
                    pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                    if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bidx_task]--;
                        pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_GESSM, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            { // TSTRF
                if (bcsc_remain_task_count[bidx_task] == 1)
                {
                    pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                    if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bidx_task]--;
                        pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_TSTRF, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);
}



// void pangulu_numeric_work_batched(
//     pangulu_int64_t ntask,
//     pangulu_task_t *tasks,
//     pangulu_common *common,
//     pangulu_block_common *block_common,
//     pangulu_block_smatrix *block_smatrix)
// {
//     pangulu_int32_t rank = block_common->rank;
//     pangulu_int32_t nproc = block_common->sum_rank_size;
//     pangulu_inblock_idx nb = block_common->nb;
//     pangulu_int32_t p = block_common->p;
//     pangulu_int32_t q = block_common->q;
//     pangulu_task_queue_t *heap = block_smatrix->heap;
//     pangulu_storage_t *storage = block_smatrix->storage;
//     char *sent_rank_flag = block_smatrix->sent_rank_flag;
//     pangulu_exblock_idx block_length = block_common->block_length;

//     pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
//     pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
//     pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
//     pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
//     pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
//     pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
//     pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

//     pangulu_int64_t *sc25_batch_tileid_capacity = &block_smatrix->sc25_batch_tileid_capacity;
//     unsigned long long **sc25_batch_tileid = &block_smatrix->sc25_batch_tileid;

//     if (ntask > *sc25_batch_tileid_capacity)
//     {
//         *sc25_batch_tileid_capacity = ntask;
//         *sc25_batch_tileid = pangulu_realloc(__FILE__, __LINE__, *sc25_batch_tileid, sizeof(unsigned long long) * *sc25_batch_tileid_capacity);
//     }
//     pangulu_int64_t ntask_no_ssssm = 0;
//     for (pangulu_int64_t i = 0; i < ntask; i++)
//     {
//         if (tasks[i].kernel_id != PANGULU_TASK_SSSSM)
//         {
//             (*sc25_batch_tileid)[ntask_no_ssssm] = tasks[i].row * block_length + tasks[i].col;
//             if (i != ntask_no_ssssm)
//             {
//                 pangulu_task_t tmp = tasks[i];
//                 tasks[i] = tasks[ntask_no_ssssm];
//                 tasks[ntask_no_ssssm] = tmp;
//             }
//             ntask_no_ssssm++;
//         }
//     }

//     pangulu_numeric_thread_param param;
//     param.pangulu_common = common;
//     param.block_common = block_common;
//     param.block_smatrix = block_smatrix;
//     // printf("> #%d SSSSM-batched\n", rank);
//     sc25_task_compute_multi_tile(NULL, ntask_no_ssssm, *sc25_batch_tileid, pangulu_sc25_batch_ssssm_callback, &param);
//     // printf("> #%d SSSSM-batched done\n", rank);
//     // sc25_task_compute_multi_tile_threadsafe(NULL, ntask_no_ssssm, *sc25_batch_tileid, pangulu_sc25_batch_ssssm_callback, &param);

// // #pragma omp parallel for num_threads(common->omp_thread)
//     // for (pangulu_int64_t itask = 0; itask < ntask_no_ssssm; itask++)
//     // {
//     //     // int tid = omp_get_thread_num();
//     //     int tid = 0;
//     //     pangulu_task_t *task = &tasks[itask];
//     //     pangulu_int16_t kernel_id = task->kernel_id;
//     //     if (kernel_id == PANGULU_TASK_GETRF)
//     //     {
//     //         // if (task->row % 100 == 0)
//     //             // printf("> #%d GETRF %d\n", rank, task->row);
//     //         // pangulu_getrf_interface(nb, task->opdst, tid);
//     //         pangulu_hybrid_batched_interface(nb, 1, task, PANGULU_DEFAULT_PLATFORM);
//     //         // printf("> #%d GETRF %d done\n", rank, task->row);
//     //     }
//     //     else if (kernel_id == PANGULU_TASK_TSTRF)
//     //     {
//     //         // printf("> #%d TSTRF %d\n", rank, task->row);
//     //         // pangulu_tstrf_interface(nb, task->opdst, task->op1, tid);
//     //         pangulu_hybrid_batched_interface(nb, 1, task, PANGULU_DEFAULT_PLATFORM);
//     //         // printf("> #%d TSTRF %d done\n", rank, task->row);
//     //     }
//     //     else if (kernel_id == PANGULU_TASK_GESSM)
//     //     {
//     //         // printf("> #%d GESSM %d\n", rank, task->row);
//     //         // pangulu_gessm_interface(nb, task->opdst, task->op1, tid);
//     //         pangulu_hybrid_batched_interface(nb, 1, task, PANGULU_DEFAULT_PLATFORM);
//     //         // printf("> #%d GESSM %d done\n", rank, task->row);
//     //     }
//     // }

//     pangulu_task_t* tasks_fetched = NULL;
//     // int ntasks_fetched = sc25_get_idle_tasks(NULL, 2, &tasks_fetched);
//     // printf("ntasks_fetched = %d\n", ntasks_fetched);
//     int ntasks_fetched = 0; // would not fetch low-prio tasks
//     while (ntask + ntasks_fetched > working_task_buf_capacity)
//     {
//         working_task_buf_capacity = (working_task_buf_capacity + ntask + ntasks_fetched) * 2;
//         working_task_buf = pangulu_realloc(__FILE__, __LINE__, working_task_buf, sizeof(pangulu_task_t) * working_task_buf_capacity);
//         tasks = working_task_buf;
//     }
//     memcpy(tasks+ntask_no_ssssm+ntasks_fetched, tasks+ntask_no_ssssm, sizeof(pangulu_task_t)*(ntask - ntask_no_ssssm));
//     memcpy(tasks+ntask_no_ssssm, tasks_fetched, sizeof(pangulu_task_t) * ntasks_fetched);
//     ntask_no_ssssm = ntask_no_ssssm + ntasks_fetched; // have ssssm from taskpool
//     ntask = ntask + ntasks_fetched;

//     if(ntask_no_ssssm > 0){
//         pangulu_hybrid_batched_interface(nb, ntask_no_ssssm, tasks);
//     }
//     // for(int i=0;i<5;i++){
//     //     sc25_idle_work(NULL, pangulu_sc25_batch_ssssm_callback, &param);
//     // }

//     pangulu_int64_t ntask_fact = 0;
//     pangulu_int64_t ntask_trsm = 0;
//     for (pangulu_int64_t itask = 0; itask < ntask_no_ssssm; itask++)
//     {
//         pangulu_task_t *task = &tasks[itask];
//         pangulu_int16_t kernel_id = task->kernel_id;
//         if (kernel_id == PANGULU_TASK_GETRF)
//         {
//             ntask_fact++;
//         }
//         else if (kernel_id == PANGULU_TASK_TSTRF)
//         {
//             ntask_trsm++;
//         }
//         else if (kernel_id == PANGULU_TASK_GESSM)
//         {
//             ntask_trsm++;
//         }
//     }
//     if (ntask_no_ssssm != 0)
//     {
//         // printf("[SC25 LOG] sc25_task_compute1 %lld %lld 0\n", ntask_fact, ntask_trsm);
//     }

//     pthread_mutex_lock(block_smatrix->info_mutex);
//     for (pangulu_int64_t itask = 0; itask < ntask; itask++)
//     {
//         pangulu_task_t *task = &tasks[itask];
//         pangulu_int16_t kernel_id = task->kernel_id;
//         pangulu_int64_t level = task->task_level;
//         pangulu_int64_t brow_task = task->row;
//         pangulu_int64_t bcol_task = task->col;
//         memset(sent_rank_flag, 0, sizeof(char) * nproc);
//         if (kernel_id == PANGULU_TASK_GETRF)
//         {
//             task->opdst->data_status = PANGULU_DATA_READY;
//             for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], level) + 1;
//                  bidx < bcsc_related_pointer[level + 1]; bidx++)
//             {
//                 pangulu_exblock_idx brow = bcsc_related_index[bidx];
//                 pangulu_int32_t target_rank = (brow % p) * q + (level % q);
//                 if (target_rank == rank)
//                 {
//                     if (bcsc_remain_task_count[bidx] == 1)
//                     {
//                         bcsc_remain_task_count[bidx]--;
//                         pangulu_task_queue_push(heap, brow, level, level, PANGULU_TASK_TSTRF, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), task->opdst, NULL, block_length, __FILE__, __LINE__);
//                     }
//                 }
//                 else
//                 {
//                     if (sent_rank_flag[target_rank] == 0)
//                     {
//                         pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
//                         sent_rank_flag[target_rank] = 1;
//                     }
//                 }
//             }
//             for (pangulu_exblock_ptr bidx = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], level) + 1;
//                  bidx < bcsr_related_pointer[level + 1]; bidx++)
//             {
//                 pangulu_exblock_idx bcol = bcsr_related_index[bidx];
//                 pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
//                 if (target_rank == rank)
//                 {
//                     if (bcsc_remain_task_count[bcsr_index_bcsc[bidx]] == 1)
//                     {
//                         bcsc_remain_task_count[bcsr_index_bcsc[bidx]]--;
//                         pangulu_task_queue_push(heap, level, bcol, level, PANGULU_TASK_GESSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx]]), task->opdst, NULL, block_length, __FILE__, __LINE__);
//                     }
//                 }
//                 else
//                 {
//                     if (sent_rank_flag[target_rank] == 0)
//                     {
//                         pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
//                         sent_rank_flag[target_rank] = 1;
//                     }
//                 }
//             }
//         }
//         else if (kernel_id == PANGULU_TASK_TSTRF)
//         {
//             task->opdst->data_status = PANGULU_DATA_READY;
//             pangulu_exblock_ptr bidx_csr_diag = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], bcol_task);
//             if ((level % p) * q + (level % q) != rank)
//             {
//                 bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]]--;
//                 if (bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]] == 0)
//                 {
//                     pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
//                 }
//             }
//             for (pangulu_exblock_ptr bidx_csr = binarysearch(bcsr_related_index, bcsr_related_pointer[brow_task], bcsr_related_pointer[brow_task + 1], bcol_task) + 1;
//                  bidx_csr < bcsr_related_pointer[brow_task + 1]; bidx_csr++)
//             {
//                 pangulu_exblock_idx bcol = bcsr_related_index[bidx_csr];
//                 pangulu_int32_t target_rank = (brow_task % p) * q + (bcol % q);
//                 if (target_rank == rank)
//                 {
//                     while ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] < bcol))
//                     {
//                         bidx_csr_diag++;
//                     }
//                     if ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] == bcol))
//                     {
//                         pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
//                         if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
//                         {
//                             bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr]]--;
//                             pangulu_task_queue_push(heap, brow_task, bcol, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr]]), task->opdst, ssssm_op2, block_length, __FILE__, __LINE__);
//                         }
//                     }
//                 }
//                 else
//                 {
//                     if (sent_rank_flag[target_rank] == 0)
//                     {
//                         pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
//                         sent_rank_flag[target_rank] = 1;
//                     }
//                 }
//             }
//         }
//         else if (kernel_id == PANGULU_TASK_GESSM)
//         {
//             task->opdst->data_status = PANGULU_DATA_READY;
//             pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], brow_task);
//             if ((level % p) * q + (level % q) != rank)
//             {
//                 bcsc_remain_task_count[bidx_diag]--;
//                 if (bcsc_remain_task_count[bidx_diag] == 0)
//                 {
//                     pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_diag]);
//                 }
//             }
//             for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task) + 1;
//                  bidx < bcsc_related_pointer[bcol_task + 1]; bidx++)
//             {
//                 pangulu_exblock_idx brow = bcsc_related_index[bidx];
//                 pangulu_int32_t target_rank = (brow % p) * q + (bcol_task % q);
//                 if (target_rank == rank)
//                 {
//                     while ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] < brow))
//                     {
//                         bidx_diag++;
//                     }
//                     if ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] == brow))
//                     {
//                         pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
//                         if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
//                         {
//                             bcsc_remain_task_count[bidx]--;
//                             pangulu_task_queue_push(heap, brow, bcol_task, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), ssssm_op1, task->opdst, block_length, __FILE__, __LINE__);
//                         }
//                     }
//                 }
//                 else
//                 {
//                     if (sent_rank_flag[target_rank] == 0)
//                     {
//                         pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
//                         sent_rank_flag[target_rank] = 1;
//                     }
//                 }
//             }
//         }
//         else if (kernel_id == PANGULU_TASK_SSSSM)
//         {
//             pangulu_exblock_ptr bidx_task = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task);
//             pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task)], bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task) + 1], PANGULU_MIN(brow_task, bcol_task));
//             if (brow_task == bcol_task)
//             {
//                 if (bcsc_remain_task_count[bidx_task] == 1)
//                 {
//                     bcsc_remain_task_count[bidx_task]--;
//                     pangulu_task_queue_push(heap, brow_task, bcol_task, brow_task, PANGULU_TASK_GETRF, brow_task, task->opdst, NULL, NULL, block_length, __FILE__, __LINE__);
//                 }
//             }
//             else if (brow_task < bcol_task)
//             { // GESSM
//                 if (bcsc_remain_task_count[bidx_task] == 1)
//                 {
//                     pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
//                     if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
//                     {
//                         bcsc_remain_task_count[bidx_task]--;
//                         pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_GESSM, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
//                     }
//                 }
//             }
//             else
//             { // TSTRF
//                 if (bcsc_remain_task_count[bidx_task] == 1)
//                 {
//                     pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
//                     if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
//                     {
//                         bcsc_remain_task_count[bidx_task]--;
//                         pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_TSTRF, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
//                     }
//                 }
//             }
//         }
//     }
//     pthread_mutex_unlock(block_smatrix->info_mutex);
// }

void pangulu_numeric_work(
    pangulu_task_t *task,
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int16_t kernel_id = task->kernel_id;
    pangulu_int64_t brow_task = task->row;
    pangulu_int64_t bcol_task = task->col;
    pangulu_int64_t level = task->task_level;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    memset(sent_rank_flag, 0, sizeof(char) * nproc);

    if (kernel_id != PANGULU_TASK_SSSSM)
    {
        pangulu_numeric_thread_param param;
        param.pangulu_common = common;
        param.block_common = block_common;
        param.block_smatrix = block_smatrix;
        sc25_task_compute(NULL, brow_task * block_length + bcol_task, pangulu_sc25_batch_ssssm_callback, &param);
    }

    if (kernel_id == PANGULU_TASK_GETRF)
    {
        // if(level%100 == 0){
        //     printf("> #%d GETRF level=%d\n", rank, level);
        // }
        pangulu_getrf_interface(nb, task->opdst, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], level) + 1;
             bidx < bcsc_related_pointer[level + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (level % q);
            if (target_rank == rank)
            {
                if (bcsc_remain_task_count[bidx] == 1)
                {
                    bcsc_remain_task_count[bidx]--;
                    pangulu_task_queue_push(heap, brow, level, level, PANGULU_TASK_TSTRF, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), task->opdst, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        for (pangulu_exblock_ptr bidx = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], level) + 1;
             bidx < bcsr_related_pointer[level + 1]; bidx++)
        {
            pangulu_exblock_idx bcol = bcsr_related_index[bidx];
            pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                if (bcsc_remain_task_count[bcsr_index_bcsc[bidx]] == 1)
                {
                    bcsc_remain_task_count[bcsr_index_bcsc[bidx]]--;
                    pangulu_task_queue_push(heap, level, bcol, level, PANGULU_TASK_GESSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx]]), task->opdst, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, level, level, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_TSTRF)
    {
        pangulu_tstrf_interface(nb, task->opdst, task->op1, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        pangulu_exblock_ptr bidx_csr_diag = binarysearch(bcsr_related_index, bcsr_related_pointer[level], bcsr_related_pointer[level + 1], bcol_task);
        if ((level % p) * q + (level % q) != rank)
        {
            bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]]--;
            if (bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr_diag]] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
            }
        }
        for (pangulu_exblock_ptr bidx_csr = binarysearch(bcsr_related_index, bcsr_related_pointer[brow_task], bcsr_related_pointer[brow_task + 1], bcol_task) + 1;
             bidx_csr < bcsr_related_pointer[brow_task + 1]; bidx_csr++)
        {
            pangulu_exblock_idx bcol = bcsr_related_index[bidx_csr];
            pangulu_int32_t target_rank = (brow_task % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                while ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] < bcol))
                {
                    bidx_csr_diag++;
                }
                if ((bidx_csr_diag < bcsr_related_pointer[level + 1]) && (bcsr_related_index[bidx_csr_diag] == bcol))
                {
                    pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr_diag]]);
                    if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bcsr_index_bcsc[bidx_csr]]--;
                        pangulu_task_queue_push(heap, brow_task, bcol, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bcsr_index_bcsc[bidx_csr]]), task->opdst, ssssm_op2, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_GESSM)
    {
        // printf("#%d POP GESSM (%d, %d) %d\n", rank, brow_task, bcol_task, level);
        pangulu_gessm_interface(nb, task->opdst, task->op1, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[level], bcsc_related_pointer[level + 1], brow_task);
        if ((level % p) * q + (level % q) != rank)
        {
            bcsc_remain_task_count[bidx_diag]--;
            if (bcsc_remain_task_count[bidx_diag] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_diag]);
            }
        }
        for (pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task) + 1;
             bidx < bcsc_related_pointer[bcol_task + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (bcol_task % q);
            if (target_rank == rank)
            {
                while ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] < brow))
                {
                    bidx_diag++;
                }
                if ((bidx_diag < bcsc_related_pointer[level + 1]) && (bcsc_related_index[bidx_diag] == brow))
                {
                    pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                    if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                    {
                        bcsc_remain_task_count[bidx]--;
                        pangulu_task_queue_push(heap, brow, bcol_task, level, PANGULU_TASK_SSSSM, level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), ssssm_op1, task->opdst, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_SSSSM)
    {
        // pangulu_ssssm_interface(nb, task->opdst, task->op1, task->op2, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        // if((task->op1->brow_pos % p) * q + (task->op1->bcol_pos % q) != rank){
        //     pangulu_exblock_ptr bidx_op1 = binarysearch(bcsc_related_index, bcsc_related_pointer[task->op1->bcol_pos], bcsc_related_pointer[task->op1->bcol_pos+1], task->op1->brow_pos);
        //     bcsc_remain_task_count[bidx_op1]--;
        //     if(bcsc_remain_task_count[bidx_op1] == 0){
        //         pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_op1]);
        //     }
        // }
        // if((task->op2->brow_pos % p) * q + (task->op2->bcol_pos % q) != rank){
        //     pangulu_exblock_ptr bidx_op2 = binarysearch(bcsc_related_index, bcsc_related_pointer[task->op2->bcol_pos], bcsc_related_pointer[task->op2->bcol_pos+1], task->op2->brow_pos);
        //     bcsc_remain_task_count[bidx_op2]--;
        //     if(bcsc_remain_task_count[bidx_op2] == 0){
        //         pangulu_storage_slot_queue_recycle(storage, &bcsc_related_draft_info[bidx_op2]);
        //     }
        // }
        pangulu_exblock_ptr bidx_task = binarysearch(bcsc_related_index, bcsc_related_pointer[bcol_task], bcsc_related_pointer[bcol_task + 1], brow_task);
        pangulu_exblock_ptr bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task)], bcsc_related_pointer[PANGULU_MIN(brow_task, bcol_task) + 1], PANGULU_MIN(brow_task, bcol_task));
        if (brow_task == bcol_task)
        {
            if (bcsc_remain_task_count[bidx_task] == 1)
            {
                bcsc_remain_task_count[bidx_task]--;
                pangulu_task_queue_push(heap, brow_task, bcol_task, brow_task, PANGULU_TASK_GETRF, brow_task, task->opdst, NULL, NULL, block_length, __FILE__, __LINE__);
            }
        }
        else if (brow_task < bcol_task)
        { // GESSM
            if (bcsc_remain_task_count[bidx_task] == 1)
            {
                pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                {
                    bcsc_remain_task_count[bidx_task]--;
                    pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_GESSM, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
        else
        { // TSTRF
            if (bcsc_remain_task_count[bidx_task] == 1)
            {
                pangulu_storage_slot_t *diag_block = pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx_diag]);
                if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                {
                    bcsc_remain_task_count[bidx_task]--;
                    pangulu_task_queue_push(heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_TSTRF, PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else
    {
        printf(PANGULU_E_K_ID);
        exit(1);
    }
}

void *pangulu_numeric_compute_thread(void *param)
{
    pangulu_numeric_thread_param *work_param = (pangulu_numeric_thread_param *)param;
    pangulu_common *common = work_param->pangulu_common;
    pangulu_block_common *block_common = work_param->block_common;
    pangulu_block_smatrix *block_smatrix = work_param->block_smatrix;
    pangulu_int32_t rank = block_common->rank;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    int pangulu_omp_num_threads = common->omp_thread;
    for (int i = 0; i < pangulu_omp_num_threads; i++)
    {
#ifdef HT_IS_OPEN
        CPU_SET((2 * (pangulu_omp_num_threads * rank + i)) % cpu_thread_count_per_node, &cpuset);
#else
        CPU_SET((pangulu_omp_num_threads * rank + i) % cpu_thread_count_per_node, &cpuset);
#endif
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        perror("pthread_setaffinity_np error");
    }
#pragma omp parallel num_threads(common->omp_thread)
    {
        int tid = omp_get_thread_num();
#ifdef HT_IS_OPEN
        bind_to_core((2 * (pangulu_omp_num_threads * rank + tid)) % cpu_thread_count_per_node);
#else
        bind_to_core((pangulu_omp_num_threads * rank + tid) % cpu_thread_count_per_node);
#endif
    }

#ifdef GPU_OPEN
    int device_num;
    pangulu_platform_get_device_num(&device_num, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_set_default_device(rank % device_num, PANGULU_DEFAULT_PLATFORM);
#endif
    pangulu_int64_t *rank_remain_task_count = &(block_smatrix->rank_remain_task_count);
    pangulu_task_queue_t *heap = block_smatrix->heap;

    // while ((*rank_remain_task_count) != 0)
    // {
    //     pangulu_task_t task = pangulu_task_queue_pop(heap);
    //     (*rank_remain_task_count)--;
    //     pangulu_numeric_work(&task, block_common, block_smatrix);
    // }

    
    
    while ((*rank_remain_task_count) != 0)
    {
        if (pangulu_task_queue_empty(heap))
        {
            //usleep(1);
            sc25_idle_work(NULL, pangulu_sc25_batch_ssssm_callback, param);
            continue;
        }
        pangulu_int64_t ntask = 0;
        while (!pangulu_task_queue_empty(heap))
        {
            pangulu_task_t task = pangulu_task_queue_pop(heap);
            while (ntask + 1 > working_task_buf_capacity)
            {
                working_task_buf_capacity = (working_task_buf_capacity + 1) * 2;
                working_task_buf = pangulu_realloc(__FILE__, __LINE__, working_task_buf, sizeof(pangulu_task_t) * working_task_buf_capacity);
            }
            working_task_buf[ntask] = task;
            ntask++;
        }
        (*rank_remain_task_count) -= ntask;
        pangulu_numeric_work_batched(ntask, working_task_buf, common, block_common, block_smatrix);
        // printf("#%d *rank_remain_task_count = %d\n", rank, *rank_remain_task_count);
    }
    pangulu_free(__FILE__, __LINE__, working_task_buf);
    return NULL;
}

void pangulu_numeric(
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pthread_t pthread;
    pangulu_numeric_thread_param param;
    param.pangulu_common = common;
    param.block_common = block_common;
    param.block_smatrix = block_smatrix;
    pthread_create(&pthread, NULL, pangulu_numeric_compute_thread, (void *)(&param));

    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int64_t *rank_remain_recv_block_count = &block_smatrix->rank_remain_recv_block_count;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    int pangulu_omp_num_threads = common->omp_thread;
#ifdef HT_IS_OPEN
    bind_to_core((2 * pangulu_omp_num_threads * rank) % cpu_thread_count_per_node);
#else
    bind_to_core((pangulu_omp_num_threads * rank) % cpu_thread_count_per_node);
#endif

    pangulu_cm_sync();
    pangulu_task_queue_clear(heap);
    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_int64_t now_level = 0; now_level < block_length; now_level++)
    {
        pangulu_int64_t now_rank = (now_level % p) * q + (now_level % q);
        if (now_rank == rank)
        {
            pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[now_level], bcsc_related_pointer[now_level + 1], now_level);
            if (bcsc_remain_task_count[bidx] == 1)
            {
                bcsc_remain_task_count[bidx]--;
                pangulu_task_queue_push(heap,
                                        now_level, now_level, now_level, PANGULU_TASK_GETRF,
                                        now_level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), NULL, NULL, block_length, __FILE__, __LINE__);
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);

    while ((*rank_remain_recv_block_count) != 0)
    {
        MPI_Status status;
        pangulu_cm_probe(&status);
        (*rank_remain_recv_block_count)--;
        pangulu_numerical_receive_message(status, block_common, block_smatrix);
    }

    pthread_join(pthread, NULL);
}

void pangulu_numeric_check(
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int64_t *rank_remain_recv_block_count = &block_smatrix->rank_remain_recv_block_count;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_int32_t n = block_common->n;

    pangulu_exblock_ptr *bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx *bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr *bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr *bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t *bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;

    calculate_type *A_rowsum = block_smatrix->A_rowsum_reordered;
    calculate_type *Ux1 = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *LxUx1 = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    memset(Ux1, 0, sizeof(calculate_type) * n);
    memset(LxUx1, 0, sizeof(calculate_type) * n);

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_idx bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            if (brow == bcol)
            {
                if ((brow % p) * q + (bcol % q) == rank)
                {
                    pangulu_uint64_t slot_addr = bcsc_related_draft_info[bidx];
                    pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                    if (slot)
                    {
                        pangulu_inblock_ptr *colptr = slot->columnpointer;
                        pangulu_inblock_idx *rowidx = slot->rowindex;
                        calculate_type *value = slot->value;
                        for (pangulu_inblock_idx col = 0; col < nb; col++)
                        {
                            for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                            {
                                pangulu_inblock_idx row = rowidx[idx];
                                if (row > col)
                                {
                                    continue;
                                }
                                Ux1[brow * nb + row] += value[idx];
                            }
                        }
                    }
                }
                continue;
            }
            if (brow > bcol)
            {
                continue;
            }
            if ((brow % p) * q + (bcol % q) == rank)
            {
                pangulu_uint64_t slot_addr = bcsc_related_draft_info[bidx];
                pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                if (slot)
                {
                    pangulu_inblock_ptr *colptr = slot->columnpointer;
                    pangulu_inblock_idx *rowidx = slot->rowindex;
                    calculate_type *value = slot->value;
                    for (pangulu_inblock_idx col = 0; col < nb; col++)
                    {
                        for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                        {
                            pangulu_inblock_idx row = rowidx[idx];
                            Ux1[brow * nb + row] += value[idx];
                        }
                    }
                }
            }
        }
    }

    if (rank == 0)
    {
        MPI_Status mpi_stat;
        calculate_type *recv_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        for (int fetch_rank = 1; fetch_rank < nproc; fetch_rank++)
        {
            MPI_Recv(recv_buf, n, MPI_VAL_TYPE, fetch_rank, 0, MPI_COMM_WORLD, &mpi_stat);
            for (pangulu_int32_t row = 0; row < n; row++)
            {
                Ux1[row] += recv_buf[row];
            }
        }
        for (int remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            MPI_Send(Ux1, n, MPI_VAL_TYPE, remote_rank, 0, MPI_COMM_WORLD);
        }
        pangulu_free(__FILE__, __LINE__, recv_buf);
    }
    else
    {
        MPI_Status mpi_stat;
        MPI_Send(Ux1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(Ux1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD, &mpi_stat);
    }

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_idx bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            if (brow == bcol)
            {
                if ((brow % p) * q + (bcol % q) == rank)
                {
                    pangulu_uint64_t slot_addr = bcsc_related_draft_info[bidx];
                    pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                    if (slot)
                    {
                        pangulu_inblock_ptr *colptr = slot->columnpointer;
                        pangulu_inblock_idx *rowidx = slot->rowindex;
                        calculate_type *value = slot->value;
                        for (pangulu_inblock_idx col = 0; col < nb; col++)
                        {
                            for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                            {
                                pangulu_inblock_idx row = rowidx[idx];
                                if (row < col)
                                {
                                    continue;
                                }
                                if (row == col)
                                {
                                    LxUx1[brow * nb + row] += Ux1[bcol * nb + col] * 1.0;
                                }
                                else
                                {
                                    LxUx1[brow * nb + row] += Ux1[bcol * nb + col] * value[idx];
                                }
                            }
                        }
                    }
                }
                continue;
            }
            if (brow < bcol)
            {
                continue;
            }
            if ((brow % p) * q + (bcol % q) == rank)
            {
                pangulu_uint64_t slot_addr = bcsc_related_draft_info[bidx];
                pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                if (slot)
                {
                    pangulu_inblock_ptr *colptr = slot->columnpointer;
                    pangulu_inblock_idx *rowidx = slot->rowindex;
                    calculate_type *value = slot->value;
                    for (pangulu_inblock_idx col = 0; col < nb; col++)
                    {
                        for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                        {
                            pangulu_inblock_idx row = rowidx[idx];
                            LxUx1[brow * nb + row] += Ux1[bcol * nb + col] * value[idx];
                        }
                    }
                }
            }
        }
    }

    if (rank == 0)
    {
        MPI_Status mpi_stat;
        calculate_type *recv_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        for (int fetch_rank = 1; fetch_rank < nproc; fetch_rank++)
        {
            MPI_Recv(recv_buf, n, MPI_VAL_TYPE, fetch_rank, 0, MPI_COMM_WORLD, &mpi_stat);
            for (pangulu_int32_t row = 0; row < n; row++)
            {
                LxUx1[row] += recv_buf[row];
            }
        }
        pangulu_free(__FILE__, __LINE__, recv_buf);
    }
    else
    {
        MPI_Send(LxUx1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        calculate_type sum = 0.0;
        calculate_type c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type num = (LxUx1[i] - A_rowsum[i]) * (LxUx1[i] - A_rowsum[i]);
            calculate_type z = num - c;
            calculate_type t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        calculate_type residual_norm2 = sqrt(sum);

        sum = 0.0;
        c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type num = A_rowsum[i] * A_rowsum[i];
            calculate_type z = num - c;
            calculate_type t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        calculate_type rhs_norm2 = sqrt(sum);
        printf("[PanguLU] numeric check : relative_residual = %le\n", residual_norm2 / rhs_norm2);
    }
}
