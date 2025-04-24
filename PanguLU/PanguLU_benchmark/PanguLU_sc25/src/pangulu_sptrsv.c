#include "pangulu_common.h"

void pangulu_sptrsv_count_task(
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_int32_t uplo
){
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_int32_t nproc = bcommon->sum_rank_size;
    pangulu_int32_t p = bcommon->p;
    pangulu_int32_t q = bcommon->q;
    pangulu_int64_t rank_remain_task_count = 0;
    pangulu_int64_t rank_remain_recv_block_count = 0;
    char* fetch_rank_flag = bsmatrix->sent_rank_flag;
    memset(bsmatrix->rhs_remain_recv_count, 0, sizeof(pangulu_int32_t) * block_length);
    memset(bsmatrix->rhs_remain_task_count, 0, sizeof(pangulu_int32_t) * block_length);
    if(uplo == PANGULU_LOWER){ // lower sptrsv
        for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
            memset(fetch_rank_flag, 0, sizeof(char) * nproc);
            for(pangulu_exblock_ptr bidx = bsmatrix->bcsr_related_pointer[brow]; bidx < bsmatrix->bcsr_related_pointer[brow+1]; bidx++){
                pangulu_exblock_idx bcol = bsmatrix->bcsr_related_index[bidx];
                if(brow < bcol){
                    break;
                }
                if((brow % p) * q + (bcol % q) == rank){
                    rank_remain_task_count++;
                    bsmatrix->rhs_remain_task_count[brow]++;
                }else{
                    if(brow != bcol){
                        fetch_rank_flag[(brow % p) * q + (bcol % q)] = 1;
                    }
                }
            }
            if((brow % p) * q + (brow % q) == rank){
                for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
                    bsmatrix->rhs_remain_recv_count[brow] += fetch_rank_flag[fetch_rank];
                    rank_remain_recv_block_count += fetch_rank_flag[fetch_rank];
                }
            }else{
                bsmatrix->rhs_remain_recv_count[brow]++; // Receive diag vector from other ranks
                rank_remain_recv_block_count++;
                bsmatrix->rhs_remain_task_count[brow]++; // Adding fake diag sptrsv task
                rank_remain_task_count++;
            }
        }
    }else if(uplo == PANGULU_UPPER){ // upper sptrsv
        for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
            memset(fetch_rank_flag, 0, sizeof(char) * nproc);
            for(pangulu_exblock_ptr bidx = bsmatrix->bcsr_related_pointer[brow]; bidx < bsmatrix->bcsr_related_pointer[brow+1]; bidx++){
                pangulu_exblock_idx bcol = bsmatrix->bcsr_related_index[bidx];
                if(brow > bcol){
                    continue;
                }
                if((brow % p) * q + (bcol % q) == rank){
                    rank_remain_task_count++;
                    bsmatrix->rhs_remain_task_count[brow]++;
                }else{
                    if(brow != bcol){
                        fetch_rank_flag[(brow % p) * q + (bcol % q)] = 1;
                    }
                }
            }
            if((brow % p) * q + (brow % q) == rank){
                for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
                    bsmatrix->rhs_remain_recv_count[brow] += fetch_rank_flag[fetch_rank];
                    rank_remain_recv_block_count += fetch_rank_flag[fetch_rank];
                }
            }else{
                bsmatrix->rhs_remain_recv_count[brow]++; // Receive diag vector from other ranks
                rank_remain_recv_block_count++;
                bsmatrix->rhs_remain_task_count[brow]++; // Adding fake diag sptrsv task
                rank_remain_task_count++;
            }
            // bsmatrix->rhs_remain_recv_count[brow] += (block_length - brow - 1);
            // rank_remain_recv_block_count += (block_length - brow - 1);
        }
    }
    bsmatrix->rank_remain_task_count = rank_remain_task_count;
    bsmatrix->rank_remain_recv_block_count = rank_remain_recv_block_count;

    printf("#%d Task : ", rank);
    for(int i=0;i<block_length;i++){
        printf("%d ", bsmatrix->rhs_remain_task_count[i]);
    }
    printf("\n");
    printf("#%d Recv : ", rank);
    for(int i=0;i<block_length;i++){
        printf("%d ", bsmatrix->rhs_remain_recv_count[i]);
    }
    printf("\n");
}

void pangulu_sptrsv_preprocessing(
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_vector *reordered_rhs
){
    pangulu_exblock_idx block_length =  bcommon->block_length;
    pangulu_inblock_idx nb = bcommon->nb;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_int32_t nproc = bcommon->sum_rank_size;
    pangulu_int16_t p = bcommon->p;
    pangulu_int16_t q = bcommon->q;
    bsmatrix->rhs = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * block_length * nb);
    if(rank == 0){
        memcpy(bsmatrix->rhs, reordered_rhs->value, sizeof(calculate_type) * reordered_rhs->row);
        for(pangulu_exblock_idx i = reordered_rhs->row; i < block_length * nb; i++){
            bsmatrix->rhs[i] = 0.0;
        }
    }else{
        memset(bsmatrix->rhs, 0, sizeof(calculate_type) * block_length * nb);
    }
    bsmatrix->recv_buffer = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb);
    bsmatrix->rhs_remain_task_count = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_length);
    bsmatrix->rhs_remain_recv_count = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_length);

    // pangulu_uint16_t* lp = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint16_t) * nproc);
    // pangulu_uint16_t* li = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint16_t) * (block_length + 1));
    // pangulu_uint16_t* up = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint16_t) * nproc);
    // pangulu_uint16_t* ui = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint16_t) * (block_length + 1));
    // memset(lp, 0xFF, sizeof(pangulu_uint16_t) * nproc);
    // memset(up, 0xFF, sizeof(pangulu_uint16_t) * nproc);

    // pangulu_exblock_ptr* bcsr_related_pointer = bsmatrix->bcsr_related_pointer;
    // pangulu_exblock_idx* bcsr_related_index = bsmatrix->bcsr_related_index;

    // // char* flag = bsmatrix->sent_rank_flag;
    // // memset(flag, 0, sizeof(char) * nproc);
    // // pangulu_int16_t prio_cursor = 0;
    // // for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
    // //     li[brow] = prio_cursor;
    // //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    // //         flag[target_rank] &= 0x2;
    // //     }
    // //     for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    // //         pangulu_exblock_idx bcol = bcsr_related_index[bidx];
    // //         pangulu_uint16_t target_rank = (brow % p) * q + (bcol % q);
    // //         flag[target_rank] |= 0x1;
    // //     }
    // //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    // //         if(flag[target_rank] == 0x1){
    // //             lp[prio_cursor] = target_rank;
    // //             prio_cursor++;
    // //             flag[target_rank] = 0x3;
    // //         }
    // //     }
    // // }
    // // li[block_length] = prio_cursor;

    // // memset(flag, 0, sizeof(char) * nproc);
    // // prio_cursor = 0;
    // // for(pangulu_exblock_idx brow = block_length-1; brow != 0xFFFFFFFF; brow--){
    // //     ui[brow+1] = prio_cursor;
    // //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    // //         flag[target_rank] &= 0x2;
    // //     }
    // //     for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    // //         pangulu_exblock_idx bcol = bcsr_related_index[bidx];
    // //         pangulu_uint16_t target_rank = (brow % p) * q + (bcol % q);
    // //         flag[target_rank] |= 0x1;
    // //     }
    // //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    // //         if(flag[target_rank] == 0x1){
    // //             up[prio_cursor] = target_rank;
    // //             prio_cursor++;
    // //             flag[target_rank] = 0x3;
    // //         }
    // //     }
    // // }
    // // ui[0] = prio_cursor;

    // char* flag = bsmatrix->sent_rank_flag;
    // memset(flag, 0, sizeof(char) * nproc);
    // pangulu_int16_t prio_cursor = 0;
    // for(pangulu_exblock_idx brow = block_length-1; brow != 0xFFFFFFFF; brow--){
    //     li[brow+1] = prio_cursor;
    //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    //         flag[target_rank] &= 0x2;
    //     }
    //     for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    //         pangulu_exblock_idx bcol = bcsr_related_index[bidx];
    //         if(bcol >= brow){
    //             continue;
    //         }
    //         pangulu_uint16_t target_rank = (brow % p) * q + (bcol % q);
    //         flag[target_rank] |= 0x1;
    //     }
    //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    //         if(flag[target_rank] == 0x1){
    //             lp[prio_cursor] = target_rank;
    //             prio_cursor++;
    //             flag[target_rank] = 0x3;
    //         }
    //     }
    // }
    // li[0] = prio_cursor;

    // memset(flag, 0, sizeof(char) * nproc);
    // prio_cursor = 0;
    // for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
    //     ui[brow] = prio_cursor;
    //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    //         flag[target_rank] &= 0x2;
    //     }
    //     for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    //         pangulu_exblock_idx bcol = bcsr_related_index[bidx];
    //         if(brow >= bcol){
    //             continue;
    //         }
    //         pangulu_uint16_t target_rank = (brow % p) * q + (bcol % q);
    //         flag[target_rank] |= 0x1;
    //     }
    //     for(pangulu_int16_t target_rank = 0; target_rank < nproc; target_rank++){
    //         if(flag[target_rank] == 0x1){
    //             up[prio_cursor] = target_rank;
    //             prio_cursor++;
    //             flag[target_rank] = 0x3;
    //         }
    //     }
    // }
    // ui[block_length] = prio_cursor;

    // printf("#%d LP : ", rank);
    // for(int i=0;i<nproc;i++){
    //     printf("%d ", lp[i]);
    // }
    // printf("\n");
    // printf("#%d LI : ", rank);
    // for(int i=0;i<=block_length;i++){
    //     printf("%d ", li[i]);
    // }
    // printf("\n");

    // printf("#%d UP : ", rank);
    // for(int i=0;i<nproc;i++){
    //     printf("%d ", up[i]);
    // }
    // printf("\n");
    // printf("#%d UI : ", rank);
    // for(int i=0;i<=block_length;i++){
    //     printf("%d ", ui[i]);
    // }
    // printf("\n");

    // bsmatrix->sptrsv_lower_rank_prio = lp;
    // bsmatrix->sptrsv_lower_rank_index = li;
    // bsmatrix->sptrsv_upper_rank_prio = up;
    // bsmatrix->sptrsv_upper_rank_index = ui;

    // pangulu_cm_sync();
    // exit(0);
}

void pangulu_sptrsv_receive_message(
    MPI_Status status,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
){
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int64_t fetch_rank = status.MPI_SOURCE;
    pangulu_int64_t tag = status.MPI_TAG;
    pangulu_storage_t* storage = block_smatrix->storage;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_exblock_idx brow = tag % block_length;
    pangulu_int16_t recv_type = tag / block_length;
    calculate_type* rhs = block_smatrix->rhs;
    calculate_type* recv_buffer = block_smatrix->recv_buffer;
    pangulu_task_queue_t* heap = block_smatrix->heap;
    pangulu_exblock_ptr* bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx* bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t* bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_int32_t* rhs_remain_task_count = block_smatrix->rhs_remain_task_count;
    pangulu_int32_t* rhs_remain_recv_count = block_smatrix->rhs_remain_recv_count;

    // tag %= (block_length * 3);

    printf("#%d Recv from=%d recv_type=%d brow=%d\n", rank, status.MPI_SOURCE, recv_type, brow);
    if(recv_type == 3){ // recv SPMV_U
        pangulu_cm_recv(recv_buffer, sizeof(calculate_type) * nb, fetch_rank, tag, block_length * 4);
        pangulu_vecadd_interface(nb, rhs + brow * nb, recv_buffer);
        pthread_mutex_lock(block_smatrix->info_mutex);
        rhs_remain_recv_count[brow]--;
        if(rank == ((brow % p) * q + (brow % q))){
            // printf("task=%d recv=%d\n", rhs_remain_task_count[brow], rhs_remain_recv_count[brow]);
            if((rhs_remain_task_count[brow]==1) && (rhs_remain_recv_count[brow]==0)){
                block_smatrix->rhs_remain_task_count[brow]--;
                pangulu_storage_slot_t* diag_slot = pangulu_storage_get_slot(
                    storage,
                    bcsc_related_draft_info[binarysearch(
                        bcsc_related_index,
                        bcsc_related_pointer[brow],
                        bcsc_related_pointer[brow+1],
                        brow
                    )]
                );
                pangulu_task_queue_push(
                    heap, brow, brow, brow,
                    PANGULU_TASK_SPTRSV_U,
                    brow, diag_slot, NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else if(recv_type == 2){ // recv SPTRSV_U
        // printf("Recv U");
        pangulu_cm_recv(rhs + brow * nb, sizeof(calculate_type) * nb, fetch_rank, tag, block_length * 4);
        pthread_mutex_lock(block_smatrix->info_mutex);
        rhs_remain_recv_count[brow]--;
        pangulu_int64_t diag_bidx = binarysearch_first_ge(
            bcsc_related_index,
            bcsc_related_pointer[brow],
            bcsc_related_pointer[brow+1],
            brow
        );
        for(pangulu_exblock_ptr bidx = bcsc_related_pointer[brow]; bidx < diag_bidx; bidx++){
            pangulu_exblock_idx brow_update = bcsc_related_index[bidx];
            if((brow_update % p) * q + (brow % q) != rank){
                continue;
            }
            rhs_remain_task_count[brow_update]--;
            pangulu_task_queue_push(
                heap, brow_update, brow, brow_update, PANGULU_TASK_SPMV_U, brow_update,
                pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]),
                NULL, NULL, block_length, __FILE__, __LINE__
            );
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else if(recv_type == 1){ // recv SPMV_L
        pangulu_cm_recv(recv_buffer, sizeof(calculate_type) * nb, fetch_rank, tag, block_length * 4);
        pangulu_vecadd_interface(nb, rhs + brow * nb, recv_buffer);
        pthread_mutex_lock(block_smatrix->info_mutex);
        rhs_remain_recv_count[brow]--;
        if(rank == ((brow % p) * q + (brow % q))){
            // printf("task=%d recv=%d\n", rhs_remain_task_count[brow], rhs_remain_recv_count[brow]);
            if((rhs_remain_task_count[brow]==1) && (rhs_remain_recv_count[brow]==0)){
                block_smatrix->rhs_remain_task_count[brow]--;
                pangulu_storage_slot_t* diag_slot = pangulu_storage_get_slot(
                    storage,
                    bcsc_related_draft_info[binarysearch(
                        bcsc_related_index,
                        bcsc_related_pointer[brow],
                        bcsc_related_pointer[brow+1],
                        brow
                    )]
                );
                pangulu_task_queue_push(
                    heap, brow, brow, brow,
                    PANGULU_TASK_SPTRSV_L,
                    brow, diag_slot, NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else if(recv_type == 0){ // recv SPTRSV_L
        pangulu_cm_recv(rhs + brow * nb, sizeof(calculate_type) * nb, fetch_rank, tag, block_length * 4);
        pthread_mutex_lock(block_smatrix->info_mutex);
        rhs_remain_recv_count[brow]--;
        pangulu_int64_t diag_bidx = binarysearch_first_ge(
            bcsc_related_index,
            bcsc_related_pointer[brow],
            bcsc_related_pointer[brow+1],
            brow
        );
        if(bcsc_related_index[diag_bidx] == brow){
            diag_bidx++;
        }
        for(pangulu_exblock_ptr bidx = diag_bidx; bidx < bcsc_related_pointer[brow+1]; bidx++){
            pangulu_exblock_idx brow_update = bcsc_related_index[bidx];
            if((brow_update % p) * q + (brow % q) != rank){
                continue;
            }
            rhs_remain_task_count[brow_update]--;
            pangulu_task_queue_push(
                heap, brow_update, brow, brow_update, PANGULU_TASK_SPMV_L, brow_update,
                pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]),
                NULL, NULL, block_length, __FILE__, __LINE__
            );
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else{
        printf("[PanguLU ERROR] Invald SpTRSV receive message type. Exit.\n");
        exit(1);
    }
}


void pangulu_sptrsv_work(
    pangulu_task_t *task,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix
){
    pangulu_int16_t kernel_id = task->kernel_id;
    pangulu_int64_t brow_task = task->row;
    pangulu_int64_t bcol_task = task->col;
    pangulu_int64_t level = task->task_level;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t* heap = block_smatrix->heap;
    pangulu_storage_t* storage = block_smatrix->storage;
    char* sent_rank_flag = block_smatrix->sent_rank_flag;
    calculate_type* rhs = block_smatrix->rhs;

    pangulu_exblock_ptr* bcsr_related_pointer = block_smatrix->bcsr_related_pointer;
    pangulu_exblock_idx* bcsr_related_index = block_smatrix->bcsr_related_index;
    pangulu_exblock_ptr* bcsr_index_bcsc = block_smatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr* bcsc_related_pointer = block_smatrix->bcsc_related_pointer;
    pangulu_exblock_idx* bcsc_related_index = block_smatrix->bcsc_related_index;
    pangulu_uint64_t* bcsc_related_draft_info = block_smatrix->bcsc_related_draft_info;
    pangulu_uint32_t* bcsc_remain_task_count = block_smatrix->bcsc_remain_task_count;
    pangulu_uint32_t* rhs_remain_recv_count = block_smatrix->rhs_remain_recv_count;
    pangulu_uint32_t* rhs_remain_task_count = block_smatrix->rhs_remain_task_count;
    pangulu_int64_t block_length = block_common->block_length;

    printf("#%d kernel_id=%d (%d, %d)\n", rank, kernel_id, brow_task, bcol_task);

    if((kernel_id == PANGULU_TASK_SPTRSV_L) || (kernel_id == PANGULU_TASK_SPTRSV_U)){
        pangulu_sptrsv_interface(nb, task->opdst, rhs+task->opdst->brow_pos*nb, (kernel_id==PANGULU_TASK_SPTRSV_L)?PANGULU_LOWER:PANGULU_UPPER);
        for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
            // if(kernel_id==PANGULU_TASK_SPTRSV_U)printf("Send U\n");
            // else printf("Send L\n");
            if(rank == target_rank){
                continue;
            }
            pangulu_cm_isend(
                rhs + task->opdst->brow_pos * nb, 
                sizeof(calculate_type) * nb, 
                target_rank, 
                block_common->block_length * ((kernel_id==PANGULU_TASK_SPTRSV_U)?2:0) + task->opdst->brow_pos,
                block_common->block_length * 4
            ); // < block_length : diag; >=block_length : -mul;
        }
        pangulu_int64_t diag_bidx = binarysearch_first_ge(
            bcsc_related_index,
            bcsc_related_pointer[task->opdst->brow_pos],
            bcsc_related_pointer[task->opdst->brow_pos+1],
            task->opdst->brow_pos
        );
        pthread_mutex_lock(block_smatrix->info_mutex);
        if(kernel_id == PANGULU_TASK_SPTRSV_U){
            for(pangulu_exblock_ptr bidx = bcsc_related_pointer[task->opdst->brow_pos]; bidx < diag_bidx; bidx++){
                pangulu_exblock_idx brow_update = bcsc_related_index[bidx];
                if((brow_update % p) * q + (task->opdst->brow_pos % q) != rank){
                    continue;
                }
                rhs_remain_task_count[brow_update]--;
                pangulu_task_queue_push(
                    heap, brow_update, task->opdst->brow_pos, brow_update, PANGULU_TASK_SPMV_U, brow_update,
                    pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]),
                    NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }else{
            if(bcsc_related_index[diag_bidx] == task->opdst->brow_pos){
                diag_bidx++;
            }
            for(pangulu_exblock_ptr bidx = diag_bidx; bidx < bcsc_related_pointer[task->opdst->brow_pos+1]; bidx++){
                pangulu_exblock_idx brow_update = bcsc_related_index[bidx];
                if((brow_update % p) * q + (task->opdst->brow_pos % q) != rank){
                    continue;
                }
                rhs_remain_task_count[brow_update]--;
                pangulu_task_queue_push(
                    heap, brow_update, task->opdst->brow_pos, brow_update, PANGULU_TASK_SPMV_L, brow_update,
                    pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]),
                    NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else if((kernel_id == PANGULU_TASK_SPMV_L) || (kernel_id == PANGULU_TASK_SPMV_U)){
        pangulu_spmv_interface(nb, task->opdst, rhs+task->opdst->bcol_pos*nb, rhs+task->opdst->brow_pos*nb);
        pthread_mutex_lock(block_smatrix->info_mutex);
        if(rank == ((task->opdst->brow_pos % p) * q + (task->opdst->brow_pos % q))){
            // printf("task=%d recv=%d\n", rhs_remain_task_count[task->opdst->brow_pos], rhs_remain_recv_count[task->opdst->brow_pos]);
            if((rhs_remain_task_count[task->opdst->brow_pos]==1) && (rhs_remain_recv_count[task->opdst->brow_pos]==0)){
                block_smatrix->rhs_remain_task_count[task->opdst->brow_pos]--;
                pangulu_storage_slot_t* diag_slot = pangulu_storage_get_slot(
                    storage,
                    bcsc_related_draft_info[binarysearch(
                        bcsc_related_index,
                        bcsc_related_pointer[task->opdst->brow_pos],
                        bcsc_related_pointer[task->opdst->brow_pos+1],
                        task->opdst->brow_pos
                    )]
                );
                pangulu_task_queue_push(
                    heap, task->opdst->brow_pos, task->opdst->brow_pos, task->opdst->brow_pos,
                    (kernel_id==PANGULU_TASK_SPMV_L)?PANGULU_TASK_SPTRSV_L:PANGULU_TASK_SPTRSV_U,
                    task->opdst->brow_pos, diag_slot, NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }else{
            if(rhs_remain_task_count[task->opdst->brow_pos]==0){
                pangulu_int32_t target_rank = (task->opdst->brow_pos % p) * q + (task->opdst->brow_pos % q);
                printf("Sending SPMV %d->%d brow=%d\n", rank, target_rank, task->opdst->brow_pos);
                pangulu_cm_isend(
                    rhs + task->opdst->brow_pos * nb, 
                    sizeof(calculate_type)*nb, 
                    target_rank, 
                    block_common->block_length * ((kernel_id==PANGULU_TASK_SPMV_U)?3:1) + task->opdst->brow_pos, 
                    block_common->block_length * 4
                );
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }else{
        printf(PANGULU_E_K_ID);
        exit(1);
    }
}

void* pangulu_sptrsv_compute_thread(void* param){
    pangulu_numeric_thread_param *work_param = (pangulu_numeric_thread_param *)param;
    pangulu_block_common *block_common = work_param->block_common;
    pangulu_block_smatrix *block_smatrix = work_param->block_smatrix;
    pangulu_int64_t *rank_remain_task_count = &(block_smatrix->rank_remain_task_count);
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_int32_t rank = block_common->rank;
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    int pangulu_omp_num_threads = 1;
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

    while ((*rank_remain_task_count) != 0)
    {
        pangulu_task_t task = pangulu_task_queue_pop(heap);
        (*rank_remain_task_count)--;
        pangulu_sptrsv_work(&task, block_common, block_smatrix);
    }
    return NULL;
}


void pangulu_sptrsv_uplo(
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_int32_t uplo
){
    pangulu_task_queue_t* heap = bsmatrix->heap;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_int32_t p = bcommon->p;
    pangulu_int32_t q = bcommon->q;
    pangulu_int32_t* rhs_remain_task_count = bsmatrix->rhs_remain_task_count;
    pangulu_int32_t* rhs_remain_recv_count = bsmatrix->rhs_remain_recv_count;
    pangulu_storage_t* storage = bsmatrix->storage;

    pangulu_exblock_ptr* bcsr_related_pointer = bsmatrix->bcsr_related_pointer;
    pangulu_exblock_idx* bcsr_related_index = bsmatrix->bcsr_related_index;
    pangulu_exblock_ptr* bcsr_index_bcsc = bsmatrix->bcsr_index_bcsc;
    pangulu_exblock_ptr* bcsc_related_pointer = bsmatrix->bcsc_related_pointer;
    pangulu_exblock_idx* bcsc_related_index = bsmatrix->bcsc_related_index;
    pangulu_uint64_t* bcsc_related_draft_info = bsmatrix->bcsc_related_draft_info;
    pangulu_uint32_t* bcsc_remain_task_count = bsmatrix->bcsc_remain_task_count;
    pangulu_int64_t* rank_remain_recv_block_count = &bsmatrix->rank_remain_recv_block_count;
    calculate_type* rhs = bsmatrix->rhs;
    pangulu_inblock_idx nb = bcommon->nb;

    pangulu_task_queue_clear(heap);
    pangulu_task_queue_cmp_strategy(heap, (uplo==PANGULU_LOWER)?0:4);
    pangulu_sptrsv_count_task(bcommon, bsmatrix, uplo);

    if(rank == 0){
        for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
            pangulu_int32_t remote_rank = (brow % p) * q + (brow % q);
            if(remote_rank == 0){
                continue;
            }
            pangulu_cm_isend(rhs + brow * nb, sizeof(calculate_type) * nb, remote_rank, 0, 1);
        }
    }else{
        for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
            if(rank == (brow % p) * q + (brow % q)){
                pangulu_cm_recv(rhs + brow * nb, sizeof(calculate_type) * nb, 0, 0, 1);
            }
        }
    }
    pangulu_cm_sync();
    for(pangulu_exblock_idx brow = 0; brow < block_length; brow++){
        if(rank != (brow % p) * q + (brow % q)){
            memset(rhs + brow * nb, 0, sizeof(calculate_type) * nb);
        }
    }

    pthread_t pthread;
    pangulu_numeric_thread_param param;
    param.block_common = bcommon;
    param.block_smatrix = bsmatrix;
    pthread_create(&pthread, NULL, pangulu_sptrsv_compute_thread, (void *)(&param));

    pangulu_cm_sync();
    pangulu_task_queue_clear(heap);
    pthread_mutex_lock(bsmatrix->info_mutex);
    for (pangulu_int64_t now_level = 0; now_level < block_length; now_level++)
    {
        pangulu_int64_t now_rank = (now_level % p) * q + (now_level % q);
        if (now_rank == rank)
        {
            if((rhs_remain_task_count[now_level] == 1) && (rhs_remain_recv_count[now_level] == 0)){
                rhs_remain_task_count[now_level]--;
                pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[now_level], bcsc_related_pointer[now_level+1], now_level);
                pangulu_task_queue_push(heap,
                    now_level, now_level, now_level, (uplo == PANGULU_LOWER) ? PANGULU_TASK_SPTRSV_L : PANGULU_TASK_SPTRSV_U,
                    now_level, pangulu_storage_get_slot(storage, bcsc_related_draft_info[bidx]), NULL, NULL, block_length, __FILE__, __LINE__
                );
            }
        }
    }
    pthread_mutex_unlock(bsmatrix->info_mutex);

    while ((*rank_remain_recv_block_count) != 0)
    {
        MPI_Status status;
        pangulu_cm_probe(&status);
        (*rank_remain_recv_block_count)--;
        pangulu_sptrsv_receive_message(status, bcommon, bsmatrix);
    }
    pthread_join(pthread, NULL);
}

void pangulu_solve(
    pangulu_block_common* block_common,
    pangulu_block_smatrix* block_smatrix,
    pangulu_vector* result
){
    printf("sptrsv_lower phase\n");
    pangulu_sptrsv_uplo(block_common, block_smatrix, PANGULU_LOWER);

    pangulu_cm_sync();
    exit(0);

    printf("sptrsv_upper phase\n");
    pangulu_sptrsv_uplo(block_common, block_smatrix, PANGULU_UPPER);

    if(block_common->rank == 0){
        memcpy(result->value, block_smatrix->rhs, sizeof(calculate_type) * result->row);
    }
}