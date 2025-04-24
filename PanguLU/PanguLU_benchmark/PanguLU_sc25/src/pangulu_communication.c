#include "pangulu_common.h"

void pangulu_cm_rank(pangulu_int32_t* rank){MPI_Comm_rank(MPI_COMM_WORLD, rank);}
void pangulu_cm_size(pangulu_int32_t* size){MPI_Comm_size(MPI_COMM_WORLD, size);}
void pangulu_cm_sync(){MPI_Barrier(MPI_COMM_WORLD);}
void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root){MPI_Bcast(buffer, count, datatype, root, MPI_COMM_WORLD);}
void pangulu_cm_isend(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    MPI_Request req;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < send_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Isend(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &req);
    }
    if(send_times == 0){
        MPI_Isend(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD, &req);
    }
}
void pangulu_cm_send(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < send_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Send(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD);
    }
    if(send_times == 0){
        MPI_Send(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD);
    }
}
void pangulu_cm_recv(char* buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    MPI_Status stat;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t recv_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < recv_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Recv(buf + iter * send_maxlen, count_current, MPI_CHAR, fetch_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &stat);
    }
    if(recv_times == 0){
        MPI_Recv(buf, 0, MPI_CHAR, fetch_rank, tag, MPI_COMM_WORLD, &stat);
    }
}
void pangulu_cm_sync_asym(int wake_rank)
{
    pangulu_int32_t sum_rank_size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&sum_rank_size);
    if (rank == wake_rank)
    {
        for (int i = 0; i < sum_rank_size; i++)
        {
            if (i != wake_rank)
            {
                MPI_Send(&sum_rank_size, 1, MPI_INT, i, 0xCAFE, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        int mpi_buf_int;
        int mpi_flag;
        MPI_Status mpi_stat;
        while (1)
        {
            mpi_flag = 0;
            MPI_Iprobe(wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_flag, &mpi_stat);
            if (mpi_flag != 0 && mpi_stat.MPI_TAG == 0xCAFE)
            {
                MPI_Recv(&mpi_buf_int, 1, MPI_INT, wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_stat);
                if (mpi_buf_int == sum_rank_size)
                {
                    break;
                }
                else
                {
                    printf(PANGULU_E_ASYM);
                    exit(1);
                }
            }
            usleep(50);
        }
    }
    pangulu_cm_sync();
}
void pangulu_cm_probe(MPI_Status *status)
{
    int have_msg=0;
    do{
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &have_msg, status);
        if(have_msg){
            return;
        }
        usleep(10);
    }while(!have_msg);
}

// void pangulu_cm_distribute_bcsc_to_distbcsc(
//     pangulu_int32_t root_rank,
//     int rootproc_free_originmatrix,
//     pangulu_exblock_idx* n,

//     pangulu_exblock_ptr** bcsc_struct_pointer,
//     pangulu_exblock_idx** bcsc_struct_index,
//     pangulu_inblock_ptr** bcsc_struct_nnz,
//     pangulu_inblock_ptr*** bcsc_inblock_pointers,
//     pangulu_inblock_idx*** bcsc_inblock_indeces,
//     calculate_type*** bcsc_values
// ){
    

// }

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
){
    struct timeval start_time;
    pangulu_time_start(&start_time);
    // printf("1.1\n");


    pangulu_int32_t nproc = 0;
    pangulu_int32_t rank;
    pangulu_cm_size(&nproc);
    pangulu_cm_rank(&rank);
    *distcsc_nproc = nproc;
    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_cm_bcast(n, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank);
    // printf("#%d 1.2\n", rank);

    rowchunk_align *= q;
    pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(*n, rowchunk_align), nproc) * rowchunk_align;

    *distcsc_proc_nnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    if(rank == root_rank){
        pangulu_exblock_ptr* columnpointer = *distcsc_pointer;
        pangulu_exblock_idx* rowindex = *distcsc_index;
        calculate_type* value_csc = NULL;
        if(distcsc_value){
            value_csc = *distcsc_value;
        }
        (*distcsc_proc_nnzptr)[0] = 0;
    // printf("#%d 1.3.1\n", rank);
        for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
    // printf("#%d 1.3.2\n", rank);
            pangulu_exblock_idx n_loc_remote = PANGULU_MIN(col_per_rank * (target_rank + 1), *n) - PANGULU_MIN(col_per_rank * target_rank, *n);
            (*distcsc_proc_nnzptr)[target_rank + 1] = columnpointer[PANGULU_MIN(col_per_rank * (target_rank + 1), *n)];
            if(rank == target_rank){
                *n_loc = n_loc_remote;
    // printf("#%d n_loc = %d\n", rank, *n_loc);

                *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                memcpy(*distcsc_pointer, &columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
                for(pangulu_exblock_idx col = 0; col < *n_loc + 1; col++){
                    (*distcsc_pointer)[col] -= col_offset;
                }
                pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
                *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
                memcpy(*distcsc_index, &rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * nnz_loc);

                if(distcsc_value){
                    *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
                    memcpy(*distcsc_value, &value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * nnz_loc);
                }
            }else{
                MPI_Send(&n_loc_remote, 1, MPI_PANGULU_EXBLOCK_IDX, target_rank, 0, MPI_COMM_WORLD);
                MPI_Send(&columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], n_loc_remote + 1, MPI_PANGULU_EXBLOCK_PTR, target_rank, 1, MPI_COMM_WORLD);
                //MPI_Send(
                //    &rowindex[(*distcsc_proc_nnzptr)[target_rank]], 
                //    (*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank], 
                //    MPI_PANGULU_EXBLOCK_IDX, target_rank, 2, MPI_COMM_WORLD
                //);
                pangulu_cm_send(&rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * ((*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 2, 10);
                if(distcsc_value){
                    // MPI_Send(
                    //     &value_csc[(*distcsc_proc_nnzptr)[target_rank]],
                    //     (*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank],
                    //     MPI_VAL_TYPE, target_rank, 3, MPI_COMM_WORLD
                    // );
                    pangulu_cm_send(&value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * ((*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 3, 10);
                    // printf("Sent %d->%d\n", rank, target_rank);
                }else{
                    int nouse = 0;
                    MPI_Send(&nouse, 1, MPI_INT, target_rank, 4, MPI_COMM_WORLD);
                }
            }
    // printf("#%d 1.3.3\n", rank);
        }
        MPI_Bcast(*distcsc_proc_nnzptr, nproc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
        if(rootproc_free_originmatrix){
            pangulu_free(__FILE__, __LINE__, columnpointer);
            pangulu_free(__FILE__, __LINE__, rowindex);
            if(distcsc_value){
                pangulu_free(__FILE__, __LINE__, value_csc);
            }
        }
    // printf("#%d 1.3.4\n", rank);
    }else{
    // printf("#%d 1.4.1\n", rank);
        MPI_Status mpi_stat;
        MPI_Recv(n_loc, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank, 0, MPI_COMM_WORLD, &mpi_stat);
    // printf("#%d n_loc = %d\n", rank, *n_loc);
        
        *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
        MPI_Recv(*distcsc_pointer, *n_loc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, 1, MPI_COMM_WORLD, &mpi_stat);
        pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
        for(pangulu_exblock_idx col = 0; col < *n_loc + 1; col++){
            (*distcsc_pointer)[col] -= col_offset;
        }
        pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
        *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
        //MPI_Recv(*distcsc_index, nnz_loc, MPI_PANGULU_EXBLOCK_IDX, root_rank, 2, MPI_COMM_WORLD, &mpi_stat);
        pangulu_cm_recv(*distcsc_index, sizeof(pangulu_exblock_idx) * nnz_loc, root_rank, 2, 10);
    // printf("#%d 1.4.2\n", rank);

        MPI_Probe(root_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_stat);
        // printf("Probe\n");
        if((mpi_stat.MPI_TAG%10) == 3){
            *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
            // MPI_Recv(*distcsc_value, nnz_loc, MPI_VAL_TYPE, root_rank, 3, MPI_COMM_WORLD, &mpi_stat);
            pangulu_cm_recv(*distcsc_value, sizeof(calculate_type) * nnz_loc, root_rank, 3, 10);
        }else if(mpi_stat.MPI_TAG == 4){
            int nouse = 0;
            MPI_Recv(&nouse, 1, MPI_INT, root_rank, 4, MPI_COMM_WORLD, &mpi_stat);
        }
    // printf("#%d 1.4.25\n", rank);

        MPI_Bcast(*distcsc_proc_nnzptr, nproc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
    // printf("#%d 1.4.3\n", rank);
    }
    // printf("#%d 1.5\n", rank);

    // printf("[PanguLU LOG] pangulu_cm_distribute_csc_to_distcsc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

// void pangulu_cm_distribute_distcsc_to_distbcsc(
//     int rootproc_free_originmatrix,
//     pangulu_exblock_idx n_glo,
//     pangulu_exblock_idx n_loc,
//     pangulu_inblock_idx block_order,
    
//     pangulu_exblock_ptr* distcsc_proc_nnzptr,
//     pangulu_exblock_ptr* distcsc_pointer,
//     pangulu_exblock_idx* distcsc_index,
//     calculate_type* distcsc_value,

//     pangulu_exblock_ptr** bcsc_struct_pointer,
//     pangulu_exblock_idx** bcsc_struct_index,
//     pangulu_exblock_ptr** bcsc_struct_nnzptr,
//     pangulu_inblock_ptr*** bcsc_inblock_pointers,
//     pangulu_inblock_idx*** bcsc_inblock_indeces,
//     calculate_type*** bcsc_values
// ){
// #define _PANGULU_SET_VALUE_SIZE(size) ((distcsc_value)?(size):(0))

//     struct timeval start_time;
//     pangulu_time_start(&start_time);

//     pangulu_int32_t rank, nproc;
//     pangulu_cm_rank(&rank);
//     pangulu_cm_size(&nproc);

//     if(distcsc_proc_nnzptr){
//         pangulu_free(__FILE__, __LINE__, distcsc_proc_nnzptr);
//     }

//     int preprocess_ompnum_separate_block = 2;

//     bind_to_core((rank * preprocess_ompnum_separate_block) % sysconf(_SC_NPROCESSORS_ONLN));

//     #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
//     {
//         bind_to_core((rank * preprocess_ompnum_separate_block + omp_get_thread_num()) % sysconf(_SC_NPROCESSORS_ONLN));
//     }

//     pangulu_int64_t p = sqrt(nproc);
//     while ((nproc % p) != 0)
//     {
//         p--;
//     }
//     pangulu_int64_t q = nproc / p;
//     pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(n_glo, block_order * q), nproc) * (block_order * q);
//     pangulu_int64_t block_length = PANGULU_ICEIL(col_per_rank, block_order);
//     pangulu_int64_t block_length_col = PANGULU_ICEIL(n_glo, block_order);
//     pangulu_exblock_ptr nnz = distcsc_pointer[n_loc];
//     pangulu_int64_t bit_length = (block_length_col + 31) / 32;
//     pangulu_int64_t block_num = 0;
//     pangulu_int64_t *block_nnz_pt;

//     pangulu_int64_t avg_nnz = PANGULU_ICEIL(nnz, preprocess_ompnum_separate_block);
//     pangulu_int64_t *block_row_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
//     for (int i = 0; i < block_length; i++)
//     {
//         block_row_nnz_pt[i] = distcsc_pointer[PANGULU_MIN(i * block_order, n_loc)];
//     }
//     block_row_nnz_pt[block_length] = distcsc_pointer[n_loc];

//     int *thread_pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (preprocess_ompnum_separate_block + 1));
//     thread_pt[0] = 0;
//     for (int i = 1; i < preprocess_ompnum_separate_block + 1; i++)
//     {
//         thread_pt[i] = binarylowerbound(block_row_nnz_pt, block_length, avg_nnz * i);
//     }
//     pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
//     block_row_nnz_pt = NULL;

//     pangulu_int64_t *block_row_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
//     memset(block_row_pt, 0, sizeof(pangulu_int64_t) * (block_length + 1));

//     unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * preprocess_ompnum_separate_block);

// #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
//     {
//         int tid = omp_get_thread_num();
//         unsigned int *tmp_bit = bit_array + bit_length * tid;

//         for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
//         {
//             memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

//             int start_row = level * block_order;
//             int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

//             for (int rid = start_row; rid < end_row; rid++)
//             {
//                 for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
//                 {
//                     pangulu_int32_t colidx = distcsc_index[idx];
//                     pangulu_int32_t block_cid = colidx / block_order;
//                     setbit(tmp_bit[block_cid / 32], block_cid % 32);
//                 }
//             }

//             pangulu_int64_t tmp_blocknum = 0;
//             for (int i = 0; i < bit_length; i++)
//             {
//                 tmp_blocknum += __builtin_popcount(tmp_bit[i]);
//             }

//             block_row_pt[level] = tmp_blocknum;
//         }
//     }
//     exclusive_scan_1(block_row_pt, block_length + 1);
//     block_num = block_row_pt[block_length];

//     block_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num + 1));
//     memset(block_nnz_pt, 0, sizeof(pangulu_int64_t) * (block_num + 1));
//     pangulu_int32_t *block_col_idx = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);

//     int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * preprocess_ompnum_separate_block);
// #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
//     {
//         int tid = omp_get_thread_num();
//         unsigned int *tmp_bit = bit_array + bit_length * tid;
//         int *tmp_count = count_array + block_length_col * tid;

//         for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
//         {
//             memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
//             memset(tmp_count, 0, sizeof(int) * block_length_col);

//             pangulu_int64_t *cur_block_nnz_pt = block_nnz_pt + block_row_pt[level];
//             pangulu_int32_t *cur_block_col_idx = block_col_idx + block_row_pt[level];

//             int start_row = level * block_order;
//             int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

//             for (int rid = start_row; rid < end_row; rid++)
//             {
//                 for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
//                 {
//                     pangulu_int32_t colidx = distcsc_index[idx];
//                     pangulu_int32_t block_cid = colidx / block_order;
//                     setbit(tmp_bit[block_cid / 32], block_cid % 32);
//                     tmp_count[block_cid]++;
//                 }
//             }

//             pangulu_int64_t cnt = 0;
//             for (int i = 0; i < block_length_col; i++)
//             {
//                 if (getbit(tmp_bit[i / 32], i % 32))
//                 {
//                     cur_block_nnz_pt[cnt] = tmp_count[i];
//                     cur_block_col_idx[cnt] = i;
//                     cnt++;
//                 }
//             }
//         }
//     }
//     pangulu_free(__FILE__, __LINE__, bit_array);
//     bit_array = NULL;
//     pangulu_free(__FILE__, __LINE__, count_array);
//     count_array = NULL;
//     exclusive_scan_1(block_nnz_pt, block_num + 1);
    
    
//     pangulu_exblock_ptr* nzblk_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
//     pangulu_exblock_ptr* nnz_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
//     memset(nzblk_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * nproc);
//     memset(nnz_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * nproc);
//     for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
//         for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
//             pangulu_exblock_idx brow = block_col_idx[bidx];
//             nzblk_each_rank_ptr[(brow % p) * q + (bcol % q) + 1]++;
//             nnz_each_rank_ptr[(brow % p) * q + (bcol % q) + 1] += (block_nnz_pt[bidx + 1] - block_nnz_pt[bidx]);
//         }
//     }
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         nzblk_each_rank_ptr[remote_rank + 1] += nzblk_each_rank_ptr[remote_rank];
//         nnz_each_rank_ptr[remote_rank + 1] += nnz_each_rank_ptr[remote_rank];
//     }
//     char** csc_draft_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         csc_draft_remote[remote_rank] = pangulu_malloc(
//             __FILE__, __LINE__, 
//             sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//             sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
//             sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
//         );
//         memset(csc_draft_remote[remote_rank], 0, 
//             sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//             sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
//             sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
//         );
//     }
    
//     for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
//         for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
//             pangulu_exblock_idx brow = block_col_idx[bidx];
//             pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
//             pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
//             pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
//             remote_bcolptr[bcol + 1]++;
//         }
//     }

//     pangulu_exblock_ptr* aid_arr_colptr_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
//         for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
//             remote_bcolptr[bcol + 1] += remote_bcolptr[bcol];
//         }
//         memcpy(&aid_arr_colptr_remote[(block_length + 1) * remote_rank], remote_bcolptr, sizeof(pangulu_exblock_ptr) * (block_length + 1));
//     }

//     for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
//         for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
//             pangulu_exblock_idx brow = block_col_idx[bidx];
//             pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
//             pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
            
//             pangulu_exblock_idx* remote_browidx = 
//                 csc_draft_remote[remote_rank] + 
//                 sizeof(pangulu_exblock_ptr) * (block_length + 1);
//             pangulu_exblock_ptr* remote_blknnzptr = 
//                 csc_draft_remote[remote_rank] + 
//                 sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                 sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
            
//             remote_browidx[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]] = brow;
//             remote_blknnzptr[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol] + 1] = nnz_in_blk;
//             aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]++;
//         }
//     }
//     pangulu_free(__FILE__, __LINE__, aid_arr_colptr_remote);
//     aid_arr_colptr_remote = NULL;
//     pangulu_free(__FILE__, __LINE__, block_nnz_pt);
//     block_nnz_pt = NULL;
    
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
//         pangulu_exblock_ptr* remote_blknnzptr = 
//                 csc_draft_remote[remote_rank] + 
//                 sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                 sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
//         for(pangulu_inblock_ptr bidx = 0; bidx < remote_bcolptr[block_length]; bidx++){
//             remote_blknnzptr[bidx + 1] += remote_blknnzptr[bidx];
//         }
//     }

//     char** block_csc_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
//     for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
//         block_csc_remote[target_rank] = pangulu_malloc(
//             __FILE__, __LINE__, 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
//             sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
//         );
//         memset(
//             block_csc_remote[target_rank], 0, 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
//             sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
//         );
//     }


// #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
//     {
//         int tid = omp_get_thread_num();
//         int* tmp_count = pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * q);

//         for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
//         {

//             memset(tmp_count, 0, sizeof(int) * block_length_col * q);
        
//             pangulu_exblock_idx start_col = level * block_order;
//             pangulu_exblock_idx end_col = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;
            
//             for (pangulu_exblock_idx col = start_col, col_in_blc = 0; col < end_col; col++, col_in_blc++)
//             {
//                 pangulu_int64_t bidx_glo = block_row_pt[level];
//                 pangulu_exblock_idx brow = block_col_idx[bidx_glo];
//                 pangulu_int32_t target_rank = (brow % p) * q + (level % q);
//                 pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
//                 pangulu_exblock_idx* remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
//                 pangulu_exblock_ptr* remote_bnnzptr = 
//                     csc_draft_remote[target_rank] + 
//                     sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                     sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
//                 pangulu_int64_t bidx = remote_bcolptr[level];

//                 pangulu_int64_t arr_len = 
//                     sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
//                     (sizeof(pangulu_inblock_idx) + _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
//                 pangulu_inblock_ptr *cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
//                 pangulu_inblock_idx *cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
//                 calculate_type *cur_block_value = NULL;
//                 if(distcsc_value){
//                     cur_block_value = (calculate_type *)(
//                         block_csc_remote[target_rank] + arr_len + 
//                         sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
//                         sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
//                     );
//                 }

//                 pangulu_exblock_ptr reorder_matrix_idx = distcsc_pointer[col];
//                 pangulu_exblock_ptr reorder_matrix_idx_ub = distcsc_pointer[col + 1];
//                 for (pangulu_exblock_ptr idx = distcsc_pointer[col]; idx < distcsc_pointer[col + 1]; idx++)
//                 {
//                     pangulu_exblock_idx row = distcsc_index[idx];
//                     brow = row / block_order;
//                     if (block_col_idx[bidx_glo] != brow)
//                     {
//                         bidx_glo = binarysearch(block_col_idx, bidx_glo, block_row_pt[level + 1], brow);
//                         target_rank = (brow % p) * q + (level % q);
//                         remote_bcolptr = csc_draft_remote[target_rank];
//                         remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
//                         remote_bnnzptr = 
//                             csc_draft_remote[target_rank] + 
//                             sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                             sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
//                         bidx = binarysearch(remote_browidx, remote_bcolptr[level], remote_bcolptr[level + 1], brow);
//                         arr_len = 
//                             sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
//                             (sizeof(pangulu_inblock_idx) + _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
//                         cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
//                         cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
//                         if(distcsc_value){
//                             cur_block_value = (calculate_type *)(
//                                 block_csc_remote[target_rank] + arr_len + 
//                                 sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
//                                 sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
//                             );
//                         }
//                     }
//                     if(distcsc_value){
//                         cur_block_value[tmp_count[(level % q) * block_length_col + brow]] = distcsc_value[reorder_matrix_idx];
//                     }
//                     reorder_matrix_idx++;
//                     cur_block_colidx[tmp_count[(level % q) * block_length_col + brow]++] = row % block_order;
//                     cur_block_rowptr[col_in_blc]++;
//                 }
//             }

//             for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
//                 pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
//                 pangulu_exblock_ptr* remote_bnnzptr = 
//                     csc_draft_remote[target_rank] + 
//                     sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                     sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
//                 for (pangulu_int64_t bidx = remote_bcolptr[level]; bidx < remote_bcolptr[level + 1]; bidx++)
//                 {
//                     pangulu_int64_t tmp_stride = bidx * (block_order + 1) * sizeof(pangulu_inblock_ptr) + remote_bnnzptr[bidx] * (sizeof(pangulu_inblock_idx) + _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)));
//                     pangulu_inblock_ptr *cur_csr_rpt = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + tmp_stride);
//                     exclusive_scan_3(cur_csr_rpt, block_order + 1);
//                 }
//             }
//         }
//         pangulu_free(__FILE__, __LINE__, tmp_count);
//         tmp_count = NULL;
//     }
//     pangulu_free(__FILE__, __LINE__, thread_pt);
//     thread_pt = NULL;
//     pangulu_free(__FILE__, __LINE__, block_row_pt);
//     block_row_pt = NULL;
//     pangulu_free(__FILE__, __LINE__, block_col_idx);
//     block_col_idx = NULL;


//     if(rootproc_free_originmatrix){
//         pangulu_free(__FILE__, __LINE__, distcsc_pointer);
//         pangulu_free(__FILE__, __LINE__, distcsc_index);
//         if(distcsc_value){
//             pangulu_free(__FILE__, __LINE__, distcsc_value); // Don't set distcsc_value to NULL.
//         }
//     }

//     // comm TAG=0
//     pangulu_cm_sync();
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         MPI_Request req;
//         pangulu_exblock_ptr nzblk_remote = nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank];
//         MPI_Isend(&nzblk_remote, 1, MPI_PANGULU_EXBLOCK_PTR, remote_rank, 0, MPI_COMM_WORLD, &req);
//     }
//     pangulu_exblock_ptr* nzblk_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         MPI_Status stat;
//         MPI_Recv(&nzblk_fetch[fetch_rank], 1, MPI_PANGULU_EXBLOCK_PTR, fetch_rank, 0, MPI_COMM_WORLD, &stat);
//     }

//     // comm TAG=1 send csc_draft_remote
//     pangulu_cm_sync();
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         MPI_Request req;
//         MPI_Isend(
//             csc_draft_remote[remote_rank], 
//             sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//                 sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
//                 sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
//             MPI_CHAR, remote_rank, 1, MPI_COMM_WORLD, &req
//         );
//     }
//     char** bstruct_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         bstruct_csc_fetch[fetch_rank] = pangulu_malloc(
//             __FILE__, __LINE__, 
//             sizeof(pangulu_exblock_ptr) * (block_length+1) + 
//             sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + 
//             sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1)
//         );
//         MPI_Status stat;
//         MPI_Recv(
//             bstruct_csc_fetch[fetch_rank],
//             sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
//             MPI_CHAR, fetch_rank, 1, MPI_COMM_WORLD, &stat
//         );
//     }
//     pangulu_cm_sync();
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         pangulu_free(__FILE__, __LINE__, csc_draft_remote[remote_rank]);
//     }
//     pangulu_free(__FILE__, __LINE__, csc_draft_remote);
//     csc_draft_remote = NULL;
    
//     // comm TAG=2 send block_csc_remote
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         MPI_Request req;
//         pangulu_int64_t send_size = sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
//             sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]) + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]);
//         MPI_Isend(block_csc_remote[remote_rank], send_size, MPI_CHAR, remote_rank, 2, MPI_COMM_WORLD, &req);
//     }
//     pangulu_free(__FILE__, __LINE__, nzblk_each_rank_ptr);
//     nzblk_each_rank_ptr = NULL;
//     pangulu_free(__FILE__, __LINE__, nnz_each_rank_ptr);
//     nnz_each_rank_ptr = NULL;
//     char** block_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         MPI_Status stat;
//         pangulu_exblock_ptr* remote_bcolptr = bstruct_csc_fetch[fetch_rank];
//         pangulu_exblock_idx* remote_browidx = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
//         pangulu_exblock_ptr* remote_bnnzptr = 
//             bstruct_csc_fetch[fetch_rank] + 
//             sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
//             sizeof(pangulu_exblock_idx) * (remote_bcolptr[block_length]);
        
//         pangulu_int64_t recv_size = 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * remote_bcolptr[block_length] + 
//             sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * remote_bnnzptr[remote_bcolptr[block_length]];
//         block_csc_fetch[fetch_rank] = pangulu_malloc(__FILE__, __LINE__, recv_size);
//         MPI_Recv(block_csc_fetch[fetch_rank], recv_size, MPI_CHAR, fetch_rank, 2, MPI_COMM_WORLD, &stat);
//     }
//     pangulu_cm_sync();
//     for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
//         pangulu_free(__FILE__, __LINE__, block_csc_remote[remote_rank]);
//     }
//     pangulu_free(__FILE__, __LINE__, block_csc_remote);
//     block_csc_remote = NULL;


//     // for(int current_rank = 0; current_rank < nproc; current_rank++){
//     //     if(rank == current_rank){
//     //         printf("rank = %d\n", rank);
//     //         for(int fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//     //             pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
//     //             pangulu_exblock_idx* bstruct_fetch_index = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1);
//     //             pangulu_exblock_ptr* bstruct_fetch_nnzptr = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank];
//     //             printf("fetch_rank = %d\n", fetch_rank);
//     //             for(int bcol = 0; bcol < block_length; bcol++){
//     //                 for(int bidx = bstruct_fetch_pointer[bcol]; bidx < bstruct_fetch_pointer[bcol + 1]; bidx++){
//     //                     int brow = bstruct_fetch_index[bidx];
//     //                     pangulu_int64_t nnzptr = bstruct_fetch_nnzptr[bidx];
//     //                     pangulu_int64_t ptr_offset = sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nnzptr;
//     //                     pangulu_inblock_ptr* colptr = block_csc_fetch[fetch_rank] + ptr_offset;
//     //                     pangulu_inblock_idx* rowidx = block_csc_fetch[fetch_rank] + ptr_offset + sizeof(pangulu_inblock_ptr) * (block_order + 1);
//     //                     calculate_type* value = block_csc_fetch[fetch_rank] + ptr_offset + sizeof(pangulu_inblock_ptr) * (block_order + 1) + sizeof(pangulu_inblock_idx) * colptr[block_order];
//     //                     printf("b(%d, %d) nnz=%d\n", brow, bcol, bidx);
//     //                     for(int col = 0; col < block_order; col++){
//     //                         printf("(%d) ", col);
//     //                         for(int idx = colptr[col]; idx < colptr[col+1]; idx++){
//     //                             printf("(%d,%lf) ", rowidx[idx], value[idx]);
//     //                         }
//     //                         printf("\n");
//     //                     }
//     //                 }
//     //             }
//     //         }
//     //         printf("\n");
//     //     }
//     //     pangulu_cm_sync();
//     // }
    
//     pangulu_exblock_ptr* struct_bcolptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
//     memset(struct_bcolptr, 0, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
//     pangulu_exblock_ptr last_fetch_rank_ptr = 0;
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
//         for(pangulu_exblock_idx brow_offset = 0; brow_offset < block_length; brow_offset++){
//             if(fetch_rank * block_length + brow_offset > block_length_col){
//                 break;
//             }
//             struct_bcolptr[fetch_rank * block_length + brow_offset] = bstruct_fetch_pointer[brow_offset] + last_fetch_rank_ptr;
//         }
//         last_fetch_rank_ptr += bstruct_fetch_pointer[block_length];
//     }
//     struct_bcolptr[block_length_col] = last_fetch_rank_ptr;

    
//     pangulu_exblock_idx* struct_browidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * struct_bcolptr[block_length_col]);
//     pangulu_exblock_ptr* struct_bnnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (struct_bcolptr[block_length_col] + 1));
//     pangulu_exblock_ptr last_fetch_bnnz_ptr = 0;
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
//         pangulu_exblock_idx* bstruct_fetch_index = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1);
//         pangulu_exblock_ptr* bstruct_fetch_nnzptr = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank];
//         for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
//             struct_browidx[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_index[bidx_offset];
//         }
//         for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
//             struct_bnnzptr[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_nnzptr[bidx_offset] + last_fetch_bnnz_ptr;
//         }
//         last_fetch_bnnz_ptr += bstruct_fetch_nnzptr[nzblk_fetch[fetch_rank]];
//     }
//     struct_bnnzptr[struct_bcolptr[block_length_col]] = last_fetch_bnnz_ptr;
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch[fetch_rank]);
//     }
//     pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch);
//     bstruct_csc_fetch = NULL;
//     pangulu_free(__FILE__, __LINE__, nzblk_fetch);
//     nzblk_fetch = NULL;

//     char* block_csc = pangulu_malloc(
//         __FILE__, __LINE__,
//         sizeof(pangulu_inblock_ptr) * (block_order + 1) * struct_bcolptr[block_length_col] + 
//         sizeof(pangulu_inblock_idx) * struct_bnnzptr[struct_bcolptr[block_length_col]] + 
//         _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[struct_bcolptr[block_length_col]]
//     );
//     for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
//         pangulu_exblock_idx bidx_ub = struct_bcolptr[PANGULU_MIN((fetch_rank + 1) * block_length, block_length_col)];
//         pangulu_exblock_idx bidx_lb = struct_bcolptr[PANGULU_MIN(fetch_rank * block_length, block_length_col)];
//         pangulu_int64_t offset = 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx_lb + 
//             sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx_lb] + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx_lb];
//         pangulu_int64_t copy_size = 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * (bidx_ub - bidx_lb) +
//             sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]) + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]);
//         memcpy(block_csc + offset, block_csc_fetch[fetch_rank], copy_size);
//         pangulu_free(__FILE__, __LINE__, block_csc_fetch[fetch_rank]);
//     }
//     pangulu_free(__FILE__, __LINE__, block_csc_fetch);
//     block_csc_fetch = NULL;

//     pangulu_inblock_ptr** inblock_pointers = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr*) * struct_bcolptr[block_length_col]);
//     pangulu_inblock_idx** inblock_indeces = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx*) * struct_bcolptr[block_length_col]);
//     calculate_type** inblock_values = NULL;
//     if(distcsc_value){
//         inblock_values = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type*) * struct_bcolptr[block_length_col]);
//     }

//     for(pangulu_exblock_ptr bidx = 0; bidx < struct_bcolptr[block_length_col]; bidx++){
//         pangulu_int64_t offset = 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx + 
//             sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx] + 
//             _PANGULU_SET_VALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx];
//         inblock_pointers[bidx] = block_csc + offset;
//         inblock_indeces[bidx] = block_csc + offset + 
//             sizeof(pangulu_inblock_ptr) * (block_order + 1);
//         if(distcsc_value){
//             inblock_values[bidx] = block_csc + offset + 
//                 sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
//                 sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx + 1] - struct_bnnzptr[bidx]);
//         }
//     }


//     *bcsc_struct_pointer = struct_bcolptr;
//     *bcsc_struct_index = struct_browidx;
//     *bcsc_struct_nnzptr = struct_bnnzptr;
//     *bcsc_inblock_pointers = inblock_pointers;
//     *bcsc_inblock_indeces = inblock_indeces;
//     if(distcsc_value){
//         *bcsc_values = inblock_values;
//     }

//     bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));

//     // printf("[PanguLU LOG] pangulu_cm_distribute_distcsc_to_distbcsc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);

// #undef _PANGULU_SET_VALUE_SIZE
// }


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
){
#define _PANGULU_SET_VALUE_SIZE(size) ((distcsc_value)?(size):(0))
#define _PANGULU_SET_BVALUE_SIZE(size) ((malloc_distbcsc_value)?(size):(0))

    if(distcsc_value){
        malloc_distbcsc_value = 1;
    }
    // printf("malloc_distbcsc_value = %d\n", malloc_distbcsc_value);

    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_int32_t rank, nproc;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&nproc);

    if(distcsc_proc_nnzptr){
        pangulu_free(__FILE__, __LINE__, distcsc_proc_nnzptr);
    }
    // printf("1.1\n");

    int preprocess_ompnum_separate_block = 2;

    bind_to_core((rank * preprocess_ompnum_separate_block) % sysconf(_SC_NPROCESSORS_ONLN));

    #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        bind_to_core((rank * preprocess_ompnum_separate_block + omp_get_thread_num()) % sysconf(_SC_NPROCESSORS_ONLN));
    }

    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(n_glo, block_order * q), nproc) * (block_order * q);
    pangulu_int64_t block_length = PANGULU_ICEIL(col_per_rank, block_order);
    pangulu_int64_t block_length_col = PANGULU_ICEIL(n_glo, block_order);
    pangulu_exblock_ptr nnz = distcsc_pointer[n_loc];
    pangulu_int64_t bit_length = (block_length_col + 31) / 32;
    pangulu_int64_t block_num = 0;
    pangulu_int64_t *block_nnz_pt;
    // printf("1.2\n");

    pangulu_int64_t avg_nnz = PANGULU_ICEIL(nnz, preprocess_ompnum_separate_block);
    pangulu_int64_t *block_row_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
    for (int i = 0; i < block_length; i++)
    {
        block_row_nnz_pt[i] = distcsc_pointer[PANGULU_MIN(i * block_order, n_loc)];
    }
    block_row_nnz_pt[block_length] = distcsc_pointer[n_loc];

    int *thread_pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (preprocess_ompnum_separate_block + 1));
    thread_pt[0] = 0;
    for (int i = 1; i < preprocess_ompnum_separate_block + 1; i++)
    {
        thread_pt[i] = binarylowerbound(block_row_nnz_pt, block_length, avg_nnz * i);
    }
    pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
    block_row_nnz_pt = NULL;

    pangulu_int64_t *block_row_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
    memset(block_row_pt, 0, sizeof(pangulu_int64_t) * (block_length + 1));

    unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * preprocess_ompnum_separate_block);

#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        unsigned int *tmp_bit = bit_array + bit_length * tid;

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {
            memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

            int start_row = level * block_order;
            int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

            for (int rid = start_row; rid < end_row; rid++)
            {
                for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
                {
                    pangulu_int32_t colidx = distcsc_index[idx];
                    pangulu_int32_t block_cid = colidx / block_order;
                    setbit(tmp_bit[block_cid / 32], block_cid % 32);
                }
            }

            pangulu_int64_t tmp_blocknum = 0;
            for (int i = 0; i < bit_length; i++)
            {
                tmp_blocknum += __builtin_popcount(tmp_bit[i]);
            }

            block_row_pt[level] = tmp_blocknum;
        }
    }
    exclusive_scan_1(block_row_pt, block_length + 1);
    block_num = block_row_pt[block_length];

    block_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num + 1));
    memset(block_nnz_pt, 0, sizeof(pangulu_int64_t) * (block_num + 1));
    pangulu_int32_t *block_col_idx = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);

    int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * preprocess_ompnum_separate_block);
#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        unsigned int *tmp_bit = bit_array + bit_length * tid;
        int *tmp_count = count_array + block_length_col * tid;

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {
            memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
            memset(tmp_count, 0, sizeof(int) * block_length_col);

            pangulu_int64_t *cur_block_nnz_pt = block_nnz_pt + block_row_pt[level];
            pangulu_int32_t *cur_block_col_idx = block_col_idx + block_row_pt[level];

            int start_row = level * block_order;
            int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

            for (int rid = start_row; rid < end_row; rid++)
            {
                for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
                {
                    pangulu_int32_t colidx = distcsc_index[idx];
                    pangulu_int32_t block_cid = colidx / block_order;
                    setbit(tmp_bit[block_cid / 32], block_cid % 32);
                    tmp_count[block_cid]++;
                }
            }

            pangulu_int64_t cnt = 0;
            for (int i = 0; i < block_length_col; i++)
            {
                if (getbit(tmp_bit[i / 32], i % 32))
                {
                    cur_block_nnz_pt[cnt] = tmp_count[i];
                    cur_block_col_idx[cnt] = i;
                    cnt++;
                }
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, bit_array);
    bit_array = NULL;
    pangulu_free(__FILE__, __LINE__, count_array);
    count_array = NULL;
    exclusive_scan_1(block_nnz_pt, block_num + 1);
    
    // printf("1.3\n");
    
    pangulu_exblock_ptr* nzblk_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    pangulu_exblock_ptr* nnz_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    memset(nzblk_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    memset(nnz_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            nzblk_each_rank_ptr[(brow % p) * q + (bcol % q) + 1]++;
            nnz_each_rank_ptr[(brow % p) * q + (bcol % q) + 1] += (block_nnz_pt[bidx + 1] - block_nnz_pt[bidx]);
        }
    }
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        nzblk_each_rank_ptr[remote_rank + 1] += nzblk_each_rank_ptr[remote_rank];
        nnz_each_rank_ptr[remote_rank + 1] += nnz_each_rank_ptr[remote_rank];
    }
    char** csc_draft_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        csc_draft_remote[remote_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
        );
        memset(csc_draft_remote[remote_rank], 0, 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
        );
    }
    
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
            pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
            pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
            remote_bcolptr[bcol + 1]++;
        }
    }
    // printf("1.4\n");

    pangulu_exblock_ptr* aid_arr_colptr_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
        for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
            remote_bcolptr[bcol + 1] += remote_bcolptr[bcol];
        }
        memcpy(&aid_arr_colptr_remote[(block_length + 1) * remote_rank], remote_bcolptr, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    }

    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
            pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
            
            pangulu_exblock_idx* remote_browidx = 
                csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1);
            pangulu_exblock_ptr* remote_blknnzptr = 
                csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
            
            remote_browidx[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]] = brow;
            remote_blknnzptr[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol] + 1] = nnz_in_blk;
            aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, aid_arr_colptr_remote);
    aid_arr_colptr_remote = NULL;
    pangulu_free(__FILE__, __LINE__, block_nnz_pt);
    block_nnz_pt = NULL;
    
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
        pangulu_exblock_ptr* remote_blknnzptr = 
                csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
        for(pangulu_inblock_ptr bidx = 0; bidx < remote_bcolptr[block_length]; bidx++){
            remote_blknnzptr[bidx + 1] += remote_blknnzptr[bidx];
        }
    }

    char** block_csc_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
        // printf("target_rank = %d, nzblk = %d, nnz = %d (%d - %d)\n", 
        //     target_rank, 
        //     nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank], 
        //     nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank],
        //     nnz_each_rank_ptr[target_rank + 1],
        //     nnz_each_rank_ptr[target_rank]
        //     );
        block_csc_remote[target_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
        );
        memset(
            block_csc_remote[target_rank], 0, 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
        );
    }
    // printf("#%d 1.5\n", rank);


#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        int* tmp_count = pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * q);

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {

            memset(tmp_count, 0, sizeof(int) * block_length_col * q);
        
            pangulu_exblock_idx start_col = level * block_order;
            pangulu_exblock_idx end_col = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;
            
            for (pangulu_exblock_idx col = start_col, col_in_blc = 0; col < end_col; col++, col_in_blc++)
            {
                pangulu_int64_t bidx_glo = block_row_pt[level];
                pangulu_exblock_idx brow = block_col_idx[bidx_glo];
                pangulu_int32_t target_rank = (brow % p) * q + (level % q);
                pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
                pangulu_exblock_idx* remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                pangulu_exblock_ptr* remote_bnnzptr = 
                    csc_draft_remote[target_rank] + 
                    sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                pangulu_int64_t bidx = remote_bcolptr[level];

                pangulu_int64_t arr_len = 
                    sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
                    (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
                pangulu_inblock_ptr *cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
                pangulu_inblock_idx *cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
                calculate_type *cur_block_value = NULL;
                if(malloc_distbcsc_value){
                    cur_block_value = (calculate_type *)(
                        block_csc_remote[target_rank] + arr_len + 
                        sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                        sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
                    );
                }

                pangulu_exblock_ptr reorder_matrix_idx = distcsc_pointer[col];
                pangulu_exblock_ptr reorder_matrix_idx_ub = distcsc_pointer[col + 1];
                for (pangulu_exblock_ptr idx = distcsc_pointer[col]; idx < distcsc_pointer[col + 1]; idx++)
                {
                    pangulu_exblock_idx row = distcsc_index[idx];
                    brow = row / block_order;
                    if (block_col_idx[bidx_glo] != brow)
                    {
                        bidx_glo = binarysearch(block_col_idx, bidx_glo, block_row_pt[level + 1], brow);
                        target_rank = (brow % p) * q + (level % q);
                        remote_bcolptr = csc_draft_remote[target_rank];
                        remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                        remote_bnnzptr = 
                            csc_draft_remote[target_rank] + 
                            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                        bidx = binarysearch(remote_browidx, remote_bcolptr[level], remote_bcolptr[level + 1], brow);
                        arr_len = 
                            sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
                            (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
                        cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
                        cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
                        if(malloc_distbcsc_value){
                            cur_block_value = (calculate_type *)(
                                block_csc_remote[target_rank] + arr_len + 
                                sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                                sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
                            );
                        }
                    }
                    if(distcsc_value){
                        cur_block_value[tmp_count[(level % q) * block_length_col + brow]] = distcsc_value[reorder_matrix_idx];
                    }else if(malloc_distbcsc_value){
                        cur_block_value[tmp_count[(level % q) * block_length_col + brow]] = 0;
                    }
                    reorder_matrix_idx++;
                    cur_block_colidx[tmp_count[(level % q) * block_length_col + brow]++] = row % block_order;
                    cur_block_rowptr[col_in_blc]++;
                }
            }

            for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
                pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
                pangulu_exblock_ptr* remote_bnnzptr = 
                    csc_draft_remote[target_rank] + 
                    sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                for (pangulu_int64_t bidx = remote_bcolptr[level]; bidx < remote_bcolptr[level + 1]; bidx++)
                {
                    pangulu_int64_t tmp_stride = bidx * (block_order + 1) * sizeof(pangulu_inblock_ptr) + remote_bnnzptr[bidx] * (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)));
                    pangulu_inblock_ptr *cur_csr_rpt = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + tmp_stride);
                    exclusive_scan_3(cur_csr_rpt, block_order + 1);
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, tmp_count);
        tmp_count = NULL;
    }
    pangulu_free(__FILE__, __LINE__, thread_pt);
    thread_pt = NULL;
    pangulu_free(__FILE__, __LINE__, block_row_pt);
    block_row_pt = NULL;
    pangulu_free(__FILE__, __LINE__, block_col_idx);
    block_col_idx = NULL;

    // printf("#%d 1.6\n", rank);

    if(rootproc_free_originmatrix){
        pangulu_free(__FILE__, __LINE__, distcsc_pointer);
        pangulu_free(__FILE__, __LINE__, distcsc_index);
        if(distcsc_value){
            pangulu_free(__FILE__, __LINE__, distcsc_value); // Don't set distcsc_value to NULL.
        }
    }

    // // comm TAG=0*
    // pangulu_cm_sync();
    // for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
    //     MPI_Request req;
    //     pangulu_exblock_ptr nzblk_remote = nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank];
    //     MPI_Isend(&nzblk_remote, 1, MPI_PANGULU_EXBLOCK_PTR, remote_rank, remote_rank, MPI_COMM_WORLD, &req);
    // }
    // pangulu_exblock_ptr* nzblk_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
    // for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
    //     MPI_Status stat;
    //     MPI_Recv(&nzblk_fetch[fetch_rank], 1, MPI_PANGULU_EXBLOCK_PTR, fetch_rank, rank, MPI_COMM_WORLD, &stat);
    // }
    
    // comm TAG=0*
    pangulu_cm_sync();
    
    pangulu_exblock_ptr* nzblk_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        MPI_Status stat;
        if(rank == fetch_rank){
            for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
                MPI_Request req;
                pangulu_exblock_ptr nzblk_remote = nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank];
                if(rank == remote_rank){
                    nzblk_fetch[fetch_rank] = nzblk_remote;
                }else{
                    MPI_Send(&nzblk_remote, 1, MPI_PANGULU_EXBLOCK_PTR, remote_rank, remote_rank, MPI_COMM_WORLD);
                }
            }
        }else{
            MPI_Recv(&nzblk_fetch[fetch_rank], 1, MPI_PANGULU_EXBLOCK_PTR, fetch_rank, rank, MPI_COMM_WORLD, &stat);
        }
    }

    // comm TAG=1 send csc_draft_remote
    pangulu_cm_sync();
    // printf("#%d 1.7\n", rank);
    // for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
    //     // MPI_Request req;
    //     // MPI_Isend(
    //     //     csc_draft_remote[remote_rank], 
    //     //     sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
    //     //         sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
    //     //         sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
    //     //     MPI_CHAR, remote_rank, 1, MPI_COMM_WORLD, &req
    //     // );
    //     pangulu_cm_isend(
    //         csc_draft_remote[remote_rank], 
    //         sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
    //             sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
    //             sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
    //         remote_rank, 1, 10
    //     );
    // }
    // char** bstruct_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    // for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
    //     bstruct_csc_fetch[fetch_rank] = pangulu_malloc(
    //         __FILE__, __LINE__, 
    //         sizeof(pangulu_exblock_ptr) * (block_length+1) + 
    //         sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + 
    //         sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1)
    //     );
    //     // MPI_Status stat;
    //     // MPI_Recv(
    //     //     bstruct_csc_fetch[fetch_rank],
    //     //     sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
    //     //     MPI_CHAR, fetch_rank, 1, MPI_COMM_WORLD, &stat
    //     // );
    //     pangulu_cm_recv(
    //         bstruct_csc_fetch[fetch_rank],
    //         sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
    //         fetch_rank, 1, 10
    //     );
    // }
    char** bstruct_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        bstruct_csc_fetch[fetch_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_exblock_ptr) * (block_length+1) + 
            sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + 
            sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1)
        );
        if(rank == fetch_rank){
            for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
                if(remote_rank == rank){
                    memcpy(bstruct_csc_fetch[fetch_rank], csc_draft_remote[remote_rank], 
                        sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1));
                }else{
                    // MPI_Request req;
                    // MPI_Isend(
                    //     csc_draft_remote[remote_rank], 
                    //     sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                    //         sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                    //         sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
                    //     MPI_CHAR, remote_rank, 1, MPI_COMM_WORLD, &req
                    // );
                    pangulu_cm_send(
                        csc_draft_remote[remote_rank], 
                        sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
                        remote_rank, 1 * nproc + remote_rank, 3 * nproc
                    );
                    // printf("#%d -> #%d S\n", rank, remote_rank);
                }
            }
        }else{
            // MPI_Status stat;
            // MPI_Recv(
            //     bstruct_csc_fetch[fetch_rank],
            //     sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
            //     MPI_CHAR, fetch_rank, 1, MPI_COMM_WORLD, &stat
            // );
            pangulu_cm_recv(
                bstruct_csc_fetch[fetch_rank],
                sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
                fetch_rank, 1 * nproc + rank, 3 * nproc
            );
            // printf("#%d -> #%d R\n", fetch_rank, rank);
        }
        pangulu_cm_sync();
        // printf("#%d sync %d\n", rank, fetch_rank);
    }
    pangulu_cm_sync();
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_free(__FILE__, __LINE__, csc_draft_remote[remote_rank]);
    }
    pangulu_free(__FILE__, __LINE__, csc_draft_remote);
    csc_draft_remote = NULL;
    // printf("#%d 1.8\n", rank);
    
    // comm TAG=2* send block_csc_remote
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        // MPI_Request req;
        pangulu_int64_t send_size = sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]);
        // MPI_Isend(block_csc_remote[remote_rank], send_size, MPI_CHAR, remote_rank, 2, MPI_COMM_WORLD, &req);
        pangulu_cm_isend(block_csc_remote[remote_rank], send_size, remote_rank, 2 * nproc + remote_rank, 3 * nproc);
    }
    pangulu_free(__FILE__, __LINE__, nzblk_each_rank_ptr);
    nzblk_each_rank_ptr = NULL;
    pangulu_free(__FILE__, __LINE__, nnz_each_rank_ptr);
    nnz_each_rank_ptr = NULL;
    char** block_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        MPI_Status stat;
        pangulu_exblock_ptr* remote_bcolptr = bstruct_csc_fetch[fetch_rank];
        pangulu_exblock_idx* remote_browidx = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
        pangulu_exblock_ptr* remote_bnnzptr = 
            bstruct_csc_fetch[fetch_rank] + 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (remote_bcolptr[block_length]);
        
        pangulu_int64_t recv_size = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * remote_bcolptr[block_length] + 
            sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * remote_bnnzptr[remote_bcolptr[block_length]];
        block_csc_fetch[fetch_rank] = pangulu_malloc(__FILE__, __LINE__, recv_size);
        // MPI_Recv(block_csc_fetch[fetch_rank], recv_size, MPI_CHAR, fetch_rank, 2, MPI_COMM_WORLD, &stat);
        pangulu_cm_recv(block_csc_fetch[fetch_rank], recv_size, fetch_rank, 2 * nproc + rank, 3 * nproc);
    }
    pangulu_cm_sync();
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_free(__FILE__, __LINE__, block_csc_remote[remote_rank]);
    }
    pangulu_free(__FILE__, __LINE__, block_csc_remote);
    block_csc_remote = NULL;

    // printf("#%d 1.9\n", rank);

    // for(int current_rank = 0; current_rank < nproc; current_rank++){
    //     if(rank == current_rank){
    //         printf("rank = %d\n", rank);
    //         for(int fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
    //             pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
    //             pangulu_exblock_idx* bstruct_fetch_index = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1);
    //             pangulu_exblock_ptr* bstruct_fetch_nnzptr = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank];
    //             printf("fetch_rank = %d\n", fetch_rank);
    //             for(int bcol = 0; bcol < block_length; bcol++){
    //                 for(int bidx = bstruct_fetch_pointer[bcol]; bidx < bstruct_fetch_pointer[bcol + 1]; bidx++){
    //                     int brow = bstruct_fetch_index[bidx];
    //                     pangulu_int64_t nnzptr = bstruct_fetch_nnzptr[bidx];
    //                     pangulu_int64_t ptr_offset = sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nnzptr;
    //                     pangulu_inblock_ptr* colptr = block_csc_fetch[fetch_rank] + ptr_offset;
    //                     pangulu_inblock_idx* rowidx = block_csc_fetch[fetch_rank] + ptr_offset + sizeof(pangulu_inblock_ptr) * (block_order + 1);
    //                     calculate_type* value = block_csc_fetch[fetch_rank] + ptr_offset + sizeof(pangulu_inblock_ptr) * (block_order + 1) + sizeof(pangulu_inblock_idx) * colptr[block_order];
    //                     printf("b(%d, %d) nnz=%d\n", brow, bcol, bidx);
    //                     for(int col = 0; col < block_order; col++){
    //                         printf("(%d) ", col);
    //                         for(int idx = colptr[col]; idx < colptr[col+1]; idx++){
    //                             printf("(%d,%lf) ", rowidx[idx], value[idx]);
    //                         }
    //                         printf("\n");
    //                     }
    //                 }
    //             }
    //         }
    //         printf("\n");
    //     }
    //     pangulu_cm_sync();
    // }
    
    pangulu_exblock_ptr* struct_bcolptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
    memset(struct_bcolptr, 0, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
    pangulu_exblock_ptr last_fetch_rank_ptr = 0;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
        for(pangulu_exblock_idx brow_offset = 0; brow_offset < block_length; brow_offset++){
            if(fetch_rank * block_length + brow_offset > block_length_col){
                break;
            }
            struct_bcolptr[fetch_rank * block_length + brow_offset] = bstruct_fetch_pointer[brow_offset] + last_fetch_rank_ptr;
        }
        last_fetch_rank_ptr += bstruct_fetch_pointer[block_length];
    }
    struct_bcolptr[block_length_col] = last_fetch_rank_ptr;

    // printf("#%d 1.10\n", rank);
    
    pangulu_exblock_idx* struct_browidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * struct_bcolptr[block_length_col]);
    pangulu_exblock_ptr* struct_bnnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (struct_bcolptr[block_length_col] + 1));
    pangulu_exblock_ptr last_fetch_bnnz_ptr = 0;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
        pangulu_exblock_idx* bstruct_fetch_index = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1);
        pangulu_exblock_ptr* bstruct_fetch_nnzptr = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank];
        for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
            struct_browidx[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_index[bidx_offset];
        }
        for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
            struct_bnnzptr[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_nnzptr[bidx_offset] + last_fetch_bnnz_ptr;
        }
        last_fetch_bnnz_ptr += bstruct_fetch_nnzptr[nzblk_fetch[fetch_rank]];
    }
    struct_bnnzptr[struct_bcolptr[block_length_col]] = last_fetch_bnnz_ptr;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch[fetch_rank]);
    }
    pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch);
    bstruct_csc_fetch = NULL;
    pangulu_free(__FILE__, __LINE__, nzblk_fetch);
    nzblk_fetch = NULL;
    // printf("#%d 1.11\n", rank);

    char* block_csc = pangulu_malloc(
        __FILE__, __LINE__,
        sizeof(pangulu_inblock_ptr) * (block_order + 1) * struct_bcolptr[block_length_col] + 
        sizeof(pangulu_inblock_idx) * struct_bnnzptr[struct_bcolptr[block_length_col]] + 
        _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[struct_bcolptr[block_length_col]]
    );
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_idx bidx_ub = struct_bcolptr[PANGULU_MIN((fetch_rank + 1) * block_length, block_length_col)];
        pangulu_exblock_idx bidx_lb = struct_bcolptr[PANGULU_MIN(fetch_rank * block_length, block_length_col)];
        pangulu_int64_t offset = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx_lb + 
            sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx_lb] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx_lb];
        pangulu_int64_t copy_size = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (bidx_ub - bidx_lb) +
            sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]);
        memcpy(block_csc + offset, block_csc_fetch[fetch_rank], copy_size);
        pangulu_free(__FILE__, __LINE__, block_csc_fetch[fetch_rank]);
    }
    pangulu_free(__FILE__, __LINE__, block_csc_fetch);
    block_csc_fetch = NULL;
    // printf("#%d 1.12\n", rank);

    pangulu_inblock_ptr** inblock_pointers = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr*) * struct_bcolptr[block_length_col]);
    pangulu_inblock_idx** inblock_indeces = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx*) * struct_bcolptr[block_length_col]);
    calculate_type** inblock_values = NULL;
    if(malloc_distbcsc_value){
        inblock_values = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type*) * struct_bcolptr[block_length_col]);
    }

    for(pangulu_exblock_ptr bidx = 0; bidx < struct_bcolptr[block_length_col]; bidx++){
        pangulu_int64_t offset = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx + 
            sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx];
        inblock_pointers[bidx] = block_csc + offset;
        inblock_indeces[bidx] = block_csc + offset + 
            sizeof(pangulu_inblock_ptr) * (block_order + 1);
        if(malloc_distbcsc_value){
            inblock_values[bidx] = block_csc + offset + 
                sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx + 1] - struct_bnnzptr[bidx]);
        }
    }

    // printf("#%d 1.13\n", rank);

    *bcsc_struct_pointer = struct_bcolptr;
    *bcsc_struct_index = struct_browidx;
    *bcsc_struct_nnzptr = struct_bnnzptr;
    *bcsc_inblock_pointers = inblock_pointers;
    *bcsc_inblock_indeces = inblock_indeces;
    if(malloc_distbcsc_value){
        *bcsc_values = inblock_values;
    }

    bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));

    // printf("[PanguLU LOG] pangulu_cm_distribute_distcsc_to_distbcsc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
    // printf("#%d 1.14\n", rank);

#undef _PANGULU_SET_BVALUE_SIZE
#undef _PANGULU_SET_VALUE_SIZE
}

pangulu_inblock_ptr* temp_rowptr = NULL;
pangulu_inblock_ptr* aid_inptr = NULL;
pangulu_inblock_idx* temp_colidx = NULL;
pangulu_inblock_ptr* temp_valueidx = NULL;

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
){
    pangulu_storage_slot_t* slot = pangulu_storage_get_slot(storage, slot_addr);
    MPI_Status mpi_stat;
    int mpi_count = 0;
    MPI_Get_count(msg_stat, MPI_CHAR, &mpi_count);
    // printf("Recv length=%d max_length=%llu (?, %u) 0x%016llX\n", mpi_count, storage->bins[PANGULU_DIGINFO_GET_BINID(slot_addr)].slot_capacity, msg_stat->MPI_TAG, slot_addr);
    MPI_Recv(slot->columnpointer, mpi_count, MPI_CHAR, msg_stat->MPI_SOURCE, msg_stat->MPI_TAG, MPI_COMM_WORLD, &mpi_stat);
    pangulu_inblock_ptr nnz = slot->columnpointer[nb];
    slot->rowindex = ((char*)slot->columnpointer) + sizeof(pangulu_inblock_ptr) * (nb + 1);
    slot->value = ((char*)slot->columnpointer) + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz;

    *bcol_pos = (msg_stat->MPI_TAG); // bcol_pos = tag, brow_pos = columnpointer[0]
    *brow_pos = slot->columnpointer[0];
    slot->columnpointer[0] = 0;

    pangulu_exblock_ptr bidx = binarysearch(bcsc_related_index, bcsc_related_pointer[*bcol_pos], bcsc_related_pointer[(*bcol_pos) + 1], *brow_pos);
    bcsc_related_draft_info[bidx] = 
        PANGULU_DIGINFO_SET_BINID(PANGULU_DIGINFO_GET_BINID(slot_addr)) |
        PANGULU_DIGINFO_SET_NNZ(nnz) |
        PANGULU_DIGINFO_SET_STOREIDX(PANGULU_DIGINFO_GET_STOREIDX(slot_addr));
    slot->brow_pos = *brow_pos;
    slot->bcol_pos = *bcol_pos;

    #ifdef PANGULU_NONSHAREDMEM
    if(temp_rowptr == NULL){
        temp_rowptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
        aid_inptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
        temp_colidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nb * nb);
        temp_valueidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * nb * nb);
    }
    slot->d_columnpointer = (((char*)(slot->d_value)) + sizeof(calculate_type) * nnz);
    slot->d_rowindex = (((char*)(slot->d_value)) + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb+1));
    pangulu_platform_memcpy(slot->d_columnpointer, slot->columnpointer, sizeof(pangulu_inblock_ptr) * (nb+1), 0, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_memcpy(slot->d_rowindex, slot->rowindex, sizeof(pangulu_inblock_idx) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_memcpy(slot->d_value, slot->value, sizeof(calculate_type) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
    if((slot->brow_pos) == (slot->bcol_pos)){
        pangulu_platform_malloc(&(slot->d_rowpointer), sizeof(pangulu_inblock_ptr) * (nb+1), PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_malloc(&(slot->d_columnindex), sizeof(pangulu_inblock_idx) * nnz, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_malloc(&(slot->d_idx_of_csc_value_for_csr), sizeof(pangulu_inblock_ptr) * nnz, PANGULU_DEFAULT_PLATFORM);

        pangulu_transpose_struct_with_valueidx_inblock(nb, slot->columnpointer, slot->rowindex, temp_rowptr, temp_colidx, temp_valueidx, aid_inptr);
        pangulu_platform_memcpy(slot->d_rowpointer, temp_rowptr, sizeof(pangulu_inblock_ptr) * (nb+1), 0, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_memcpy(slot->d_columnindex, temp_colidx, sizeof(pangulu_inblock_idx) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_memcpy(slot->d_idx_of_csc_value_for_csr, temp_valueidx, sizeof(pangulu_inblock_ptr) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
        slot->have_csr_data = 1;
    }
    #endif
}

void pangulu_cm_isend_block(
    pangulu_storage_slot_t* slot,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx brow_pos,
    pangulu_exblock_idx bcol_pos,
    pangulu_int32_t target_rank
){
    // if(slot->data_status != PANGULU_DATA_READY){
    //     printf("[PanguLU WANRING] Sending data not ready.\n");
    // }
    pangulu_int32_t rank;
    pangulu_cm_rank(&rank);
    // printf("#%d Send (%d, %d) to %d\n", rank, slot->brow_pos, slot->bcol_pos, target_rank);
    MPI_Request mpi_req;
    pangulu_int64_t size = 
        sizeof(pangulu_inblock_ptr) * (nb + 1) +
        sizeof(pangulu_inblock_idx) * slot->columnpointer[nb] + 
        sizeof(calculate_type) * slot->columnpointer[nb];
    // printf("nnz=%d\n", slot->columnpointer[nb]);
    slot->columnpointer[0] = brow_pos;
    MPI_Isend(slot->columnpointer, size, MPI_CHAR, target_rank, bcol_pos, MPI_COMM_WORLD, &mpi_req);
}
