#include "pangulu_common.h"

extern pangulu_int32_t **ssssm_hash_lu;
extern pangulu_int32_t **ssssm_hash_l_row;
extern pangulu_int32_t **ssssm_hash_u_col;
extern calculate_type **ssssm_l_value;
extern calculate_type **ssssm_u_value;
extern calculate_type **temp_a_value;
extern pangulu_int32_t **getrf_diagIndex_csc;
extern pangulu_int32_t **getrf_nextptr;
extern calculate_type **TEMP_calculate_type;
extern pangulu_inblock_ptr **TEMP_pangulu_inblock_ptr;
extern pangulu_int32_t **hd_getrf_nnzu;
extern calculate_type* getrf_dense_buf_d;

void pangulu_preprocessing(
    pangulu_common *common,
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_origin_smatrix *reorder_matrix,
    pangulu_int32_t nthread)
{
    pangulu_exblock_idx n = bcommon->n;
    pangulu_exblock_idx n_loc = 0;
    pangulu_exblock_idx nb = bcommon->nb;
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_int32_t nproc = bcommon->sum_rank_size;

    bind_to_core((rank) % nproc);

    pangulu_exblock_ptr *distcsc_proc_nnzptr;
    pangulu_exblock_ptr *distcsc_symbolic_proc_nnzptr;
    pangulu_exblock_ptr *distcsc_pointer;
    pangulu_exblock_idx *distcsc_index;
    calculate_type *distcsc_value;
    pangulu_exblock_ptr *bcsc_nofill_pointer;
    pangulu_exblock_idx *bcsc_nofill_index;
    pangulu_exblock_ptr *bcsc_nofill_blknnzptr;
    pangulu_inblock_ptr **bcsc_nofill_inblk_pointers;
    pangulu_inblock_idx **bcsc_nofill_inblk_indeces;
    calculate_type **bcsc_nofill_inblk_values;
    pangulu_exblock_ptr *bcsc_pointer;
    pangulu_exblock_idx *bcsc_index;
    pangulu_exblock_ptr *bcsc_blknnzptr;
    pangulu_inblock_ptr **bcsc_inblk_pointers;
    pangulu_inblock_idx **bcsc_inblk_indeces;
    calculate_type **bcsc_inblk_values;

    pangulu_exblock_ptr *bcsc_related_pointer;
    pangulu_exblock_idx *bcsc_related_index;
    pangulu_uint64_t *bcsc_related_draft_info;

    if (rank == 0)
    {
        pangulu_convert_csr_to_csc(
            1, n,
            &reorder_matrix->rowpointer,
            &reorder_matrix->columnindex,
            &reorder_matrix->value,
            &reorder_matrix->columnpointer,
            &reorder_matrix->rowindex,
            &reorder_matrix->value_csc);
        distcsc_pointer = reorder_matrix->columnpointer;
        distcsc_index = reorder_matrix->rowindex;
        distcsc_value = reorder_matrix->value_csc;
        reorder_matrix->columnpointer = NULL;
        reorder_matrix->rowindex = NULL;
        reorder_matrix->value_csc = NULL;
    }

    pangulu_cm_sync();

    //  distribute to processes
    pangulu_cm_distribute_csc_to_distcsc(
        0, 1, &n, nb, &nproc, &n_loc,
        &distcsc_proc_nnzptr,
        &distcsc_pointer,
        &distcsc_index,
        &distcsc_value);

    pangulu_cm_sync();
    // printf("#%d A 1\n", rank);

    // for(int target_rank = 0; target_rank < nproc; target_rank++){
    //     if(rank == target_rank){
    //         printf("#%d\n", rank);
    //         for(int i = 0; i < n_loc; i++){
    //             printf("(%d) : ", i);
    //             for(int j=distcsc_pointer[i]; j < distcsc_pointer[i+1]; j++){
    //                 printf("(%d, %.1le) ", distcsc_index[j], distcsc_value[j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    //     sleep(1);
    //     pangulu_cm_sync();
    // }

    pangulu_cm_distribute_distcsc_to_distbcsc(
        1, 1, n, n_loc, nb,
        distcsc_proc_nnzptr,
        distcsc_pointer,
        distcsc_index,
        distcsc_value,
        &bcsc_nofill_pointer,
        &bcsc_nofill_index,
        &bcsc_nofill_blknnzptr,
        &bcsc_nofill_inblk_pointers,
        &bcsc_nofill_inblk_indeces,
        &bcsc_nofill_inblk_values);

    pangulu_cm_sync();
    // printf("#%d A 2\n", rank);/

    // for(int target_rank = 0; target_rank < nproc; target_rank++){
    //     if(rank == target_rank){
    //         printf("#%d\n", rank);
    //         for(int i=0; i<block_length; i++){
    //             for(int j=bcsc_nofill_pointer[i]; j < bcsc_nofill_pointer[i+1]; j++){
    //                 printf("#%d (%d, %d) %d\n", rank, bcsc_nofill_index[j], i, bcsc_nofill_blknnzptr[j+1] - bcsc_nofill_blknnzptr[j]);
    //                 for(int ii = 0; ii < nb; ii++){
    //                     printf("(%d) : ", ii);
    //                     for(int jj=bcsc_nofill_inblk_pointers[j][ii]; jj < bcsc_nofill_inblk_pointers[j][ii+1]; jj++){
    //                         printf("(%d, %.1le) ", bcsc_nofill_inblk_indeces[j][jj], bcsc_nofill_inblk_values[j][jj]);
    //                     }
    //                     printf("\n");
    //                 }
    //             }
    //         }
    //     }
    //     sleep(1);
    //     pangulu_cm_sync();
    // }

    if (rank == 0)
    {
        //  generate full symbolic struct
        pangulu_convert_halfsymcsc_to_csc_struct(
            1, 0, n,
            &bsmatrix->symbolic_rowpointer,
            &bsmatrix->symbolic_columnindex,
            &bsmatrix->symbolic_rowpointer,
            &bsmatrix->symbolic_columnindex);

        // printf("%d %d %d\n", n, n, bsmatrix->symbolic_nnz);
        // for(int i=0;i<n;i++){
        //     for(int j=bsmatrix->symbolic_rowpointer[i]; j < bsmatrix->symbolic_rowpointer[i+1]; j++){
        //         printf("%d %d %d\n", bsmatrix->symbolic_columnindex[j]+1, i+1, 1);
        //     }
        // }
        // exit(0);
    }

    pangulu_cm_sync_asym(0);
    // pangulu_cm_sync();

    //  distribute to processes
    pangulu_cm_distribute_csc_to_distcsc(
        0, 1, &n, nb, &nproc, &n_loc,
        &distcsc_symbolic_proc_nnzptr,
        &bsmatrix->symbolic_rowpointer,
        &bsmatrix->symbolic_columnindex,
        NULL);

    pangulu_cm_sync();

    struct timeval start_time;
    pangulu_time_start(&start_time);
    pangulu_sort_exblock_struct(n_loc, bsmatrix->symbolic_rowpointer, bsmatrix->symbolic_columnindex, 0);
    // printf("[PanguLU LOG] SORT time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);

    pangulu_cm_sync_asym(nproc - 1);

    pangulu_cm_distribute_distcsc_to_distbcsc(
        1, 1, n, n_loc, nb,
        distcsc_symbolic_proc_nnzptr,
        bsmatrix->symbolic_rowpointer,
        bsmatrix->symbolic_columnindex,
        NULL,
        &bcsc_pointer,
        &bcsc_index,
        &bcsc_blknnzptr,
        &bcsc_inblk_pointers,
        &bcsc_inblk_indeces,
        &bcsc_inblk_values);

    pangulu_cm_sync();

    // for(int target_rank = 0; target_rank < nproc; target_rank++){
    //     if(rank == target_rank){
    //         for(int i=0; i<block_length; i++){
    //             for(int j=bcsc_pointer[i]; j < bcsc_pointer[i+1]; j++){
    //                 printf("#%d (%d, %d) %d\n", rank, bcsc_index[j], i, bcsc_blknnzptr[j]);
    //                 for(int ii = 0; ii < nb; ii++){
    //                     printf("(%d) : ", ii);
    //                     for(int jj=bcsc_inblk_pointers[j][ii]; jj < bcsc_inblk_pointers[j][ii+1]; jj++){
    //                         printf("%d ", bcsc_inblk_indeces[j][jj]);
    //                     }
    //                     printf("\n");
    //                 }
    //             }
    //         }
    //     }
    //     sleep(1);
    //     pangulu_cm_sync();
    // }

    // bcsc_inblk_values = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type*) * bcsc_pointer[block_length]);
    // calculate_type* bcsc_inblk_value_store = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * bcsc_blknnzptr[bcsc_pointer[block_length]]);
    // memset(bcsc_inblk_value_store, 0, sizeof(calculate_type) * bcsc_blknnzptr[bcsc_pointer[block_length]]);
    // for(pangulu_exblock_ptr bidx = 0; bidx < bcsc_pointer[block_length]; bidx++){
    //     bcsc_inblk_values[bidx] = &bcsc_inblk_value_store[bcsc_blknnzptr[bidx]];
    // }
    pangulu_convert_bcsc_fill_value_to_struct(
        1, n, nb,
        bcsc_nofill_pointer,
        bcsc_nofill_index,
        bcsc_nofill_blknnzptr,
        bcsc_nofill_inblk_pointers,
        bcsc_nofill_inblk_indeces,
        bcsc_nofill_inblk_values,
        bcsc_pointer,
        bcsc_index,
        bcsc_blknnzptr,
        bcsc_inblk_pointers,
        bcsc_inblk_indeces,
        bcsc_inblk_values);
    bcsc_nofill_pointer = NULL;
    bcsc_nofill_index = NULL;
    bcsc_nofill_blknnzptr = NULL;
    bcsc_nofill_inblk_pointers = NULL;
    bcsc_nofill_inblk_indeces = NULL;
    bcsc_nofill_inblk_values = NULL;

    // for(int target_rank = 0; target_rank < nproc; target_rank++){
    //     if(rank == target_rank){
    //         for(int i=0; i<block_length; i++){
    //             for(int j=bcsc_pointer[i]; j < bcsc_pointer[i+1]; j++){
    //                 printf("#%d (%d, %d) %d\n", rank, bcsc_index[j], i, bcsc_blknnzptr[j]);
    //                 for(int ii = 0; ii < nb; ii++){
    //                     printf("(%d) : ", ii);
    //                     for(int jj=bcsc_inblk_pointers[j][ii]; jj < bcsc_inblk_pointers[j][ii+1]; jj++){
    //                         printf("(%d, %.1le) ", bcsc_inblk_indeces[j][jj], bcsc_inblk_values[j][jj]);
    //                     }
    //                     printf("\n");
    //                 }
    //             }
    //         }
    //     }
    //     sleep(1);
    //     pangulu_cm_sync();
    // }

    pangulu_time_start(&start_time);
    //  1. bcsc_related_pointer, bcsc_related_index, bcsc_related_draft_info
    pangulu_digest_coo_t *digest_info_local = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_digest_coo_t) * bcsc_pointer[block_length]);
    pangulu_convert_bcsc_to_digestcoo(
        block_length,
        bcsc_pointer,
        bcsc_index,
        bcsc_blknnzptr,
        digest_info_local);
    // 1.1 Receive block-level struct info of related ranks.
    pangulu_digest_coo_t **digest_info_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_digest_coo_t *) * nproc);
    pangulu_exblock_ptr *digest_info_arr_len = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
    memset(digest_info_arr, 0, sizeof(pangulu_digest_coo_t *) * nproc);
    memset(digest_info_arr_len, 0, sizeof(pangulu_exblock_ptr) * nproc);
    digest_info_arr[rank] = digest_info_local;
    digest_info_arr_len[rank] = bcsc_pointer[block_length];
    pangulu_int32_t current_rank_prow = rank / bcommon->q;
    pangulu_int32_t current_rank_pcol = rank % bcommon->q;
    for (pangulu_int32_t prow = 0; prow < bcommon->p; prow++)
    {
        for (pangulu_int32_t pcol = 0; pcol < bcommon->q; pcol++)
        {
            if (((current_rank_prow == prow) || (current_rank_pcol == pcol)) && (prow * bcommon->q + pcol != rank))
            {
                MPI_Request req;
                MPI_Isend(
                    digest_info_local,
                    sizeof(pangulu_digest_coo_t) * bcsc_pointer[block_length],
                    MPI_CHAR,
                    prow * bcommon->q + pcol,
                    0,
                    MPI_COMM_WORLD,
                    &req);
            }
        }
    }
    pangulu_int32_t need_to_recv = 0;
    for (pangulu_int32_t prow = 0; prow < bcommon->p; prow++)
    {
        for (pangulu_int32_t pcol = 0; pcol < bcommon->q; pcol++)
        {
            if (((current_rank_prow == prow) || (current_rank_pcol == pcol)) && (prow * bcommon->q + pcol != rank))
            {
                need_to_recv++;
            }
        }
    }
    for (pangulu_int32_t i = 0; i < need_to_recv; i++)
    {
        MPI_Status stat;
        int message_length;
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
        MPI_Get_count(&stat, MPI_CHAR, &message_length);
        digest_info_arr_len[stat.MPI_SOURCE] = message_length / sizeof(pangulu_digest_coo_t);
        digest_info_arr[stat.MPI_SOURCE] = pangulu_malloc(__FILE__, __LINE__, message_length);
        MPI_Recv(digest_info_arr[stat.MPI_SOURCE], message_length, MPI_CHAR, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
    }

    // 1.2 Assemble struct info of related ranks into bcsc_related_*.
    bcsc_related_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    memset(bcsc_related_pointer, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    for (pangulu_uint32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
    {
        if (digest_info_arr[remote_rank])
        {
            for (pangulu_exblock_ptr coo_idx = 0; coo_idx < digest_info_arr_len[remote_rank]; coo_idx++)
            {
                bcsc_related_pointer[digest_info_arr[remote_rank][coo_idx].col + 1]++;
                // printf("#%d contain %d (%d, %d) \n", rank, remote_rank, digest_info_arr[remote_rank][coo_idx].row, digest_info_arr[remote_rank][coo_idx].col);
            }
        }
    }
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        bcsc_related_pointer[bcol + 1] += bcsc_related_pointer[bcol];
    }
    bcsc_related_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * bcsc_related_pointer[block_length]);
    bcsc_related_draft_info = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * bcsc_related_pointer[block_length]);
    // memset(bcsc_related_draft_info, -1, sizeof(pangulu_uint64_t) * bcsc_related_pointer[block_length]);
    for (pangulu_exblock_ptr bidx = 0; bidx < bcsc_related_pointer[block_length]; bidx++)
    {
        bcsc_related_draft_info[bidx] = 0xFFFFFFFFFFFFFFFF;
    }
    pangulu_exblock_ptr *bcsc_aid_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    memcpy(bcsc_aid_arr, bcsc_related_pointer, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    for (pangulu_uint32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
    {
        if (digest_info_arr[remote_rank])
        {
            for (pangulu_exblock_ptr coo_idx = 0; coo_idx < digest_info_arr_len[remote_rank]; coo_idx++)
            {
                bcsc_related_index[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]] = digest_info_arr[remote_rank][coo_idx].row;
                if (remote_rank == rank)
                {
                    bcsc_related_draft_info[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]] = 0;
                    bcsc_related_draft_info[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]] |= PANGULU_DIGINFO_SET_STOREIDX(coo_idx);
                    // printf("(%d, %d) %d\n", digest_info_arr[remote_rank][coo_idx].row, digest_info_arr[remote_rank][coo_idx].col, coo_idx);
                }
                else
                {
                    bcsc_related_draft_info[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]] = PANGULU_DIGINFO_SET_BINID(7);
                }
                bcsc_related_draft_info[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]] |= PANGULU_DIGINFO_SET_NNZ(digest_info_arr[remote_rank][coo_idx].nnz);
                // printf("nnz = %u 0x%X\n", digest_info_arr[remote_rank][coo_idx].nnz, digest_info_arr[remote_rank][coo_idx].nnz);
                // printf("0x%016llX %d\n", bcsc_related_draft_info[bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]], bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]);
                bcsc_aid_arr[digest_info_arr[remote_rank][coo_idx].col]++;
            }
        }
    }
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        pangulu_kvsort2(bcsc_related_index, bcsc_related_draft_info, bcsc_related_pointer[bcol], bcsc_related_pointer[bcol + 1] - 1);
    }
    pangulu_cm_sync();

    // for(int i=0;i<block_length;i++){
    //     printf("%d ", bcsc_aid_arr[i]);
    // }
    // printf("\n");

    pangulu_free(__FILE__, __LINE__, bcsc_aid_arr);
    bcsc_aid_arr = NULL;
    for (pangulu_uint32_t i = 0; i < nproc; i++)
    {
        if (digest_info_arr[i])
        {
            pangulu_free(__FILE__, __LINE__, digest_info_arr[i]);
            digest_info_arr[i] = NULL;
        }
    }
    pangulu_free(__FILE__, __LINE__, digest_info_arr);
    pangulu_free(__FILE__, __LINE__, digest_info_arr_len);
    digest_info_local = NULL;
    digest_info_arr = NULL;
    digest_info_arr_len = NULL;

    // printf("[PanguLU LOG] Get 1 time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);

    //  2. bcsr_related_pointer, bcsr_related_index, bcsr_index_bcsc
    pangulu_time_start(&start_time);
    pangulu_exblock_ptr *bcsr_related_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    pangulu_exblock_idx *bcsr_related_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * bcsc_related_pointer[block_length]);
    pangulu_exblock_ptr *bcsr_index_bcsc = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * bcsc_related_pointer[block_length]);
    memset(bcsr_related_pointer, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_ptr brow = bcsc_related_index[bidx];
            bcsr_related_pointer[brow + 1]++;
        }
    }
    for (pangulu_exblock_idx brow = 0; brow < block_length; brow++)
    {
        bcsr_related_pointer[brow + 1] += bcsr_related_pointer[brow];
    }

    pangulu_exblock_ptr *bcsr_aid_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    memcpy(bcsr_aid_pointer, bcsr_related_pointer, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_ptr brow = bcsc_related_index[bidx];
            bcsr_related_index[bcsr_aid_pointer[brow]] = bcol;
            bcsr_index_bcsc[bcsr_aid_pointer[brow]] = bidx;
            bcsr_aid_pointer[brow]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, bcsr_aid_pointer);
    // printf("[PanguLU LOG] Get 2 time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);

    //  3. block_remain_task_count
    //  4. rank_remain_task_count
    pangulu_time_start(&start_time);
    pangulu_int64_t rank_remain_task_count = 0;
    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_int32_t *bcsc_remain_task_count = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * bcsc_related_pointer[block_length]);
    memset(bcsc_remain_task_count, 0, sizeof(pangulu_int32_t) * bcsc_related_pointer[block_length]);
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            if (((brow % p) * q + (bcol % q)) == rank)
            {
                pangulu_exblock_ptr bidx1 = bcsc_related_pointer[bcol];
                pangulu_exblock_ptr bidx2 = bcsr_related_pointer[brow];
                pangulu_exblock_idx blevel_ub = PANGULU_MIN(brow, bcol);
                pangulu_int64_t current_block_remain_task_count = 1;
                pangulu_int64_t bidx_diag = binarysearch(bcsc_related_index, bcsc_related_pointer[blevel_ub], bcsc_related_pointer[blevel_ub + 1], blevel_ub);
                if ((bcsc_related_index[bidx_diag] % p) * q + (blevel_ub % q) != rank)
                {
                    bcsc_remain_task_count[bidx_diag]++;
                }
                while (bcsc_related_index[bidx1] < blevel_ub && bcsr_related_index[bidx2] < blevel_ub)
                {
                    if (bcsc_related_index[bidx1] < bcsr_related_index[bidx2])
                    {
                        bidx1++;
                    }
                    else if (bcsc_related_index[bidx1] > bcsr_related_index[bidx2])
                    {
                        bidx2++;
                    }
                    else
                    {
                        if (((bcsc_related_index[bidx1] % p) * q + (bcol % q)) != rank)
                        {
                            bcsc_remain_task_count[bidx1]++;
                        }
                        if (((brow % p) * q + (bcsr_related_index[bidx2] % q)) != rank)
                        {
                            bcsc_remain_task_count[bcsr_index_bcsc[bidx2]]++;
                        }
                        bidx1++;
                        bidx2++;
                        current_block_remain_task_count++;
                    }
                }
                rank_remain_task_count += current_block_remain_task_count;
                bcsc_remain_task_count[bidx] += current_block_remain_task_count;
            }
        }
    }

    // for(int i=0;i<block_length;i++){
    //     printf("(%d) ", bcsc_related_pointer[i+1] - bcsc_related_pointer[i]);
    //     int idx = bcsc_related_pointer[i];
    //     // for(int bidx = bcsc_related_pointer[i]; bidx < bcsc_related_pointer[i+1]; bidx++){
    //     //     printf("%d ", bcsc_related_index[bidx]);
    //     // }
    //     // printf("\n");
    //     for(int j=0;j<block_length;j++){
    //         if(idx >= bcsc_related_pointer[i+1]){
    //             break;
    //         }
    //         if(bcsc_related_index[idx] > j){
    //             printf("    ");
    //         }else{
    //             printf("%3d ", bcsc_remain_task_count[idx]);
    //             idx++;
    //         }
    //     }
    //     printf(" \n");
    // }

    //  5. rank_remain_recv_block_count
    // pangulu_uint64_t* maxbrow_upper_idx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * PANGULU_ICEIL(block_length, q));
    // pangulu_uint64_t* maxbcol_lower_idx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * PANGULU_ICEIL(block_length, p));
    // pangulu_int64_t rank_remain_recv_block_count = 0;
    // for(pangulu_exblock_idx brow = rank / q; brow < block_length; brow+=p){
    //     char break_flag = 0;
    //     for(pangulu_uint64_t bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    //         pangulu_exblock_idx bcol = bcsr_related_index[bidx];
    //         if(((brow % p) * q + (bcol % q)) == rank){
    //             maxbcol_lower_idx[brow / p] |= (1ULL<<63); // upper triangle part needs current row
    //             maxbcol_lower_idx[brow / p] &= (1ULL<<63);
    //             maxbcol_lower_idx[brow / p] |= bidx;
    //             if(bcol >= brow){
    //                 break_flag = 1;
    //                 break;
    //             }
    //         }
    //     }
    //     if(break_flag == 0){
    //         maxbcol_lower_idx[brow / p]++;
    //     }
    // }
    // for(pangulu_exblock_idx bcol = rank % q; bcol < block_length; bcol+=q){
    //     char break_flag = 0;
    //     for(pangulu_uint64_t bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++){
    //         pangulu_exblock_idx brow = bcsc_related_index[bidx];
    //         if(((brow % p) * q + (bcol % q)) == rank){
    //             maxbrow_upper_idx[bcol / q] |= (1ULL<<63);
    //             maxbrow_upper_idx[bcol / q] &= (1ULL<<63);
    //             maxbrow_upper_idx[bcol / q] |= bidx;
    //             if(brow >= bcol){
    //                 break_flag = 1;
    //                 break;
    //             }
    //         }
    //     }
    //     if(break_flag == 0){
    //         maxbrow_upper_idx[bcol / q]++;
    //     }
    // }
    // // for(int i=0;i<(block_length / q);i++){
    // //     printf("%016llX ", maxbrow_upper_idx[i]);
    // // }
    // // printf("\n");
    // // for(int i=0;i<(block_length / p);i++){
    // //     printf("%016llX ", maxbcol_lower_idx[i]);
    // // }
    // // printf("\n");
    // for(pangulu_exblock_idx bprow = 0; bprow < block_length / p; bprow++){
    //     pangulu_exblock_idx brow = bprow * p + (rank / q);
    //     if(maxbcol_lower_idx[bprow] & (1ULL<<63)){
    //         pangulu_exblock_ptr row_nzblk_lower = 0;
    //         for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bcsr_related_pointer[brow + 1]; bidx++){
    //             if(bcsr_related_index[bidx] >= brow){
    //                 break;
    //             }
    //             if(((brow % p) * q + (bcsr_related_index[bidx] % q)) != rank){
    //                 row_nzblk_lower++;
    //             }
    //         }
    //         rank_remain_recv_block_count += row_nzblk_lower;
    //     }else{
    //         pangulu_exblock_ptr row_nzblk_lower = 0;
    //         for(pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < maxbcol_lower_idx[bprow]; bidx++){
    //             if(((brow % p) * q + (bcsr_related_index[bidx] % q)) != rank){
    //                 row_nzblk_lower++;
    //             }
    //         }
    //         rank_remain_recv_block_count += row_nzblk_lower;
    //     }
    // }
    // for(pangulu_exblock_idx bpcol = 0; bpcol < block_length / q; bpcol++){
    //     pangulu_exblock_idx bcol = bpcol * q + (rank % q);
    //     if(maxbrow_upper_idx[bpcol] & (1ULL<<63)){
    //         pangulu_exblock_ptr col_nzblk_upper = 0;
    //         for(pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++){
    //             if(bcsc_related_index[bidx] >= bcol){
    //                 break;
    //             }
    //             if(((bcsc_related_index[bidx] % p) * q + (bcol % q)) != rank){
    //                 col_nzblk_upper++;
    //             }
    //         }
    //         rank_remain_recv_block_count += col_nzblk_upper;
    //     }else{
    //         pangulu_exblock_ptr col_nzblk_upper = 0;
    //         for(pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < maxbrow_upper_idx[bpcol]; bidx++){
    //             if(((bcsc_related_index[bidx] % p) * q + (bcol % q)) != rank){
    //                 col_nzblk_upper++;
    //             }
    //         }
    //         rank_remain_recv_block_count += col_nzblk_upper;
    //     }
    // }
    // pangulu_free(__FILE__, __LINE__, maxbrow_upper_idx);
    // pangulu_free(__FILE__, __LINE__, maxbcol_lower_idx);
    // // printf("#%d rank_remain_task_count = %d\n", rank, rank_remain_task_count);
    // char* pivot_fetch_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * block_length);
    // memset(pivot_fetch_flag, 0, sizeof(char) * block_length);
    // for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
    //     for(pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bcsc_related_pointer[bcol + 1]; bidx++){
    //         pangulu_exblock_idx brow = bcsc_related_index[bidx];
    //         if((brow % p) * q + (bcol % q) == rank){
    //             pangulu_exblock_idx blevel = PANGULU_MIN(brow, bcol);
    //             if((blevel % p) * q + (blevel % q) != rank){
    //                 if(!pivot_fetch_flag[blevel]){
    //                     pivot_fetch_flag[blevel] = 1;
    //                     rank_remain_recv_block_count++;
    //                 }
    //             }
    //         }
    //     }
    // }
    // pangulu_free(__FILE__, __LINE__, pivot_fetch_flag);
    // // printf("#%d rank_remain_recv_block_count = %d\n", rank, rank_remain_recv_block_count);
    // // printf("[PanguLU LOG] Get 3, 4, 5 time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);

    char *bcsc_remote_block_need_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * bcsc_related_pointer[block_length]);
    memset(bcsc_remote_block_need_flag, 0, sizeof(char) * bcsc_related_pointer[block_length]);
    pangulu_int64_t rank_remain_recv_block_count = 0;
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        pangulu_int64_t bidx_maxrow = bcsc_related_pointer[bcol + 1] - 1;
        while (bidx_maxrow >= (pangulu_int64_t)bcsc_related_pointer[bcol])
        {
            if ((bcsc_related_index[bidx_maxrow] % p) * q + (bcol % q) == rank)
            {
                break;
            }
            bidx_maxrow--;
        }
        if (bidx_maxrow < (pangulu_int64_t)bcsc_related_pointer[bcol])
        {
            continue;
        }
        for (pangulu_exblock_ptr bidx = bcsc_related_pointer[bcol]; bidx < bidx_maxrow; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_related_index[bidx];
            if (brow > bcol)
            {
                break;
            }
            if ((brow % p) * q + (bcol % q) != rank)
            {
                bcsc_remote_block_need_flag[bidx] = 1;
            }
        }
    }
    for (pangulu_exblock_idx brow = 0; brow < block_length; brow++)
    {
        pangulu_int64_t bidx_maxcol = bcsr_related_pointer[brow + 1] - 1;
        while (bidx_maxcol >= (pangulu_int64_t)bcsr_related_pointer[brow])
        {
            if ((brow % p) * q + (bcsr_related_index[bidx_maxcol] % q) == rank)
            {
                break;
            }
            bidx_maxcol--;
        }
        if (bidx_maxcol < (pangulu_int64_t)bcsr_related_pointer[brow])
        {
            continue;
        }
        for (pangulu_exblock_ptr bidx = bcsr_related_pointer[brow]; bidx < bidx_maxcol; bidx++)
        {
            pangulu_exblock_idx bcol = bcsr_related_index[bidx];
            if (bcol > brow)
            {
                break;
            }
            if ((brow % p) * q + (bcol % q) != rank)
            {
                bcsc_remote_block_need_flag[bcsr_index_bcsc[bidx]] = 1;
            }
        }
    }
    for (pangulu_exblock_ptr bidx = 0; bidx < bcsc_related_pointer[block_length]; bidx++)
    {
        if (bcsc_remote_block_need_flag[bidx] == 1)
        {
            rank_remain_recv_block_count++;
        }
    }
    pangulu_free(__FILE__, __LINE__, bcsc_remote_block_need_flag);

    // printf("rank_remain_recv_block_count = %d\n", rank_remain_recv_block_count);

    // Configurable parameters about memory usage
    // pangulu_int64_t pangulu_heap_capacity = 10000000;
    pangulu_int64_t pangulu_heap_capacity = rank_remain_task_count + 1;
    // pangulu_int64_t pangulu_storage_slot_capacity[7] = {
    //     0 /*[0] must be 0*/,
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (PANGULU_MIN(5, nb * nb / 1024)),
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb / 256),
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb / 64),
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb / 16),
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb / 4),
    //     sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb / 1)
    // };
    pangulu_int64_t pangulu_storage_slot_capacity[7] = {
        0 /*[0] must be 0*/,
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (PANGULU_MIN(5, nb * nb / 1024)),
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * PANGULU_MIN(nb, 2)),
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * PANGULU_MIN(nb, 4)),
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * PANGULU_MIN(nb, 6)),
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * PANGULU_MIN(nb, 32)),
        sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nb * nb)};
    // pangulu_int32_t pangulu_storage_slot_count[7] = {
    //     0 /*[0] must be 0*/,
    //     20000, 10000, 5000, 1000, 1000, 200
    // };
    // pangulu_int32_t pangulu_storage_slot_count[7] = {
    //     0 /*[0] must be 0*/,
    //     1000000/nb, 1000000/nb, 1000000/nb, 1000000/nb, 1000000/nb, PANGULU_MAX(100000000/nb/nb, 5)
    // };

#ifdef PANGULU_NONSHAREDMEM
    pangulu_int64_t basic_param = 8;
#else
    pangulu_int64_t basic_param = 100;
#endif
    if(rank == 0){
    	printf("recv/send buf : %lld\n", basic_param);
    }
    pangulu_int32_t pangulu_storage_slot_count[7];
    if (nproc == 1)
    {
        printf("[PanguLU Info] Only 1 process detected in MPI_COMM_WORLD. Singleton mode enabled.\n");
        pangulu_int32_t tmp[7] = {
            0 /*[0] must be 0*/,
            0, 0, 0, 0, 0, 0};
        memcpy(pangulu_storage_slot_count, tmp, sizeof(pangulu_int32_t) * 7);
    }
    else
    {
        pangulu_int32_t tmp[7] = {
            0 /*[0] must be 0*/,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))),
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))),
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))),
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))),
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))),
            PANGULU_MAX(100 * basic_param * block_length / nb / PANGULU_MAX(1, nproc / (p + q)), 5)};
        memcpy(pangulu_storage_slot_count, tmp, sizeof(pangulu_int32_t) * 7);
    }

    // if(rank == 0){
    //     printf("[PanguLU LOG] pangulu_storage_slot_capacity : ");
    //     for(int i=0;i<7;i++){
    //         printf("[%d]=%lld ", i, pangulu_storage_slot_capacity[i]);
    //     }
    //     printf("\n");
    //     printf("[PanguLU LOG] pangulu_storage_slot_count : ");
    //     for(int i=0;i<7;i++){
    //         printf("[%d]=%lld ", i, pangulu_storage_slot_count[i]);
    //     }
    //     printf("\n");
    // }

    //  6. pangulu_storage_t
    //  7. pangulu_storage_slot_t
    //  8. pangulu_storage_bin_t
    bsmatrix->storage = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_t));
    pangulu_storage_init(
        bsmatrix->storage, pangulu_storage_slot_capacity, pangulu_storage_slot_count, block_length,
        bcsc_pointer, bcsc_index, bcsc_blknnzptr, bcsc_inblk_pointers, bcsc_inblk_indeces, bcsc_inblk_values, nb);
    //  9. pangulu_task_pool_t
    // 10. pangulu_task_heap_t
    bsmatrix->heap = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_queue_t));
    pangulu_task_queue_init(bsmatrix->heap, pangulu_heap_capacity);

    bsmatrix->bcsr_related_pointer = bcsr_related_pointer;
    bsmatrix->bcsr_related_index = bcsr_related_index;
    bsmatrix->bcsr_index_bcsc = bcsr_index_bcsc;
    bsmatrix->bcsc_related_pointer = bcsc_related_pointer;
    bsmatrix->bcsc_related_index = bcsc_related_index;
    bsmatrix->bcsc_related_draft_info = bcsc_related_draft_info;
    bsmatrix->bcsc_remain_task_count = bcsc_remain_task_count;
    bsmatrix->bcsc_pointer = bcsc_pointer;
    bsmatrix->bcsc_index = bcsc_index;
    bsmatrix->bcsc_blknnzptr = bcsc_blknnzptr;

    bsmatrix->rank_remain_task_count = rank_remain_task_count;
    bsmatrix->rank_remain_recv_block_count = rank_remain_recv_block_count;

    // printf("#%d rank_remain_task_count=%d rank_remain_recv_block_count=%d\n", rank, bsmatrix->rank_remain_task_count, bsmatrix->rank_remain_recv_block_count);

    bsmatrix->sent_rank_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * nproc);
    bsmatrix->info_mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(bsmatrix->info_mutex, NULL);

    bsmatrix->sc25_batch_tileid = NULL;
    bsmatrix->sc25_batch_tileid_capacity = 0;

    #ifdef PANGULU_NONSHAREDMEM
    ssssm_hash_lu = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_lu, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_l_row = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_l_row, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_u_col = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_u_col, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_l_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(ssssm_l_value, 0, sizeof(calculate_type *) * common->omp_thread);
    ssssm_u_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(ssssm_u_value, 0, sizeof(calculate_type *) * common->omp_thread);
    temp_a_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(temp_a_value, 0, sizeof(calculate_type *) * common->omp_thread);
    getrf_diagIndex_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(getrf_diagIndex_csc, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    getrf_nextptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(getrf_nextptr, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    TEMP_calculate_type = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(TEMP_calculate_type, 0, sizeof(calculate_type *) * common->omp_thread);
    TEMP_pangulu_inblock_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * common->omp_thread);
    memset(TEMP_pangulu_inblock_ptr, 0, sizeof(pangulu_inblock_ptr *) * common->omp_thread);
    hd_getrf_nnzu = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(hd_getrf_nnzu, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    // getrf_dense_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type*) * nb * nb);
    pangulu_platform_malloc(&getrf_dense_buf_d, sizeof(calculate_type) * nb * nb, PANGULU_DEFAULT_PLATFORM);
    #endif


    // if(rank == 0)
    // for(int i=0;i<block_length;i++){
    //     printf("(%d) : ", i);
    //     for(int j=bcsr_related_pointer[i]; j < bcsr_related_pointer[i+1]; j++){
    //         printf("%d ", bcsr_related_index[j]);
    //     }
    //     printf("\n");
    // }
}
