#include "pangulu_common.h"

pangulu_stat_t global_stat;

void pangulu_init(pangulu_exblock_idx pangulu_n, pangulu_exblock_ptr pangulu_nnz, pangulu_exblock_ptr *csr_rowptr, pangulu_exblock_idx *csr_colidx, calculate_type *csr_value, pangulu_init_options *init_options, void **pangulu_handle)
{
    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;
    common->n = pangulu_n;

    if (rank == 0)
    {
        if (init_options == NULL)
        {
            printf(PANGULU_E_OPTION_IS_NULLPTR);
            exit(1);
        }
        if (init_options->nb == 0)
        {
            printf(PANGULU_E_NB_IS_ZERO);
            exit(1);
        }
    }

    common->nb = init_options->nb;
    common->sum_rank_size = size;
    common->omp_thread = init_options->nthread;
    pangulu_cm_bcast(&common->n, 1, MPI_PANGULU_EXBLOCK_IDX, 0);
    pangulu_cm_bcast(&common->nb, 1, MPI_PANGULU_INBLOCK_IDX, 0);

    pangulu_int64_t tmp_p = sqrt(common->sum_rank_size);
    while (((common->sum_rank_size) % tmp_p) != 0)
    {
        tmp_p--;
    }

    common->p = tmp_p;
    common->q = common->sum_rank_size / tmp_p;
    pangulu_origin_smatrix *origin_smatrix = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
    pangulu_init_pangulu_origin_smatrix(origin_smatrix);

    if (rank == 0)
    {
        origin_smatrix->row = pangulu_n;
        origin_smatrix->column = pangulu_nnz;
        origin_smatrix->rowpointer = csr_rowptr;
        origin_smatrix->columnindex = csr_colidx;
        origin_smatrix->nnz = pangulu_nnz;
        origin_smatrix->value = csr_value;
        if (origin_smatrix->row == 0)
        {
            printf(PANGULU_E_ROW_IS_ZERO);
            exit(1);
        }
    }

    pangulu_int32_t p = common->p;
    pangulu_int32_t q = common->q;
    pangulu_int32_t nb = common->nb;
    pangulu_cm_sync();
    pangulu_cm_bcast(&origin_smatrix->row, 1, MPI_PANGULU_INT64_T, 0);
    common->n = origin_smatrix->row;
    pangulu_int64_t n = common->n;
    omp_set_num_threads(init_options->nthread);
#if defined(OPENBLAS_CONFIG_H) || defined(OPENBLAS_VERSION)
    openblas_set_num_threads(1);
#endif
    if (rank == 0)
    {
#ifdef ADAPTIVE_KERNEL_SELECTION
        printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON);
#else
        printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF);
#endif
#ifdef PANGULU_GPU_COMPLEX_FALLBACK_FLAG
        printf(PANGULU_W_COMPLEX_FALLBACK);
#endif
        printf(PANGULU_I_BASIC_INFO);
    }

#ifdef GPU_OPEN
    int device_num;
    pangulu_platform_get_device_num(&device_num, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_set_default_device(rank%device_num, PANGULU_DEFAULT_PLATFORM);
#endif

    pangulu_block_smatrix *block_smatrix = (pangulu_block_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_smatrix));
    pangulu_init_pangulu_block_smatrix(block_smatrix);
    pangulu_block_common *block_common = (pangulu_block_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_common));
    block_common->rank = rank;
    block_common->p = p;
    block_common->q = q;
    block_common->nb = nb;
    block_common->n = n;
    block_common->block_length = PANGULU_ICEIL(n, nb);
    block_common->sum_rank_size = common->sum_rank_size;
    sc25_init(NULL, sizeof(pangulu_task_t));

    pangulu_origin_smatrix *reorder_matrix = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
    pangulu_init_pangulu_origin_smatrix(reorder_matrix);

    struct timeval time_start;
    double elapsed_time;
    
    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_reordering(
        block_smatrix,
        origin_smatrix,
        reorder_matrix
    );
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_REORDER);}

    if(rank == 0){
        block_smatrix->A_rowsum_reordered = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        memset(block_smatrix->A_rowsum_reordered, 0, sizeof(calculate_type) * n);
        for(pangulu_exblock_idx row = 0; row < n; row++){
            for(pangulu_exblock_ptr idx = reorder_matrix->rowpointer[row]; idx < reorder_matrix->rowpointer[row+1]; idx++){
                block_smatrix->A_rowsum_reordered[row]+=reorder_matrix->value[idx];
            }
        }
    }

    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    if (rank == 0)
    {
        pangulu_symbolic(block_common,
                         block_smatrix,
                         reorder_matrix);
    }
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_SYMBOLIC);}

    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_preprocessing(
        common,
        block_common,
        block_smatrix,
        reorder_matrix,
        init_options->nthread);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_PRE);}

    pangulu_free(__FILE__, __LINE__, origin_smatrix);
    origin_smatrix = NULL;
    pangulu_free(__FILE__, __LINE__, reorder_matrix);
    reorder_matrix = NULL;

    pangulu_cm_sync();
    mem_usage();

    (*pangulu_handle) = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_handle_t));
    (*(pangulu_handle_t **)pangulu_handle)->block_common = block_common;
    (*(pangulu_handle_t **)pangulu_handle)->block_smatrix = block_smatrix;
    (*(pangulu_handle_t **)pangulu_handle)->commmon = common;
}

void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);

    if (rank == 0)
    {
        if (gstrf_options == NULL)
        {
            printf(PANGULU_E_GSTRF_OPTION_IS_NULLPTR);
            exit(1);
        }
    }

    struct timeval time_start;
    double elapsed_time;

    //for(int bcol = 0; bcol < block_common->block_length; bcol++){
    //    for(int bidx = block_smatrix->bcsc_related_pointer[bcol]; bidx < block_smatrix->bcsc_related_pointer[bcol+1]; bidx++){
    //        int brow = block_smatrix->bcsc_related_index[bidx];
    //        pangulu_uint64_t slot_addr = block_smatrix->bcsc_related_draft_info[bidx];
    //        pangulu_storage_slot_t* slot = &(block_smatrix->storage->bins[0].slots[PANGULU_DIGINFO_GET_STOREIDX(slot_addr)]);
    //
    //        pangulu_inblock_ptr* colptr = slot->columnpointer;
    //        pangulu_inblock_idx* rowidx = slot->rowindex;
    //        calculate_type* value = slot->value;
    //        // printf("Tile(%d, %d):\n", brow, bcol);
    //        for(int col = 0; col < block_common->nb; col++){
    //            // printf("(col=%3d) : ", col);
    //            for(int idx = colptr[col]; idx < colptr[col + 1]; idx++){
    //                value[idx] = 1.0;
    //                // printf("%3d(%9.2le) ", rowidx[idx], value[idx]);
    //            }
    //            // printf("\n");
    //        }
    //    }
    //}

    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_numeric(common,
                    block_common,
                    block_smatrix);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_NUMERICAL);}

    if(rank == 0){
        printf("rank\ttime_getrf\ttime_tstrf\ttime_gessm\ttime_ssssm\n");
        fflush(stdout);
    }
    pangulu_cm_sync();
    for(int i = 0; i < size; i++){
        if(rank == i){
            printf("#%d\t%lf\t%lf\t%lf\t%lf\n", rank, global_stat.time_getrf, global_stat.time_tstrf, global_stat.time_gessm, global_stat.time_ssssm);
            fflush(stdout);
        }
        pangulu_cm_sync();
        usleep(10);
    }

    // for(int bcol = 0; bcol < block_common->block_length; bcol++){
    //     for(int bidx = block_smatrix->bcsc_related_pointer[bcol]; bidx < block_smatrix->bcsc_related_pointer[bcol+1]; bidx++){
    //         int brow = block_smatrix->bcsc_related_index[bidx];
    //         pangulu_uint64_t slot_addr = block_smatrix->bcsc_related_draft_info[bidx];
    //         pangulu_storage_slot_t* slot = &(block_smatrix->storage->bins[0].slots[PANGULU_DIGINFO_GET_STOREIDX(slot_addr)]);

    //         pangulu_inblock_ptr* colptr = slot->columnpointer;
    //         pangulu_inblock_idx* rowidx = slot->rowindex;
    //         calculate_type* value = slot->value;
    //         printf("Tile(%d, %d):\n", brow, bcol);
    //         for(int col = 0; col < block_common->nb; col++){
    //             printf("(col=%3d) : ", col);
    //             for(int idx = colptr[col]; idx < colptr[col + 1]; idx++){
    //                 printf("%3d(%9.2le) ", rowidx[idx], value[idx]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    pangulu_cm_sync();
    mem_usage();
    pangulu_numeric_check(common, block_common, block_smatrix);
    pangulu_cm_sync();
    
    // int nb = block_common->nb;
    // for(int bcol = 0; bcol < block_common->block_length; bcol++){
    //     for(int bidx = block_smatrix->bcsc_related_pointer[bcol]; bidx < block_smatrix->bcsc_related_pointer[bcol+1]; bidx++){
    //         int brow = block_smatrix->bcsc_related_index[bidx];
    //         pangulu_uint64_t slot_addr = block_smatrix->bcsc_related_draft_info[bidx];
    //         pangulu_storage_slot_t* slot = &(block_smatrix->storage->bins[0].slots[PANGULU_DIGINFO_GET_STOREIDX(slot_addr)]);
    //         pangulu_inblock_ptr* colptr = slot->columnpointer;
    //         pangulu_inblock_idx* rowidx = slot->rowindex;
    //         calculate_type* value = slot->value;
    //         // printf("Tile(%d, %d):\n", brow, bcol);
    //         for(int col = 0; col < block_common->nb; col++){
    //             // printf("(col=%3d) : ", col);
    //             // for(int idx = colptr[col]; idx < colptr[col + 1]; idx++){
    //             //     printf("%3d(%9.2le) ", rowidx[idx], value[idx]);
    //             // }
    //             // printf("\n");

    //             if(bidx == block_smatrix->bcsc_related_pointer[bcol]){
    //                 int idx = colptr[col];
    //                 printf("%d %lf\n", col + bcol * nb, value[colptr[col]]);
    //                 if(fabs(value[colptr[col]] - 1.0) > 0.1){
    //                     printf("first val in col %d is %lf\n", col + bcol * nb, value[colptr[col]]);
    //                     exit(1);
    //                 }
    //             }

    //         }
    //     }
    // }
    
    if(rank == 0){
        printf("\n\n");
    }
    MPI_Finalize();
    exit(0);
}

void pangulu_gstrs(calculate_type *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);

    if (rank == 0)
    {
        if (gstrs_options == NULL)
        {
            printf(PANGULU_E_GSTRS_OPTION_IS_NULLPTR);
            exit(1);
        }
    }

    pangulu_int64_t vector_length = common->n;
    pangulu_vector *x_vector = NULL;
    pangulu_vector *b_vector = NULL;
    pangulu_vector *answer_vector = NULL;

    if (rank == 0)
    {
        x_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        b_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        answer_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        b_vector->row = common->n;
        b_vector->value = rhs;
        pangulu_init_pangulu_vector(x_vector, vector_length);
        pangulu_init_pangulu_vector(answer_vector, vector_length);
        // for(int i=0; i<b_vector->row; i++){
        //     printf("%f ", b_vector->value[i]);
        // }
        // printf("\n");
        pangulu_reorder_vector_b_tran(block_smatrix->row_perm, block_smatrix->metis_perm, block_smatrix->row_scale, b_vector, answer_vector);
        // for(int i=0; i<answer_vector->row; i++){
        //     printf("%f ", answer_vector->value[i]);
        // }
        // printf("\n");
    }

    pangulu_sptrsv_preprocessing(
        block_common,
        block_smatrix,
        answer_vector);

    struct timeval time_start;
    double elapsed_time;
    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_solve(block_common, block_smatrix, answer_vector);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_SPTRSV);}

    // check sptrsv answer
    // pangulu_sptrsv_vector_gather(block_common, block_smatrix, answer_vector);
    

    if (rank == 0)
    {
        pangulu_reorder_vector_x_tran(block_smatrix, answer_vector, x_vector);

        // for(int i=0;i<common->nb * block_common->block_length;i++){
        //     printf("%f ", answer_vector->value[i]);
        // }
        // printf("\n");

        for (int i = 0; i < common->n; i++)
        {
            rhs[i] = x_vector->value[i];
        }

        pangulu_destroy_pangulu_vector(x_vector);
        pangulu_destroy_pangulu_vector(answer_vector);
        pangulu_free(__FILE__, __LINE__, b_vector);
    }
}

void pangulu_gssv(calculate_type *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_gstrf(gstrf_options, pangulu_handle);
    pangulu_gstrs(rhs, gstrs_options, pangulu_handle);
}

void pangulu_finalize(void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    sc25_finalize(NULL);
    // pangulu_destroy(block_common, block_smatrix);

    pangulu_free(__FILE__, __LINE__, block_common);
    pangulu_free(__FILE__, __LINE__, block_smatrix);
    pangulu_free(__FILE__, __LINE__, common);
    pangulu_free(__FILE__, __LINE__, *(pangulu_handle_t **)pangulu_handle);
}