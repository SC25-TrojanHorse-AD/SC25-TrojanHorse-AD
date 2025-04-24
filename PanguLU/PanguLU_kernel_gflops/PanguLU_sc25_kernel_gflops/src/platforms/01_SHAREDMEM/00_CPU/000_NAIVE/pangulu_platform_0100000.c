#define PANGULU_PLATFORM_ENV
#include "../../../../pangulu_common.h"
#include "cblas.h"

pangulu_int32_t** getrf_diagIndex_csc = NULL;
pangulu_int32_t** getrf_nextptr = NULL;
pangulu_int32_t **ssssm_hash_lu = NULL;
pangulu_int32_t **ssssm_hash_l_row = NULL;
pangulu_int32_t **ssssm_hash_u_col = NULL;
calculate_type **ssssm_l_value = NULL;
calculate_type **ssssm_u_value = NULL;
calculate_type **temp_a_value;
calculate_type** TEMP_calculate_type;
pangulu_inblock_ptr** TEMP_pangulu_inblock_ptr;

void pangulu_platform_0100000_malloc(void** platform_address, size_t size){
    *platform_address = pangulu_malloc(__FILE__, __LINE__, size);
}

void pangulu_platform_0100000_synchronize(){}

void pangulu_platform_0100000_memset(void* s, int c, size_t n){
    memset(s, c, n);
}

void pangulu_platform_0100000_create_stream(void** stream){}

void pangulu_platform_0100000_memcpy(void *dst, const void *src, size_t count, unsigned int kind){
    memcpy(dst, src, count);
}

void pangulu_platform_0100000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void* stream){
    memcpy(dst, src, count);
}

void pangulu_platform_0100000_free(void* devptr){
    free(devptr);
}

void pangulu_platform_0100000_get_device_num(int* device_num){
    *device_num = 1;
}

void pangulu_platform_0100000_set_default_device(int device_num){}

void pangulu_platform_0100000_get_device_name(char* name, int device_num){
    strcpy(name, "CPU");
}

void pangulu_platform_0100000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    int tid
){
    // printf("GETRF opdst before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }

    if(!getrf_diagIndex_csc[tid]){
        getrf_diagIndex_csc[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb + 1));
    }
    if(!getrf_nextptr[tid]){
        getrf_nextptr[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * nb);
    }
    pangulu_inblock_idx n = nb;
    for (int i = 0; i < n; i++)
    {
        getrf_diagIndex_csc[tid][i] = binarysearch_inblk(opdst->rowindex, ((i==0)?0:opdst->columnpointer[i]), opdst->columnpointer[i+1], i);
    }

    for (int i = 0; i < n; i++)
    {
        if (((i==0)?0:opdst->columnpointer[i]) == opdst->columnpointer[i + 1])
        {
            continue;
        }
        for(int j = 0; j < i; j++){
            getrf_nextptr[tid][j] = getrf_diagIndex_csc[tid][j];
        }
        for(int j = ((i==0)?0:opdst->columnpointer[i]); j <= getrf_diagIndex_csc[tid][i]; j++){
            pangulu_inblock_idx row_target = opdst->rowindex[j];
            for(int k = ((i==0)?0:opdst->columnpointer[i]); k < j; k++){
                pangulu_inblock_idx row_vec = opdst->rowindex[k];
                while((getrf_nextptr[tid][row_vec] < opdst->columnpointer[row_vec + 1]) && (opdst->rowindex[getrf_nextptr[tid][row_vec]] < row_target)){
                    getrf_nextptr[tid][row_vec]++;
                }
                if((getrf_nextptr[tid][row_vec] < opdst->columnpointer[row_vec + 1]) && (opdst->rowindex[getrf_nextptr[tid][row_vec]] == row_target)){
                    opdst->value[j] -= opdst->value[k] * opdst->value[getrf_nextptr[tid][row_vec]];
                }
            }
        }
        calculate_type pivot = opdst->value[getrf_diagIndex_csc[tid][i]];
        for(int j = getrf_diagIndex_csc[tid][i] + 1; j < opdst->columnpointer[i + 1]; j++){
            pangulu_inblock_idx row_target = opdst->rowindex[j];
            for(int k = ((i==0)?0:opdst->columnpointer[i]); k < j; k++){
                if(opdst->rowindex[k] >= i){
                    break;
                }
                pangulu_inblock_idx row_vec = opdst->rowindex[k];
                while((getrf_nextptr[tid][row_vec] < opdst->columnpointer[row_vec + 1]) && (opdst->rowindex[getrf_nextptr[tid][row_vec]] < row_target)){
                    getrf_nextptr[tid][row_vec]++;
                }
                if((getrf_nextptr[tid][row_vec] < opdst->columnpointer[row_vec + 1]) && (opdst->rowindex[getrf_nextptr[tid][row_vec]] == row_target)){
                    opdst->value[j] -= opdst->value[k] * opdst->value[getrf_nextptr[tid][row_vec]];
                }
            }
            opdst->value[j] /= pivot;
        }
    }
    
    // printf("GETRF opdst after:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }
}


void pangulu_platform_0100000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
){
    // printf("TSTRF diag:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdiag->columnpointer[i]);j<opdiag->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdiag->value[j]);
    //     }
    //     printf("\n");
    // }
    // printf("TSTRF rhs before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }
    
    pangulu_inblock_ptr* U_colptr = opdiag->columnpointer;
    pangulu_inblock_idx* U_rowidx = opdiag->rowindex;
    calculate_type* u_value = opdiag->value;
    pangulu_inblock_ptr* A_colptr = opdst->columnpointer;
    pangulu_inblock_idx* A_rowidx = opdst->rowindex;
    calculate_type* a_value = opdst->value;
    pangulu_inblock_idx n = nb;

    // if(TEMP_calculate_type_len < n){
    //     calculate_type* TEMP_calculate_type_old = TEMP_calculate_type;
    //     TEMP_calculate_type = (calculate_type*)pangulu_realloc(__FILE__, __LINE__, TEMP_calculate_type, n*sizeof(calculate_type));
    //     if(TEMP_calculate_type == NULL){
    //         pangulu_free(__FILE__, __LINE__, TEMP_calculate_type_old);
    //         TEMP_calculate_type_len = 0;
    //         printf("[ERROR] kernel error : CPU sparse tstrf : realloc TEMP_calculate_type failed.\n");
    //         return;
    //     }
    //     TEMP_calculate_type_len = n;
    // }
    
    // if(TEMP_pangulu_inblock_ptr_len < n){
    //     pangulu_inblock_ptr* TEMP_int64_old = TEMP_pangulu_inblock_ptr;
    //     TEMP_pangulu_inblock_ptr = (pangulu_inblock_ptr*)pangulu_realloc(__FILE__, __LINE__, TEMP_pangulu_inblock_ptr, n*sizeof(pangulu_inblock_ptr));
    //     if(TEMP_pangulu_inblock_ptr == NULL){
    //         pangulu_free(__FILE__, __LINE__, TEMP_int64_old);
    //         TEMP_pangulu_inblock_ptr_len = 0;
    //         printf("[ERROR] kernel error : CPU sparse tstrf : realloc TEMP_int64 failed.\n");
    //         return;
    //     }
    //     TEMP_pangulu_inblock_ptr_len = n;
    // }

    if(!TEMP_calculate_type[tid]){
        TEMP_calculate_type[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    }
    
    if(!TEMP_pangulu_inblock_ptr[tid]){
        TEMP_pangulu_inblock_ptr[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * n);
    }

    pangulu_inblock_ptr* U_next_array = TEMP_pangulu_inblock_ptr[tid];
    calculate_type* A_major_column = TEMP_calculate_type[tid];
    memcpy(U_next_array, U_colptr, sizeof(pangulu_inblock_ptr) * n);
    U_next_array[0] = 0;
    for(pangulu_int64_t i=0;i<n;i++){ // A的每列作为主列
        memset(A_major_column, 0, sizeof(calculate_type)*n);
        pangulu_int32_t u_pivot_idx = binarysearch_inblk(U_rowidx, ((i==0)?0:U_colptr[i]), U_colptr[i+1], i);
        calculate_type U_pivot = u_value[u_pivot_idx]; //这里i本来应是A主列的列号，也是U主元的行号。U的主元在对角线上，因此，i也是U的主元的列号。
        // #pragma omp parallel for
        for(pangulu_int64_t j=((i==0)?0:A_colptr[i]);j<A_colptr[i+1];j++){
            A_major_column[A_rowidx[j]] = (a_value[j] /= U_pivot);
        }
        // #pragma omp parallel for
        for(pangulu_int64_t k=i+1;k<n;k++){ // 遍历A的副列
            if(U_next_array[k] >= U_colptr[k+1]/*U_next_array[k]跑到了下一列*/ || U_rowidx[U_next_array[k]] > i/*U的第k列中，下一个要访问的元素的行号大于A当前主列号i*/){
                continue;
            }
            for(pangulu_int64_t j=((k==0)?0:A_colptr[k]);j<A_colptr[k+1];j++){ // 遍历A的副列k中的每个元素
                a_value[j] -= u_value[U_next_array[k]] * A_major_column[A_rowidx[j]];
            }
            U_next_array[k]++;
        }
    }

    // printf("TSTRF rhs after:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=opdst->columnpointer[i];j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }
}


void pangulu_platform_0100000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
){
    // printf("GESSM diag:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdiag->columnpointer[i]);j<opdiag->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdiag->value[j]);
    //     }
    //     printf("\n");
    // }
    // printf("GESSM rhs before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }

    pangulu_inblock_ptr *a_columnpointer = opdst->columnpointer;
    pangulu_inblock_idx *a_rowidx = opdst->rowindex;
    calculate_type *a_value = opdst->value;
    pangulu_inblock_ptr *l_columnpointer = opdiag->columnpointer;
    pangulu_inblock_idx *l_rowidx = opdiag->rowindex;
    calculate_type *l_value = opdiag->value;
    pangulu_int64_t n = nb;

    if(temp_a_value[tid] == NULL){
        temp_a_value[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    }

// #pragma omp parallel for num_threads(pangulu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = ((i==0)?0:a_columnpointer[i]); j < a_columnpointer[i + 1]; j++)
        {
            int idx = a_rowidx[j];
            temp_a_value[tid][i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

// #pragma omp parallel for num_threads(pangulu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = ((i==0)?0:a_columnpointer[i]); j < a_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = a_rowidx[j];
            a_value[j] = temp_a_value[tid][i * n + idx];
            pangulu_int32_t diag_idx = binarysearch_inblk(l_rowidx, ((idx==0)?0:l_columnpointer[idx]), l_columnpointer[idx+1], idx);
            for (pangulu_int64_t k = diag_idx + 1; k < l_columnpointer[idx + 1]; k++)
            {
                temp_a_value[tid][i * n + l_rowidx[k]] -= l_value[k] * a_value[j];
            }
        }
    }


    // printf("GESSM rhs after:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=opdst->columnpointer[i];j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }
}

void pangulu_platform_0100000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* op1,
    pangulu_storage_slot_t* op2,
    int tid
){
    // printf("SSSSM op1 before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:op1->columnpointer[i]);j<op1->columnpointer[i+1];j++){
    //         printf("%6.2lf ", op1->value[j]);
    //     }
    //     printf("\n");
    // }

    // printf("SSSSM op2 before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:op2->columnpointer[i]);j<op2->columnpointer[i+1];j++){
    //         printf("%6.2lf ", op2->value[j]);
    //     }
    //     printf("\n");
    // }

    // printf("SSSSM opdst before:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }

    if(ssssm_hash_lu[tid] == NULL){
        ssssm_hash_lu[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_l_row[tid] == NULL){
        ssssm_hash_l_row[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_u_col[tid] == NULL){
        ssssm_hash_u_col[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_l_value[tid] == NULL){
        ssssm_l_value[tid] = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
        memset(ssssm_l_value[tid], 0, sizeof(calculate_type) * nb * nb);
    }
    if(ssssm_u_value[tid] == NULL){
        ssssm_u_value[tid] = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
        memset(ssssm_u_value[tid], 0, sizeof(calculate_type) * nb * nb);
    }
    if(temp_a_value[tid] == NULL){
        temp_a_value[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    }

    int n = nb;
    int L_row_num = 0, U_col_num = 0, LU_rc_num = 0;
    int *blas_dense_hash_col_LU = NULL;
    int *blas_dense_hash_row_L = NULL;

    blas_dense_hash_col_LU = ssssm_hash_lu[tid];
    blas_dense_hash_row_L = ssssm_hash_l_row[tid];
    for (int i = 0; i < n; i++)
    {
        blas_dense_hash_row_L[i] = -1;
    }
    for (int i = 0; i < n; i++)
    {
        if (op1->columnpointer[i + 1] > ((i==0)?0:op1->columnpointer[i]))
        {
            blas_dense_hash_col_LU[i] = LU_rc_num;
            LU_rc_num++;
        }
    }
    for (int i = 0; i < n; i++)
    {
        int col_begin = ((i==0)?0:op1->columnpointer[i]);
        int col_end = op1->columnpointer[i + 1];
        for (int j = col_begin; j < col_end; j++)
        {
            int L_row = op1->rowindex[j];
            if (blas_dense_hash_row_L[L_row] == -1) // 如果当前行未标记
            {
                blas_dense_hash_row_L[L_row] = L_row_num;
                L_row_num++;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        int begin = ((i==0)?0:op2->columnpointer[i]);
        int end = op2->columnpointer[i + 1];
        if (end > begin)
        {
            calculate_type *U_temp_value = ssssm_u_value[tid] + U_col_num * LU_rc_num; // op2 column based

            for (int j = begin; j < end; j++)
            {
                int U_row = op2->rowindex[j];
                if (op1->columnpointer[U_row + 1] > ((U_row==0)?0:op1->columnpointer[U_row]) > 0)
                { // only store the remain data
                    U_temp_value[blas_dense_hash_col_LU[U_row]] = op2->value[j];
                }
            }
            ssssm_hash_u_col[tid][U_col_num] = i;
            U_col_num++;
        }
    }
    for (int i = 0; i < n; i++)
    {
        int col_begin = ((i==0)?0:op1->columnpointer[i]);
        int col_end = op1->columnpointer[i + 1];
        calculate_type *temp_data = ssssm_l_value[tid] + L_row_num * blas_dense_hash_col_LU[i];
        for (int j = col_begin; j < col_end; j++)
        {
            temp_data[blas_dense_hash_row_L[op1->rowindex[j]]] = op1->value[j];
        }
    }
    int m = L_row_num;
    int k = LU_rc_num;
    n = U_col_num;

    calculate_type alpha = 1.0, beta = 0.0;
#if defined(CALCULATE_TYPE_CR64)
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, &beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_R64)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_R32)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_CR32)
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, &beta, temp_a_value[tid], m);
#else
#error[PanguLU Compile Error] Unsupported value type for BLAS library.
#endif

    memset(ssssm_l_value[tid], 0, sizeof(calculate_type) * m * k);
    memset(ssssm_u_value[tid], 0, sizeof(calculate_type) * k * n);


    #pragma omp critical
    {
        for (int i = 0; i < U_col_num; i++)
        {
            int col_num = ssssm_hash_u_col[tid][i];
            calculate_type *temp_value = temp_a_value[tid] + i * m;
            int j_begin = ((col_num==0)?0:opdst->columnpointer[col_num]);
            int j_end = opdst->columnpointer[col_num + 1];
            for (int j = j_begin; j < j_end; j++)
            {
                int row = opdst->rowindex[j];
                if (blas_dense_hash_row_L[row] != -1)
                {
                    int row_index = blas_dense_hash_row_L[row];
                    // #pragma omp atomic
                    opdst->value[j] -= temp_value[row_index];
                }
            }
        }
    }

    // printf("SSSSM opdst after:\n");
    // for(int i=0;i<nb;i++){
    //     printf("(%d):", i);
    //     for(int j=((i==0)?0:opdst->columnpointer[i]);j<opdst->columnpointer[i+1];j++){
    //         printf("%6.2lf ", opdst->value[j]);
    //     }
    //     printf("\n");
    // }
}

void pangulu_platform_0100000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
){
    
}


void pangulu_platform_0100000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
){
    // need nthread
}



void pangulu_platform_0100000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* a,
    calculate_type* x,
    calculate_type* y
){
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
    if(nb > 0){
        for(int idx = 0; idx < a->columnpointer[1]; idx++){
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[0];
        }
    }
    for(int col = 1; col < nb; col++){
        for(int idx = a->columnpointer[col]; idx < a->columnpointer[col+1]; idx++){
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[col];
        }
    }

    // for(int i=0;i<y->row; i++){
    //     printf("%6.2lf ", y->value[i]);
    // }
    // printf("\n");
}

void pangulu_platform_0100000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval, 
    calculate_type *xval
){
    // printf("vecadd\n");
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        bval[i] += xval[i];
    }
}

void pangulu_platform_0100000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type* xval,
    pangulu_int64_t uplo
){
    pangulu_int64_t col=nb;  
    pangulu_inblock_ptr *csc_column_ptr_tmp=s->columnpointer;
    pangulu_inblock_idx *csc_row_idx_tmp=s->rowindex;
    calculate_type *cscVal_tmp = s->value;

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
    
    if(uplo==0){
        // printf("sptrsv_lower kernel\n");
        for(pangulu_int64_t i=0;i<col;i++) 
        {
            pangulu_int32_t diag_idx = binarysearch_inblk(
                csc_row_idx_tmp,
                (i==0)?0:csc_column_ptr_tmp[i],
                csc_column_ptr_tmp[i+1],
                i
            );
            if(diag_idx == -1){
                xval[i]=0.0;
                continue;
            }
            for(pangulu_int64_t j=diag_idx+1;j<csc_column_ptr_tmp[i+1];j++)
            {
                pangulu_inblock_idx row=csc_row_idx_tmp[j];
                xval[row]-=cscVal_tmp[j]*xval[i];
            }
        }
    }else{
        // printf("sptrsv_upper kernel\n");
        for(pangulu_int64_t i=col-1;i>=0;i--) 
        {
            pangulu_int32_t diag_idx = binarysearch_inblk(
                csc_row_idx_tmp,
                (i==0)?0:csc_column_ptr_tmp[i],
                csc_column_ptr_tmp[i+1],
                i
            );
            if(diag_idx != -1){
                if(fabs(cscVal_tmp[diag_idx])>PANGULU_SPTRSV_TOL)
                xval[i]=xval[i]/cscVal_tmp[diag_idx];
                else
                xval[i]=xval[i]/PANGULU_SPTRSV_TOL;
            }
            else{
                xval[i]=0.0;
                continue;
            }
            for(pangulu_int64_t j=diag_idx-1;j>=((i==0)?0:csc_column_ptr_tmp[i]);j--)
            {
                pangulu_inblock_idx row=csc_row_idx_tmp[j];
                xval[row]-=cscVal_tmp[j]*xval[i];
            }
        }
    }
    // for(int i=0; i<nb; i++){
    //     printf("%f ", x->value[i]);
    // }
    // printf("\n");
    
}
