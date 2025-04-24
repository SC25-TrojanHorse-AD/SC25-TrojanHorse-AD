#include "pangulu_common.h"

// void pangulu_convert_csc_to_bcsc(
//     pangulu_exblock_idx n,
//     pangulu_exblock_ptr* csc_pointer,
//     pangulu_exblock_idx* csc_index,
//     calculate_type* csc_value,
//     pangulu_inblock_idx block_order,
//     pangulu_exblock_ptr** bcsc_struct_pointer,
//     pangulu_exblock_idx** bcsc_struct_index,
//     pangulu_inblock_ptr** bcsc_struct_nnz,
//     pangulu_inblock_ptr*** bcsc_inblock_pointers,
//     pangulu_inblock_idx*** bcsc_inblock_indeces,
//     calculate_type*** bcsc_values
// ){
    
// }

void pangulu_convert_csr_to_csc(
    int free_csrmatrix,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** csr_pointer,
    pangulu_exblock_idx** csr_index,
    calculate_type** csr_value,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index,
    calculate_type** csc_value
){
    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_exblock_ptr* rowpointer = *csr_pointer;
    pangulu_exblock_idx* columnindex = *csr_index;
    calculate_type* value = NULL;
    if(csr_value){
        value = *csr_value;
    }

    pangulu_exblock_ptr nnz = rowpointer[n];
    
    pangulu_exblock_ptr* columnpointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_ptr* aid_ptr_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx* rowindex = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    calculate_type* value_csc = NULL;
    if(value){
        value_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    }

    memset(columnpointer, 0, sizeof(pangulu_exblock_ptr) * (n + 1));
    for(pangulu_exblock_idx row = 0; row < n; row++){
        for(pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row+1]; idx++){
            pangulu_exblock_idx col = columnindex[idx];
            columnpointer[col+1]++;
        }
    }
    for(pangulu_exblock_idx col = 0; col < n; col++){
        columnpointer[col+1] += columnpointer[col];
    }
    memcpy(aid_ptr_arr, columnpointer, sizeof(pangulu_exblock_ptr) * (n + 1));

    for(pangulu_exblock_idx row = 0; row < n; row++){
        for(pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row+1]; idx++){
            pangulu_exblock_idx col = columnindex[idx];
            rowindex[aid_ptr_arr[col]] = row;
            if(value){
                value_csc[aid_ptr_arr[col]] = value[idx];
            }
            aid_ptr_arr[col]++;
        }
    }

    pangulu_free(__FILE__, __LINE__, aid_ptr_arr);
    if(free_csrmatrix){
        pangulu_free(__FILE__, __LINE__, *csr_pointer);
        pangulu_free(__FILE__, __LINE__, *csr_index);
        pangulu_free(__FILE__, __LINE__, *csr_value);
        *csr_pointer = NULL;
        *csr_index = NULL;
        *csr_value = NULL;
    }
    *csc_pointer = columnpointer;
    *csc_index = rowindex;
    if(csc_value){
        *csc_value = value_csc;
    }
    // printf("[PanguLU LOG] pangulu_convert_csr_to_csc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

void pangulu_convert_halfsymcsc_to_csc_struct(
    int free_halfmatrix,
    int if_colsort,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** half_csc_pointer,
    pangulu_exblock_idx** half_csc_index,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index
){
    struct timeval start_time;
    pangulu_time_start(&start_time);
    struct timeval start_time1;

    pangulu_exblock_ptr* half_ptr = *half_csc_pointer;
    pangulu_exblock_idx* half_idx = *half_csc_index;
    pangulu_exblock_ptr nnz = half_ptr[n];
    pangulu_exblock_ptr* csc_ptr = (pangulu_exblock_ptr*)malloc(sizeof(pangulu_exblock_ptr) * (n + 1));
    memset(csc_ptr, 0, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx* local_csc_idx = (pangulu_exblock_idx*)malloc(sizeof(pangulu_exblock_idx) * nnz * 2);
    for (pangulu_exblock_idx col = 0; col < n; col++) {
        for (pangulu_exblock_ptr row_idx = half_ptr[col]; row_idx < half_ptr[col + 1]; row_idx++) {
            pangulu_exblock_idx row = half_idx[row_idx];
            csc_ptr[col + 1]++;
            if (row != col) {
                csc_ptr[row + 1]++;
            }
        }
    }
    for (pangulu_exblock_idx i = 1; i <= n; i++) {
        csc_ptr[i] += csc_ptr[i - 1];
    }


    pangulu_exblock_idx* col_offset = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
    memset(col_offset, 0, sizeof(pangulu_exblock_idx) * n);
    for (pangulu_exblock_idx col = 0; col < n; col++) {
        for (pangulu_exblock_ptr row_idx = half_ptr[col]; row_idx < half_ptr[col + 1]; row_idx++) {
            pangulu_exblock_idx row = half_idx[row_idx];
            local_csc_idx[csc_ptr[col] + col_offset[col]] = row;
            col_offset[col]++;
            if (row != col) {
                local_csc_idx[csc_ptr[row] + col_offset[row]] = col;
                col_offset[row]++;
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, col_offset);


    // pangulu_exblock_ptr* aid_arr = (pangulu_exblock_ptr*)malloc(sizeof(pangulu_exblock_ptr) * (n + 1));
    // memcpy(aid_arr, csc_ptr, sizeof(pangulu_exblock_ptr) * (n + 1));
    // pangulu_time_start(&start_time1);
    // for (pangulu_exblock_idx col = 0; col < n; col++) {
    //     for (pangulu_exblock_idx row_idx = (half_idx[half_ptr[col]]==col)?half_ptr[col]+1:half_ptr[col]; row_idx < half_ptr[col + 1]; row_idx++) {
    //         pangulu_exblock_idx row = half_idx[row_idx];
    //         local_csc_idx[aid_arr[col]] = row;
    //         aid_arr[col]++;
    //     }
    // }
    // // printf("[PanguLU LOG] pangulu_convert_halfsymcsc_to_csc_struct Step 1 time : %.2lf ms\n", pangulu_time_stop(&start_time1) * 1e3);
    // pangulu_time_start(&start_time1);
    // // pangulu_exblock_idx chunk_len = 32768;
    // // pangulu_exblock_idx nchunk = PANGULU_ICEIL(n, chunk_len);
    // pangulu_exblock_idx nchunk = 2;
    // pangulu_exblock_idx chunk_len = PANGULU_ICEIL(n, nchunk);
    // // printf("nchunk = %d chunk_len = %d\n", nchunk, chunk_len);
    // for(pangulu_exblock_idx ichunk = 0; ichunk < nchunk; ichunk++){
    //     pangulu_exblock_idx chunk_lb = PANGULU_MIN(ichunk * chunk_len, n);
    //     pangulu_exblock_idx chunk_ub = PANGULU_MIN((ichunk+1) * chunk_len, n);
    //     // printf("ichunk = %d %lld~%lld\n", ichunk, half_ptr[chunk_lb], half_ptr[chunk_ub]);
    //     for (pangulu_exblock_idx col = 0; col < n; col++) {
    //         for (pangulu_exblock_idx row_idx = half_ptr[col]; row_idx < half_ptr[col + 1]; row_idx++) {
    //             pangulu_exblock_idx row = half_idx[row_idx];
    //             if(row >= chunk_lb && row < chunk_ub){
    //                 local_csc_idx[aid_arr[row]] = col;
    //                 aid_arr[row]++;
    //             }
    //         }
    //     }
    // }
    // // printf("[PanguLU LOG] pangulu_convert_halfsymcsc_to_csc_struct Step 2 time : %.2lf ms\n", pangulu_time_stop(&start_time1) * 1e3);
    // pangulu_free(__FILE__, __LINE__, aid_arr);


    if (free_halfmatrix) {
        pangulu_free(__FILE__, __LINE__, *half_csc_pointer);
        pangulu_free(__FILE__, __LINE__, *half_csc_index);
        *half_csc_pointer = NULL;
        *half_csc_index = NULL;
    }

    if(if_colsort){
        pangulu_int32_t rank, nproc;
        pangulu_cm_rank(&rank);
        pangulu_cm_size(&nproc);
        int nthread_sort = 2;
        bind_to_core((rank * nthread_sort) % sysconf(_SC_NPROCESSORS_ONLN));
        #pragma omp parallel num_threads(nthread_sort)
        {
            bind_to_core((rank * nthread_sort + omp_get_thread_num()) % sysconf(_SC_NPROCESSORS_ONLN));
        }
        pangulu_sort_exblock_struct(n, csc_ptr, local_csc_idx, nthread_sort);
        bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));
    }
    
    *csc_pointer = csc_ptr;
    *csc_index = pangulu_realloc(__FILE__, __LINE__, local_csc_idx, sizeof(pangulu_exblock_idx) * csc_ptr[n]);
    // printf("full nnz = %llu\n", csc_ptr[n]);

    printf("[PanguLU LOG] pangulu_convert_halfsymcsc_to_csc_struct time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

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
){
    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_exblock_idx block_length = PANGULU_ICEIL(n , nb);

    for(pangulu_exblock_idx sp = 0; sp < block_length; sp++)
    {
        pangulu_exblock_ptr ssi = struct_bcsc_struct_pointer[sp];
        for(pangulu_exblock_ptr vsi = value_bcsc_struct_pointer[sp]; vsi < value_bcsc_struct_pointer[sp + 1]; vsi++)
        {
            while((struct_bcsc_struct_index[ssi] != value_bcsc_struct_index[vsi]) && (ssi < struct_bcsc_struct_pointer[sp+1]))
            {
                ssi++;
            }
            if(ssi >= struct_bcsc_struct_pointer[sp+1]){
                break;
            }
            for(pangulu_exblock_idx ip = 0; ip < nb; ip++)
            {
                pangulu_inblock_ptr sii = struct_bcsc_inblock_pointers[ssi][ip];
                for(pangulu_exblock_ptr vii = value_bcsc_inblock_pointers[vsi][ip]; vii < value_bcsc_inblock_pointers[vsi][ip + 1]; vii++)
                {
                    while((struct_bcsc_inblock_indeces[ssi][sii] != value_bcsc_inblock_indeces[vsi][vii]) && (sii < struct_bcsc_inblock_pointers[ssi][ip+1]))
                    {
                        sii++;
                    }
                    if(sii >= struct_bcsc_inblock_pointers[ssi][ip+1]){
                        break;
                    }
                    struct_bcsc_values[ssi][sii] = value_bcsc_values[vsi][vii];
                }
            }
        }
    }
    
    if(free_valuebcsc)
    {
        if(value_bcsc_struct_pointer[block_length] > 0){
            pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_pointers[0]);            
        }
        pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_pointers);
        pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_indeces);
        pangulu_free(__FILE__, __LINE__, value_bcsc_values);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_pointer);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_index);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_nnzptr);
    }

    // printf("[PanguLU LOG] pangulu_convert_bcsc_fill_value_to_struct time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

void pangulu_convert_bcsc_to_digestcoo(
    pangulu_exblock_idx block_length,
    const pangulu_exblock_ptr* bcsc_struct_pointer,
    const pangulu_exblock_idx* bcsc_struct_index,
    const pangulu_exblock_ptr* bcsc_struct_nnzptr,
    pangulu_digest_coo_t* digest_info
){
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = bcsc_struct_pointer[bcol]; bidx < bcsc_struct_pointer[bcol + 1]; bidx++){
            pangulu_exblock_idx brow = bcsc_struct_index[bidx];
            pangulu_exblock_ptr bnnz = bcsc_struct_nnzptr[bidx + 1] - bcsc_struct_nnzptr[bidx];
            digest_info[bidx].row = brow;
            digest_info[bidx].col = bcol;
            digest_info[bidx].nnz = bnnz;
        }
    }
}
