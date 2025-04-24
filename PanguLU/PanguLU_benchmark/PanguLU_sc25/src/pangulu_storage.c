#include "pangulu_common.h"


// int alloc_cnt = 0;

pangulu_int32_t
pangulu_storage_bin_navail(
    pangulu_storage_bin_t* bin
){
    // printf("pangulu_storage_bin_navail %d %d %d %d\n", bin->queue_head, bin->queue_tail, bin->slot_count, (bin->queue_tail + (bin->slot_count + 1) - bin->queue_head) % (bin->slot_count + 1));
    return (bin->queue_tail + (bin->slot_count + 1) - bin->queue_head) % (bin->slot_count + 1);
}

pangulu_int32_t
pangulu_storage_slot_queue_alloc(
    pangulu_storage_bin_t* bin
){
    // alloc_cnt++;
    // printf("alloc_cnt = %d\n", alloc_cnt);
    // printf("Storage slot util : %.1lf%%\n", 100.0 * ((bin->queue_tail + (bin->slot_count + 1) - bin->queue_head) % (bin->slot_count + 1)) / (bin->slot_count + 1));
    if(bin->queue_head == bin->queue_tail){
        printf("[PanguLU ERROR] pangulu_storage_slot_queue_alloc : bin.slot_capacity=%d No empty slot to allocate.\n", bin->slot_capacity);
        exit(1);
    }
    pangulu_int32_t slot_idx = bin->avail_slot_queue[bin->queue_head];
    bin->queue_head = (bin->queue_head + 1) % (bin->slot_count + 1);
    return slot_idx;
}

// void pangulu_storage_slot_queue_recycle(
//     pangulu_storage_t* storage,
//     pangulu_uint64_t* slot_addr
// ){ 
//     pthread_mutex_lock(storage->mutex);
//     pangulu_storage_bin_t* bin = &storage->bins[PANGULU_DIGINFO_GET_BINID(*slot_addr)];
//     #ifdef PANGULU_NONSHAREDMEM
//     pangulu_storage_slot_t* slot = &bin->slots[PANGULU_DIGINFO_GET_STOREIDX(*slot_addr)];
//     if((slot->brow_pos) == (slot->bcol_pos)){
//         pangulu_platform_free(slot->d_rowpointer, PANGULU_DEFAULT_PLATFORM);
//         pangulu_platform_free(slot->d_columnindex, PANGULU_DEFAULT_PLATFORM);
//         pangulu_platform_free(slot->d_idx_of_csc_value_for_csr, PANGULU_DEFAULT_PLATFORM);
//         slot->d_rowpointer = NULL;
//         slot->d_columnindex = NULL;
//         slot->d_idx_of_csc_value_for_csr = NULL;
//         slot->have_csr_data = 0;
//     }
//     #endif
//     bin->slots[PANGULU_DIGINFO_GET_STOREIDX(*slot_addr)].data_status = PANGULU_DATA_INVALID;
//     if(bin->queue_head == ((bin->queue_tail + 1) % (bin->slot_count + 1))){
//         printf("[PanguLU ERROR] pangulu_storage_slot_queue_recycle full.\n");
//         exit(1);
//     }
//     // printf("RECYCLE 0x%016llX\n", *slot_addr);
//     bin->avail_slot_queue[bin->queue_tail] = PANGULU_DIGINFO_GET_STOREIDX(*slot_addr);
//     bin->queue_tail = (bin->queue_tail + 1) % (bin->slot_count + 1);
//     *slot_addr = PANGULU_DIGINFO_SET_NNZ(PANGULU_DIGINFO_GET_NNZ(*slot_addr));
//     pthread_mutex_unlock(storage->mutex);
// }

// Yida : Change avail_slot_queue from queue to stack, reducing memory footprint.
void pangulu_storage_slot_queue_recycle(
    pangulu_storage_t* storage,
    pangulu_uint64_t* slot_addr
){ 
    pthread_mutex_lock(storage->mutex);
    pangulu_storage_bin_t* bin = &storage->bins[PANGULU_DIGINFO_GET_BINID(*slot_addr)];
    #ifdef PANGULU_NONSHAREDMEM
    pangulu_storage_slot_t* slot = &bin->slots[PANGULU_DIGINFO_GET_STOREIDX(*slot_addr)];
    if((slot->brow_pos) == (slot->bcol_pos)){
        pangulu_platform_free(slot->d_rowpointer, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_free(slot->d_columnindex, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_free(slot->d_idx_of_csc_value_for_csr, PANGULU_DEFAULT_PLATFORM);
        slot->d_rowpointer = NULL;
        slot->d_columnindex = NULL;
        slot->d_idx_of_csc_value_for_csr = NULL;
        slot->have_csr_data = 0;
    }
    #endif
    bin->slots[PANGULU_DIGINFO_GET_STOREIDX(*slot_addr)].data_status = PANGULU_DATA_INVALID;
    if(bin->queue_head == ((bin->queue_tail + 1) % (bin->slot_count + 1))){
        printf("[PanguLU ERROR] pangulu_storage_slot_queue_recycle full.\n");
        exit(1);
    }
    bin->queue_head = (bin->queue_head - 1 + (bin->slot_count + 1)) % (bin->slot_count + 1);
    bin->avail_slot_queue[bin->queue_head] = PANGULU_DIGINFO_GET_STOREIDX(*slot_addr);
    *slot_addr = PANGULU_DIGINFO_SET_NNZ(PANGULU_DIGINFO_GET_NNZ(*slot_addr));
    pthread_mutex_unlock(storage->mutex);
}

pangulu_uint64_t
pangulu_storage_allocate_slot(
    pangulu_storage_t* storage,
    pangulu_int64_t size
){
    pthread_mutex_lock(storage->mutex);
    pangulu_uint64_t slot_addr = 0xFFFFFFFFFFFFFFFF;
    for(pangulu_int32_t bin_id = 1; bin_id < storage->n_bin; bin_id++){
        if(storage->bins[bin_id].slot_capacity >= size){
            if(pangulu_storage_bin_navail(&storage->bins[bin_id]) == 0){
                // printf("[PanguLU WARNING] Bin %d is full. Trying bin %d.\n", bin_id, bin_id+1);
                continue;
            }
            pangulu_int32_t slot_idx = pangulu_storage_slot_queue_alloc(&storage->bins[bin_id]);
            storage->bins[bin_id].slots[slot_idx].data_status = PANGULU_DATA_PREPARING;
            slot_addr = 0;
            slot_addr |= PANGULU_DIGINFO_SET_STOREIDX(slot_idx);
            slot_addr |= PANGULU_DIGINFO_SET_BINID(bin_id);
            break;
        }
    }
    if(slot_addr == 0xFFFFFFFFFFFFFFFF){
        printf("[PanguLU ERROR] PanguLU storage is full.\n");
        exit(1);
    }
    pthread_mutex_unlock(storage->mutex);
    return slot_addr;
}

pangulu_storage_slot_t*
pangulu_storage_get_slot(
    pangulu_storage_t* storage,
    pangulu_uint64_t slot_addr
){
    if(PANGULU_DIGINFO_GET_BINID(slot_addr) == 7){
        return NULL;
    }
    return &(storage->bins[PANGULU_DIGINFO_GET_BINID(slot_addr)].slots[PANGULU_DIGINFO_GET_STOREIDX(slot_addr)]);
}

void pangulu_storage_bin_init(
    pangulu_storage_bin_t* bin,
    pangulu_int32_t bin_id,
    pangulu_int64_t slot_capacity,
    pangulu_int32_t slot_count
){
    bin->slot_count = slot_count;
    bin->slot_capacity = slot_capacity;
    bin->slots = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_slot_t) * slot_count);
    bin->avail_slot_queue = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (slot_count + 1));
    bin->queue_head = 0;
    bin->queue_tail = slot_count;
    memset(bin->slots, 0, sizeof(pangulu_storage_slot_t) * slot_count);
    // printf("bin_id=%d, capacity=%lld\n", bin_id, slot_capacity);
    for(pangulu_int32_t i = 0; i < slot_count; i++){
        bin->slots[i].columnpointer = pangulu_malloc(__FILE__, __LINE__, slot_capacity);
        #ifdef PANGULU_NONSHAREDMEM
        pangulu_platform_malloc(&(bin->slots[i].d_value), slot_capacity, PANGULU_DEFAULT_PLATFORM);
        #endif
        bin->avail_slot_queue[i] = i;
    }
}

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
){
    pangulu_int64_t storage_size = sizeof(pangulu_storage_bin_t) * storage->n_bin;
    storage->n_bin = 7;
    storage->bins = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_bin_t) * storage->n_bin);
    for(pangulu_int32_t i_bin = 1; i_bin < storage->n_bin; i_bin++){
        pangulu_storage_bin_init(&storage->bins[i_bin], i_bin, slot_capacity[i_bin], slot_count[i_bin]);
        storage_size += sizeof(pangulu_storage_slot_t) * slot_count[i_bin];
        storage_size += sizeof(pangulu_int32_t) * (slot_count[i_bin] + 1);
        storage_size += slot_capacity[i_bin] * slot_count[i_bin];
    }
    // printf("[PanguLU LOG] pangulu_storage_init memory per rank : %.1lf MB\n", ((double)storage_size) / 1e6);
    
    // Init bin 0.
    pangulu_storage_bin_t* bin0 = &storage->bins[0];
    bin0->slot_count = bcsc_pointer[block_length];
    bin0->slot_capacity = 0;
    bin0->slots = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_slot_t) * bcsc_pointer[block_length]);
    bin0->avail_slot_queue = NULL;
    bin0->queue_head = 0;
    bin0->queue_tail = 0;
    #ifdef PANGULU_NONSHAREDMEM
    pangulu_inblock_ptr* temp_rowptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
    pangulu_inblock_ptr* aid_inptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1));
    pangulu_inblock_idx* temp_colidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nb * nb);
    pangulu_inblock_ptr* temp_valueidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * nb * nb);
    
    pangulu_inblock_ptr* csccolptrl = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb+1));
    pangulu_inblock_idx* cscrowidxl = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nb * nb);
    pangulu_inblock_ptr* csccolptru = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb+1));
    pangulu_inblock_idx* cscrowidxu = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nb * nb);
    pangulu_int32_t* nnzu = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * nb);
    #endif
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = bcsc_pointer[bcol]; bidx < bcsc_pointer[bcol+1]; bidx++){
            bin0->slots[bidx].columnpointer = bcsc_inblk_pointers[bidx];
            bin0->slots[bidx].rowindex = bcsc_inblk_indeces[bidx];
            bin0->slots[bidx].value = bcsc_inblk_values[bidx];
            bin0->slots[bidx].data_status = PANGULU_DATA_PREPARING;
            bin0->slots[bidx].brow_pos = bcsc_index[bidx];
            bin0->slots[bidx].bcol_pos = bcol;
            #ifdef PANGULU_NONSHAREDMEM
            pangulu_platform_malloc(&(bin0->slots[bidx].d_columnpointer), sizeof(pangulu_inblock_ptr) * (nb+1), PANGULU_DEFAULT_PLATFORM);
            pangulu_platform_malloc(&(bin0->slots[bidx].d_rowindex), sizeof(pangulu_inblock_idx) * bcsc_inblk_pointers[bidx][nb], PANGULU_DEFAULT_PLATFORM);
            pangulu_platform_malloc(&(bin0->slots[bidx].d_value), sizeof(calculate_type) * bcsc_inblk_pointers[bidx][nb], PANGULU_DEFAULT_PLATFORM);
            pangulu_platform_memcpy(bin0->slots[bidx].d_columnpointer, bin0->slots[bidx].columnpointer, sizeof(pangulu_inblock_ptr) * (nb+1), 0, PANGULU_DEFAULT_PLATFORM);
            pangulu_platform_memcpy(bin0->slots[bidx].d_rowindex, bin0->slots[bidx].rowindex, sizeof(pangulu_inblock_idx) * bcsc_inblk_pointers[bidx][nb], 0, PANGULU_DEFAULT_PLATFORM);
            pangulu_platform_memcpy(bin0->slots[bidx].d_value, bin0->slots[bidx].value, sizeof(calculate_type) * bcsc_inblk_pointers[bidx][nb], 0, PANGULU_DEFAULT_PLATFORM);
            if(bcol <= bcsc_index[bidx]){
                pangulu_storage_slot_t* slot = &(bin0->slots[bidx]);
                pangulu_uint64_t nnz = bcsc_inblk_pointers[bidx][nb];
                pangulu_uint64_t size = (sizeof(calculate_type) + sizeof(pangulu_inblock_idx)) * nnz + sizeof(pangulu_inblock_ptr) * (nb+1);
                pangulu_platform_malloc(&(slot->d_rowpointer), sizeof(pangulu_inblock_ptr) * (nb+1), PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_malloc(&(slot->d_columnindex), sizeof(pangulu_inblock_idx) * nnz, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_malloc(&(slot->d_idx_of_csc_value_for_csr), sizeof(pangulu_inblock_ptr) * nnz, PANGULU_DEFAULT_PLATFORM);

                pangulu_transpose_struct_with_valueidx_inblock(nb, slot->columnpointer, slot->rowindex, temp_rowptr, temp_colidx, temp_valueidx, aid_inptr);
                pangulu_platform_memcpy(slot->d_rowpointer, temp_rowptr, sizeof(pangulu_inblock_ptr) * (nb+1), 0, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_columnindex, temp_colidx, sizeof(pangulu_inblock_idx) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_idx_of_csc_value_for_csr, temp_valueidx, sizeof(pangulu_inblock_ptr) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
                slot->have_csr_data = 1;
            }

            if(bcol == bcsc_index[bidx]){
                // pangulu_inblock_ptr *d_csccolptrl_upperbound;
                // pangulu_inblock_idx *d_cscrowidxl_upperbound;
                // pangulu_inblock_ptr *d_csccolptru_upperbound;
                // pangulu_inblock_idx *d_cscrowidxu_upperbound;
                // pangulu_int32_t *d_nnzu;

                pangulu_storage_slot_t* slot = &(bin0->slots[bidx]);
                pangulu_inblock_ptr u_ptr = 0;
                pangulu_inblock_ptr l_ptr = 0;
                
                csccolptru[0] = 0;
                csccolptrl[0] = 0;
                for(pangulu_inblock_idx col = 0; col < nb; col++){
                    nnzu[col] = 0;
                    for(pangulu_inblock_ptr idx = (col==0?0:slot->columnpointer[col]); idx < slot->columnpointer[col+1]; idx++){
                        pangulu_inblock_idx row = slot->rowindex[idx];
                        if(row <= col){ // U
                            cscrowidxu[u_ptr] = row;
                            u_ptr++;
                            nnzu[col]++;
                        }
                        if(row >= col){ // L
                            cscrowidxl[l_ptr] = row;
                            l_ptr++;
                        }
                    }
                    csccolptru[col+1] = u_ptr;
                    csccolptrl[col+1] = l_ptr;
                }
                
                pangulu_platform_malloc(&(slot->d_csccolptrl_upperbound), sizeof(pangulu_inblock_ptr)*(nb+1), PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_csccolptrl_upperbound, csccolptrl, sizeof(pangulu_inblock_ptr)*(nb+1), 0, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_malloc(&(slot->d_csccolptru_upperbound), sizeof(pangulu_inblock_ptr)*(nb+1), PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_csccolptru_upperbound, csccolptru, sizeof(pangulu_inblock_ptr)*(nb+1), 0, PANGULU_DEFAULT_PLATFORM);

                pangulu_platform_malloc(&(slot->d_cscrowidxl_upperbound), sizeof(pangulu_inblock_idx)*l_ptr, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_cscrowidxl_upperbound, cscrowidxl, sizeof(pangulu_inblock_idx)*l_ptr, 0, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_malloc(&(slot->d_cscrowidxu_upperbound), sizeof(pangulu_inblock_idx)*u_ptr, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_cscrowidxu_upperbound, cscrowidxu, sizeof(pangulu_inblock_idx)*u_ptr, 0, PANGULU_DEFAULT_PLATFORM);

                pangulu_platform_malloc(&(slot->d_nnzu), sizeof(pangulu_int32_t)*nb, PANGULU_DEFAULT_PLATFORM);
                pangulu_platform_memcpy(slot->d_nnzu, nnzu, sizeof(pangulu_int32_t)*nb, 0, PANGULU_DEFAULT_PLATFORM);
            }
            #endif
        }
    }

    #ifdef PANGULU_NONSHAREDMEM
    pangulu_free(__FILE__, __LINE__, temp_rowptr);
    pangulu_free(__FILE__, __LINE__, temp_colidx);
    pangulu_free(__FILE__, __LINE__, aid_inptr);
    #endif

    storage->mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(storage->mutex, NULL);
}