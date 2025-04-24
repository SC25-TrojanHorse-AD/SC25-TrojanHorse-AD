#include "pangulu_common.h"

pangulu_int64_t pangulu_task_queue_alloc(
    pangulu_task_queue_t *tq
){
    if(tq->task_storage_avail_queue_head == tq->task_storage_avail_queue_tail){
        printf("[PanguLU ERROR] pangulu_task_queue_alloc empty\n");
        exit(1);
    }
    pangulu_int64_t store_idx = tq->task_storage_avail_queue_head;
    tq->task_storage_avail_queue_head = (tq->task_storage_avail_queue_head + 1) % (tq->capacity + 1);
    return tq->task_storage_avail_queue[store_idx];
}

void pangulu_task_queue_recycle(
    pangulu_task_queue_t* tq,
    pangulu_int64_t store_idx
){
    if(tq->task_storage_avail_queue_head == (tq->task_storage_avail_queue_tail + 1) % (tq->capacity + 1)){
        printf("[PanguLU ERROR] pangulu_task_queue_recycle full\n");
        exit(1);
    }
    tq->task_storage_avail_queue[tq->task_storage_avail_queue_tail] = store_idx;
    tq->task_storage_avail_queue_tail = (tq->task_storage_avail_queue_tail + 1) % (tq->capacity + 1);
}

void pangulu_task_queue_cmp_strategy(
    pangulu_task_queue_t* tq,
    pangulu_int32_t cmp_strategy
){
    tq->cmp_strategy = cmp_strategy;
}

void pangulu_task_queue_init(pangulu_task_queue_t *heap, pangulu_int64_t capacity)
{
    pangulu_int64_t size = 0;

    pangulu_task_t *compare_queue = (pangulu_task_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_t) * capacity);
    size += sizeof(pangulu_task_t) * capacity;
    pangulu_int64_t *task_index_heap = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * capacity);
    size += sizeof(pangulu_int64_t) * capacity;
    heap->task_storage = compare_queue;
    heap->task_index_heap = task_index_heap;
    heap->capacity = capacity;
    heap->length = 0;
    // heap->nnz_flag = 0;
    heap->heap_bsem = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_bsem_t));
    heap->heap_bsem->mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(heap->heap_bsem->mutex, NULL);
    heap->task_storage_avail_queue = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (capacity + 1));
    size += sizeof(pangulu_int64_t) * (capacity + 1);
    heap->task_storage_avail_queue_head = 0;
    heap->task_storage_avail_queue_tail = capacity;
    for(pangulu_int64_t i = 0; i < capacity; i++){
        heap->task_storage_avail_queue[i] = i;
    }
    pangulu_task_queue_cmp_strategy(heap, 0);
    // printf("[PanguLU LOG] pangulu_task_queue_init size = %.1lf MB\n", ((double)size) / 1e6);
}

pangulu_task_queue_t *pangulu_task_queue_destory(pangulu_task_queue_t *heap)
{
    if (heap != NULL)
    {
        pangulu_free(__FILE__, __LINE__, heap->task_storage);
        pangulu_free(__FILE__, __LINE__, heap->task_index_heap);
        heap->length = 0;
        // heap->nnz_flag = 0;
        heap->capacity = 0;
    }
    pangulu_free(__FILE__, __LINE__, heap);
    return NULL;
}

void pangulu_task_queue_clear(pangulu_task_queue_t *heap)
{
    heap->length = 0;
    heap->task_storage_avail_queue_head = 0;
    heap->task_storage_avail_queue_tail = heap->capacity;
    for(pangulu_int64_t i = 0; i < heap->capacity; i++){
        heap->task_storage_avail_queue[i] = i;
    }
}

char pangulu_task_compare(
    pangulu_task_t *compare_queue, 
    pangulu_int64_t a, 
    pangulu_int64_t b, 
    pangulu_int32_t heap_select
){
    if (0 == heap_select)
    {
        if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
        {
            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            return compare_flag_a < compare_flag_b;
        }
        else
        {
            return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
        }
    }
    else if (1 == heap_select)
    {
        if (compare_queue[a].kernel_id == compare_queue[b].kernel_id)
        {

            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            if (compare_flag_a == compare_flag_b)
            {
                return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
            }
            else
            {
                return compare_flag_a < compare_flag_b;
            }
        }
        else
        {
            return compare_queue[a].kernel_id < compare_queue[b].kernel_id;
        }
    }
    else if (2 == heap_select)
    {
        if (compare_queue[a].kernel_id == compare_queue[b].kernel_id)
        {
            if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
            {
                pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
                pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;
                return compare_flag_a < compare_flag_b;
            }
            else
            {
                return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
            }
        }
        else
        {
            return compare_queue[a].kernel_id < compare_queue[b].kernel_id;
        }
    }
    else if (3 == heap_select)
    {
        pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
        pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;
        return compare_flag_a < compare_flag_b;
    }
    else if (4 == heap_select)
    {
        if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
        {
            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            return compare_flag_a > compare_flag_b;
        }
        else
        {
            return compare_queue[a].compare_flag > compare_queue[b].compare_flag;
        }
    }
    else
    {
        printf(PANGULU_E_INVALID_HEAP_SELECT);
        exit(1);
    }
}

void pangulu_task_swap(pangulu_int64_t *task_index_heap, pangulu_int64_t a, pangulu_int64_t b)
{
    pangulu_int64_t temp = task_index_heap[a];
    task_index_heap[a] = task_index_heap[b];
    task_index_heap[b] = temp;
}

// void pangulu_task_queue_push(
//     pangulu_task_queue_t *heap, 
//     pangulu_int64_t row, 
//     pangulu_int64_t col, 
//     pangulu_int64_t task_level, 
//     pangulu_int64_t kernel_id, 
//     pangulu_int64_t compare_flag,
//     pangulu_storage_slot_t* opdst,
//     pangulu_storage_slot_t* op1,
//     pangulu_storage_slot_t* op2,
//     pangulu_int64_t block_length,
//     const char* file,
//     int line
// ){
//     pthread_mutex_lock(heap->heap_bsem->mutex);
//     pangulu_task_t *task_storage = heap->task_storage;
//     pangulu_int64_t *task_index_heap = heap->task_index_heap;

//     // printf("[PanguLU LOG] Heap usage : %.1lf%%\n", 100.0 * heap->length / heap->capacity);
//     // int rank = -1;
//     // pangulu_cm_rank(&rank);
//     // if(kernel_id == PANGULU_TASK_SPTRSV_L)printf("> #%d (%d, %d) kernel=%d level=%d %s:%d heap_queue : %d %d\n", rank, row, col, kernel_id, task_level, file, line, heap->task_storage_avail_queue_head, heap->task_storage_avail_queue_tail);


//     if (heap->length >= heap->capacity)
//     {
//         printf(PANGULU_E_HEAP_FULL);
//         exit(1);
//     }
//     pangulu_int64_t store_idx = pangulu_task_queue_alloc(heap);
//     task_storage[store_idx].row = row;
//     task_storage[store_idx].col = col;
//     task_storage[store_idx].task_level = task_level;
//     task_storage[store_idx].kernel_id = kernel_id;
//     task_storage[store_idx].compare_flag = compare_flag;
//     task_storage[store_idx].opdst = opdst;
//     task_storage[store_idx].op1 = op1;
//     task_storage[store_idx].op2 = op2;
//     task_index_heap[heap->length] = store_idx;
//     pangulu_int64_t son = heap->length;
//     pangulu_int64_t parent = (son - 1) / 2;
//     while (son != 0 && parent >= 0)
//     {
//         if (pangulu_task_compare(task_storage, task_index_heap[son], task_index_heap[parent], heap->cmp_strategy))
//         {
//             pangulu_task_swap(task_index_heap, son, parent);
//         }
//         else
//         {
//             break;
//         }
//         son = parent;
//         parent = (son - 1) / 2;
//     }
//     heap->length++;
//     pthread_mutex_unlock(heap->heap_bsem->mutex);

//     // pangulu_task_queue_display(heap);
// }

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
){
    pthread_mutex_lock(heap->heap_bsem->mutex);
    if(kernel_id == PANGULU_TASK_SSSSM){
        pangulu_task_t store_task;
        store_task.row = row;
        store_task.col = col;
        store_task.task_level = task_level;
        store_task.kernel_id = kernel_id;
        store_task.compare_flag = compare_flag;
        store_task.opdst = opdst;
        store_task.op1 = op1;
        store_task.op2 = op2;
        sc25_task_store(NULL, row * block_length + col, &store_task);
    }
    pangulu_task_t *task_storage = heap->task_storage;
    pangulu_int64_t *task_index_heap = heap->task_index_heap;
    if (heap->length >= heap->capacity)
    {
        printf(PANGULU_E_HEAP_FULL);
        exit(1);
    }
    pangulu_int64_t store_idx = pangulu_task_queue_alloc(heap);
    task_storage[store_idx].row = row;
    task_storage[store_idx].col = col;
    task_storage[store_idx].task_level = task_level;
    task_storage[store_idx].kernel_id = kernel_id;
    task_storage[store_idx].compare_flag = compare_flag;
    task_storage[store_idx].opdst = opdst;
    task_storage[store_idx].op1 = op1;
    task_storage[store_idx].op2 = op2;
    task_index_heap[heap->length] = store_idx;
    pangulu_int64_t son = heap->length;
    pangulu_int64_t parent = (son - 1) / 2;
    while (son != 0 && parent >= 0)
    {
        if (pangulu_task_compare(task_storage, task_index_heap[son], task_index_heap[parent], heap->cmp_strategy))
        {
            pangulu_task_swap(task_index_heap, son, parent);
        }
        else
        {
            break;
        }
        son = parent;
        parent = (son - 1) / 2;
    }
    heap->length++;
    pthread_mutex_unlock(heap->heap_bsem->mutex);
}

char pangulu_task_queue_empty(pangulu_task_queue_t *heap)
{
    return !(heap->length);
}

pangulu_task_t pangulu_task_queue_delete(pangulu_task_queue_t *heap)
{
    pthread_mutex_lock(heap->heap_bsem->mutex);
    if (pangulu_task_queue_empty(heap))
    {
        printf(PANGULU_E_HEAP_EMPTY);
        exit(1);
    }
    pangulu_int64_t *task_index_heap = heap->task_index_heap;
    pangulu_task_swap(task_index_heap, heap->length - 1, 0);
    // pangulu_task_queue_adjust(heap, 0, length - 1);

    pangulu_task_t *task_storage = heap->task_storage;
    pangulu_int64_t top = 0;
    pangulu_int64_t left = top * 2 + 1;
    while (left < (heap->length - 1))
    {
        if ((left + 1) < (heap->length - 1) && pangulu_task_compare(task_storage, task_index_heap[left + 1], task_index_heap[left], heap->cmp_strategy))
        {
            left = left + 1;
        }
        if (pangulu_task_compare(task_storage, task_index_heap[left], task_index_heap[top], heap->cmp_strategy))
        {
            pangulu_task_swap(task_index_heap, left, top);
            top = left;
            left = 2 * top + 1;
        }
        else
        {
            break;
        }
    }

    heap->length--;
    pangulu_task_t ret = heap->task_storage[task_index_heap[heap->length]];
    pangulu_task_queue_recycle(heap, task_index_heap[heap->length]);
    pthread_mutex_unlock(heap->heap_bsem->mutex);
    return ret;
}


pangulu_task_t pangulu_task_queue_pop(pangulu_task_queue_t *heap)
{
    // pangulu_bsem_t *heap_bsem = heap->heap_bsem;
    // pthread_mutex_t *heap_mutex = heap_bsem->mutex;

    // pthread_mutex_lock(heap_mutex);
    // if (pangulu_task_queue_empty(heap) == 1)
    // {
        // heap_bsem->v = 0;
        // while (heap_bsem->v == 0)
        while(pangulu_task_queue_empty(heap))
        {
            // wait
            // pthread_cond_wait(heap_bsem->cond, heap_bsem->mutex);
            usleep(1);
        }
    // }
    
    // pangulu_task_queue_display(heap);
    pangulu_task_t task = pangulu_task_queue_delete(heap);
    // pangulu_task_queue_display(heap);
    // heap_bsem->v = 1;
    // pthread_mutex_unlock(heap_mutex);
    return task;
}

void pangulu_task_queue_display(pangulu_task_queue_t *heap)
{
    printf(PANGULU_I_HEAP_LEN);
    // for (pangulu_int64_t i = 0; i < heap->length; i++)
    // {
    //     printf(FMT_PANGULU_INT64_T " ", heap->task_index_heap[i]);
    // }
    // printf("\n");
    for (pangulu_int64_t i = 0; i < heap->length; i++)
    {
        pangulu_int64_t now = heap->task_index_heap[i];
        printf("[PanguLU Debug] (" FMT_PANGULU_EXBLOCK_IDX 
            ", " FMT_PANGULU_EXBLOCK_IDX 
            ") level = " FMT_PANGULU_EXBLOCK_IDX
            " compare_flag = " FMT_PANGULU_INT64_T
            " kernel_id = " FMT_PANGULU_INT16_T "\n",
            heap->task_storage[now].row,
            heap->task_storage[now].col,
            heap->task_storage[now].task_level,
            heap->task_storage[now].compare_flag,
            heap->task_storage[now].kernel_id);
    }
}