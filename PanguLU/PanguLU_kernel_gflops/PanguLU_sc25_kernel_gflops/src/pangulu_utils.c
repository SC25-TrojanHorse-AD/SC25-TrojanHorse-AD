#include "pangulu_common.h"

void pangulu_init_pangulu_origin_smatrix(pangulu_origin_smatrix* s){
    memset(s, 0, sizeof(pangulu_origin_smatrix));
}
void pangulu_init_pangulu_block_smatrix(pangulu_block_smatrix* bs){
    memset(bs, 0, sizeof(pangulu_block_smatrix));
}
void pangulu_time_start(struct timeval* start){
    gettimeofday(start, NULL);
}
double pangulu_time_stop(struct timeval* start){
    struct timeval end;
    gettimeofday(&end, NULL);
    double time = ((double)end.tv_sec - start->tv_sec) + ((double)end.tv_usec - start->tv_usec) * 1e-6;
    return time;
}

void pangulu_add_diagonal_element(pangulu_origin_smatrix *s)
{
    double ZERO_ELEMENT = 1e-8;

    pangulu_int64_t diagonal_add = 0;
    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr *new_rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 5));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        char flag = 0;
        for (pangulu_exblock_ptr j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            if (s->columnindex[j] == i)
            {
                flag = 1;
                break;
            }
        }
        new_rowpointer[i] = s->rowpointer[i] + diagonal_add;
        diagonal_add += (!flag);
    }
    new_rowpointer[n] = s->rowpointer[n] + diagonal_add;

    pangulu_exblock_idx *new_columnindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * new_rowpointer[n]);
    calculate_type *new_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * new_rowpointer[n]);

    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        if ((new_rowpointer[i + 1] - new_rowpointer[i]) == (s->rowpointer[i + 1] - s->rowpointer[i]))
        {
            for (pangulu_exblock_ptr j = new_rowpointer[i], k = s->rowpointer[i]; j < new_rowpointer[i + 1]; j++, k++)
            {
                new_columnindex[j] = s->columnindex[k];
                new_value[j] = s->value[k];
            }
        }
        else
        {
            char flag = 0;
            for (pangulu_exblock_ptr j = new_rowpointer[i], k = s->rowpointer[i]; k < s->rowpointer[i + 1]; j++, k++)
            {
                if (s->columnindex[k] < i)
                {
                    new_columnindex[j] = s->columnindex[k];
                    new_value[j] = s->value[k];
                }
                else if (s->columnindex[k] > i)
                {
                    if (flag == 0)
                    {
                        new_columnindex[j] = i;
                        new_value[j] = ZERO_ELEMENT;
                        k--;
                        flag = 1;
                    }
                    else
                    {
                        new_columnindex[j] = s->columnindex[k];
                        new_value[j] = s->value[k];
                    }
                }
                else
                {
                    printf(PANGULU_E_ADD_DIA);
                    exit(1);
                }
            }
            if (flag == 0)
            {
                new_columnindex[new_rowpointer[i + 1] - 1] = i;
                new_value[new_rowpointer[i + 1] - 1] = ZERO_ELEMENT;
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, s->rowpointer);
    pangulu_free(__FILE__, __LINE__, s->columnindex);
    pangulu_free(__FILE__, __LINE__, s->value);
    s->rowpointer = new_rowpointer;
    s->columnindex = new_columnindex;
    s->value = new_value;
    s->nnz = new_rowpointer[n];
}

int cmp_exidx_asc(const void* a, const void* b){
    if((*(pangulu_exblock_idx*)a) > (*(pangulu_exblock_idx*)b)){
        return 1;
    }else if((*(pangulu_exblock_idx*)a) < (*(pangulu_exblock_idx*)b)){
        return -1;
    }else{
        return 0;
    }
}

void pangulu_kvsort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end)
{
    if(val){
        if (start < end)
        {
            pangulu_int64_t pivot;
            pangulu_int64_t i, j, k;

            // k = choose_pivot(start, end);
            k = (start + end) / 2;
            swap_index_1(&key[start], &key[k]);
            swap_value(&val[start], &val[k]);
            pivot = key[start];

            i = start + 1;
            j = end;
            while (i <= j)
            {
                while ((i <= end) && (key[i] <= pivot))
                    i++;
                while ((j >= start) && (key[j] > pivot))
                    j--;
                if (i < j)
                {
                    swap_index_1(&key[i], &key[j]);
                    swap_value(&val[i], &val[j]);
                }
            }

            // swap two elements
            swap_index_1(&key[start], &key[j]);
            swap_value(&val[start], &val[j]);

            // recursively sort the lesser key
            pangulu_kvsort(key, val, start, j - 1);
            pangulu_kvsort(key, val, j + 1, end);
        }
    }else{
        qsort(key + start, end + 1 - start , sizeof(pangulu_exblock_idx), cmp_exidx_asc);
    }
}

void pangulu_kvsort2(pangulu_exblock_idx *key, pangulu_uint64_t *val, pangulu_int64_t start, pangulu_int64_t end)
{
    if(val){
        if (start < end)
        {
            pangulu_int64_t pivot;
            pangulu_int64_t i, j, k;

            // k = choose_pivot(start, end);
            k = (start + end) / 2;
            swap_index_1(&key[start], &key[k]);
            swap_value2(&val[start], &val[k]);
            pivot = key[start];

            i = start + 1;
            j = end;
            while (i <= j)
            {
                while ((i <= end) && (key[i] <= pivot))
                    i++;
                while ((j >= start) && (key[j] > pivot))
                    j--;
                if (i < j)
                {
                    swap_index_1(&key[i], &key[j]);
                    swap_value2(&val[i], &val[j]);
                }
            }

            // swap two elements
            swap_index_1(&key[start], &key[j]);
            swap_value2(&val[start], &val[j]);

            // recursively sort the lesser key
            pangulu_kvsort2(key, val, start, j - 1);
            pangulu_kvsort2(key, val, j + 1, end);
        }
    }else{
        qsort(key + start, end + 1 - start , sizeof(pangulu_exblock_idx), cmp_exidx_asc);
    }
}

void pangulu_sort_pangulu_origin_smatrix(pangulu_origin_smatrix *s)
{
    for (pangulu_exblock_idx i = 0; i < s->row; i++)
    {
        pangulu_kvsort(s->columnindex, s->value, s->rowpointer[i], s->rowpointer[i + 1] - 1);
    }
}

void pangulu_sort_exblock_struct(
    pangulu_exblock_idx n,
    pangulu_exblock_ptr* pointer,
    pangulu_exblock_idx* index,
    pangulu_int32_t nthread
){
    if(nthread <= 0){
        nthread = sysconf(_SC_NPROCESSORS_ONLN);
    }
    
    #pragma omp parallel num_threads(nthread)
    {
        bind_to_core(omp_get_thread_num() % sysconf(_SC_NPROCESSORS_ONLN));
    }

    #pragma omp parallel for num_threads(nthread) schedule(guided)
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_kvsort(index, NULL, pointer[i], pointer[i + 1] - 1);
    }
}

void swap_index_1(pangulu_exblock_idx *a, pangulu_exblock_idx *b)
{
    pangulu_exblock_idx tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_value(calculate_type *a, calculate_type *b)
{
    calculate_type tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_value2(pangulu_uint64_t *a, pangulu_uint64_t *b)
{
    pangulu_uint64_t tmp = *a;
    *a = *b;
    *b = tmp;
}

int binarylowerbound(const pangulu_int64_t *arr, int len, pangulu_int64_t value)
{
    int left = 0;
    int right = len;
    while (left < right)
    {
        int mid = (left + right) / 2;
        // value <= arr[mid] ? (right = mid) : (left = mid + 1);
        value < arr[mid] ? (right = mid) : (left = mid + 1);
    }
    return left;
}

void exclusive_scan_1(pangulu_int64_t *input, int length)
{
    if (length == 0 || length == 1)
        return;

    pangulu_int64_t old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void exclusive_scan_3(unsigned int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    unsigned int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

pangulu_int64_t binarysearch(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target)
{
    pangulu_int64_t low = left;
    pangulu_int64_t high = right;
    pangulu_int64_t mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}

pangulu_int64_t binarysearch_first_ge(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target)
{
    pangulu_int64_t low = left;
    pangulu_int64_t high = right;
    pangulu_int64_t mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return low;
}

pangulu_int64_t binarysearch_last_le(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target)
{
    pangulu_int64_t low = left;
    pangulu_int64_t high = right;
    pangulu_int64_t mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return high;
}

pangulu_int32_t binarysearch_inblk(const pangulu_inblock_idx *arr, pangulu_int32_t left, pangulu_int32_t right, pangulu_int32_t target)
{
    pangulu_int32_t low = left;
    pangulu_int32_t high = right;
    pangulu_int32_t mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}

void mem_usage()
{
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    long long int mymem = r_usage.ru_maxrss;
    long long int total;
    MPI_Allreduce(&mymem, &total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        fprintf(stdout, "Total memory usage: %.4f MB\n", (double)((double)(total) / (double)(1024)));
        fflush(stdout);
    }
}


void pangulu_zero_pangulu_vector(pangulu_vector *v)
{
    for (int i = 0; i < v->row; i++)
    {
        v->value[i] = 0.0;
    }
}


void pangulu_init_pangulu_vector(pangulu_vector *b, pangulu_int64_t n)
{
    b->value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        b->value[i] = (calculate_type)0.0;
    }
    b->row = n;
}

pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *v)
{
    pangulu_free(__FILE__, __LINE__, v->value);
    v->value = NULL;
    pangulu_free(__FILE__, __LINE__, v);
    return NULL;
}

void pangulu_transpose_struct_with_valueidx_inblock(
    const pangulu_inblock_idx nb,
    const pangulu_inblock_ptr* in_ptr,
    const pangulu_inblock_idx* in_idx,
    pangulu_inblock_ptr* out_ptr,
    pangulu_inblock_idx* out_idx,
    pangulu_inblock_ptr* out_valueidx,
    pangulu_inblock_ptr* aid_ptr
){
    memset(out_ptr, 0, sizeof(pangulu_inblock_ptr) * (nb + 1));
    for(pangulu_inblock_idx i = 0; i<nb;i++){
        for(pangulu_inblock_ptr j = in_ptr[i]; j < in_ptr[i+1]; j++){
            out_ptr[in_idx[j] + 1]++;
        }
    }
    for(pangulu_inblock_idx i = 0; i <nb;i++){
        out_ptr[i+1] += out_ptr[i];
    }
    memcpy(aid_ptr, out_ptr, sizeof(pangulu_inblock_ptr) * (nb));
    for(pangulu_inblock_idx i = 0; i<nb;i++){
        for(pangulu_inblock_ptr j = in_ptr[i]; j < in_ptr[i+1]; j++){
            out_idx[aid_ptr[in_idx[j]]] = i;
            out_valueidx[aid_ptr[in_idx[j]]] = j;
            aid_ptr[in_idx[j]]++;
        }
    }
}