#include "pangulu_common.h"

void pangulu_a_plus_at(
    const pangulu_exblock_idx n,    /* number of columns in reorder_matrix A. */
    const pangulu_exblock_ptr nz,   /* number of nonzeros in reorder_matrix A */
    pangulu_exblock_ptr *colptr,    /* column pointer of size n+1 for reorder_matrix A. */
    pangulu_exblock_idx *rowind,    /* row indices of size nz for reorder_matrix A. */
    pangulu_exblock_ptr *bnz,       /* out - on exit, returns the actual number of nonzeros in reorder_matrix A'+A. */
    pangulu_exblock_ptr **b_colptr, /* out - size n+1 */
    pangulu_exblock_idx **b_rowind  /* out - size *bnz */
)
{

    register pangulu_exblock_idx i, j, k, col;
    register pangulu_exblock_ptr num_nz;
    pangulu_exblock_ptr *t_colptr;
    pangulu_exblock_idx *t_rowind; /* a column oriented form of T = A' */
    pangulu_int32_t *marker;

    marker = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, n * sizeof(pangulu_int32_t));
    t_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, (n + 1) * sizeof(pangulu_exblock_ptr));
    t_rowind = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, nz * sizeof(pangulu_exblock_idx));

    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i)
        marker[i] = 0;
    for (j = 0; j < n; ++j)
    {
        for (i = colptr[j]; i < colptr[j + 1]; ++i)
            ++marker[rowind[i]];
    }

    t_colptr[0] = 0;
    for (i = 0; i < n; ++i)
    {
        t_colptr[i + 1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose the reorder_matrix from A to T */
    for (j = 0; j < n; ++j)
    {
        for (i = colptr[j]; i < colptr[j + 1]; ++i)
        {
            col = rowind[i];
            t_rowind[marker[col]] = j;
            ++marker[col];
        }
    }

    /* ----------------------------------------------------------------
       compute B = A + T, where column j of B is:

       Struct (B_*j) = Struct (A_*k) UNION Struct (T_*k)

       do not include the diagonal entry
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i)
        marker[i] = -1;

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    for (j = 0; j < n; ++j)
    {
        /* Flag the diagonal so it's not included in the B reorder_matrix */
        // marker[j] = j;

        /* Add pattern of column A_*k to B_*j */
        for (i = colptr[j]; i < colptr[j + 1]; ++i)
        {
            k = rowind[i];
            if (marker[k] != j)
            {
                marker[k] = j;
                ++num_nz;
            }
        }

        /* Add pattern of column T_*k to B_*j */
        for (i = t_colptr[j]; i < t_colptr[j + 1]; ++i)
        {
            k = t_rowind[i];
            if (marker[k] != j)
            {
                marker[k] = j;
                ++num_nz;
            }
        }
    }
    *bnz = num_nz;

    /* Allocate storage for A+A' */
    *b_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, (n + 1) * sizeof(pangulu_exblock_ptr));
    *b_rowind = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, *bnz * sizeof(pangulu_exblock_idx));

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i)
        marker[i] = -1;

    /* Compute each column of B, one at a time */
    num_nz = 0;
    for (j = 0; j < n; ++j)
    {
        (*b_colptr)[j] = num_nz;

        /* Flag the diagonal so it's not included in the B reorder_matrix */
        // marker[j] = j;

        /* Add pattern of column A_*k to B_*j */
        for (i = colptr[j]; i < colptr[j + 1]; ++i)
        {
            k = rowind[i];
            if (marker[k] != j)
            {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }

        /* Add pattern of column T_*k to B_*j */
        for (i = t_colptr[j]; i < t_colptr[j + 1]; ++i)
        {
            k = t_rowind[i];
            if (marker[k] != j)
            {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }
    }
    (*b_colptr)[n] = num_nz;

    pangulu_free(__FILE__, __LINE__, marker);
    pangulu_free(__FILE__, __LINE__, t_colptr);
    pangulu_free(__FILE__, __LINE__, t_rowind);
} /* at_plus_a_dist */

void add_prune(pangulu_symbolic_node_t *prune, pangulu_symbolic_node_t *prune_next, pangulu_int64_t num, pangulu_int64_t num_value, pangulu_int64_t p)
{
    prune[num].value++;
    prune_next[p].value = num_value;
    prune_next[p].next = NULL;
    pangulu_symbolic_node_t *p2 = &prune[num];
    for (;;)
    {
        if (p2->next == NULL)
        {
            break;
        }
        p2 = p2->next;
    }
    p2->next = &prune_next[p];
}

void pangulu_symbolic_symmetric(
    pangulu_exblock_idx n, pangulu_exblock_ptr nnz, pangulu_exblock_idx *ai, pangulu_exblock_ptr *ap,
    pangulu_exblock_ptr **symbolic_rowpointer, pangulu_exblock_idx **symbolic_columnindex,
    pangulu_inblock_idx nb, pangulu_exblock_idx block_length,
    pangulu_exblock_ptr *symbolic_nnz)
{
    pangulu_exblock_ptr realloc_capacity = nnz;
    pangulu_exblock_idx *L_r_idx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, realloc_capacity * sizeof(pangulu_exblock_idx)); // include diagonal
    pangulu_exblock_ptr *L_c_ptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, (n + 1) * sizeof(pangulu_exblock_ptr));
    L_c_ptr[0] = 0;

    pangulu_symbolic_node_t *prune = (pangulu_symbolic_node_t *)pangulu_malloc(__FILE__, __LINE__, n * sizeof(pangulu_symbolic_node_t));
    pangulu_symbolic_node_t *prune_next = (pangulu_symbolic_node_t *)pangulu_malloc(__FILE__, __LINE__, n * sizeof(pangulu_symbolic_node_t));
    pangulu_symbolic_node_t *p1;

    pangulu_int64_t *work_space = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, n * sizeof(pangulu_int64_t));
    pangulu_int64_t *merge = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, n * sizeof(pangulu_int64_t));

    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        work_space[i] = -1;
        prune[i].value = 0;
        prune[i].next = NULL;
        prune_next[i].value = -1;
        prune_next[i].next = NULL;
    }
    pangulu_int64_t L_maxsize = realloc_capacity;
    pangulu_int64_t L_size = 0;

    pangulu_int64_t row = -1;
    pangulu_int64_t num_merge = 0;

    pangulu_int64_t p = 0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {

        pangulu_exblock_idx n_rows = ap[i + 1] - ap[i];

        for (pangulu_exblock_idx k = 0; k < n_rows; k++)
        {

            row = (ai + ap[i])[k];
            if (row >= i)
            {
                work_space[row] = i;
                L_r_idx[L_size] = row;
                L_size++;
                if (L_size + 1 > L_maxsize)
                {
                    L_r_idx = (pangulu_exblock_idx *)pangulu_realloc(__FILE__, __LINE__, L_r_idx, (L_maxsize + realloc_capacity) * sizeof(pangulu_exblock_idx));
                    L_maxsize = L_maxsize + realloc_capacity;
                }
            }
        }

        num_merge = prune[i].value;
        p1 = &prune[i];
        for (pangulu_int64_t k = 0;; k++)
        {
            if (p1->next == NULL)
                break;
            p1 = p1->next;
            merge[k] = p1->value;
        }
        for (pangulu_int64_t k = 0; k < num_merge; k++)
        {
            row = merge[k];
            pangulu_int64_t min = L_c_ptr[row];
            pangulu_int64_t max = L_c_ptr[row + 1];
            for (pangulu_int64_t j = min; j < max; j++)
            {
                pangulu_int64_t crow = L_r_idx[j];

                if (crow > i && work_space[crow] != i)
                {
                    work_space[crow] = i;
                    L_r_idx[L_size] = crow;
                    L_size++;
                    if (L_size + 1 > L_maxsize)
                    {
                        L_r_idx = (pangulu_exblock_idx *)pangulu_realloc(__FILE__, __LINE__, L_r_idx, (L_maxsize + realloc_capacity) * sizeof(pangulu_exblock_idx));
                        L_maxsize = L_maxsize + realloc_capacity;
                    }
                }
            }
        }
        L_c_ptr[i + 1] = L_size;

        if (L_c_ptr[i + 1] - L_c_ptr[i] > 1)
        {
            pangulu_int64_t todo_prune = n + 1;
            for (pangulu_int64_t k = L_c_ptr[i]; k < L_c_ptr[i + 1]; k++)
            {
                if (todo_prune > L_r_idx[k] && L_r_idx[k] > i)
                    todo_prune = L_r_idx[k];
            }
            add_prune(prune, prune_next, todo_prune, i, p);
            p++;
        }
    }
    pangulu_free(__FILE__, __LINE__, work_space);
    pangulu_free(__FILE__, __LINE__, merge);
    pangulu_free(__FILE__, __LINE__, prune);
    pangulu_free(__FILE__, __LINE__, prune_next);

    *symbolic_nnz = L_size * 2 - n;
    *symbolic_rowpointer = L_c_ptr;
    *symbolic_columnindex = L_r_idx;

    printf(PANGULU_I_SYMBOLIC_NONZERO);
}

void pangulu_symbolic(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix,
                      pangulu_origin_smatrix *reorder_matrix)
{
    pangulu_exblock_ptr *symmetric_rowpointer = NULL;
    pangulu_exblock_idx *symmetric_columnindex = NULL;
    pangulu_exblock_ptr symmetric_nnz;
    pangulu_a_plus_at(reorder_matrix->row, reorder_matrix->nnz,
                      reorder_matrix->rowpointer, reorder_matrix->columnindex,
                      &symmetric_nnz, &symmetric_rowpointer, &symmetric_columnindex);
    pangulu_exblock_ptr *symbolic_rowpointer = NULL;
    pangulu_exblock_idx *symbolic_columnindex = NULL;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_exblock_idx block_length = block_common->block_length;

    pangulu_symbolic_symmetric(reorder_matrix->row, symmetric_nnz, symmetric_columnindex, symmetric_rowpointer,
                               &symbolic_rowpointer, &symbolic_columnindex,
                               nb, block_length,
                               &block_smatrix->symbolic_nnz);

    pangulu_free(__FILE__, __LINE__, symmetric_rowpointer);
    pangulu_free(__FILE__, __LINE__, symmetric_columnindex);

    block_smatrix->symbolic_rowpointer = symbolic_rowpointer;
    block_smatrix->symbolic_columnindex = symbolic_columnindex;
}
