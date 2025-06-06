/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

#include "superlu_ddefs.h"
#ifdef GPU_ACC
#include "gpu_api_utils.h"
#endif

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Re-distribute A on the 2D process mesh.
 *
 * Arguments
 * =========
 *
 * A      (input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T.
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * colptr (output) int*
 *
 * rowind (output) int*
 *
 * a      (output) double*
 *
 * Return value
 * ============
 *   > 0, working storage (in bytes) required to perform redistribution.
 *        (excluding LU factor size)
 * </pre>
 */
int_t dReDistribute_A(
	SuperMatrix *A, 
	dScalePermstruct_t *ScalePermstruct,
	Glu_freeable_t *Glu_freeable, 
	int_t *xsup, 
	int_t *supno,
	gridinfo_t *grid, 
	int_t *colptr[], 
	int_t *rowind[],
	double *a[]
){
	NRformat_loc *Astore;
	int_t *perm_r; /* row permutation vector */
	int_t *perm_c; /* column permutation vector */
	int_t i, irow, fst_row, j, jcol, k, gbi, gbj, n, m_loc, jsize, nnz_tot;
	int_t nnz_loc; /* number of local nonzeros */
	int_t SendCnt; /* number of remote nonzeros to be sent */
	int_t RecvCnt; /* number of remote nonzeros to be sent */
	int_t *nnzToSend, *nnzToRecv, maxnnzToRecv;
	int_t *ia, *ja, **ia_send, *index, *itemp = NULL;
	int_t *ptr_to_send;
	double *aij, **aij_send, *nzval, *dtemp = NULL;
	double *nzval_a;
	int iam, it, p, procs;
	MPI_Request *send_req;
	MPI_Status status;

	/* ------------------------------------------------------------
	   INITIALIZATION.
	   ------------------------------------------------------------*/
	iam = grid->iam;
	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	procs = grid->nprow * grid->npcol;
	Astore = (NRformat_loc *)A->Store;
	n = A->ncol;
	m_loc = Astore->m_loc;
	fst_row = Astore->fst_row;
	nnzToRecv = intCalloc_dist(2 * procs);
	nnzToSend = nnzToRecv + procs;

	/* ------------------------------------------------------------
	   COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
	   THEN ALLOCATE SPACE.
	   THIS ACCOUNTS FOR THE FIRST PASS OF A.
	   ------------------------------------------------------------*/
	for (i = 0; i < m_loc; ++i)
	{
		for (j = Astore->rowptr[i]; j < Astore->rowptr[i + 1]; ++j)
		{
			irow = perm_c[perm_r[i + fst_row]]; /* Row number in Pc*Pr*A */
			jcol = Astore->colind[j];
			gbi = BlockNum(irow);
			gbj = BlockNum(jcol);
			p = PNUM(PROW(gbi, grid), PCOL(gbj, grid), grid);
			++nnzToSend[p];
		}
	}

	/* All-to-all communication */
	MPI_Alltoall(nnzToSend, 1, mpi_int_t, nnzToRecv, 1, mpi_int_t,
				 grid->comm);

	maxnnzToRecv = 0;
	nnz_loc = SendCnt = RecvCnt = 0;

	for (p = 0; p < procs; ++p)
	{
		if (p != iam)
		{
			SendCnt += nnzToSend[p];
			RecvCnt += nnzToRecv[p];
			maxnnzToRecv = SUPERLU_MAX(nnzToRecv[p], maxnnzToRecv);
		}
		else
		{
			nnz_loc += nnzToRecv[p];
			/*assert(nnzToSend[p] == nnzToRecv[p]);*/
		}
	}
	k = nnz_loc + RecvCnt; /* Total nonzeros ended up in my process. */

	/* Allocate space for storing the triplets after redistribution. */
	if (k)
	{ /* count can be zero. */
		if (!(ia = intMalloc_dist(2 * k)))
			ABORT("Malloc fails for ia[].");
		if (!(aij = doubleMalloc_dist(k)))
			ABORT("Malloc fails for aij[].");
		ja = ia + k;
	}

	/* Allocate temporary storage for sending/receiving the A triplets. */
	if (procs > 1)
	{
		if (!(send_req = (MPI_Request *)
				  SUPERLU_MALLOC(2 * procs * sizeof(MPI_Request))))
			ABORT("Malloc fails for send_req[].");
		if (!(ia_send = (int_t **)SUPERLU_MALLOC(procs * sizeof(int_t *))))
			ABORT("Malloc fails for ia_send[].");
		if (!(aij_send = (double **)SUPERLU_MALLOC(procs * sizeof(double *))))
			ABORT("Malloc fails for aij_send[].");
		if (SendCnt)
		{ /* count can be zero */
			if (!(index = intMalloc_dist(2 * SendCnt)))
				ABORT("Malloc fails for index[].");
			if (!(nzval = doubleMalloc_dist(SendCnt)))
				ABORT("Malloc fails for nzval[].");
		}
		if (!(ptr_to_send = intCalloc_dist(procs)))
			ABORT("Malloc fails for ptr_to_send[].");
		if (maxnnzToRecv)
		{ /* count can be zero */
			if (!(itemp = intMalloc_dist(2 * maxnnzToRecv)))
				ABORT("Malloc fails for itemp[].");
			if (!(dtemp = doubleMalloc_dist(maxnnzToRecv)))
				ABORT("Malloc fails for dtemp[].");
		}

		for (i = 0, j = 0, p = 0; p < procs; ++p)
		{
			if (p != iam)
			{
				if (nnzToSend[p] > 0)
					ia_send[p] = &index[i];
				i += 2 * nnzToSend[p]; /* ia/ja indices alternate */
				if (nnzToSend[p] > 0)
					aij_send[p] = &nzval[j];
				j += nnzToSend[p];
			}
		}
	} /* if procs > 1 */

	if (!(*colptr = intCalloc_dist(n + 1)))
		ABORT("Malloc fails for *colptr[].");

	/* ------------------------------------------------------------
	   LOAD THE ENTRIES OF A INTO THE (IA,JA,AIJ) STRUCTURES TO SEND.
	   THIS ACCOUNTS FOR THE SECOND PASS OF A.
	   ------------------------------------------------------------*/
	nnz_loc = 0; /* Reset the local nonzero count. */
	nzval_a = Astore->nzval;
	for (i = 0; i < m_loc; ++i)
	{
		for (j = Astore->rowptr[i]; j < Astore->rowptr[i + 1]; ++j)
		{
			irow = perm_c[perm_r[i + fst_row]]; /* Row number in Pc*Pr*A */
			jcol = Astore->colind[j];
			gbi = BlockNum(irow);
			gbj = BlockNum(jcol);
			p = PNUM(PROW(gbi, grid), PCOL(gbj, grid), grid);

			if (p != iam)
			{ /* remote */
				k = ptr_to_send[p];
				ia_send[p][k] = irow;
				ia_send[p][k + nnzToSend[p]] = jcol;
				aij_send[p][k] = nzval_a[j];
				++ptr_to_send[p];
			}
			else
			{ /* local */
				ia[nnz_loc] = irow;
				ja[nnz_loc] = jcol;
				aij[nnz_loc] = nzval_a[j];
				++nnz_loc;
				++(*colptr)[jcol]; /* Count nonzeros in each column */
			}
		}
	}

	/* ------------------------------------------------------------
	   PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
	   NOTE: Can possibly use MPI_Alltoallv.
	   ------------------------------------------------------------*/
	for (p = 0; p < procs; ++p)
	{
		if (p != iam && nnzToSend[p] > 0)
		{
			// if ( p != iam ) {
			it = 2 * nnzToSend[p];
			MPI_Isend(ia_send[p], it, mpi_int_t,
					  p, iam, grid->comm, &send_req[p]);
			it = nnzToSend[p];
			MPI_Isend(aij_send[p], it, MPI_DOUBLE,
					  p, iam + procs, grid->comm, &send_req[procs + p]);
		}
	}

	for (p = 0; p < procs; ++p)
	{
		if (p != iam && nnzToRecv[p] > 0)
		{
			// if ( p != iam ) {
			it = 2 * nnzToRecv[p];
			MPI_Recv(itemp, it, mpi_int_t, p, p, grid->comm, &status);
			it = nnzToRecv[p];
			MPI_Recv(dtemp, it, MPI_DOUBLE, p, p + procs,
					 grid->comm, &status);
			for (i = 0; i < nnzToRecv[p]; ++i)
			{
				ia[nnz_loc] = itemp[i];
				jcol = itemp[i + nnzToRecv[p]];
				/*assert(jcol<n);*/
				ja[nnz_loc] = jcol;
				aij[nnz_loc] = dtemp[i];
				++nnz_loc;
				++(*colptr)[jcol]; /* Count nonzeros in each column */
			}
		}
	}

	for (p = 0; p < procs; ++p)
	{
		if (p != iam && nnzToSend[p] > 0)
		{	// cause two of the tests to hang
			// if ( p != iam ) {
			MPI_Wait(&send_req[p], &status);
			MPI_Wait(&send_req[procs + p], &status);
		}
	}

	/* ------------------------------------------------------------
	   DEALLOCATE TEMPORARY STORAGE
	   ------------------------------------------------------------*/

	SUPERLU_FREE(nnzToRecv);

	if (procs > 1)
	{
		SUPERLU_FREE(send_req);
		SUPERLU_FREE(ia_send);
		SUPERLU_FREE(aij_send);
		if (SendCnt)
		{
			SUPERLU_FREE(index);
			SUPERLU_FREE(nzval);
		}
		SUPERLU_FREE(ptr_to_send);
		if (maxnnzToRecv)
		{
			SUPERLU_FREE(itemp);
			SUPERLU_FREE(dtemp);
		}
	}

	/* ------------------------------------------------------------
	   CONVERT THE TRIPLET FORMAT INTO THE CCS FORMAT.
	   ------------------------------------------------------------*/
	if (nnz_loc)
	{ /* nnz_loc can be zero */
		if (!(*rowind = intMalloc_dist(nnz_loc)))
			ABORT("Malloc fails for *rowind[].");
		if (!(*a = doubleMalloc_dist(nnz_loc)))
			ABORT("Malloc fails for *a[].");
	}

	/* Initialize the array of column pointers */
	k = 0;
	jsize = (*colptr)[0];
	(*colptr)[0] = 0;
	for (j = 1; j < n; ++j)
	{
		k += jsize;
		jsize = (*colptr)[j];
		(*colptr)[j] = k;
	}

	/* Copy the triplets into the column oriented storage */
	for (i = 0; i < nnz_loc; ++i)
	{
		j = ja[i];
		k = (*colptr)[j];
		(*rowind)[k] = ia[i];
		(*a)[k] = aij[i];
		++(*colptr)[j];
	}

	/* Reset the column pointers to the beginning of each column */
	for (j = n; j > 0; --j)
		(*colptr)[j] = (*colptr)[j - 1];
	(*colptr)[0] = 0;

	if (nnz_loc)
	{
		SUPERLU_FREE(ia);
		SUPERLU_FREE(aij);
	}

	return 0;
} /* dReDistribute_A */

/*
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 *
 * Purpose
 * =======
 *   Distribute the matrix onto the 2D process mesh.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t*
 *        options->Fact specifies whether or not the L and U structures will be re-used.
 *        = SamePattern_SameRowPerm: L and U structures are input, and
 *                                   unchanged on exit.
 *        = DOFACT or SamePattern: L and U structures are computed and output.
 *
 * n      (input) int
 *        Dimension of the matrix.
 *
 * A      (input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T. The type of A can be:
 *        Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * LUstruct (input/output) dLUstruct_t*
 *        Data structures for L and U factors.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   > 0, working storage required (in bytes).
 *
 */
float pddistribute(
	superlu_dist_options_t *options,
	int_t n,
	SuperMatrix *A, // (Yida) : 分布式CSR SLU_NR_loc
	dScalePermstruct_t *ScalePermstruct,
	Glu_freeable_t *Glu_freeable,
	dLUstruct_t *LUstruct,
	gridinfo_t *grid)
{
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t bnnz, fsupc, fsupc1, i, ii, irow, istart, j, ib, jb, jj, k, k1, len, len1, nsupc;
	int_t nlb;	/* local block rows*/
	int_t ljb;	/* local block column number */
	int_t nrbl; /* number of L blocks in current block column */
	int_t nrbu; /* number of U blocks in current block column */
	int_t gb;	/* global block number; 0 < gb <= nsuper */
	int_t lb;	/* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
	int_t ub, gik, iklrow, fnz;
	int iam, jbrow, kcol, krow, mycol, myrow, pr, pc;
	int_t mybufmax[NBUFFERS];
	double *a;
	int_t *asub, *xa;
	int_t *xsup = Glu_persist->xsup; /* supernode and column mapping */
	int_t *supno = Glu_persist->supno;
	int_t *lsub, *xlsub, *usub, *usub1, *xusub;
	int_t nsupers;
	int_t next_lind;				  /* next available position in index[*] */
	int_t next_lval;				  /* next available position in nzval[*] */
	int_t *index;					  /* indices consist of headers and row subscripts */
	int_t *index_srt;				  /* indices consist of headers and row subscripts */
	int *index1;					  /* temporary pointer to array of int */
	double *lusup, *lusup_srt, *uval; /* nonzero values in L and U */
	double **Lnzval_bc_ptr;			  /* size ceil(NSUPERS/Pc) */

	int_t **Lrowind_bc_ptr;			 /* size ceil(NSUPERS/Pc) */
	int_t **Lindval_loc_bc_ptr;		 /* size ceil(NSUPERS/Pc)                 */

	int_t *Unnz;				/* size ceil(NSUPERS/Pc)                 */
	double **Unzval_br_ptr;		/* size ceil(NSUPERS/Pr) */
	int_t **Ufstnz_br_ptr;		/* size ceil(NSUPERS/Pr) */

	int_t *Urbs, *Urbs1;	   /* Number of row blocks in each block column of U. */
	Ucb_indptr_t **Ucb_indptr; /* Vertical linked list pointing to Uindex[] */
	int_t **Ucb_valptr; /* Vertical linked list pointing to Unzval[] */

	/*-- Counts to be used in factorization. --*/
	int *ToRecv, *ToSendD, **ToSendR;

	/*-- Counts to be used in lower triangular solve. --*/
	int *fmod;			/* Modification count for L-solve.        */
	int **fsendx_plist; /* Column process list to send down Xk.   */
	int nfrecvx = 0;	/* Number of Xk I will receive.           */
	int nfsendx = 0;	/* Number of Xk I will send               */
	int kseen;

	/*-- Counts to be used in upper triangular solve. --*/
	int *bmod;			/* Modification count for U-solve.        */
	int **bsendx_plist; /* Column process list to send down Xk.   */
	int nbrecvx = 0;	/* Number of Xk I will receive.           */
	int nbsendx = 0;	/* Number of Xk I will send               */
	int_t *ilsum;		/* starting position of each supernode in
			   the full array (local)                 */

	/*-- Auxiliary arrays; freed on return --*/
	int_t *rb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
	int_t *Urb_length; /* U block length; size ceil(NSUPERS/Pr)             */
	int_t *Urb_indptr; /* pointers to U index[]; size ceil(NSUPERS/Pr)      */
	int_t *Urb_fstnz;  /* # of fstnz in a block row; size ceil(NSUPERS/Pr)  */
	int_t *Ucbs;	   /* number of column blocks in a block row            */
	int_t *Lrb_length; /* L block length; size ceil(NSUPERS/Pr)             */
	int_t *Lrb_number; /* global block number; size ceil(NSUPERS/Pr)        */
	int_t *Lrb_indptr; /* pointers to L index[]; size ceil(NSUPERS/Pr)      */
	int_t *Lrb_valptr; /* pointers to L nzval[]; size ceil(NSUPERS/Pr)      */
	int_t **nzrows;
	double *dense, *dense_col; /* SPA */
	double zero = 0.0;
	int_t ldaspa; /* LDA of SPA */
	int_t iword, dword;
	float mem_use = 0.0;
	float memTRS = 0.; /* memory allocated for storing the meta-data for triangular solve (positive number)*/

	int_t *lloc;
	double **Linv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	double **Uinv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	int_t idx_indx, idx_lusup;
	int_t nbrow;
	int_t ik, lk, knsupc;
	int_t uu;
	int_t nub;

	/* Initialization. */
	iam = grid->iam;
	myrow = MYROW(iam, grid);
	mycol = MYCOL(iam, grid);
	for (i = 0; i < NBUFFERS; ++i)
		mybufmax[i] = 0;
	nsupers = supno[n - 1] + 1;

	iword = sizeof(int_t);
	dword = sizeof(double);

	dReDistribute_A(A, ScalePermstruct, Glu_freeable, xsup, supno,
					grid, &xa, &asub, &a);



	if (options->Fact == SamePattern_SameRowPerm)
	{
		/* We can propagate the new values of A into the existing
		   L and U data structures.            */
		ilsum = Llu->ilsum;
		ldaspa = Llu->ldalsum;
		if (!(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options))))
			ABORT("Calloc fails for SPA dense[].");
		nrbu = CEILING(nsupers, grid->nprow); /* No. of local block rows */
		if (!(Urb_length = intCalloc_dist(nrbu)))
			ABORT("Calloc fails for Urb_length[].");
		if (!(Urb_indptr = intMalloc_dist(nrbu)))
			ABORT("Malloc fails for Urb_indptr[].");
		Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
		Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
		Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
		Unzval_br_ptr = Llu->Unzval_br_ptr;

		mem_use += 2.0 * nrbu * iword + ldaspa * sp_ienv_dist(3, options) * dword;

		/* Initialize Uval to zero. */
		for (lb = 0; lb < nrbu; ++lb)
		{
			Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
			index = Ufstnz_br_ptr[lb];
			if (index)
			{
				uval = Unzval_br_ptr[lb];
				len = index[1];
				for (i = 0; i < len; ++i)
					uval[i] = zero;
			} /* if index != NULL */
		} /* for lb ... */

		for (jb = 0; jb < nsupers; ++jb)
		{ /* Loop through each block column */
			pc = PCOL(jb, grid);
			if (mycol == pc)
			{ /* Block column jb in my process column */
				fsupc = FstBlockC(jb);
				nsupc = SuperSize(jb);

				/* Scatter A into SPA (for L), or into U directly. */
				for (j = fsupc, dense_col = dense; j < FstBlockC(jb + 1); ++j)
				{
					for (i = xa[j]; i < xa[j + 1]; ++i)
					{
						irow = asub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							if (gb < jb)
							{ /* in U */
								index = Ufstnz_br_ptr[lb];
								uval = Unzval_br_ptr[lb];
								while ((k = index[Urb_indptr[lb]]) < jb)
								{
									/* Skip nonzero values in this block */
									Urb_length[lb] += index[Urb_indptr[lb] + 1];
									/* Move pointer to the next block */
									Urb_indptr[lb] += UB_DESCRIPTOR + SuperSize(k);
								}
								/*assert(k == jb);*/
								/* start fstnz */
								istart = Urb_indptr[lb] + UB_DESCRIPTOR;
								len = Urb_length[lb];
								fsupc1 = FstBlockC(gb + 1);
								k = j - fsupc;
								/* Sum the lengths of the leading columns */
								for (jj = 0; jj < k; ++jj)
									len += fsupc1 - index[istart++];
								/*assert(irow>=index[istart]);*/
								uval[len + irow - index[istart]] = a[i];
							}
							else
							{ /* in L; put in SPA first */
								irow = ilsum[lb] + irow - FstBlockC(gb);
								dense_col[irow] = a[i];
							}
						}
					} /* for i ... */
					dense_col += ldaspa;
				} /* for j ... */

				/* Gather the values of A from SPA into Lnzval[]. */
				ljb = LBj(jb, grid); /* Local block number */
				index = Lrowind_bc_ptr[ljb];
				if (index)
				{
					nrbl = index[0]; /* Number of row blocks. */
					len = index[1];	 /* LDA of lusup[]. */
					lusup = Lnzval_bc_ptr[ljb];
					next_lind = BC_HEADER;
					next_lval = 0;
					for (jj = 0; jj < nrbl; ++jj)
					{
						gb = index[next_lind++];
						len1 = index[next_lind++]; /* Rows in the block. */
						lb = LBi(gb, grid);
						for (bnnz = 0; bnnz < len1; ++bnnz)
						{
							irow = index[next_lind++]; /* Global index. */
							irow = ilsum[lb] + irow - FstBlockC(gb);
							k = next_lval++;
							for (j = 0, dense_col = dense; j < nsupc; ++j)
							{
								lusup[k] = dense_col[irow];
								dense_col[irow] = zero;
								k += len;
								dense_col += ldaspa;
							}
						} /* for bnnz ... */
					} /* for jj ... */
				} /* if index ... */
			} /* if mycol == pc */
		} /* for jb ... */

		SUPERLU_FREE(dense);
		SUPERLU_FREE(Urb_length);
		SUPERLU_FREE(Urb_indptr);
	}
	else
	{
		/* options->Fact is not SamePattern_SameRowPerm */
		/* ------------------------------------------------------------
	       FIRST TIME CREATING THE L AND U DATA STRUCTURES.
		   ------------------------------------------------------------*/
		/* We first need to set up the L and U data structures and then
		 * propagate the values of A into them.
		 */
		lsub = Glu_freeable->lsub; /* compressed L subscripts */
		xlsub = Glu_freeable->xlsub;
		usub = Glu_freeable->usub; /* compressed U subscripts */ // (Yida) : U.CSC.rowidx
		xusub = Glu_freeable->xusub; // (Yida) : U.CSC.colptr

		if (!(ToRecv = (int *)SUPERLU_MALLOC(nsupers * sizeof(int))))
			ABORT("Malloc fails for ToRecv[].");
		for (i = 0; i < nsupers; ++i)
			ToRecv[i] = 0;

		k = CEILING(nsupers, grid->npcol); /* Number of local column blocks */
		if (!(ToSendR = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for ToSendR[].");
		j = k * grid->npcol;
		if (!(index1 = SUPERLU_MALLOC(j * sizeof(int))))
			ABORT("Malloc fails for index[].");

		mem_use += (float)k * sizeof(int_t *) + (j + nsupers) * iword;

		for (i = 0; i < j; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->npcol)
			ToSendR[i] = &index1[j];
		k = CEILING(nsupers, grid->nprow); /* Number of local block rows */

		/* Pointers to the beginning of each block row of U. */
		if (!(Unzval_br_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
			ABORT("Malloc fails for Unzval_br_ptr[].");
		if (!(Ufstnz_br_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Ufstnz_br_ptr[].");

		if (!(ToSendD = SUPERLU_MALLOC(k * sizeof(int))))
			ABORT("Malloc fails for ToSendD[].");
		for (i = 0; i < k; ++i)
			ToSendD[i] = NO;
		if (!(ilsum = intMalloc_dist(k + 1)))
			ABORT("Malloc fails for ilsum[].");

		/* Auxiliary arrays used to set up U block data structures.
		   They are freed on return. */
		if (!(rb_marker = intCalloc_dist(k)))
			ABORT("Calloc fails for rb_marker[].");
		if (!(Urb_length = intCalloc_dist(k)))
			ABORT("Calloc fails for Urb_length[].");
		if (!(Urb_indptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Urb_indptr[].");
		if (!(Urb_fstnz = intCalloc_dist(k)))
			ABORT("Calloc fails for Urb_fstnz[].");
		if (!(Ucbs = intCalloc_dist(k)))
			ABORT("Calloc fails for Ucbs[].");

		mem_use += 2.0 * k * sizeof(int_t *) + (7 * k + 1) * iword;

		/* Compute ldaspa and ilsum[]. */
		ldaspa = 0; // (Yida) : 完整矩阵在本地的行数或列数
		ilsum[0] = 0; // (Yida) : 在一个本地列块中，每个行块的第一行是本地列块的第几行
		for (gb = 0; gb < nsupers; ++gb)
		{
			if (myrow == PROW(gb, grid))
			{
				i = SuperSize(gb);
				ldaspa += i;
				lb = LBi(gb, grid);
				ilsum[lb + 1] = ilsum[lb] + i;
			}
		}

		/* ------------------------------------------------------------
		   COUNT NUMBER OF ROW BLOCKS AND THE LENGTH OF EACH BLOCK IN U.
		   THIS ACCOUNTS FOR ONE-PASS PROCESSING OF G(U). (Yida) : Pass 1 of LU
		   ------------------------------------------------------------*/
		/* Loop through each supernode column. */
		// printf("[Yida] nsupers=%d\n", nsupers);
		for (jb = 0; jb < nsupers; ++jb)
		{
			pc = PCOL(jb, grid);
			fsupc = FstBlockC(jb);
			nsupc = SuperSize(jb);
			// printf("[Yida] jb=%d fsupc=%d nsupc=%d\n", jb, fsupc, nsupc);
			
			/* Loop through each column in the block. */
			for (j = fsupc; j < fsupc + nsupc; ++j)
			{
				// printf("[Yida] j=%d\n", j);
				/* usub[*] contains only "first nonzero" in each segment. */
				for (i = xusub[j]; i < xusub[j + 1]; ++i)
				{
					irow = usub[i]; /* First nonzero of the segment. */
					gb = BlockNum(irow);
					// printf("[Yida] i=%d irow=%d gb=%d\n", i, irow, gb);
					kcol = PCOL(gb, grid);
					ljb = LBj(gb, grid); // (Yida) : ljb * Pc + kcol == gb
					if (mycol == kcol && mycol != pc)
						ToSendR[ljb][pc] = YES;
					pr = PROW(gb, grid);
					lb = LBi(gb, grid);
					if (mycol == pc)
					{
						if (myrow == pr)
						{
							ToSendD[lb] = YES;
							/* Count nonzeros in entire block row. */
							Urb_length[lb] += FstBlockC(gb + 1) - irow; // (Yida) : Why to get triangle sum?
							// printf("[Yida] (%d,%d) Urb_length[lb]=%d %d-%d\n", pc, pr, Urb_length[lb], FstBlockC(gb + 1), irow);
							
							if (rb_marker[lb] <= jb)
							{ /* First see the block */
								rb_marker[lb] = jb + 1;
								Urb_fstnz[lb] += nsupc;
								++Ucbs[lb]; /* Number of column blocks in block row lb. */
							}
							ToRecv[gb] = 1;
						}
						else
							ToRecv[gb] = 2; /* Do I need 0, 1, 2 ? */
					}
				} /* for i ... */
			} /* for j ... */
		} /* for jb ... */

		/* Set up the initial pointers for each block row in U. */
		nrbu = CEILING(nsupers, grid->nprow); /* Number of local block rows */
		for (lb = 0; lb < nrbu; ++lb)
		{
			len = Urb_length[lb];
			rb_marker[lb] = 0; /* Reset block marker. */
			// printf("[Yida] lb=%d len=%d\n", lb, len);
			if (len)
			{
				/* Add room for descriptors */
				len1 = Urb_fstnz[lb] /* nnz in brow */ + BR_HEADER + Ucbs[lb] /* bnz in brow */ * UB_DESCRIPTOR;
				if (!(index = intMalloc_dist(len1 + 1)))
					ABORT("Malloc fails for Uindex[].");
				Ufstnz_br_ptr[lb] = index;
				if (!(Unzval_br_ptr[lb] = doubleMalloc_dist(len)))
					ABORT("Malloc fails for Unzval_br_ptr[*][].");

				mybufmax[2] = SUPERLU_MAX(mybufmax[2], len1);
				mybufmax[3] = SUPERLU_MAX(mybufmax[3], len);
				index[0] = Ucbs[lb]; /* Number of column blocks */
				index[1] = len;		 /* Total length of nzval[] */
				index[2] = len1;	 /* Total length of index[] */
				index[len1] = -1;	 /* End marker */
			}
			else
			{
				Ufstnz_br_ptr[lb] = NULL;
				Unzval_br_ptr[lb] = NULL;
			}
			Urb_length[lb] = 0;			/* Reset block length. */
			Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
			Urb_fstnz[lb] = BR_HEADER;
		} /* for lb ... */

		SUPERLU_FREE(Ucbs);

		mem_use -= 2.0 * k * iword;

		/* Auxiliary arrays used to set up L block data structures.
		   They are freed on return.
		   k is the number of local row blocks.   */
		if (!(Lrb_length = intCalloc_dist(k)))
			ABORT("Calloc fails for Lrb_length[].");
		if (!(Lrb_number = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_number[].");
		if (!(Lrb_indptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_indptr[].");
		if (!(Lrb_valptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_valptr[].");
		if (!(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options)))) // (Yida) : dense : 列主序，长度ldaspa * superlu_maxsup
			ABORT("Calloc fails for SPA dense[].");

		/* These counts will be used for triangular solves. */
		if (!(fmod = int32Calloc_dist(k)))
			ABORT("Calloc fails for fmod[].");
		if (!(bmod = int32Calloc_dist(k)))
			ABORT("Calloc fails for bmod[].");

		/* ------------------------------------------------ */
		mem_use += 6.0 * k * iword + ldaspa * sp_ienv_dist(3, options) * dword;

		k = CEILING(nsupers, grid->npcol); /* Number of local block columns */
		/* Pointers to the beginning of each block column of L. */
		if (!(Lnzval_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
			ABORT("Malloc fails for Lnzval_bc_ptr[].");
		Lnzval_bc_ptr[k - 1] = NULL;
		if (!(Lrowind_bc_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Lrowind_bc_ptr[].");
		Lrowind_bc_ptr[k - 1] = NULL;

		if (!(Lindval_loc_bc_ptr =
				  (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Lindval_loc_bc_ptr[].");
		Lindval_loc_bc_ptr[k - 1] = NULL;

		if (!(Linv_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
		{
			fprintf(stderr, "Malloc fails for Linv_bc_ptr[].");
		}
		if (!(Uinv_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
		{
			fprintf(stderr, "Malloc fails for Uinv_bc_ptr[].");
		}
		Linv_bc_ptr[k - 1] = NULL;
		Uinv_bc_ptr[k - 1] = NULL;

		if (!(Unnz =
				  (int_t *)SUPERLU_MALLOC(k * sizeof(int_t))))
			ABORT("Malloc fails for Unnz[].");

		/* These lists of processes will be used for triangular solves. */
		if (!(fsendx_plist = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for fsendx_plist[].");
		len = k * grid->nprow;
		if (!(index1 = int32Malloc_dist(len)))
			ABORT("Malloc fails for fsendx_plist[0]");
		for (i = 0; i < len; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
			fsendx_plist[i] = &index1[j];
		if (!(bsendx_plist = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for bsendx_plist[].");
		if (!(index1 = int32Malloc_dist(len)))
			ABORT("Malloc fails for bsendx_plist[0]");
		for (i = 0; i < len; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
			bsendx_plist[i] = &index1[j];
		/* -------------------------------------------------------------- */
		mem_use += 4.0 * k * sizeof(int_t *) + 2.0 * len * iword;
		memTRS += k * sizeof(int_t *) + 2.0 * k * sizeof(double *) + k * iword; // acount for Lindval_loc_bc_ptr, Unnz, Linv_bc_ptr,Uinv_bc_ptr

		/*------------------------------------------------------------
		  PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
		  THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U. (Yida) : Pass 2 of LU
		  ------------------------------------------------------------*/
		for (jb = 0; jb < nsupers; ++jb)
		{ /* for each block column ... */
			pc = PCOL(jb, grid);
			if (mycol == pc)
			{ /* Block column jb in my process column */
				fsupc = FstBlockC(jb);
				nsupc = SuperSize(jb);
				ljb = LBj(jb, grid); /* Local block number */
				// printf("[LYDSCHED] #%d ljb = %d\n", iam, ljb);

				/* Scatter A into SPA. */
				for (j = fsupc, dense_col = dense; j < FstBlockC(jb + 1); ++j) // (Yida) : 对当前supercol的每列
				{
					for (i = xa[j]; i < xa[j + 1]; ++i)
					{
						irow = asub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							irow = ilsum[lb] + irow - FstBlockC(gb);
							dense_col[irow] = a[i];
						}
					}
					dense_col += ldaspa; // (Yida) : dense_col右移一列
				} /* for j ... */

				jbrow = PROW(jb, grid);

				/*------------------------------------------------
				 * SET UP U BLOCKS.
				 *------------------------------------------------*/
				kseen = 0;
				dense_col = dense;
				/* Loop through each column in the block column. */
				for (j = fsupc; j < FstBlockC(jb + 1); ++j)
				{
					istart = xusub[j];
					/* NOTE: Only the first nonzero index of the segment
					   is stored in usub[]. */
					for (i = istart; i < xusub[j + 1]; ++i)
					{
						irow = usub[i]; /* First nonzero in the segment. */
						gb = BlockNum(irow);
						pr = PROW(gb, grid);
						if (pr != jbrow &&
							myrow == jbrow && /* diag. proc. owning jb */
							bsendx_plist[ljb][pr] == SLU_EMPTY)
						{
							bsendx_plist[ljb][pr] = YES;
							++nbsendx;
						}
						if (myrow == pr)
						{
							lb = LBi(gb, grid); /* Local block number */
							index = Ufstnz_br_ptr[lb];
							uval = Unzval_br_ptr[lb];
							fsupc1 = FstBlockC(gb + 1);
							if (rb_marker[lb] <= jb)
							{ /* First time see the block */
								rb_marker[lb] = jb + 1;
								Urb_indptr[lb] = Urb_fstnz[lb];
								;
								index[Urb_indptr[lb]] = jb; /* Descriptor */
								Urb_indptr[lb] += UB_DESCRIPTOR;
								/* Record the first location in index[] of the
								   next block */
								Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;
								len = Urb_indptr[lb]; /* Start fstnz in index */
								index[len - 1] = 0;
								for (k = 0; k < nsupc; ++k)
									index[len + k] = fsupc1;
								if (gb != jb)	/* Exclude diagonal block. */
									++bmod[lb]; /* Mod. count for back solve */
								if (kseen == 0 && myrow != jbrow)
								{
									++nbrecvx;
									kseen = 1;
								}
							}
							else
							{						  /* Already saw the block */
								len = Urb_indptr[lb]; /* Start fstnz in index */
							}
							jj = j - fsupc;
							index[len + jj] = irow;
							/* Load the numerical values */
							k = fsupc1 - irow;	 /* No. of nonzeros in segment */
							index[len - 1] += k; /* Increment block length in
										Descriptor */
							irow = ilsum[lb] + irow - FstBlockC(gb);
							for (ii = 0; ii < k; ++ii)
							{
								uval[Urb_length[lb]++] = dense_col[irow + ii];
								dense_col[irow + ii] = zero;
							}
						} /* if myrow == pr ... */
					} /* for i ... */
					dense_col += ldaspa;
				} /* for j ... */

				/*------------------------------------------------
				 * SET UP L BLOCKS.
				 *------------------------------------------------*/

				/* Count number of blocks and length of each block. */
				nrbl = 0;
				len = 0; /* Number of row subscripts I own. */
				kseen = 0;
				istart = xlsub[fsupc];
				// printf("[LYDSCHED] #%d fsupc=%d %d-%d\n", iam, fsupc, istart, xlsub[fsupc+1]);
				for (i = istart; i < xlsub[fsupc + 1]; ++i)
				{
					irow = lsub[i];
					// printf("[LYDSCHED] #%d i=%d irow=%d\n", iam, i, irow);
					gb = BlockNum(irow); /* Global block number */
					pr = PROW(gb, grid); /* Process row owning this block */
					if (pr != jbrow &&
						myrow == jbrow && /* diag. proc. owning jb */
						fsendx_plist[ljb][pr] == SLU_EMPTY /* first time */)
					{
						fsendx_plist[ljb][pr] = YES;
						++nfsendx;
					}
					if (myrow == pr)
					{
						lb = LBi(gb, grid); /* Local block number */
						if (rb_marker[lb] <= jb)
						{ /* First see this block */
							rb_marker[lb] = jb + 1;
							Lrb_length[lb] = 1;
							Lrb_number[nrbl++] = gb;
							if (gb != jb)	/* Exclude diagonal block. */
								++fmod[lb]; /* Mod. count for forward solve */
							if (kseen == 0 && myrow != jbrow)
							{
								++nfrecvx;
								kseen = 1;
							}
							// printf("[LYDSCHED] #%d jb=%d i=%d irow=%d nrbl=%d\n", iam, jb, i, irow, nrbl);
						}
						else
						{
							++Lrb_length[lb];
						}
						++len;
					}
				} /* for i ... */

				if (nrbl)
				{ /* Do not ensure the blocks are sorted! */
					/* Set up the initial pointers for each block in
					   index[] and nzval[]. */
					/* Add room for descriptors */
					len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
					if (!(index = intMalloc_dist(len1)))
						ABORT("Malloc fails for index[]");

					if (!(lusup = (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double))))
						ABORT("Malloc fails for lusup[]");
					if (!(Lindval_loc_bc_ptr[ljb] = intCalloc_dist(nrbl * 3)))
						ABORT("Malloc fails for Lindval_loc_bc_ptr[ljb][]");

					myrow = MYROW(iam, grid);
					krow = PROW(jb, grid);
					if (myrow == krow)
					{ /* diagonal block */
						if (!(Linv_bc_ptr[ljb] = (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double))))
							ABORT("Malloc fails for Linv_bc_ptr[ljb][]");
						if (!(Uinv_bc_ptr[ljb] = (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double))))
							ABORT("Malloc fails for Uinv_bc_ptr[ljb][]");
					}
					else
					{
						Linv_bc_ptr[ljb] = NULL;
						Uinv_bc_ptr[ljb] = NULL;
					}

					mybufmax[0] = SUPERLU_MAX(mybufmax[0], len1);
					mybufmax[1] = SUPERLU_MAX(mybufmax[1], len * nsupc);
					mybufmax[4] = SUPERLU_MAX(mybufmax[4], len);
					memTRS += nrbl * 3.0 * iword + 2.0 * nsupc * nsupc * dword; // acount for Lindval_loc_bc_ptr[ljb],Linv_bc_ptr[ljb],Uinv_bc_ptr[ljb]
					index[0] = nrbl;											/* Number of row blocks */
					index[1] = len;												/* LDA of the nzval[] */
					next_lind = BC_HEADER;
					next_lval = 0;
					for (k = 0; k < nrbl; ++k)
					{
						gb = Lrb_number[k];
						lb = LBi(gb, grid);
						len = Lrb_length[lb];
						Lindval_loc_bc_ptr[ljb][k] = lb;
						Lindval_loc_bc_ptr[ljb][k + nrbl] = next_lind;
						Lindval_loc_bc_ptr[ljb][k + nrbl * 2] = next_lval;
						Lrb_length[lb] = 0;		 /* Reset vector of block length */
						index[next_lind++] = gb; /* Descriptor */
						index[next_lind++] = len;
						Lrb_indptr[lb] = next_lind;
						Lrb_valptr[lb] = next_lval;
						next_lind += len;
						next_lval += len;
					}
					/* Propagate the compressed row subscripts to Lindex[],
							   and the initial values of A from SPA into Lnzval[]. */
					len = index[1]; /* LDA of lusup[] */
					for (i = istart; i < xlsub[fsupc + 1]; ++i)
					{
						irow = lsub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							k = Lrb_indptr[lb]++; /* Random access a block */
							index[k] = irow;
							k = Lrb_valptr[lb]++;
							irow = ilsum[lb] + irow - FstBlockC(gb);
							for (j = 0, dense_col = dense; j < nsupc; ++j)
							{
								lusup[k] = dense_col[irow];
								dense_col[irow] = 0.0;
								k += len;
								dense_col += ldaspa;
							}
						}
					} /* for i ... */

					Lrowind_bc_ptr[ljb] = index;
					Lnzval_bc_ptr[ljb] = lusup;

					/* sort Lindval_loc_bc_ptr[ljb], Lrowind_bc_ptr[ljb]
								   and Lnzval_bc_ptr[ljb] here.  */
					if (nrbl > 1)
					{
						krow = PROW(jb, grid);
						if (myrow == krow)
						{ /* skip the diagonal block */
							uu = nrbl - 2;
							lloc = &Lindval_loc_bc_ptr[ljb][1];
						}
						else
						{
							uu = nrbl - 1;
							lloc = Lindval_loc_bc_ptr[ljb];
						}
						quickSortM(lloc, 0, uu, nrbl, 0, 3);
					}

					if (!(index_srt = intMalloc_dist(len1)))
						ABORT("Malloc fails for index_srt[]");
					if (!(lusup_srt = (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double))))
						ABORT("Malloc fails for lusup_srt[]");

					idx_indx = BC_HEADER;
					idx_lusup = 0;
					for (jj = 0; jj < BC_HEADER; jj++)
						index_srt[jj] = index[jj];

					for (i = 0; i < nrbl; i++)
					{
						nbrow = index[Lindval_loc_bc_ptr[ljb][i + nrbl] + 1];
						for (jj = 0; jj < LB_DESCRIPTOR + nbrow; jj++)
						{
							index_srt[idx_indx++] = index[Lindval_loc_bc_ptr[ljb][i + nrbl] + jj];
						}

						Lindval_loc_bc_ptr[ljb][i + nrbl] = idx_indx - LB_DESCRIPTOR - nbrow;

						for (jj = 0; jj < nbrow; jj++)
						{
							k = idx_lusup;
							k1 = Lindval_loc_bc_ptr[ljb][i + nrbl * 2] + jj;
							for (j = 0; j < nsupc; ++j)
							{
								lusup_srt[k] = lusup[k1];
								k += len;
								k1 += len;
							}
							idx_lusup++;
						}
						Lindval_loc_bc_ptr[ljb][i + nrbl * 2] = idx_lusup - nbrow;
					}

					SUPERLU_FREE(lusup);
					SUPERLU_FREE(index);

					Lrowind_bc_ptr[ljb] = index_srt;
					Lnzval_bc_ptr[ljb] = lusup_srt;
				}
				else
				{
					Lrowind_bc_ptr[ljb] = NULL;
					Lnzval_bc_ptr[ljb] = NULL;
					Linv_bc_ptr[ljb] = NULL;
					Uinv_bc_ptr[ljb] = NULL;
					Lindval_loc_bc_ptr[ljb] = NULL;
				} /* if nrbl ... */
			} /* if mycol == pc */
		} /* for jb ... */

		/////////////////////////////////////////////////////////////////

		/* Set up additional pointers for the index and value arrays of U.
		   nub is the number of local block columns. */
		nub = CEILING(nsupers, grid->npcol); /* Number of local block columns. */
		if (!(Urbs = (int_t *)intCalloc_dist(2 * nub)))
			ABORT("Malloc fails for Urbs[]"); /* Record number of nonzero
								 blocks in a block column. */
		Urbs1 = Urbs + nub;
		if (!(Ucb_indptr = SUPERLU_MALLOC(nub * sizeof(Ucb_indptr_t *))))
			ABORT("Malloc fails for Ucb_indptr[]");
		if (!(Ucb_valptr = SUPERLU_MALLOC(nub * sizeof(int_t *))))
			ABORT("Malloc fails for Ucb_valptr[]");

		nlb = CEILING(nsupers, grid->nprow); /* Number of local block rows. */

		/* Count number of row blocks in a block column.
		   One pass of the skeleton graph of U. */
		for (lk = 0; lk < nlb; ++lk)
		{
			usub1 = Ufstnz_br_ptr[lk];
			if (usub1)
			{ /* Not an empty block row. */
				/* usub1[0] -- number of column blocks in this block row. */
				i = BR_HEADER; /* Pointer in index array. */
				for (lb = 0; lb < usub1[0]; ++lb)
				{				  /* For all column blocks. */
					k = usub1[i]; /* Global block number */
					++Urbs[LBj(k, grid)];
					i += UB_DESCRIPTOR + SuperSize(k);
				}
			}
		}

		/* Set up the vertical linked lists for the row blocks.
		   One pass of the skeleton graph of U. */
		for (lb = 0; lb < nub; ++lb)
		{
			if (Urbs[lb])
			{ /* Not an empty block column. */
				if (!(Ucb_indptr[lb] = SUPERLU_MALLOC(Urbs[lb] * sizeof(Ucb_indptr_t))))
					ABORT("Malloc fails for Ucb_indptr[lb][]");
				if (!(Ucb_valptr[lb] = (int_t *)intMalloc_dist(Urbs[lb])))
					ABORT("Malloc fails for Ucb_valptr[lb][]");
			}
			else
			{
				Ucb_valptr[lb] = NULL;
				Ucb_indptr[lb] = NULL;
			}
		}
		for (lk = 0; lk < nlb; ++lk)
		{ /* For each block row. */
			usub1 = Ufstnz_br_ptr[lk];
			if (usub1)
			{				   /* Not an empty block row. */
				i = BR_HEADER; /* Pointer in index array. */
				j = 0;		   /* Pointer in nzval array. */

				for (lb = 0; lb < usub1[0]; ++lb)
				{						/* For all column blocks. */
					k = usub1[i];		/* Global block number, column-wise. */
					ljb = LBj(k, grid); /* Local block number, column-wise. */
					Ucb_indptr[ljb][Urbs1[ljb]].lbnum = lk;

					Ucb_indptr[ljb][Urbs1[ljb]].indpos = i;
					Ucb_valptr[ljb][Urbs1[ljb]] = j;

					++Urbs1[ljb];
					j += usub1[i + 1];
					i += UB_DESCRIPTOR + SuperSize(k);
				}
			}
		}

		/* Count the nnzs per block column */
		for (lb = 0; lb < nub; ++lb)
		{
			Unnz[lb] = 0;
			k = lb * grid->npcol + mycol; /* Global block number, column-wise. */
			knsupc = SuperSize(k);
			for (ub = 0; ub < Urbs[lb]; ++ub)
			{
				ik = Ucb_indptr[lb][ub].lbnum; /* Local block number, row-wise. */
				i = Ucb_indptr[lb][ub].indpos; /* Start of the block in usub[]. */
				i += UB_DESCRIPTOR;
				gik = ik * grid->nprow + myrow; /* Global block number, row-wise. */
				iklrow = FstBlockC(gik + 1);
				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = Ufstnz_br_ptr[ik][i + jj];
					if (fnz < iklrow)
					{
						Unnz[lb] += iklrow - fnz;
					}
				} /* for jj ... */
			}
		}

		Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
		Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
		Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
		Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
		Llu->Unzval_br_ptr = Unzval_br_ptr;
		Llu->Unnz = Unnz;
		Llu->ToRecv = ToRecv;
		Llu->ToSendD = ToSendD;
		Llu->ToSendR = ToSendR;
		Llu->fmod = fmod;
		Llu->fsendx_plist = fsendx_plist;
		Llu->nfrecvx = nfrecvx;
		Llu->nfsendx = nfsendx;
		Llu->bmod = bmod;
		Llu->bsendx_plist = bsendx_plist;
		Llu->nbrecvx = nbrecvx;
		Llu->nbsendx = nbsendx;
		Llu->ilsum = ilsum;
		Llu->ldalsum = ldaspa;
		Llu->Linv_bc_ptr = Linv_bc_ptr;
		Llu->Uinv_bc_ptr = Uinv_bc_ptr;
		Llu->Urbs = Urbs;
		Llu->Ucb_indptr = Ucb_indptr;
		Llu->Ucb_valptr = Ucb_valptr;

		SUPERLU_FREE(rb_marker);
		SUPERLU_FREE(Urb_fstnz);
		SUPERLU_FREE(Urb_length);
		SUPERLU_FREE(Urb_indptr);
		SUPERLU_FREE(Lrb_length);
		SUPERLU_FREE(Lrb_number);
		SUPERLU_FREE(Lrb_indptr);
		SUPERLU_FREE(Lrb_valptr);
		SUPERLU_FREE(dense);

		/* Find the maximum buffer size. */
		MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t,
					  MPI_MAX, grid->comm);

		k = CEILING(nsupers, grid->nprow); /* Number of local block rows */
		if (!(Llu->mod_bit = int32Malloc_dist(k)))
			ABORT("Malloc fails for mod_bit[].");

	} /* else fact != SamePattern_SameRowPerm */

	if (xa[A->ncol] > 0)
	{ /* may not have any entries on this process. */
		SUPERLU_FREE(asub);
		SUPERLU_FREE(a);
	}
	SUPERLU_FREE(xa);
	LUstruct->trf3Dpart = NULL;

	return (mem_use + memTRS);

} /* PDDISTRIBUTE */

/*
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 *
 * Purpose
 * =======
 *   Distribute the matrix onto the 2D process mesh on all girds based on supernodeMask
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t*
 *        options->Fact specifies whether or not the L and U structures will be re-used.
 *        = SamePattern_SameRowPerm: L and U structures are input, and
 *                                   unchanged on exit.
 *        = DOFACT or SamePattern: L and U structures are computed and output.
 *
 * n      (input) int
 *        Dimension of the matrix.
 *
 * A      (input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T. The type of A can be:
 *        Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * LUstruct (input/output) dLUstruct_t*
 *        Data structures for L and U factors.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   > 0, working storage required (in bytes).
 *
 */
float pddistribute_allgrid(
	superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	dScalePermstruct_t *ScalePermstruct,
	Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
	gridinfo_t *grid, int *supernodeMask)
{
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t bnnz, fsupc, fsupc1, i, ii, irow, istart, j, ib, jb, jj, k, k1,
		len, len1, nsupc, masked;
	int_t lib;	/* local block row number */
	int_t nlb;	/* local block rows*/
	int_t ljb;	/* local block column number */
	int_t nrbl; /* number of L blocks in current block column */
	int_t nrbu; /* number of U blocks in current block column */
	int_t gb;	/* global block number; 0 < gb <= nsuper */
	int_t lb;	/* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
	int_t ub, gik, iklrow, fnz;
	int iam, jbrow, kcol, krow, mycol, myrow, pc, pr;
	int_t mybufmax[NBUFFERS];
	NRformat_loc *Astore;
	double *a;
	int_t *asub, *xa;
	int_t *xa_begin, *xa_end;
	int_t *xsup = Glu_persist->xsup; /* supernode and column mapping */
	int_t *supno = Glu_persist->supno;
	int_t *lsub, *xlsub, *usub, *usub1, *xusub;
	int_t nsupers;
	int_t next_lind;				  /* next available position in index[*] */
	int_t next_lval;				  /* next available position in nzval[*] */
	int_t *index;					  /* indices consist of headers and row subscripts */
	int_t *index_srt;				  /* indices consist of headers and row subscripts */
	int *index1;					  /* temporary pointer to array of int */
	double *lusup, *lusup_srt, *uval; /* nonzero values in L and U */
	double **Lnzval_bc_ptr;			  /* size ceil(NSUPERS/Pc) */
	double *Lnzval_bc_dat;			  /* size: sum of sizes of Lnzval_bc_ptr[lk])  */
	long int *Lnzval_bc_offset;		  /* size ceil(NSUPERS/Pc)                 */

	int_t **Lrowind_bc_ptr;			 /* size ceil(NSUPERS/Pc) */
	int_t *Lrowind_bc_dat;			 /* size: sum of sizes of Lrowind_bc_ptr[lk])   */
	long int *Lrowind_bc_offset;	 /* size ceil(NSUPERS/Pc)                 */
	int_t **Lindval_loc_bc_ptr;		 /* size ceil(NSUPERS/Pc)                 */
	int_t *Lindval_loc_bc_dat;		 /* size: sum of sizes of Lindval_loc_bc_ptr[lk]) */
	long int *Lindval_loc_bc_offset; /* size ceil(NSUPERS/Pc)                 */

	int_t *Unnz;				/* size ceil(NSUPERS/Pc)                 */
	double **Unzval_br_ptr;		/* size ceil(NSUPERS/Pr) */
	double *Unzval_br_dat;		/* size: sum of sizes of Unzval_br_ptr[lk]) */
	long int *Unzval_br_offset; /* size ceil(NSUPERS/Pr)    */
	long int Unzval_br_cnt = 0;
	int_t **Ufstnz_br_ptr;		/* size ceil(NSUPERS/Pr) */
	int_t *Ufstnz_br_dat;		/* size: sum of sizes of Ufstnz_br_ptr[lk]) */
	long int *Ufstnz_br_offset; /* size ceil(NSUPERS/Pr)    */
	long int Ufstnz_br_cnt = 0;

	C_Tree *LBtree_ptr; /* size ceil(NSUPERS/Pc) */
	C_Tree *LRtree_ptr; /* size ceil(NSUPERS/Pr) */
	C_Tree *UBtree_ptr; /* size ceil(NSUPERS/Pc) */
	C_Tree *URtree_ptr; /* size ceil(NSUPERS/Pr) */
	int msgsize;

	int_t *Urbs, *Urbs1;	   /* Number of row blocks in each block column of U. */
	Ucb_indptr_t **Ucb_indptr; /* Vertical linked list pointing to Uindex[] */
	Ucb_indptr_t *Ucb_inddat;
	long int *Ucb_indoffset;
	long int Ucb_indcnt = 0;
	int_t **Ucb_valptr; /* Vertical linked list pointing to Unzval[] */
	int_t *Ucb_valdat;
	long int *Ucb_valoffset;
	long int Ucb_valcnt = 0;

	/*-- Counts to be used in factorization. --*/
	int *ToRecv, *ToSendD, **ToSendR;

	/*-- Counts to be used in lower triangular solve. --*/
	int *fmod;			/* Modification count for L-solve.        */
	int **fsendx_plist; /* Column process list to send down Xk.   */
	int nfrecvx = 0;	/* Number of Xk I will receive.           */
	int nfsendx = 0;	/* Number of Xk I will send               */
	int kseen;

	/*-- Counts to be used in upper triangular solve. --*/
	int *bmod;			/* Modification count for U-solve.        */
	int **bsendx_plist; /* Column process list to send down Xk.   */
	int nbrecvx = 0;	/* Number of Xk I will receive.           */
	int nbsendx = 0;	/* Number of Xk I will send               */
	int_t *ilsum;		/* starting position of each supernode in
			   the full array (local)                 */

	/*-- Auxiliary arrays; freed on return --*/
	int_t *rb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
	int_t *Urb_length; /* U block length; size ceil(NSUPERS/Pr)             */
	int_t *Urb_indptr; /* pointers to U index[]; size ceil(NSUPERS/Pr)      */
	int_t *Urb_fstnz;  /* # of fstnz in a block row; size ceil(NSUPERS/Pr)  */
	int_t *Ucbs;	   /* number of column blocks in a block row            */
	int_t *Lrb_length; /* L block length; size ceil(NSUPERS/Pr)             */
	int_t *Lrb_number; /* global block number; size ceil(NSUPERS/Pr)        */
	int_t *Lrb_indptr; /* pointers to L index[]; size ceil(NSUPERS/Pr)      */
	int_t *Lrb_valptr; /* pointers to L nzval[]; size ceil(NSUPERS/Pr)      */
	int_t *ActiveFlag;
	int_t *ActiveFlagAll;
	int_t Iactive;
	int *ranks;
	int_t *idxs;
	int_t **nzrows;
	double rseed;
	int rank_cnt, rank_cnt_ref, Root;
	double *dense, *dense_col; /* SPA */
	double zero = 0.0;
	int_t ldaspa; /* LDA of SPA */
	int_t iword, dword;
	float mem_use = 0.0;
	float memTRS = 0.; /* memory allocated for storing the meta-data for triangular solve (positive number)*/

	int *mod_bit; // Sherry 1/16/2022: changed to 'int'
	int *frecv, *brecv;
	int_t *lloc;
	double **Linv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	double *Linv_bc_dat;	  /* size: sum of sizes of Linv_bc_ptr[lk]) */
	long int *Linv_bc_offset; /* size ceil(NSUPERS/Pc)              */
	double **Uinv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	double *Uinv_bc_dat;	  /* size: sum of sizes of Uinv_bc_ptr[lk]) */
	long int *Uinv_bc_offset; /* size ceil(NSUPERS/Pc)     */
	double *SeedSTD_BC, *SeedSTD_RD;
	int_t idx_indx, idx_lusup;
	int_t nbrow;
	int_t ik, il, lk, rel, knsupc, idx_r;
	int_t lptr1_tmp, idx_i, idx_v, m, uu;
	int_t nub;
	int tag;

#if (PRNTlevel >= 1)
	int_t nLblocks = 0, nUblocks = 0;
#endif
#if (PROFlevel >= 1)
	double t, t_u, t_l;
	int_t u_blks;
#endif

	/* Initialization. */
	iam = grid->iam;
	myrow = MYROW(iam, grid);
	mycol = MYCOL(iam, grid);
	for (i = 0; i < NBUFFERS; ++i)
		mybufmax[i] = 0;
	nsupers = supno[n - 1] + 1;
	Astore = (NRformat_loc *)A->Store;

	// #if ( PRNTlevel>=1 )
	iword = sizeof(int_t);
	dword = sizeof(double);
	// #endif

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Enter pddistribute_allgrid()");
#endif
#if (PROFlevel >= 1)
	t = SuperLU_timer_();
#endif

	dReDistribute_A(A, ScalePermstruct, Glu_freeable, xsup, supno,
					grid, &xa, &asub, &a);

#if (PROFlevel >= 1)
	t = SuperLU_timer_() - t;
	if (!iam)
		printf("--------\n"
			   ".. Phase 1 - ReDistribute_A time: %.2f\t\n",
			   t);
#endif

	if (options->Fact == SamePattern_SameRowPerm)
	{

#if (PROFlevel >= 1)
		t_l = t_u = 0;
		u_blks = 0;
#endif
		/* We can propagate the new values of A into the existing
		   L and U data structures.            */
		ilsum = Llu->ilsum;
		ldaspa = Llu->ldalsum;
		if (!(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options))))
			ABORT("Calloc fails for SPA dense[].");
		nrbu = CEILING(nsupers, grid->nprow); /* No. of local block rows */
		if (!(Urb_length = intCalloc_dist(nrbu)))
			ABORT("Calloc fails for Urb_length[].");
		if (!(Urb_indptr = intMalloc_dist(nrbu)))
			ABORT("Malloc fails for Urb_indptr[].");
		Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
		Lindval_loc_bc_ptr = Llu->Lindval_loc_bc_ptr;
		Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
		Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
		Unzval_br_ptr = Llu->Unzval_br_ptr;
		Unnz = Llu->Unnz;

		mem_use += 2.0 * nrbu * iword + ldaspa * sp_ienv_dist(3, options) * dword;

#if (PROFlevel >= 1)
		t = SuperLU_timer_();
#endif

		/* Initialize Uval to zero. */
		for (lb = 0; lb < nrbu; ++lb)
		{
			Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
			index = Ufstnz_br_ptr[lb];
			if (index)
			{
				uval = Unzval_br_ptr[lb];
				len = index[1];
				for (i = 0; i < len; ++i)
					uval[i] = zero;
			} /* if index != NULL */
		} /* for lb ... */

		for (jb = 0; jb < nsupers; ++jb)
		{ /* Loop through each block column */
			pc = PCOL(jb, grid);
			if (mycol == pc)
			{ /* Block column jb in my process column */
				fsupc = FstBlockC(jb);
				nsupc = SuperSize(jb);

				/* Scatter A into SPA (for L), or into U directly. */
				for (j = fsupc, dense_col = dense; j < FstBlockC(jb + 1); ++j)
				{
					for (i = xa[j]; i < xa[j + 1]; ++i)
					{
						irow = asub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							if (gb < jb)
							{ /* in U */
								index = Ufstnz_br_ptr[lb];
								uval = Unzval_br_ptr[lb];
								if (index)
								{
									while ((k = index[Urb_indptr[lb]]) < jb)
									{
										/* Skip nonzero values in this block */
										Urb_length[lb] += index[Urb_indptr[lb] + 1];
										/* Move pointer to the next block */
										Urb_indptr[lb] += UB_DESCRIPTOR + SuperSize(k);
									}
									/*assert(k == jb);*/
									/* start fstnz */
									istart = Urb_indptr[lb] + UB_DESCRIPTOR;
									len = Urb_length[lb];
									fsupc1 = FstBlockC(gb + 1);
									k = j - fsupc;
									/* Sum the lengths of the leading columns */
									for (jj = 0; jj < k; ++jj)
										len += fsupc1 - index[istart++];
									/*assert(irow>=index[istart]);*/
									uval[len + irow - index[istart]] = a[i];
								}
							}
							else
							{ /* in L; put in SPA first */
								irow = ilsum[lb] + irow - FstBlockC(gb);
								dense_col[irow] = a[i];
							}
						}
					} /* for i ... */
					dense_col += ldaspa;
				} /* for j ... */

#if (PROFlevel >= 1)
				t_u += SuperLU_timer_() - t;
				t = SuperLU_timer_();
#endif

				/* Gather the values of A from SPA into Lnzval[]. */
				ljb = LBj(jb, grid); /* Local block number */
				index = Lrowind_bc_ptr[ljb];
				if (index)
				{
					nrbl = index[0]; /* Number of row blocks. */
					len = index[1];	 /* LDA of lusup[]. */
					lusup = Lnzval_bc_ptr[ljb];
					next_lind = BC_HEADER;
					next_lval = 0;
					for (jj = 0; jj < nrbl; ++jj)
					{
						gb = index[next_lind++];
						len1 = index[next_lind++]; /* Rows in the block. */
						lb = LBi(gb, grid);
						for (bnnz = 0; bnnz < len1; ++bnnz)
						{
							irow = index[next_lind++]; /* Global index. */
							irow = ilsum[lb] + irow - FstBlockC(gb);
							k = next_lval++;
							for (j = 0, dense_col = dense; j < nsupc; ++j)
							{
								lusup[k] = dense_col[irow];
								dense_col[irow] = zero;
								k += len;
								dense_col += ldaspa;
							}
						} /* for bnnz ... */
					} /* for jj ... */
				} /* if index ... */
#if (PROFlevel >= 1)
				t_l += SuperLU_timer_() - t;
#endif
			} /* if mycol == pc */
		} /* for jb ... */

		SUPERLU_FREE(dense);
		SUPERLU_FREE(Urb_length);
		SUPERLU_FREE(Urb_indptr);
		mem_use -= 2.0 * nrbu * iword + ldaspa * sp_ienv_dist(3, options) * dword;

#if (PROFlevel >= 1)
		if (!iam)
			printf(".. 2nd distribute time: L %.2f\tU %.2f\tu_blks %d\tnrbu %d\n",
				   t_l, t_u, u_blks, nrbu);
#endif
	}
	else
	{	/* options->Fact is not SamePattern_SameRowPerm */
		/* ------------------------------------------------------------
	   FIRST TIME CREATING THE L AND U DATA STRUCTURES.
	   ------------------------------------------------------------*/

#if (PROFlevel >= 1)
		t_l = t_u = 0;
		u_blks = 0;
#endif
		/* We first need to set up the L and U data structures and then
		 * propagate the values of A into them.
		 */
		lsub = Glu_freeable->lsub; /* compressed L subscripts */
		xlsub = Glu_freeable->xlsub;
		usub = Glu_freeable->usub; /* compressed U subscripts */
		xusub = Glu_freeable->xusub;

		if (!(ToRecv = (int *)SUPERLU_MALLOC(nsupers * sizeof(int))))
			ABORT("Malloc fails for ToRecv[].");
		for (i = 0; i < nsupers; ++i)
			ToRecv[i] = 0;

		k = CEILING(nsupers, grid->npcol); /* Number of local column blocks */
		if (!(ToSendR = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for ToSendR[].");
		j = k * grid->npcol;
		if (!(index1 = SUPERLU_MALLOC(j * sizeof(int))))
			ABORT("Malloc fails for index[].");

		mem_use += (float)k * sizeof(int_t *) + (j + nsupers) * iword;

		for (i = 0; i < j; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->npcol)
			ToSendR[i] = &index1[j];
		k = CEILING(nsupers, grid->nprow); /* Number of local block rows */

		/* Pointers to the beginning of each block row of U. */
		if (!(Unzval_br_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
			ABORT("Malloc fails for Unzval_br_ptr[].");
		if (!(Ufstnz_br_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Ufstnz_br_ptr[].");
		if (!(ToSendD = SUPERLU_MALLOC(k * sizeof(int))))
			ABORT("Malloc fails for ToSendD[].");
		for (i = 0; i < k; ++i)
			ToSendD[i] = NO;
		if (!(ilsum = intMalloc_dist(k + 1)))
			ABORT("Malloc fails for ilsum[].");

		/* Auxiliary arrays used to set up U block data structures.
		   They are freed on return. */
		if (!(rb_marker = intCalloc_dist(k)))
			ABORT("Calloc fails for rb_marker[].");
		if (!(Urb_length = intCalloc_dist(k)))
			ABORT("Calloc fails for Urb_length[].");
		if (!(Urb_indptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Urb_indptr[].");
		if (!(Urb_fstnz = intCalloc_dist(k)))
			ABORT("Calloc fails for Urb_fstnz[].");
		if (!(Ucbs = intCalloc_dist(k)))
			ABORT("Calloc fails for Ucbs[].");

		mem_use += 2.0 * k * sizeof(int_t *) + (7 * k + 1) * iword;

		/* Compute ldaspa and ilsum[]. */
		ldaspa = 0;
		ilsum[0] = 0;
		for (gb = 0; gb < nsupers; ++gb)
		{
			if (myrow == PROW(gb, grid))
			{
				i = SuperSize(gb);
				ldaspa += i;
				lb = LBi(gb, grid);
				ilsum[lb + 1] = ilsum[lb] + i;
			}
		}

#if (PROFlevel >= 1)
		t = SuperLU_timer_();
#endif
		/* ------------------------------------------------------------
		   COUNT NUMBER OF ROW BLOCKS AND THE LENGTH OF EACH BLOCK IN U.
		   THIS ACCOUNTS FOR ONE-PASS PROCESSING OF G(U).
		   ------------------------------------------------------------*/

		/* Loop through each supernode column. */
		for (jb = 0; jb < nsupers; ++jb)
		{
			pc = PCOL(jb, grid);
			fsupc = FstBlockC(jb);
			nsupc = SuperSize(jb);
			/* Loop through each column in the block. */
			for (j = fsupc; j < fsupc + nsupc; ++j)
			{
				/* usub[*] contains only "first nonzero" in each segment. */
				for (i = xusub[j]; i < xusub[j + 1]; ++i)
				{
					irow = usub[i]; /* First nonzero of the segment. */
					gb = BlockNum(irow);
					kcol = PCOL(gb, grid);
					ljb = LBj(gb, grid);
					if (mycol == kcol && mycol != pc)
						ToSendR[ljb][pc] = YES;
					pr = PROW(gb, grid);
					lb = LBi(gb, grid);
					if (mycol == pc)
					{
						if (myrow == pr)
						{
							ToSendD[lb] = YES;
							/* Count nonzeros in entire block row. */
							Urb_length[lb] += FstBlockC(gb + 1) - irow;
							if (rb_marker[lb] <= jb)
							{ /* First see the block */
								rb_marker[lb] = jb + 1;
								Urb_fstnz[lb] += nsupc;
								++Ucbs[lb]; /* Number of column blocks
										   in block row lb. */
#if (PRNTlevel >= 1)
								++nUblocks;
#endif
							}
							ToRecv[gb] = 1;
						}
						else
							ToRecv[gb] = 2; /* Do I need 0, 1, 2 ? */
					}
				} /* for i ... */
			} /* for j ... */
		} /* for jb ... */

		/* Set up the initial pointers for each block row in U. */
		nrbu = CEILING(nsupers, grid->nprow); /* Number of local block rows */
		for (lb = 0; lb < nrbu; ++lb)
		{
			ib = myrow + lb * grid->nprow; /* not sure */
			len = Urb_length[lb];
			rb_marker[lb] = 0; /* Reset block marker. */
			if (len)
			{
				/* Add room for descriptors */
				len1 = Urb_fstnz[lb] + BR_HEADER + Ucbs[lb] * UB_DESCRIPTOR;
				mybufmax[2] = SUPERLU_MAX(mybufmax[2], len1);
				mybufmax[3] = SUPERLU_MAX(mybufmax[3], len);

				if (supernodeMask[ib] > 0)
				{ // YL: added supernode mask here
					if (!(index = intMalloc_dist(len1 + 1)))
						ABORT("Malloc fails for Uindex[].");
					Ufstnz_br_ptr[lb] = index;
					if (!(Unzval_br_ptr[lb] = doubleMalloc_dist(len)))
						ABORT("Malloc fails for Unzval_br_ptr[*][].");
					mem_use += len * dword + (len1 + 1) * iword;
					index[0] = Ucbs[lb]; /* Number of column blocks */
					index[1] = len;		 /* Total length of nzval[] */
					index[2] = len1;	 /* Total length of index[] */
					index[len1] = -1;	 /* End marker */
				}
				else
				{
					Ufstnz_br_ptr[lb] = NULL;
					Unzval_br_ptr[lb] = NULL;
				}
			}
			else
			{
				Ufstnz_br_ptr[lb] = NULL;
				Unzval_br_ptr[lb] = NULL;
			}
			Urb_length[lb] = 0;			/* Reset block length. */
			Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
			Urb_fstnz[lb] = BR_HEADER;
		} /* for lb ... */

		SUPERLU_FREE(Ucbs);

#if (PROFlevel >= 1)
		t = SuperLU_timer_() - t;
		if (!iam)
			printf(".. Phase 2 - setup U strut time: %.2f\t\n", t);
#endif

		mem_use -= 2.0 * k * iword;

		/* Auxiliary arrays used to set up L block data structures.
		   They are freed on return.
		   k is the number of local row blocks.   */
		if (!(Lrb_length = intCalloc_dist(k)))
			ABORT("Calloc fails for Lrb_length[].");
		if (!(Lrb_number = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_number[].");
		if (!(Lrb_indptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_indptr[].");
		if (!(Lrb_valptr = intMalloc_dist(k)))
			ABORT("Malloc fails for Lrb_valptr[].");
		if (!(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options))))
			ABORT("Calloc fails for SPA dense[].");

		/* These counts will be used for triangular solves. */
		if (!(fmod = int32Calloc_dist(k)))
			ABORT("Calloc fails for fmod[].");
		if (!(bmod = int32Calloc_dist(k)))
			ABORT("Calloc fails for bmod[].");

		/* ------------------------------------------------ */
		mem_use += 6.0 * k * iword + ldaspa * sp_ienv_dist(3, options) * dword;

		k = CEILING(nsupers, grid->npcol); /* Number of local block columns */

		/* Pointers to the beginning of each block column of L. */
		if (!(Lnzval_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
			ABORT("Malloc fails for Lnzval_bc_ptr[].");
		Lnzval_bc_ptr[k - 1] = NULL;
		if (!(Lrowind_bc_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Lrowind_bc_ptr[].");
		Lrowind_bc_ptr[k - 1] = NULL;

		if (!(Lindval_loc_bc_ptr =
				  (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
			ABORT("Malloc fails for Lindval_loc_bc_ptr[].");
		Lindval_loc_bc_ptr[k - 1] = NULL;
		if (!(Linv_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
		{
			fprintf(stderr, "Malloc fails for Linv_bc_ptr[].");
		}
		if (!(Uinv_bc_ptr =
				  (double **)SUPERLU_MALLOC(k * sizeof(double *))))
		{
			fprintf(stderr, "Malloc fails for Uinv_bc_ptr[].");
		}
		Linv_bc_ptr[k - 1] = NULL;
		Uinv_bc_ptr[k - 1] = NULL;
		if (!(Unnz =
				  (int_t *)SUPERLU_MALLOC(k * sizeof(int_t))))
			ABORT("Malloc fails for Unnz[].");

		/* These lists of processes will be used for triangular solves. */
		if (!(fsendx_plist = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for fsendx_plist[].");
		len = k * grid->nprow;
		if (!(index1 = int32Malloc_dist(len)))
			ABORT("Malloc fails for fsendx_plist[0]");
		for (i = 0; i < len; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
			fsendx_plist[i] = &index1[j];
		if (!(bsendx_plist = (int **)SUPERLU_MALLOC(k * sizeof(int *))))
			ABORT("Malloc fails for bsendx_plist[].");
		if (!(index1 = int32Malloc_dist(len)))
			ABORT("Malloc fails for bsendx_plist[0]");
		for (i = 0; i < len; ++i)
			index1[i] = SLU_EMPTY;
		for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
			bsendx_plist[i] = &index1[j];
		/* -------------------------------------------------------------- */
		mem_use += 4.0 * k * sizeof(int_t *) + 2.0 * len * iword;
		memTRS += k * sizeof(int_t *) + 2.0 * k * sizeof(double *) + k * iword; // acount for Lindval_loc_bc_ptr, Unnz, Linv_bc_ptr,Uinv_bc_ptr

		/*------------------------------------------------------------
		  PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
		  THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U.
		  ------------------------------------------------------------*/
		long int Linv_bc_cnt = 0;
		long int Uinv_bc_cnt = 0;
		long int Lrowind_bc_cnt = 0;
		long int Lnzval_bc_cnt = 0;
		long int Lindval_loc_bc_cnt = 0;

		for (jb = 0; jb < nsupers; ++jb)
		{ /* for each block column ... */
			pc = PCOL(jb, grid);
			if (mycol == pc)
			{ /* Block column jb in my process column */
				fsupc = FstBlockC(jb);
				nsupc = SuperSize(jb);
				ljb = LBj(jb, grid); /* Local block number */

				/* Scatter A into SPA. */
				for (j = fsupc, dense_col = dense; j < FstBlockC(jb + 1); ++j)
				{
					for (i = xa[j]; i < xa[j + 1]; ++i)
					{
						irow = asub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							irow = ilsum[lb] + irow - FstBlockC(gb);
							dense_col[irow] = a[i];
						}
					}
					dense_col += ldaspa;
				} /* for j ... */

				jbrow = PROW(jb, grid);

				/*------------------------------------------------
				 * SET UP U BLOCKS.
				 *------------------------------------------------*/
#if (PROFlevel >= 1)
				t = SuperLU_timer_();
#endif
				kseen = 0;
				dense_col = dense;
				/* Loop through each column in the block column. */
				for (j = fsupc; j < FstBlockC(jb + 1); ++j)
				{
					istart = xusub[j];
					/* NOTE: Only the first nonzero index of the segment
					   is stored in usub[]. */
					for (i = istart; i < xusub[j + 1]; ++i)
					{
						irow = usub[i]; /* First nonzero in the segment. */
						gb = BlockNum(irow);
						pr = PROW(gb, grid);
						if (pr != jbrow &&
							myrow == jbrow && /* diag. proc. owning jb */
							bsendx_plist[ljb][pr] == SLU_EMPTY)
						{
							bsendx_plist[ljb][pr] = YES;
							++nbsendx;
						}
						if (myrow == pr)
						{ // YL: added supernode mask here, TODO: double check bmod
							if (supernodeMask[gb] > 0)
							{
								lb = LBi(gb, grid); /* Local block number */
								index = Ufstnz_br_ptr[lb];
								uval = Unzval_br_ptr[lb];
								fsupc1 = FstBlockC(gb + 1);
								if (rb_marker[lb] <= jb)
								{ /* First time see
				   the block       */
									rb_marker[lb] = jb + 1;
									Urb_indptr[lb] = Urb_fstnz[lb];
									;
									index[Urb_indptr[lb]] = jb; /* Descriptor */
									Urb_indptr[lb] += UB_DESCRIPTOR;
									/* Record the first location in index[] of the
									next block */
									Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;
									len = Urb_indptr[lb]; /* Start fstnz in index */
									index[len - 1] = 0;
									for (k = 0; k < nsupc; ++k)
										index[len + k] = fsupc1;
									if (gb != jb)	/* Exclude diagonal block. */
										++bmod[lb]; /* Mod. count for back solve */
									if (kseen == 0 && myrow != jbrow)
									{
										++nbrecvx;
										kseen = 1;
									}
								}
								else
								{						  /* Already saw the block */
									len = Urb_indptr[lb]; /* Start fstnz in index */
								}
								jj = j - fsupc;
								index[len + jj] = irow;
								/* Load the numerical values */
								k = fsupc1 - irow;	 /* No. of nonzeros in segment */
								index[len - 1] += k; /* Increment block length in
										  Descriptor */
								irow = ilsum[lb] + irow - FstBlockC(gb);
								for (ii = 0; ii < k; ++ii)
								{
									uval[Urb_length[lb]++] = dense_col[irow + ii];
									dense_col[irow + ii] = zero;
								}
							}
							else
							{
								lb = LBi(gb, grid); /* Local block number */
								uval = Unzval_br_ptr[lb];
								fsupc1 = FstBlockC(gb + 1);
								if (rb_marker[lb] <= jb)
								{ /* First time see
				   the block       */
									rb_marker[lb] = jb + 1;
									Urb_indptr[lb] = Urb_fstnz[lb];
									;
									Urb_indptr[lb] += UB_DESCRIPTOR;
									/* Record the first location in index[] of the
									next block */
									Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;

									if (gb != jb)	/* Exclude diagonal block. */
										++bmod[lb]; /* Mod. count for back solve */
									if (kseen == 0 && myrow != jbrow)
									{
										++nbrecvx;
										kseen = 1;
									}
								}
							}

						} /* if myrow == pr ... */
					} /* for i ... */
					dense_col += ldaspa;
				} /* for j ... */

#if (PROFlevel >= 1)
				t_u += SuperLU_timer_() - t;
				t = SuperLU_timer_();
#endif
				/*------------------------------------------------
				 * SET UP L BLOCKS.
				 *------------------------------------------------*/

				/* Count number of blocks and length of each block. */
				nrbl = 0;
				len = 0; /* Number of row subscripts I own. */
				kseen = 0;
				istart = xlsub[fsupc];
				for (i = istart; i < xlsub[fsupc + 1]; ++i)
				{
					irow = lsub[i];
					gb = BlockNum(irow); /* Global block number */
					pr = PROW(gb, grid); /* Process row owning this block */
					if (pr != jbrow &&
						myrow == jbrow && /* diag. proc. owning jb */
						fsendx_plist[ljb][pr] == SLU_EMPTY /* first time */)
					{
						fsendx_plist[ljb][pr] = YES;
						++nfsendx;
					}
					if (myrow == pr)
					{
						lb = LBi(gb, grid); /* Local block number */
						if (rb_marker[lb] <= jb)
						{ /* First see this block */
							rb_marker[lb] = jb + 1;
							Lrb_length[lb] = 1;
							Lrb_number[nrbl++] = gb;
							if (gb != jb)	/* Exclude diagonal block. */
								++fmod[lb]; /* Mod. count for forward solve */
							if (kseen == 0 && myrow != jbrow)
							{
								++nfrecvx;
								kseen = 1;
							}
#if (PRNTlevel >= 1)
							++nLblocks;
#endif
						}
						else
						{
							++Lrb_length[lb];
						}
						++len;
					}
				} /* for i ... */

				if (nrbl)
				{ /* Do not ensure the blocks are sorted! */
					if (supernodeMask[jb] > 0)
					{ // YL: added supernode mask here
						/* Set up the initial pointers for each block in
						   index[] and nzval[]. */
						/* Add room for descriptors */
						len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
						if (!(index = intMalloc_dist(len1)))
							ABORT("Malloc fails for index[]");
						if (!(lusup = (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double))))
							ABORT("Malloc fails for lusup[]");
						if (!(Lindval_loc_bc_ptr[ljb] = intCalloc_dist(nrbl * 3)))
							ABORT("Malloc fails for Lindval_loc_bc_ptr[ljb][]");
						myrow = MYROW(iam, grid);
						krow = PROW(jb, grid);
						if (myrow == krow)
						{ /* diagonal block */
							if (!(Linv_bc_ptr[ljb] = (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double))))
								ABORT("Malloc fails for Linv_bc_ptr[ljb][]");
							if (!(Uinv_bc_ptr[ljb] = (double *)SUPERLU_MALLOC(nsupc * nsupc * sizeof(double))))
								ABORT("Malloc fails for Uinv_bc_ptr[ljb][]");
						}
						else
						{
							Linv_bc_ptr[ljb] = NULL;
							Uinv_bc_ptr[ljb] = NULL;
						}

						mybufmax[0] = SUPERLU_MAX(mybufmax[0], len1);
						mybufmax[1] = SUPERLU_MAX(mybufmax[1], len * nsupc);
						mybufmax[4] = SUPERLU_MAX(mybufmax[4], len);
						mem_use += len * nsupc * dword + (len1)*iword;
						memTRS += nrbl * 3.0 * iword + 2.0 * nsupc * nsupc * dword; // acount for Lindval_loc_bc_ptr[ljb],Linv_bc_ptr[ljb],Uinv_bc_ptr[ljb]
						index[0] = nrbl;											/* Number of row blocks */
						index[1] = len;												/* LDA of the nzval[] */
						next_lind = BC_HEADER;
						next_lval = 0;
						for (k = 0; k < nrbl; ++k)
						{
							gb = Lrb_number[k];
							lb = LBi(gb, grid);
							len = Lrb_length[lb];
							Lindval_loc_bc_ptr[ljb][k] = lb;
							Lindval_loc_bc_ptr[ljb][k + nrbl] = next_lind;
							Lindval_loc_bc_ptr[ljb][k + nrbl * 2] = next_lval;
							Lrb_length[lb] = 0;		 /* Reset vector of block length */
							index[next_lind++] = gb; /* Descriptor */
							index[next_lind++] = len;
							Lrb_indptr[lb] = next_lind;
							Lrb_valptr[lb] = next_lval;
							next_lind += len;
							next_lval += len;
						}
						/* Propagate the compressed row subscripts to Lindex[],
								   and the initial values of A from SPA into Lnzval[]. */
						len = index[1]; /* LDA of lusup[] */
						for (i = istart; i < xlsub[fsupc + 1]; ++i)
						{
							irow = lsub[i];
							gb = BlockNum(irow);
							if (myrow == PROW(gb, grid))
							{
								lb = LBi(gb, grid);
								k = Lrb_indptr[lb]++; /* Random access a block */
								index[k] = irow;
								k = Lrb_valptr[lb]++;
								irow = ilsum[lb] + irow - FstBlockC(gb);
								for (j = 0, dense_col = dense; j < nsupc; ++j)
								{
									lusup[k] = dense_col[irow];
									dense_col[irow] = 0.0;
									k += len;
									dense_col += ldaspa;
								}
							}
						} /* for i ... */

						Lrowind_bc_ptr[ljb] = index;
						Lnzval_bc_ptr[ljb] = lusup;

						/* sort Lindval_loc_bc_ptr[ljb], Lrowind_bc_ptr[ljb]
									   and Lnzval_bc_ptr[ljb] here.  */
						if (nrbl > 1)
						{
							krow = PROW(jb, grid);
							if (myrow == krow)
							{ /* skip the diagonal block */
								uu = nrbl - 2;
								lloc = &Lindval_loc_bc_ptr[ljb][1];
							}
							else
							{
								uu = nrbl - 1;
								lloc = Lindval_loc_bc_ptr[ljb];
							}
							quickSortM(lloc, 0, uu, nrbl, 0, 3);
						}

						if (!(index_srt = intMalloc_dist(len1)))
							ABORT("Malloc fails for index_srt[]");
						if (!(lusup_srt = (double *)SUPERLU_MALLOC(len * nsupc * sizeof(double))))
							ABORT("Malloc fails for lusup_srt[]");

						idx_indx = BC_HEADER;
						idx_lusup = 0;
						for (jj = 0; jj < BC_HEADER; jj++)
							index_srt[jj] = index[jj];

						for (i = 0; i < nrbl; i++)
						{
							nbrow = index[Lindval_loc_bc_ptr[ljb][i + nrbl] + 1];
							for (jj = 0; jj < LB_DESCRIPTOR + nbrow; jj++)
							{
								index_srt[idx_indx++] = index[Lindval_loc_bc_ptr[ljb][i + nrbl] + jj];
							}

							Lindval_loc_bc_ptr[ljb][i + nrbl] = idx_indx - LB_DESCRIPTOR - nbrow;

							for (jj = 0; jj < nbrow; jj++)
							{
								k = idx_lusup;
								k1 = Lindval_loc_bc_ptr[ljb][i + nrbl * 2] + jj;
								for (j = 0; j < nsupc; ++j)
								{
									lusup_srt[k] = lusup[k1];
									k += len;
									k1 += len;
								}
								idx_lusup++;
							}
							Lindval_loc_bc_ptr[ljb][i + nrbl * 2] = idx_lusup - nbrow;
						}

						SUPERLU_FREE(lusup);
						SUPERLU_FREE(index);

						Lrowind_bc_ptr[ljb] = index_srt;
						Lnzval_bc_ptr[ljb] = lusup_srt;
					}
					else
					{ // if(supernodeMask[jb]==0)

						/* Set up the initial pointers for each block in
						index[] and nzval[]. */
						/* Add room for descriptors */
						len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;

						myrow = MYROW(iam, grid);
						krow = PROW(jb, grid);

						mybufmax[0] = SUPERLU_MAX(mybufmax[0], len1);
						mybufmax[1] = SUPERLU_MAX(mybufmax[1], len * nsupc);
						mybufmax[4] = SUPERLU_MAX(mybufmax[4], len);

						/* YL: need to zero out dense_col even if supernodeMask[jb]=0 for this column. */
						for (i = istart; i < xlsub[fsupc + 1]; ++i)
						{
							irow = lsub[i];
							gb = BlockNum(irow);
							if (myrow == PROW(gb, grid))
							{
								lb = LBi(gb, grid);
								irow = ilsum[lb] + irow - FstBlockC(gb);
								for (j = 0, dense_col = dense; j < nsupc; ++j)
								{
									dense_col[irow] = zero;
									dense_col += ldaspa;
								}
							}
						} /* for i ... */

						Lrowind_bc_ptr[ljb] = NULL;
						Lnzval_bc_ptr[ljb] = NULL;
						Linv_bc_ptr[ljb] = NULL;
						Uinv_bc_ptr[ljb] = NULL;
						Lindval_loc_bc_ptr[ljb] = NULL;
					}
				}
				else
				{
					Lrowind_bc_ptr[ljb] = NULL;
					Lnzval_bc_ptr[ljb] = NULL;
					Linv_bc_ptr[ljb] = NULL;
					Uinv_bc_ptr[ljb] = NULL;
					Lindval_loc_bc_ptr[ljb] = NULL;
				} /* if nrbl ... */
#if (PROFlevel >= 1)
				t_l += SuperLU_timer_() - t;
#endif
			} /* if mycol == pc */

		} /* for jb ... */

		/////////////////////////////////////////////////////////////////

		/* Set up additional pointers for the index and value arrays of U.
		   nub is the number of local block columns. */
		nub = CEILING(nsupers, grid->npcol); /* Number of local block columns. */
		if (!(Urbs = (int_t *)intCalloc_dist(2 * nub)))
			ABORT("Malloc fails for Urbs[]"); /* Record number of nonzero
								 blocks in a block column. */
		Urbs1 = Urbs + nub;
		if (!(Ucb_indptr = SUPERLU_MALLOC(nub * sizeof(Ucb_indptr_t *))))
			ABORT("Malloc fails for Ucb_indptr[]");
		if (!(Ucb_valptr = SUPERLU_MALLOC(nub * sizeof(int_t *))))
			ABORT("Malloc fails for Ucb_valptr[]");
		mem_use += nub * sizeof(Ucb_indptr_t *) + nub * sizeof(int_t *) + (2 * nub) * iword;

		nlb = CEILING(nsupers, grid->nprow); /* Number of local block rows. */

		/* Count number of row blocks in a block column.
		   One pass of the skeleton graph of U. */
		for (lk = 0; lk < nlb; ++lk)
		{
			usub1 = Ufstnz_br_ptr[lk];
			// YL: no need to supernode mask here ????
			if (usub1)
			{ /* Not an empty block row. */
				/* usub1[0] -- number of column blocks in this block row. */
				i = BR_HEADER; /* Pointer in index array. */
				for (lb = 0; lb < usub1[0]; ++lb)
				{				  /* For all column blocks. */
					k = usub1[i]; /* Global block number */
					++Urbs[LBj(k, grid)];
					i += UB_DESCRIPTOR + SuperSize(k);
				}
			}
		}

		/* Set up the vertical linked lists for the row blocks.
		   One pass of the skeleton graph of U. */
		for (lb = 0; lb < nub; ++lb)
		{
			// YL: no need to add supernode mask here ????
			if (Urbs[lb])
			{ /* Not an empty block column. */
				if (!(Ucb_indptr[lb] = SUPERLU_MALLOC(Urbs[lb] * sizeof(Ucb_indptr_t))))
					ABORT("Malloc fails for Ucb_indptr[lb][]");
				if (!(Ucb_valptr[lb] = (int_t *)intMalloc_dist(Urbs[lb])))
					ABORT("Malloc fails for Ucb_valptr[lb][]");
				mem_use += Urbs[lb] * sizeof(Ucb_indptr_t) + (Urbs[lb]) * iword;
			}
			else
			{
				Ucb_valptr[lb] = NULL;
				Ucb_indptr[lb] = NULL;
			}
		}
		for (lk = 0; lk < nlb; ++lk)
		{ /* For each block row. */
			usub1 = Ufstnz_br_ptr[lk];
			// YL: no need to add supernode mask here ????
			if (usub1)
			{				   /* Not an empty block row. */
				i = BR_HEADER; /* Pointer in index array. */
				j = 0;		   /* Pointer in nzval array. */

				for (lb = 0; lb < usub1[0]; ++lb)
				{						/* For all column blocks. */
					k = usub1[i];		/* Global block number, column-wise. */
					ljb = LBj(k, grid); /* Local block number, column-wise. */
					Ucb_indptr[ljb][Urbs1[ljb]].lbnum = lk;

					Ucb_indptr[ljb][Urbs1[ljb]].indpos = i;
					Ucb_valptr[ljb][Urbs1[ljb]] = j;

					++Urbs1[ljb];
					j += usub1[i + 1];
					i += UB_DESCRIPTOR + SuperSize(k);
				}
			}
		}

		/* Count the nnzs per block column */
		for (lb = 0; lb < nub; ++lb)
		{
			Unnz[lb] = 0;
			k = lb * grid->npcol + mycol; /* Global block number, column-wise. */
			knsupc = SuperSize(k);
			for (ub = 0; ub < Urbs[lb]; ++ub)
			{
				ik = Ucb_indptr[lb][ub].lbnum; /* Local block number, row-wise. */
				i = Ucb_indptr[lb][ub].indpos; /* Start of the block in usub[]. */
				i += UB_DESCRIPTOR;
				gik = ik * grid->nprow + myrow; /* Global block number, row-wise. */
				iklrow = FstBlockC(gik + 1);
				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = Ufstnz_br_ptr[ik][i + jj];
					if (fnz < iklrow)
					{
						Unnz[lb] += iklrow - fnz;
					}
				} /* for jj ... */
			}
		}

		// for (int lb = 0; lb < nub; ++lb) {
		// 	printf("ID %5d lb %5d, supernodeMask[lb] %5d, Unnz[lb] %5d\n",supernodeMask[0],lb, supernodeMask[lb], Unnz[lb]);
		// }

		Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
		Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
		Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
		Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
		Llu->Unzval_br_ptr = Unzval_br_ptr;
		Llu->Unnz = Unnz;
		Llu->ToRecv = ToRecv;
		Llu->ToSendD = ToSendD;
		Llu->ToSendR = ToSendR;
		Llu->fmod = fmod;
		Llu->fsendx_plist = fsendx_plist;
		Llu->nfrecvx = nfrecvx;
		Llu->nfsendx = nfsendx;
		Llu->bmod = bmod;
		Llu->bsendx_plist = bsendx_plist;
		Llu->nbrecvx = nbrecvx;
		Llu->nbsendx = nbsendx;
		Llu->ilsum = ilsum;
		Llu->ldalsum = ldaspa;
		Llu->Linv_bc_ptr = Linv_bc_ptr;
		Llu->Uinv_bc_ptr = Uinv_bc_ptr;
		Llu->Urbs = Urbs;
		Llu->Ucb_indptr = Ucb_indptr;
		Llu->Ucb_valptr = Ucb_valptr;

#if (PRNTlevel >= 1)
		if (!iam)
			printf(".. # L blocks " IFMT "\t# U blocks " IFMT "\n",
				   nLblocks, nUblocks);
#endif

		SUPERLU_FREE(rb_marker);
		SUPERLU_FREE(Urb_fstnz);
		SUPERLU_FREE(Urb_length);
		SUPERLU_FREE(Urb_indptr);
		SUPERLU_FREE(Lrb_length);
		SUPERLU_FREE(Lrb_number);
		SUPERLU_FREE(Lrb_indptr);
		SUPERLU_FREE(Lrb_valptr);
		SUPERLU_FREE(dense);

		k = CEILING(nsupers, grid->nprow); /* Number of local block rows */
		mem_use -= (k * 8) * iword + ldaspa * sp_ienv_dist(3, options) * dword;

		/* Find the maximum buffer size. */
		MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t,
					  MPI_MAX, grid->comm);

		k = CEILING(nsupers, grid->nprow); /* Number of local block rows */
		if (!(Llu->mod_bit = int32Malloc_dist(k)))
			ABORT("Malloc fails for mod_bit[].");

#if (PROFlevel >= 1)
		if (!iam)
			printf(".. 1st distribute time:\n "
				   "\tL\t%.2f\n\tU\t%.2f\n"
				   "\tu_blks %d\tnrbu %d\n--------\n",
				   t_l, t_u, u_blks, nrbu);
#endif

	} /* else fact != SamePattern_SameRowPerm */

	if (xa[A->ncol] > 0)
	{ /* may not have any entries on this process. */
		SUPERLU_FREE(asub);
		SUPERLU_FREE(a);
	}
	SUPERLU_FREE(xa);

#if (DEBUGlevel >= 1)
	/* Memory allocated but not freed:
	   ilsum, fmod, fsendx_plist, bmod, bsendx_plist  */
	CHECK_MALLOC(iam, "Exit pddistribute_allgrid()");
#endif

	return (mem_use + memTRS);

} /* PDDISTRIBUTE_ALLGRID */

/*
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 *
 * Purpose
 * =======
 *   Distribute the matrix onto the 2D process mesh on all girds based on supernodeMask
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t*
 *        options->Fact specifies whether or not the L and U structures will be re-used.
 *        = SamePattern_SameRowPerm: L and U structures are input, and
 *                                   unchanged on exit.
 *        = DOFACT or SamePattern: L and U structures are computed and output.
 *
 * n      (input) int
 *        Dimension of the matrix.
 *
 * A      (input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T. The type of A can be:
 *        Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * LUstruct (input/output) dLUstruct_t*
 *        Data structures for L and U factors.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   > 0, working storage required (in bytes).
 *
 */
float pddistribute_allgrid_index_only(
	superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	dScalePermstruct_t *ScalePermstruct,
	Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
	gridinfo_t *grid, int *supernodeMask)
{
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t bnnz, fsupc, fsupc1, i, ii, irow, istart, j, ib, jb, jj, k, k1,
		len, len1, nsupc, masked;
	int_t lib;	/* local block row number */
	int_t nlb;	/* local block rows*/
	int_t ljb;	/* local block column number */
	int_t nrbl; /* number of L blocks in current block column */
	int_t nrbu; /* number of U blocks in current block column */
	int_t gb;	/* global block number; 0 < gb <= nsuper */
	int_t lb;	/* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
	int_t ub, gik, iklrow, fnz;
	int iam, jbrow, kcol, krow, mycol, myrow, pc, pr;
	int_t mybufmax[NBUFFERS];
	NRformat_loc *Astore;
	double *a;
	int_t *asub, *xa;
	int_t *xa_begin, *xa_end;
	int_t *xsup = Glu_persist->xsup; /* supernode and column mapping */
	int_t *supno = Glu_persist->supno;
	int_t *lsub, *xlsub, *usub, *usub1, *xusub;
	int_t nsupers;
	int_t next_lind;				  /* next available position in index[*] */
	int_t next_lval;				  /* next available position in nzval[*] */
	int_t *index;					  /* indices consist of headers and row subscripts */
	int_t *index_srt;				  /* indices consist of headers and row subscripts */
	int *index1;					  /* temporary pointer to array of int */
	double *lusup, *lusup_srt, *uval; /* nonzero values in L and U */
	double **Lnzval_bc_ptr;			  /* size ceil(NSUPERS/Pc) */
	double *Lnzval_bc_dat;			  /* size: sum of sizes of Lnzval_bc_ptr[lk])  */
	long int *Lnzval_bc_offset;		  /* size ceil(NSUPERS/Pc)                 */

	int_t **Lrowind_bc_ptr;			 /* size ceil(NSUPERS/Pc) */
	int_t *Lrowind_bc_dat;			 /* size: sum of sizes of Lrowind_bc_ptr[lk])   */
	long int *Lrowind_bc_offset;	 /* size ceil(NSUPERS/Pc)                 */
	int_t **Lindval_loc_bc_ptr;		 /* size ceil(NSUPERS/Pc)                 */
	int_t *Lindval_loc_bc_dat;		 /* size: sum of sizes of Lindval_loc_bc_ptr[lk]) */
	long int *Lindval_loc_bc_offset; /* size ceil(NSUPERS/Pc)                 */

	int_t *Unnz;				/* size ceil(NSUPERS/Pc)                 */
	double **Unzval_br_ptr;		/* size ceil(NSUPERS/Pr) */
	double *Unzval_br_dat;		/* size: sum of sizes of Unzval_br_ptr[lk]) */
	long int *Unzval_br_offset; /* size ceil(NSUPERS/Pr)    */
	long int Unzval_br_cnt = 0;
	int_t **Ufstnz_br_ptr;		/* size ceil(NSUPERS/Pr) */
	int_t *Ufstnz_br_dat;		/* size: sum of sizes of Ufstnz_br_ptr[lk]) */
	long int *Ufstnz_br_offset; /* size ceil(NSUPERS/Pr)    */
	long int Ufstnz_br_cnt = 0;

	C_Tree *LBtree_ptr; /* size ceil(NSUPERS/Pc) */
	C_Tree *LRtree_ptr; /* size ceil(NSUPERS/Pr) */
	C_Tree *UBtree_ptr; /* size ceil(NSUPERS/Pc) */
	C_Tree *URtree_ptr; /* size ceil(NSUPERS/Pr) */
	int msgsize;

	int kseen;

	/*-- Counts to be used in upper triangular solve. --*/
	int *bmod;			/* Modification count for U-solve.        */
	int **bsendx_plist; /* Column process list to send down Xk.   */
	int nbrecvx = 0;	/* Number of Xk I will receive.           */
	int nbsendx = 0;	/* Number of Xk I will send               */
	int_t *ilsum;		/* starting position of each supernode in
			   the full array (local)                 */

	/*-- Auxiliary arrays; freed on return --*/
	int_t *rb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
	int_t *Urb_length; /* U block length; size ceil(NSUPERS/Pr)             */
	int_t *Urb_indptr; /* pointers to U index[]; size ceil(NSUPERS/Pr)      */
	int_t *Urb_fstnz;  /* # of fstnz in a block row; size ceil(NSUPERS/Pr)  */
	int_t *Ucbs;	   /* number of column blocks in a block row            */
	int_t *Lrb_length; /* L block length; size ceil(NSUPERS/Pr)             */
	int_t *Lrb_number; /* global block number; size ceil(NSUPERS/Pr)        */
	int_t *Lrb_indptr; /* pointers to L index[]; size ceil(NSUPERS/Pr)      */
	int_t *Lrb_valptr; /* pointers to L nzval[]; size ceil(NSUPERS/Pr)      */
	int_t *ActiveFlag;
	int_t *ActiveFlagAll;
	int_t Iactive;
	int *ranks;
	int_t *idxs;
	int_t **nzrows;
	double rseed;
	int rank_cnt, rank_cnt_ref, Root;
	double *dense, *dense_col; /* SPA */
	double zero = 0.0;
	int_t ldaspa; /* LDA of SPA */
	int_t iword, dword;
	float mem_use = 0.0;
	float memTRS = 0.; /* memory allocated for storing the meta-data for triangular solve (positive number)*/

	int *mod_bit; // Sherry 1/16/2022: changed to 'int'
	int *frecv, *brecv;
	int_t *lloc;
	double **Linv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	double *Linv_bc_dat;	  /* size: sum of sizes of Linv_bc_ptr[lk]) */
	long int *Linv_bc_offset; /* size ceil(NSUPERS/Pc)              */
	double **Uinv_bc_ptr;	  /* size ceil(NSUPERS/Pc) */
	double *Uinv_bc_dat;	  /* size: sum of sizes of Uinv_bc_ptr[lk]) */
	long int *Uinv_bc_offset; /* size ceil(NSUPERS/Pc)     */
	double *SeedSTD_BC, *SeedSTD_RD;
	int_t idx_indx, idx_lusup;
	int_t nbrow;
	int_t ik, il, lk, rel, knsupc, idx_r;
	int_t lptr1_tmp, idx_i, idx_v, m, uu;
	int_t nub;
	int tag;

#if (PRNTlevel >= 1)
	int_t nLblocks = 0, nUblocks = 0;
#endif
#if (PROFlevel >= 1)
	double t, t_u, t_l;
	int_t u_blks;
#endif

	/* Initialization. */
	iam = grid->iam;
	myrow = MYROW(iam, grid);
	mycol = MYCOL(iam, grid);
	nsupers = supno[n - 1] + 1;
	Astore = (NRformat_loc *)A->Store;

	// #if ( PRNTlevel>=1 )
	iword = sizeof(int_t);
	dword = sizeof(double);
	// #endif

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Enter pddistribute_allgrid_index_only()");
#endif
#if (PROFlevel >= 1)
	t = SuperLU_timer_();
#endif

	dReDistribute_A(A, ScalePermstruct, Glu_freeable, xsup, supno,
					grid, &xa, &asub, &a);

#if (PROFlevel >= 1)
	t = SuperLU_timer_() - t;
	if (!iam)
		printf("--------\n"
			   ".. Phase 1 - ReDistribute_A time: %.2f\t\n",
			   t);
#endif
		/* ------------------------------------------------------------
	   FIRST TIME CREATING THE L AND U DATA STRUCTURES.
	   ------------------------------------------------------------*/

#if (PROFlevel >= 1)
	t_l = t_u = 0;
	u_blks = 0;
#endif
	/* We first need to set up the L and U data structures and then
	 * propagate the values of A into them.
	 */
	lsub = Glu_freeable->lsub; /* compressed L subscripts */
	xlsub = Glu_freeable->xlsub;
	usub = Glu_freeable->usub; /* compressed U subscripts */
	xusub = Glu_freeable->xusub;

	k = CEILING(nsupers, grid->npcol); /* Number of local column blocks */
	j = k * grid->npcol;

	k = CEILING(nsupers, grid->nprow); /* Number of local block rows */

	/* Pointers to the beginning of each block row of U. */
	if (!(Ufstnz_br_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
		ABORT("Malloc fails for Ufstnz_br_ptr[].");
	if (!(ilsum = intMalloc_dist(k + 1)))
		ABORT("Malloc fails for ilsum[].");

	/* Auxiliary arrays used to set up U block data structures.
	   They are freed on return. */
	if (!(rb_marker = intCalloc_dist(k)))
		ABORT("Calloc fails for rb_marker[].");
	if (!(Urb_length = intCalloc_dist(k)))
		ABORT("Calloc fails for Urb_length[].");
	if (!(Urb_indptr = intMalloc_dist(k)))
		ABORT("Malloc fails for Urb_indptr[].");
	if (!(Urb_fstnz = intCalloc_dist(k)))
		ABORT("Calloc fails for Urb_fstnz[].");
	if (!(Ucbs = intCalloc_dist(k)))
		ABORT("Calloc fails for Ucbs[].");

	mem_use += 2.0 * k * sizeof(int_t *) + (7 * k + 1) * iword;

	/* Compute ldaspa and ilsum[]. */
	ldaspa = 0;
	ilsum[0] = 0;
	for (gb = 0; gb < nsupers; ++gb)
	{
		if (myrow == PROW(gb, grid))
		{
			i = SuperSize(gb);
			ldaspa += i;
			lb = LBi(gb, grid);
			ilsum[lb + 1] = ilsum[lb] + i;
		}
	}

#if (PROFlevel >= 1)
	t = SuperLU_timer_();
#endif
	/* ------------------------------------------------------------
	   COUNT NUMBER OF ROW BLOCKS AND THE LENGTH OF EACH BLOCK IN U.
	   THIS ACCOUNTS FOR ONE-PASS PROCESSING OF G(U).
	   ------------------------------------------------------------*/

	/* Loop through each supernode column. */
	for (jb = 0; jb < nsupers; ++jb)
	{
		pc = PCOL(jb, grid);
		fsupc = FstBlockC(jb);
		nsupc = SuperSize(jb);
		/* Loop through each column in the block. */
		for (j = fsupc; j < fsupc + nsupc; ++j)
		{
			/* usub[*] contains only "first nonzero" in each segment. */
			for (i = xusub[j]; i < xusub[j + 1]; ++i)
			{
				irow = usub[i]; /* First nonzero of the segment. */
				gb = BlockNum(irow);
				kcol = PCOL(gb, grid);
				ljb = LBj(gb, grid);
				pr = PROW(gb, grid);
				lb = LBi(gb, grid);
				if (mycol == pc)
				{
					if (myrow == pr)
					{
						/* Count nonzeros in entire block row. */
						Urb_length[lb] += FstBlockC(gb + 1) - irow;
						if (rb_marker[lb] <= jb)
						{ /* First see the block */
							rb_marker[lb] = jb + 1;
							Urb_fstnz[lb] += nsupc;
							++Ucbs[lb]; /* Number of column blocks
									   in block row lb. */
#if (PRNTlevel >= 1)
							++nUblocks;
#endif
						}
					}
				}
			} /* for i ... */
		} /* for j ... */
	} /* for jb ... */

	/* Set up the initial pointers for each block row in U. */
	nrbu = CEILING(nsupers, grid->nprow); /* Number of local block rows */
	for (lb = 0; lb < nrbu; ++lb)
	{
		ib = myrow + lb * grid->nprow; /* not sure */
		len = Urb_length[lb];
		rb_marker[lb] = 0; /* Reset block marker. */
		if (len)
		{
			/* Add room for descriptors */
			len1 = Urb_fstnz[lb] + BR_HEADER + Ucbs[lb] * UB_DESCRIPTOR;

			if (supernodeMask[ib] > 0)
			{ // YL: added supernode mask here
				if (!(index = intMalloc_dist(len1 + 1)))
					ABORT("Malloc fails for Uindex[].");
				Ufstnz_br_ptr[lb] = index;
				index[0] = Ucbs[lb]; /* Number of column blocks */
				index[1] = len;		 /* Total length of nzval[] */
				index[2] = len1;	 /* Total length of index[] */
				index[len1] = -1;	 /* End marker */
			}
			else
			{
				Ufstnz_br_ptr[lb] = NULL;
			}
		}
		else
		{
			Ufstnz_br_ptr[lb] = NULL;
		}
		Urb_length[lb] = 0;			/* Reset block length. */
		Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
		Urb_fstnz[lb] = BR_HEADER;
	} /* for lb ... */

	SUPERLU_FREE(Ucbs);

#if (PROFlevel >= 1)
	t = SuperLU_timer_() - t;
	if (!iam)
		printf(".. Phase 2 - setup U strut time: %.2f\t\n", t);
#endif

	mem_use -= 2.0 * k * iword;

	/* Auxiliary arrays used to set up L block data structures.
	   They are freed on return.
	   k is the number of local row blocks.   */
	if (!(Lrb_length = intCalloc_dist(k)))
		ABORT("Calloc fails for Lrb_length[].");
	if (!(Lrb_number = intMalloc_dist(k)))
		ABORT("Malloc fails for Lrb_number[].");
	if (!(Lrb_indptr = intMalloc_dist(k)))
		ABORT("Malloc fails for Lrb_indptr[].");

	/* ------------------------------------------------ */
	mem_use += 6.0 * k * iword + ldaspa * sp_ienv_dist(3, options) * dword;

	k = CEILING(nsupers, grid->npcol); /* Number of local block columns */

	/* Pointers to the beginning of each block column of L. */
	if (!(Lrowind_bc_ptr = (int_t **)SUPERLU_MALLOC(k * sizeof(int_t *))))
		ABORT("Malloc fails for Lrowind_bc_ptr[].");
	Lrowind_bc_ptr[k - 1] = NULL;

	/*------------------------------------------------------------
	  PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
	  THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U.
	  ------------------------------------------------------------*/
	long int Lrowind_bc_cnt = 0;
	for (jb = 0; jb < nsupers; ++jb)
	{ /* for each block column ... */
		pc = PCOL(jb, grid);
		if (mycol == pc)
		{ /* Block column jb in my process column */
			fsupc = FstBlockC(jb);
			nsupc = SuperSize(jb);
			ljb = LBj(jb, grid); /* Local block number */

			jbrow = PROW(jb, grid);

			/*------------------------------------------------
			 * SET UP U BLOCKS.
			 *------------------------------------------------*/
#if (PROFlevel >= 1)
			t = SuperLU_timer_();
#endif
			/* Loop through each column in the block column. */
			for (j = fsupc; j < FstBlockC(jb + 1); ++j)
			{
				istart = xusub[j];
				/* NOTE: Only the first nonzero index of the segment
				   is stored in usub[]. */
				for (i = istart; i < xusub[j + 1]; ++i)
				{
					irow = usub[i]; /* First nonzero in the segment. */
					gb = BlockNum(irow);
					pr = PROW(gb, grid);
					if (myrow == pr)
					{ // YL: added supernode mask here
						if (supernodeMask[gb] > 0)
						{
							lb = LBi(gb, grid); /* Local block number */
							index = Ufstnz_br_ptr[lb];
							fsupc1 = FstBlockC(gb + 1);
							if (rb_marker[lb] <= jb)
							{ /* First time see
			   the block       */
								rb_marker[lb] = jb + 1;
								Urb_indptr[lb] = Urb_fstnz[lb];
								;
								index[Urb_indptr[lb]] = jb; /* Descriptor */
								Urb_indptr[lb] += UB_DESCRIPTOR;
								/* Record the first location in index[] of the
								next block */
								Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;
								len = Urb_indptr[lb]; /* Start fstnz in index */
								index[len - 1] = 0;
								for (k = 0; k < nsupc; ++k)
									index[len + k] = fsupc1;
							}
							else
							{						  /* Already saw the block */
								len = Urb_indptr[lb]; /* Start fstnz in index */
							}
							jj = j - fsupc;
							index[len + jj] = irow;
							/* Load the numerical values */
							k = fsupc1 - irow;	 /* No. of nonzeros in segment */
							index[len - 1] += k; /* Increment block length in
									  Descriptor */
							irow = ilsum[lb] + irow - FstBlockC(gb);
						}
						else
						{
							lb = LBi(gb, grid); /* Local block number */
							fsupc1 = FstBlockC(gb + 1);
							if (rb_marker[lb] <= jb)
							{ /* First time see
			   the block       */
								rb_marker[lb] = jb + 1;
								Urb_indptr[lb] = Urb_fstnz[lb];
								;
								Urb_indptr[lb] += UB_DESCRIPTOR;
								/* Record the first location in index[] of the
								next block */
								Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;
							}
						}

					} /* if myrow == pr ... */
				} /* for i ... */
			} /* for j ... */

#if (PROFlevel >= 1)
			t_u += SuperLU_timer_() - t;
			t = SuperLU_timer_();
#endif
			/*------------------------------------------------
			 * SET UP L BLOCKS.
			 *------------------------------------------------*/

			/* Count number of blocks and length of each block. */
			nrbl = 0;
			len = 0; /* Number of row subscripts I own. */
			istart = xlsub[fsupc];
			for (i = istart; i < xlsub[fsupc + 1]; ++i)
			{
				irow = lsub[i];
				gb = BlockNum(irow); /* Global block number */
				pr = PROW(gb, grid); /* Process row owning this block */
				if (myrow == pr)
				{
					lb = LBi(gb, grid); /* Local block number */
					if (rb_marker[lb] <= jb)
					{ /* First see this block */
						rb_marker[lb] = jb + 1;
						Lrb_length[lb] = 1;
						Lrb_number[nrbl++] = gb;
#if (PRNTlevel >= 1)
						++nLblocks;
#endif
					}
					else
					{
						++Lrb_length[lb];
					}
					++len;
				}
			} /* for i ... */

			if (nrbl)
			{ /* Do not ensure the blocks are sorted! */
				if (supernodeMask[jb] > 0)
				{ // YL: added supernode mask here
					/* Set up the initial pointers for each block in
					   index[] and nzval[]. */
					/* Add room for descriptors */
					len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
					if (!(index = intMalloc_dist(len1)))
						ABORT("Malloc fails for index[]");
					myrow = MYROW(iam, grid);
					krow = PROW(jb, grid);

					index[0] = nrbl; /* Number of row blocks */
					index[1] = len;	 /* LDA of the nzval[] */
					next_lind = BC_HEADER;
					next_lval = 0;
					for (k = 0; k < nrbl; ++k)
					{
						gb = Lrb_number[k];
						lb = LBi(gb, grid);
						len = Lrb_length[lb];
						Lrb_length[lb] = 0;		 /* Reset vector of block length */
						index[next_lind++] = gb; /* Descriptor */
						index[next_lind++] = len;
						Lrb_indptr[lb] = next_lind;
						next_lind += len;
						next_lval += len;
					}
					/* Propagate the compressed row subscripts to Lindex[],
							   and the initial values of A from SPA into Lnzval[]. */
					len = index[1]; /* LDA of lusup[] */
					for (i = istart; i < xlsub[fsupc + 1]; ++i)
					{
						irow = lsub[i];
						gb = BlockNum(irow);
						if (myrow == PROW(gb, grid))
						{
							lb = LBi(gb, grid);
							k = Lrb_indptr[lb]++; /* Random access a block */
							index[k] = irow;
							irow = ilsum[lb] + irow - FstBlockC(gb);
						}
					} /* for i ... */

					Lrowind_bc_ptr[ljb] = index;
				}
				else
				{ // if(supernodeMask[jb]==0)
					Lrowind_bc_ptr[ljb] = NULL;
				}
			}
			else
			{
				Lrowind_bc_ptr[ljb] = NULL;
			} /* if nrbl ... */
#if (PROFlevel >= 1)
			t_l += SuperLU_timer_() - t;
#endif
		} /* if mycol == pc */

	} /* for jb ... */

	/////////////////////////////////////////////////////////////////

	Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
	Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
	Llu->ldalsum = ldaspa;

#if (PRNTlevel >= 1)
	if (!iam)
		printf(".. # L blocks " IFMT "\t# U blocks " IFMT "\n",
			   nLblocks, nUblocks);
#endif

	SUPERLU_FREE(rb_marker);
	SUPERLU_FREE(Urb_fstnz);
	SUPERLU_FREE(Urb_length);
	SUPERLU_FREE(Urb_indptr);
	SUPERLU_FREE(Lrb_length);
	SUPERLU_FREE(Lrb_number);
	SUPERLU_FREE(Lrb_indptr);
	SUPERLU_FREE(ilsum);

	k = CEILING(nsupers, grid->nprow); /* Number of local block rows */

#if (PROFlevel >= 1)
	if (!iam)
		printf(".. 1st distribute time:\n "
			   "\tL\t%.2f\n\tU\t%.2f\n"
			   "\tu_blks %d\tnrbu %d\n--------\n",
			   t_l, t_u, u_blks, nrbu);
#endif

	if (xa[A->ncol] > 0)
	{ /* may not have any entries on this process. */
		SUPERLU_FREE(asub);
		SUPERLU_FREE(a);
	}
	SUPERLU_FREE(xa);
	LUstruct->trf3Dpart = NULL;

#if (DEBUGlevel >= 1)
	/* Memory allocated but not freed:
	   ilsum, fmod, fsendx_plist, bmod, bsendx_plist  */
	CHECK_MALLOC(iam, "Exit pddistribute_allgrid_index_only()");
#endif

	return (mem_use);

} /* PDDISTRIBUTE_ALLGRID_INDEX_ONLY */
