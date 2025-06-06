/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Performs a symbolic factorization
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999

  Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

  THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
  EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

  Permission is hereby granted to use or copy this program for any
  purpose, provided the above notices are retained on all copies.
  Permission to modify the code and to distribute modified code is
  granted, provided the above notices are retained, and a notice that
  the code was modified is included with the above copyright notice.
 * </pre>
 */

/*
 * Modified by X. S. Li.
 */

#include "superlu_ddefs.h"

/* What type of supernodes we want */
#define T2_SUPER

/*
 * Internal protypes
 */
static void relax_snode(int_t, int_t *, int_t, int_t *, int_t *);
static int_t snode_dfs(SuperMatrix *, const int_t, const int_t, int_t *,
					   int_t *, Glu_persist_t *, Glu_freeable_t *);
static int_t column_dfs(superlu_dist_options_t *, SuperMatrix *,
						const int_t, int_t *, int_t *, int_t *,
						int_t *, int_t *, int_t *, int_t *, int_t *,
						Glu_persist_t *, Glu_freeable_t *);
static int_t pivotL(const int_t, int_t *, int_t *,
					Glu_persist_t *, Glu_freeable_t *);
static int_t set_usub(const int_t, const int_t, const int_t, int_t *, int_t *,
					  Glu_persist_t *, Glu_freeable_t *);
static void pruneL(const int_t, const int_t *, const int_t, const int_t,
				   const int_t *, const int_t *, int_t *,
				   Glu_persist_t *, Glu_freeable_t *);

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   symbfact() performs a symbolic factorization on matrix A and sets up
 *   the nonzero data structures which are suitable for supernodal Gaussian
 *   elimination with no pivoting (GENP). This routine features:
 *        o depth-first search (DFS)
 *        o supernodes
 *        o symmetric structure pruning
 *
 * Return value
 * ============
 *   < 0, number of bytes needed for LSUB.
 *   = 0, matrix dimension is 1.
 *   > 0, number of bytes allocated when out of memory.
 * </pre>
 */
int_t symbfact
/************************************************************************/
(
	superlu_dist_options_t *options, /* input options */
	int pnum,						 /* process number */
	SuperMatrix *A,					 /* original matrix A permuted by columns (input) */
	int_t *perm_c,					 /* column permutation vector (input) */
	int_t *etree,					 /* column elimination tree (input) */
	Glu_persist_t *Glu_persist,		 /* output */
	Glu_freeable_t *Glu_freeable	 /* output */
)
{

	int_t m, n, min_mn, j, i, k, irep, nseg, pivrow, info;
	int_t *iwork, *perm_r, *segrep, *repfnz;
	int_t *xprune, *marker, *parent, *xplore;
	int_t relax, *desc, *relax_end;
	int_t nnzLU, nnzLSUB;
	int_t nnzL, nnzU;

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(pnum, "Enter symbfact()");
#endif

	m = A->nrow;
	n = A->ncol;
	min_mn = SUPERLU_MIN(m, n);

	/* Allocate storage common to the symbolic factor routines */
	info = symbfact_SubInit(options, DOFACT, NULL, 0, m, n,
							((NCPformat *)A->Store)->nnz,
							Glu_persist, Glu_freeable);
	if (info != 0)
		return info;

	iwork = (int_t *)intMalloc_dist(6 * m + 2 * n);
	perm_r = iwork;
	segrep = iwork + m;
	repfnz = segrep + m;
	marker = repfnz + m;
	parent = marker + m;
	xplore = parent + m;
	xprune = xplore + m;
	relax_end = xprune + n;
	relax = sp_ienv_dist(2, options);
	ifill_dist(perm_r, m, SLU_EMPTY);
	ifill_dist(repfnz, m, SLU_EMPTY);
	ifill_dist(marker, m, SLU_EMPTY);
	Glu_persist->supno[0] = -1;
	Glu_persist->xsup[0] = 0;
	Glu_freeable->xlsub[0] = 0;
	Glu_freeable->xusub[0] = 0;

	/*for (j = 0; j < n; ++j) iperm_c[perm_c[j]] = j;*/

	/* Identify relaxed supernodes. */
	if (!(desc = intMalloc_dist(n + 1)))
		ABORT("Malloc fails for desc[]");
	;
	relax_snode(n, etree, relax, desc, relax_end);
	SUPERLU_FREE(desc);

	for (j = 0; j < min_mn;)
	{
		if (relax_end[j] != SLU_EMPTY)
		{					  /* beginning of a relaxed snode */
			k = relax_end[j]; /* end of the relaxed snode */

			/* Determine union of the row structure of supernode (j:k). */
			if ((info = snode_dfs(A, j, k, xprune, marker,
								  Glu_persist, Glu_freeable)) != 0)
				return info;

			for (i = j; i <= k; ++i)
				pivotL(i, perm_r, &pivrow, Glu_persist, Glu_freeable);

			j = k + 1;
		}
		else
		{
			/* Perform a symbolic factorization on column j, and detects
			   whether column j starts a new supernode. */
			if ((info = column_dfs(options, A, j, perm_r, &nseg, segrep, repfnz,
								   xprune, marker, parent, xplore,
								   Glu_persist, Glu_freeable)) != 0)
				return info;

			// (Yida) add : set each column of current U.block_row, length = 0 or length = SuperSize
			// for(int lyd_i=0;lyd_i<m;lyd_i++){
			// 	// if(repfnz[lyd_i] == SLU_EMPTY){

			// 	// }else if(repfnz[lyd_i] == Glu_persist->xsup[Glu_persist->supno[lyd_i]]){

			// 	// }else if(repfnz[lyd_i] == Glu_persist->xsup[Glu_persist->supno[lyd_i]+1]){

			// 	// }else{
			// 	// 	repfnz[lyd_i] = Glu_persist->xsup[Glu_persist->supno[lyd_i]];
			// 	// 	// printf("j=%d repfnz[%d]=%d xsup[%d]=%d xsup[%d]=%d\n", j, lyd_i, repfnz[lyd_i], 
			// 	// 	// 	Glu_persist->supno[lyd_i], Glu_persist->xsup[Glu_persist->supno[lyd_i]],
			// 	// 	// 	Glu_persist->supno[lyd_i]+1, Glu_persist->xsup[Glu_persist->supno[lyd_i]+1]);
			// 	// }

			// 	if(repfnz[lyd_i]!=SLU_EMPTY && repfnz[lyd_i]!=Glu_persist->xsup[Glu_persist->supno[lyd_i]+1]){
			// 		repfnz[lyd_i] = Glu_persist->xsup[Glu_persist->supno[lyd_i]];
			// 	}
			// }
			// int lyd_nsup = Glu_persist->supno[m-1]+1;
			// printf("lyd_nsup = %d\n", lyd_nsup);
			// for(int lyd_isup = 0; lyd_isup = lyd_nsup; lyd_isup++){
			// 	int lyd_nextsup_firstrow = Glu_persist->xsup[lyd_isup+1];
			// 	int lyd_thissup_firstrow = Glu_persist->xsup[lyd_isup];
			// 	for(int lyd_col = lyd_thissup_firstrow; lyd_col < lyd_nextsup_firstrow; lyd_col++){
			// 		if(repfnz[lyd_col]!=SLU_EMPTY && repfnz[lyd_col]!=lyd_nextsup_firstrow){
			// 			repfnz[lyd_col] = lyd_thissup_firstrow;
			// 		}
			// 	}
			// }
			//

			/* Copy the U-segments to usub[*]. */
			if ((info = set_usub(min_mn, j, nseg, segrep, repfnz,
								 Glu_persist, Glu_freeable)) != 0)
				return info;

			pivotL(j, perm_r, &pivrow, Glu_persist, Glu_freeable);

			/* Prune columns [0:j-1] using column j. */
			pruneL(j, perm_r, pivrow, nseg, segrep, repfnz, xprune,
				   Glu_persist, Glu_freeable);

			/* Reset repfnz[*] to prepare for the next column. */
			for (i = 0; i < nseg; i++)
			{
				irep = segrep[i];
				repfnz[irep] = SLU_EMPTY;
			}

			++j;
		} /* else */
	} /* for j ... */

	countnz_dist(min_mn, xprune, &nnzL, &nnzU, Glu_persist, Glu_freeable);
	Glu_freeable->nnzLU = nnzL + nnzU - min_mn;
	/* Apply perm_r to L; Compress LSUB array. */
	nnzLSUB = fixupL_dist(min_mn, perm_r, Glu_persist, Glu_freeable);

	if (!pnum && (options->PrintStat == YES))
	{
		nnzLU = nnzL + nnzU - min_mn;
		printf("\tMatrix size min_mn  " IFMT "\n", min_mn);
		printf("\tNonzeros in L       " IFMT "\n", nnzL);
		printf("\tNonzeros in U       " IFMT "\n", nnzU);
		printf("\tnonzeros in L+U     " IFMT "\n", nnzLU);
		printf("\tnonzeros in LSUB    " IFMT "\n", nnzLSUB);
	}
	SUPERLU_FREE(iwork);

#if (PRNTlevel >= 3)
	PrintInt10("lsub", Glu_freeable->xlsub[n], Glu_freeable->lsub);
	PrintInt10("xlsub", n + 1, Glu_freeable->xlsub);
	PrintInt10("xprune", n, xprune);
	PrintInt10("usub", Glu_freeable->xusub[n], Glu_freeable->usub);
	PrintInt10("xusub", n + 1, Glu_freeable->xusub);
	PrintInt10("supno", n, Glu_persist->supno);
	PrintInt10("xsup", (Glu_persist->supno[n]) + 2, Glu_persist->xsup);
#endif

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(pnum, "Exit symbfact()");
#endif

	/* return (-i); */
	return (-nnzLSUB);

} /* SYMBFACT */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   relax_snode() identifies the initial relaxed supernodes, assuming that
 *   the matrix has been reordered according to an postorder of the etree.
 * </pre>
 */
static void relax_snode
	/************************************************************************/
	(
		const int_t n,	   /* number of columns in the matrix (input) */
		int_t *et,		   /* column elimination tree (input) */
		const int_t relax, /* max no of columns allowed in a relaxed snode (input) */
		int_t *desc,	   /* number of descendants of each etree node. */
		int_t *relax_end   /* last column in a supernode (output) */
	)
{

	register int_t j, parent, nsuper;
	register int_t fsupc; /* beginning of a snode */

	ifill_dist(relax_end, n, SLU_EMPTY);
	ifill_dist(desc, n + 1, 0);
	nsuper = 0;

	/* Compute the number of descendants of each node in the etree. */
	for (j = 0; j < n; j++)
	{
		parent = et[j];
		if (parent != n) /* not the dummy root */
			desc[parent] += desc[j] + 1;
	}

	/* Identify the relaxed supernodes by postorder traversal of the etree. */
	for (j = 0; j < n;)
	{
		parent = et[j];
		fsupc = j;
		while (parent != n && desc[parent] < relax)
		{
			j = parent;
			parent = et[j];
		}
		/* Found a supernode with j being the last column. */
		relax_end[fsupc] = j; /* Last column is recorded. */
		++nsuper;
		++j;
		/* Search for a new leaf. */
		while (desc[j] != 0 && j < n)
			++j;
	}

#if (DEBUGlevel >= 1)
	printf(".. No of relaxed snodes: " IFMT "\trelax: " IFMT "\n", nsuper, relax);
#endif
} /* RELAX_SNODE */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    snode_dfs() determines the union of the row structures of those
 *    columns within the relaxed snode.
 *    Note: The relaxed snodes are leaves of the supernodal etree, therefore,
 *    the part outside the rectangular supernode must be zero.
 *
 * Return value
 * ============
 *    0   success;
 *   >0   number of bytes allocated when run out of memory.
 * </pre>
 */
static int_t snode_dfs
	/************************************************************************/
	(
		SuperMatrix *A,				/* original matrix A permuted by columns (input) */
		const int_t jcol,			/* beginning of the supernode (input) */
		const int_t kcol,			/* end of the supernode (input) */
		int_t *xprune,				/* pruned location in each adjacency list (output) */
		int_t *marker,				/* working array of size m */
		Glu_persist_t *Glu_persist, /* global LU data structures (modified) */
		Glu_freeable_t *Glu_freeable)
{

	NCPformat *Astore;
	int_t *asub, *xa_begin, *xa_end;
	register int_t i, k, ifrom, ito, nextl, new_next;
	int_t nsuper, krow, kmark, mem_error;
	int_t *xsup, *supno;
	int_t *lsub, *xlsub;
	int_t nzlmax, nextu;

	Astore = A->Store;
	asub = Astore->rowind;
	xa_begin = Astore->colbeg;
	xa_end = Astore->colend;
	xsup = Glu_persist->xsup;
	supno = Glu_persist->supno;
	lsub = Glu_freeable->lsub;
	xlsub = Glu_freeable->xlsub;
	nzlmax = Glu_freeable->nzlmax;
	nsuper = ++supno[jcol]; /* Next available supernode number */
	nextl = xlsub[jcol];
	nextu = Glu_freeable->xusub[jcol];

	for (i = jcol; i <= kcol; i++)
	{
		/* For each nonzero in A[*,i] */
		for (k = xa_begin[i]; k < xa_end[i]; ++k)
		{
			krow = asub[k];
			kmark = marker[krow];
			if (kmark != kcol)
			{ /* First time visit krow */
				marker[krow] = kcol;
				lsub[nextl++] = krow;
				if (nextl >= nzlmax)
				{
					if ((mem_error = symbfact_SubXpand(A->ncol, jcol, nextl,
													   (MemType)LSUB, &nzlmax,
													   Glu_freeable)))
						return (mem_error);
					lsub = Glu_freeable->lsub;
				}
			}
		}
		supno[i] = nsuper;
		Glu_freeable->xusub[i + 1] = nextu; /* Tidy up the pointers in usub[*]. */
	}

	/* Supernode > 1, then make a copy of the subscripts for pruning */
	if (jcol < kcol)
	{
		new_next = nextl + (nextl - xlsub[jcol]);
		while (new_next > nzlmax)
		{
			if ((mem_error = symbfact_SubXpand(A->ncol, jcol, nextl, (MemType)LSUB,
											   &nzlmax, Glu_freeable)))
				return (mem_error);
			lsub = Glu_freeable->lsub;
		}
		ito = nextl;
		for (ifrom = xlsub[jcol]; ifrom < nextl;)
			lsub[ito++] = lsub[ifrom++];
		for (i = jcol + 1; i <= kcol; i++)
			xlsub[i] = nextl;
		nextl = ito;
	}

	xsup[nsuper + 1] = kcol + 1;
	supno[kcol + 1] = nsuper;
	xprune[kcol] = nextl;
	xlsub[kcol + 1] = nextl;
#if (PRNTlevel >= 3)
	printf(".. snode_dfs(): (%8d:%8d) nextl %d\n", jcol, kcol, nextl);
#endif
	return 0;
} /* SNODE_DFS */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   column_dfs() performs a symbolic factorization on column jcol, and
 *   detects the supernode boundary. This routine uses the row indices of
 *   A[*,jcol] to start the depth-first search (DFS).
 *
 * Output
 * ======
 *   A supernode representative is the last column of a supernode.
 *   The nonzeros in U[*,j] are segments that end at supernodal
 *   representatives. The routine returns a list of such supernodal
 *   representatives ( segrep[*] ) in topological order of the DFS that
 *   generates them. The location of the first nonzero in each such
 *   supernodal segment is also returned ( repfnz[*] ).
 *
 * Data structure
 * ==============
 *   (lsub, xlsub):
 *      lsub[*] contains the compressed subscripts of the supernodes;
 *      xlsub[j] points to the starting location of the j-th column in
 *               lsub[*];
 *	Storage: original row subscripts in A.
 *
 *      During the course of symbolic factorization, we also use
 *	(lsub, xlsub, xprune) for the purpose of symmetric pruning.
 *      For each supernode {s,s+1,...,t=s+r} with first column s and last
 *	column t, there are two subscript sets,  the last column
 *      structures (for pruning) will be removed in the end.
 *        o lsub[j], j = xlsub[s], ..., xlsub[s+1]-1
 *          is the structure of column s (i.e. structure of this supernode).
 *          It is used for the storage of numerical values.
 *	  o lsub[j], j = xlsub[t], ..., xlsub[t+1]-1
 *	    is the structure of the last column t of this supernode.
 *	    It is for the purpose of symmetric pruning. Therefore, the
 *	    structural subscripts can be rearranged without making physical
 *	    interchanges among the numerical values.
 *
 *      (1) if t > s, only the subscript sets for column s and column t
 *          are stored. Column t represents pruned adjacency structure.
 *
 *                  --------------------------------------------
 *          lsub[*]    ... |   col s    |   col t   | ...
 *                  --------------------------------------------
 *                          ^            ^           ^
 *                       xlsub[s]    xlsub[s+1]  xlsub[t+1]
 *                                       :           :
 *                                       :         xprune[t]
 *                                   xlsub[t]
 *                                   xprune[s]
 *
 *      (2) if t == s, i.e., a singleton supernode, the same subscript set
 *          is used for both G(L) and pruned graph:
 *
 *                  --------------------------------------
 *          lsub[*]    ... |      s     | ...
 *                  --------------------------------------
 *                          ^            ^
 *                       xlsub[s]   xlsub[s+1]
 *                                  xprune[s]
 *
 *       DFS will traverse the second subscript list, i.e., the part of the
 *       pruned graph.
 *
 * Local parameters
 * ================
 *   nseg: no of segments in current U[*,j]
 *   jsuper: jsuper=EMPTY if column j does not belong to the same
 *	supernode as j-1. Otherwise, jsuper=nsuper.
 *
 *   marker: A-row --> A-row/col (0/1)
 *   repfnz: SuperA-col --> PA-row
 *   parent: SuperA-col --> SuperA-col
 *   xplore: SuperA-col --> index to L-structure
 *
 * Return value
 * ============
 *     0  success;
 *   > 0  number of bytes allocated when run out of space.
 * </pre>
 */
static int_t column_dfs
	/************************************************************************/
	(
		superlu_dist_options_t *options,
		SuperMatrix *A,				/* original matrix A permuted by columns (input) */
		const int_t jcol,			/* current column number (input) */
		int_t *perm_r,				/* row permutation vector (input) */
		int_t *nseg,				/* number of U-segments in column jcol (output) */
		int_t *segrep,				/* list of U-segment representatives (output) */
		int_t *repfnz,				/* list of first nonzeros in the U-segments (output) */
		int_t *xprune,				/* pruned location in each adjacency list (output) */
		int_t *marker,				/* working array of size m */
		int_t *parent,				/* working array of size m */
		int_t *xplore,				/* working array of size m */
		Glu_persist_t *Glu_persist, /* global LU data structures (modified) */
		Glu_freeable_t *Glu_freeable)
{

	NCPformat *Astore;
	int_t *asub, *xa_begin, *xa_end;
	int_t jcolp1, jcolm1, jsuper, nsuper, nextl;
	int_t k, krep, krow, kmark, kperm;
	int_t fsupc; /* first column of a supernode */
	int_t myfnz; /* first nonzero column of a U-segment */
	int_t chperm, chmark, chrep, kchild;
	int_t xdfs, maxdfs, kpar, oldrep;
	int_t jptr, jm1ptr;
	int_t ito, ifrom, istop; /* used to compress row subscripts */
	int_t *xsup, *supno, *lsub, *xlsub;
	int_t nzlmax;
	static int_t first = 1, maxsuper;
	int_t mem_error;

	/* Initializations */
	Astore = A->Store;
	asub = Astore->rowind;
	xa_begin = Astore->colbeg;
	xa_end = Astore->colend;
	xsup = Glu_persist->xsup;
	supno = Glu_persist->supno;
	lsub = Glu_freeable->lsub;
	xlsub = Glu_freeable->xlsub;
	nzlmax = Glu_freeable->nzlmax;
	jcolp1 = jcol + 1;
	jcolm1 = jcol - 1;
	jsuper = nsuper = supno[jcol];
	nextl = xlsub[jcol];
	if (first)
	{
		maxsuper = sp_ienv_dist(3, options);
		first = 0;
	}

	*nseg = 0;

	/* For each nonzero in A[*,jcol] perform depth-first search. */
	for (k = xa_begin[jcol]; k < xa_end[jcol]; ++k)
	{
		krow = asub[k];
		kmark = marker[krow];

		/* krow was visited before, go to the next nonzero. */
		if (kmark == jcol)
			continue;

		/*
		 * For each unmarked neighber krow of jcol ...
		 */
		marker[krow] = jcol; /* mark as "visited" */
		kperm = perm_r[krow];

		if (kperm == SLU_EMPTY)
		{
			/* ---------------
			 *  krow is in L
			 * ---------------
			 * place it in structure of L[*,jcol].
			 */
			lsub[nextl++] = krow; /* krow is indexed into A */
			if (nextl >= nzlmax)
			{
				if ((mem_error = symbfact_SubXpand(A->ncol, jcol, nextl, (MemType)LSUB,
												   &nzlmax, Glu_freeable)))
					return (mem_error);
				lsub = Glu_freeable->lsub;
			}
			if (kmark != jcolm1)
				jsuper = SLU_EMPTY; /* Row index subset test */
		}
		else
		{
			/* ---------------
			 *  krow is in U
			 * ---------------
			 * If its supernode krep has been explored, update repfnz[*].
			 */
			krep = xsup[supno[kperm] + 1] - 1;
			myfnz = repfnz[krep];

			if (myfnz != SLU_EMPTY)
			{ /* krep was visited before */
				if (kperm < myfnz)
					repfnz[krep] = kperm;
				/* continue; */
			}
			else
			{
				/* Otherwise perform DFS, starting at krep */
				oldrep = SLU_EMPTY;
				parent[krep] = oldrep;
				repfnz[krep] = kperm;
				xdfs = xlsub[krep];
				maxdfs = xprune[krep];

				do
				{
					/*
					 * For each unmarked kchild of krep
					 */
					while (xdfs < maxdfs)
					{
						kchild = lsub[xdfs++];
						chmark = marker[kchild];

						if (chmark != jcol)
						{ /* Not reached yet */
							marker[kchild] = jcol;
							chperm = perm_r[kchild];

							/* Case kchild is in L: place it in L[*,k] */
							if (chperm == SLU_EMPTY)
							{
								lsub[nextl++] = kchild;
								if (nextl >= nzlmax)
								{
									if ((mem_error =
											 symbfact_SubXpand(A->ncol, jcol, nextl,
															   (MemType)LSUB, &nzlmax,
															   Glu_freeable)))
										return (mem_error);
									lsub = Glu_freeable->lsub;
								}
								if (chmark != jcolm1)
									jsuper = SLU_EMPTY;
							}
							else
							{
								/* Case kchild is in U:
								 * chrep = its supernode-rep. If its rep
								 * has been explored, update its repfnz[*].
								 */
								chrep = xsup[supno[chperm] + 1] - 1;
								myfnz = repfnz[chrep];
								if (myfnz != SLU_EMPTY)
								{ /* Visited before */
									if (chperm < myfnz)
										repfnz[chrep] = chperm;
								}
								else
								{
									/* Continue DFS at sup-rep of kchild */
									xplore[krep] = xdfs;
									oldrep = krep;
									krep = chrep; /* Go deeper down G(L') */
									parent[krep] = oldrep;
									repfnz[krep] = chperm;
									xdfs = xlsub[krep];
									maxdfs = xprune[krep];
								} /* else */
							} /* else */
						} /* if chmark != jcol */

					} /* while */

					/* krow has no more unexplored neighbors:
					 *    place supernode-rep krep in postorder DFS;
					 *    backtrack DFS to its parent.
					 */
					segrep[*nseg] = krep;
					++(*nseg);
					kpar = parent[krep]; /* Pop from stack; recurse */
					if (kpar == SLU_EMPTY)
						break; /* DFS done */
					krep = kpar;
					xdfs = xplore[krep];
					maxdfs = xprune[krep];
				} while (kpar != SLU_EMPTY); /* Until empty stack */
			} /* else */
		} /* else: krow is in U */
	} /* for each nonzero in A[*, jcol] */

	/* Check to see if jcol belongs in the same supernode as jcol-1 */
	if (jcol == 0)
	{ /* Do nothing for column 0 */
		nsuper = supno[0] = 0;
	}
	else
	{
		fsupc = xsup[nsuper];
		jptr = xlsub[jcol]; /* Not compressed yet */
		jm1ptr = xlsub[jcolm1];

#ifdef T2_SUPER
		if ((nextl - jptr != jptr - jm1ptr - 1))
			jsuper = SLU_EMPTY;
#endif
		/* Make sure the number of columns in a supernode doesn't
		   exceed threshold. */
		if (jcol - fsupc >= maxsuper)
			jsuper = SLU_EMPTY;

		/* If jcol starts a new supernode, reclaim storage space in
		 * lsub[*] from the previous supernode. Note we only store
		 * the subscript set of the first and last columns of
		 * a supernode. (first for G(L'), last for pruned graph)
		 */
		if (jsuper == SLU_EMPTY)
		{ /* Starts a new supernode */
			if ((fsupc < jcolm1 - 1))
			{ /* >= 3 columns in nsuper */
#ifdef CHK_COMPRESS
				printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
				ito = xlsub[fsupc + 1];
				xlsub[jcolm1] = ito;
				istop = ito + jptr - jm1ptr;
				xprune[jcolm1] = istop; /* Initialize xprune[jcol-1] */
				xlsub[jcol] = istop;
				for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
					lsub[ito] = lsub[ifrom];
				nextl = ito; /* = istop + length(jcol) */
			}
			++nsuper;
			supno[jcol] = nsuper;
		} /* if a new supernode */

	} /* else: jcol > 0 */

	/* Tidy up the pointers before exit */
	xsup[nsuper + 1] = jcolp1;
	supno[jcolp1] = nsuper;
	xprune[jcol] = nextl; /* Initialize an upper bound for pruning. */
	xlsub[jcolp1] = nextl;
	return 0;
} /* COLUMN_DFS */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   pivotL() interchanges row subscripts so that each diagonal block of a
 *   supernode in L has the row subscripts sorted in order of pivots.
 *   The row subscripts in the off-diagonal block are not sorted.
 * </pre>
 */
static int_t pivotL
	/************************************************************************/
	(
		const int_t jcol,			/* current column number     (input)    */
		int_t *perm_r,				/* row permutation vector    (output)   */
		int_t *pivrow,				/* the pivot row index       (output)   */
		Glu_persist_t *Glu_persist, /* global LU data structures (modified) */
		Glu_freeable_t *Glu_freeable)
{

	int_t fsupc; /* first column in the supernode */
	int_t nsupc; /* number of columns in the supernode */
	int_t nsupr; /* number of rows in the supernode */
	int_t lptr;	 /* point_ts to the first subscript of the supernode */
	int_t diag, diagind;
	int_t *lsub_ptr;
	int_t isub, itemp;
	int_t *lsub, *xlsub;

	/* Initialization. */
	lsub = Glu_freeable->lsub;
	xlsub = Glu_freeable->xlsub;
	fsupc = (Glu_persist->xsup)[(Glu_persist->supno)[jcol]];
	nsupc = jcol - fsupc; /* excluding jcol; nsupc >= 0 */
	lptr = xlsub[fsupc];
	nsupr = xlsub[fsupc + 1] - lptr;
	lsub_ptr = &lsub[lptr]; /* start of row indices of the supernode */

	/* Search for diagonal element. */
	/* diagind = iperm_c[jcol];*/
	diagind = jcol;
	diag = SLU_EMPTY;
	for (isub = nsupc; isub < nsupr; ++isub)
		if (lsub_ptr[isub] == diagind)
		{
			diag = isub;
			break;
		}

	/* Diagonal pivot exists? */
	if (diag == SLU_EMPTY)
	{
		printf("At column " IFMT ", ", jcol);
		ABORT("pivotL() encounters zero diagonal");
	}

	/* Record pivot row. */
	*pivrow = lsub_ptr[diag];
	perm_r[*pivrow] = jcol; /* perm_r[] should be Identity. */
	/*assert(*pivrow==jcol);*/

	/* Interchange row subscripts. */
	if (diag != nsupc)
	{
		itemp = lsub_ptr[diag];
		lsub_ptr[diag] = lsub_ptr[nsupc];
		lsub_ptr[nsupc] = itemp;
	}

	return 0;
} /* PIVOTL */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   set_usub() sets up data structure to store supernodal segments in U.
 *   The supernodal segments in each column are stored in topological order.
 *
 * NOTE
 * ====
 *   For each supernodal segment, we only store the index of the first
 *   nonzero index, rather than the indices of the whole segment, because
 *   those indices can be generated from first nonzero and supnodal
 *   representative.
 *   Therefore, for G(U), we store the "skeleton" of it.
 * </pre>
 */
static int_t set_usub
	/************************************************************************/
	(
		const int_t n,				/* total number of columns (input) */
		const int_t jcol,			/* current column number (input) */
		const int_t nseg,			/* number of supernodal segments in U[*,jcol] (input) */
		int_t *segrep,				/* list of U-segment representatives (output) */
		int_t *repfnz,				/* list of first nonzeros in the U-segments (output) */
		Glu_persist_t *Glu_persist, /* global LU data structures (modified) */
		Glu_freeable_t *Glu_freeable)
{

	int_t ksub, krep, ksupno;
	int_t k, kfnz;
	int_t jsupno, nextu;
	int_t new_next, mem_error;
	int_t *supno;
	int_t *usub, *xusub;
	int_t nzumax;

	supno = Glu_persist->supno;
	usub = Glu_freeable->usub;
	xusub = Glu_freeable->xusub;
	nzumax = Glu_freeable->nzumax;
	jsupno = supno[jcol];
	nextu = xusub[jcol];

	new_next = nextu + nseg;
	while (new_next > nzumax)
	{
		if ((mem_error = symbfact_SubXpand(n, jcol, nextu, (MemType)USUB, &nzumax,
										   Glu_freeable)))
			return (mem_error);
		usub = Glu_freeable->usub;
	}

	/* We store U-segments in topological order. */
	k = nseg - 1;
	for (ksub = 0; ksub < nseg; ++ksub)
	{
		krep = segrep[k--];
		ksupno = supno[krep];

		if (ksupno != jsupno)
		{ /* Should go into usub[*] */
			kfnz = repfnz[krep];
			if (kfnz != SLU_EMPTY)
			{ /* Nonzero U-segment */
				usub[nextu++] = kfnz;

				/*	    	fsupc = xsup[ksupno];
							isub = xlsub[fsupc] + kfnz - fsupc;
						irow = lsub[isub];
						usub[nextu++] = perm_r[irow];*/
			} /* if ... */
		} /* if ... */
	} /* for each segment... */

	xusub[jcol + 1] = nextu; /* Close U[*,jcol] */
	return 0;
} /* SET_USUB */

/************************************************************************/
static void pruneL
	/************************************************************************/
	(
		const int_t jcol,			/* in */
		const int_t *perm_r,		/* in */
		const int_t pivrow,			/* in */
		const int_t nseg,			/* in */
		const int_t *segrep,		/* in */
		const int_t *repfnz,		/* in */
		int_t *xprune,				/* out */
		Glu_persist_t *Glu_persist, /* global LU data structures (modified) */
		Glu_freeable_t *Glu_freeable)
{
	/*
	 * Purpose
	 * =======
	 *   pruneL() prunes the L-structure of supernodes whose L-structure
	 *   contains the current pivot row "pivrow".
	 *
	 */
	int_t jsupno, irep, irep1, kmin, kmax, krow;
	int_t i, ktemp;
	int_t do_prune; /* logical variable */
	int_t *supno;
	int_t *lsub, *xlsub;

	supno = Glu_persist->supno;
	lsub = Glu_freeable->lsub;
	xlsub = Glu_freeable->xlsub;

	/*
	 * For each supernode-rep irep in U[*,j]
	 */
	jsupno = supno[jcol];
	for (i = 0; i < nseg; i++)
	{
		irep = segrep[i];
		irep1 = irep + 1;

		/* Do not prune with a zero U-segment */
		if (repfnz[irep] == SLU_EMPTY)
			continue;

		/*
		 * If irep has not been pruned & it has a nonzero in row L[pivrow,i]
		 */
		do_prune = FALSE;
		if (supno[irep] != jsupno)
		{
			if (xprune[irep] >= xlsub[irep1])
			{
				kmin = xlsub[irep];
				kmax = xlsub[irep1] - 1;
				for (krow = kmin; krow <= kmax; ++krow)
					if (lsub[krow] == pivrow)
					{
						do_prune = TRUE;
						break;
					}
			}

			if (do_prune)
			{
				/* Do a quicksort-type partition. */
				while (kmin <= kmax)
				{
					if (perm_r[lsub[kmax]] == SLU_EMPTY)
						kmax--;
					else if (perm_r[lsub[kmin]] != SLU_EMPTY)
						kmin++;
					else
					{ /* kmin below pivrow, and kmax above pivrow:
					   * 	   interchange the two subscripts
					   */
						ktemp = lsub[kmin];
						lsub[kmin] = lsub[kmax];
						lsub[kmax] = ktemp;
						kmin++;
						kmax--;
					}
				} /* while */
				xprune[irep] = kmin; /* Pruning */
#if (DEBUGlevel >= 3)
				printf(".. pruneL(): use col %d: xprune[%d] = %d\n",
					   jcol, irep, kmin);
#endif
			} /* if do_prune */
		} /* if */
	} /* for each U-segment ... */
} /* PRUNEL */
