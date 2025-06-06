/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Elimination tree computation and layout routines
 *
 * <pre>
 *  Implementation of disjoint set union routines.
 *  Elements are integers in 0..n-1, and the
 *  names of the sets themselves are of type int.
 *
 *  Calls are:
 *  initialize_disjoint_sets (n) initial call.
 *  s = make_set (i)             returns a set containing only i.
 *  s = link (t, u)		 returns s = t union u, destroying t and u.
 *  s = find (i)		 return name of set containing i.
 *  finalize_disjoint_sets 	 final call.
 *
 *  This implementation uses path compression but not weighted union.
 *  See Tarjan's book for details.
 *  John Gilbert, CMI, 1987.
 *
 *  Implemented path-halving by XL 7/5/95.
 * </pre>
 */

#include <stdio.h>
#include <stdlib.h>
#include "superlu_ddefs.h"
#include "superlu_defs.h"

static int_t *mxCallocInt(int_t n)
{
	register int_t i;
	int_t *buf;

	buf = (int_t *)SUPERLU_MALLOC(n * sizeof(int_t));
	if (buf)
		for (i = 0; i < n; i++)
			buf[i] = 0;
	return (buf);
}

static void initialize_disjoint_sets(
	int_t n,
	int_t **pp /* parent array for sets */
)
{
	if (!((*pp) = mxCallocInt(n)))
		ABORT("mxCallocInit fails for pp[]");
}

static int_t make_set(
	int_t i,
	int_t *pp /* parent array for sets */
)
{
	pp[i] = i;
	return i;
}

static int_t link(
	int_t s,
	int_t t,
	int_t *pp)
{
	pp[s] = t;
	return t;
}

/* PATH HALVING */
static int_t find(
	int_t i,
	int_t *pp)
{
	register int_t p, gp;

	p = pp[i];
	gp = pp[p];
	while (gp != p)
	{
		pp[i] = gp;
		i = gp;
		p = pp[i];
		gp = pp[p];
	}
	return (p);
}

static void finalize_disjoint_sets(
	int_t *pp)
{
	SUPERLU_FREE(pp);
}

/*! \brief Symmetric elimination tree
 *
 * <pre>
 *      p = spsymetree (A);
 *
 *      Find the elimination tree for symmetric matrix A.
 *      This uses Liu's algorithm, and runs in time O(nz*log n).
 *
 *      Input:
 *        Square sparse matrix A.  No check is made for symmetry;
 *        elements below and on the diagonal are ignored.
 *        Numeric values are ignored, so any explicit zeros are
 *        treated as nonzero.
 *      Output:
 *        Integer array of parents representing the etree, with n
 *        meaning a root of the elimination forest.
 *      Note:
 *        This routine uses only the upper triangle, while sparse
 *        Cholesky (as in spchol.c) uses only the lower.  Matlab's
 *        dense Cholesky uses only the upper.  This routine could
 *        be modified to use the lower triangle either by transposing
 *        the matrix or by traversing it by rows with auxiliary
 *        pointer and link arrays.
 *
 *      John R. Gilbert, Xerox, 10 Dec 1990
 *      Based on code by JRG dated 1987, 1988, and 1990.
 *      Modified by X.S. Li, November 1999.
 * </pre>
 */
int sp_symetree_dist(
	int_t *acolst, int_t *acolend, /* column starts and ends past 1 */
	int_t *arow,				   /* row indices of A */
	int_t n,					   /* dimension of A */
	int_t *parent				   /* parent in elim tree */
)
{
	int_t *root; /* root of subtee of etree 	*/
	int_t rset, cset;
	int_t row, col;
	int_t rroot;
	int_t p;
	int_t *pp;

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(0, "Enter sp_symetree()");
#endif

	root = mxCallocInt(n);
	initialize_disjoint_sets(n, &pp);

	for (col = 0; col < n; col++)
	{
		cset = make_set(col, pp);
		root[cset] = col;
		parent[col] = n; /* Matlab */ // (Yida) : Vertex whose parent is n (vtx n is not exist) is root.
		for (p = acolst[col]; p < acolend[col]; p++)
		{
			row = arow[p];
			if (row >= col)
				continue;
			rset = find(row, pp);
			rroot = root[rset];
			if (rroot != col)
			{
				parent[rroot] = col;
				cset = link(cset, rset, pp);
				root[cset] = col;
			}
		}
	}
	SUPERLU_FREE(root);
	finalize_disjoint_sets(pp);

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(0, "Exit sp_symetree()");
#endif
	return 0;
} /* SP_SYMETREE_DIST */

/*! \brief Nonsymmetric elimination tree
 *
 * <pre>
 *      Find the elimination tree for A'*A.
 *      This uses something similar to Liu's algorithm.
 *      It runs in time O(nz(A)*log n) and does not form A'*A.
 *
 *      Input:
 *        Sparse matrix A.  Numeric values are ignored, so any
 *        explicit zeros are treated as nonzero.
 *      Output:
 *        Integer array of parents representing the elimination
 *        tree of the symbolic product A'*A.  Each vertex is a
 *        column of A, and nc means a root of the elimination forest.
 *
 *      John R. Gilbert, Xerox, 10 Dec 1990
 *      Based on code by JRG dated 1987, 1988, and 1990.
 * </pre>
 */
int sp_coletree_dist(
	int_t *acolst, int_t *acolend, /* column start and end past 1 */
	int_t *arow,				   /* row indices of A */
	int_t nr, int_t nc,			   /* dimension of A */
	int_t *parent				   /* parent in elim tree */
)
{
	int_t *root;	 /* root of subtee of etree 	*/
	int_t *firstcol; /* first nonzero col in each row*/
	int_t rset, cset;
	int_t row, col;
	int_t rroot;
	int_t p;
	int_t *pp;

#if (DEBUGlevel >= 1)
	int iam = 0;
	CHECK_MALLOC(iam, "Enter sp_coletree()");
#endif

	root = mxCallocInt(nc);
	initialize_disjoint_sets(nc, &pp);

	/* Compute firstcol[row] = first nonzero column in row */

	firstcol = mxCallocInt(nr);
	for (row = 0; row < nr; firstcol[row++] = nc)
		;
	for (col = 0; col < nc; col++)
		for (p = acolst[col]; p < acolend[col]; p++)
		{
			row = arow[p];
			firstcol[row] = SUPERLU_MIN(firstcol[row], col);
		}

	/* Compute etree by Liu's algorithm for symmetric matrices,
		   except use (firstcol[r],c) in place of an edge (r,c) of A.
	   Thus each row clique in A'*A is replaced by a star
	   centered at its first vertex, which has the same fill. */

	for (col = 0; col < nc; col++)
	{
		cset = make_set(col, pp);
		root[cset] = col;
		parent[col] = nc; /* Matlab */
		for (p = acolst[col]; p < acolend[col]; p++)
		{
			row = firstcol[arow[p]];
			if (row >= col)
				continue;
			rset = find(row, pp);
			rroot = root[rset];
			if (rroot != col)
			{
				parent[rroot] = col;
				cset = link(cset, rset, pp);
				root[cset] = col;
			}
		}
	}

	SUPERLU_FREE(root);
	SUPERLU_FREE(firstcol);
	finalize_disjoint_sets(pp);

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit sp_coletree()");
#endif
	return 0;
} /* SP_COLETREE_DIST */

/*! \brief Depth-first search from vertext
 *
 * <pre>
 *  q = TreePostorder_dist (n, p);
 *
 *	Postorder a tree.
 *	Input:
 *	  p is a vector of parent pointers for a forest whose
 *        vertices are the integers 0 to n-1; p[root]==n.
 *	Output:
 *	  q is a vector indexed by 0..n-1 such that q[i] is the
 *	  i-th vertex in a postorder numbering of the tree.
 *
 *        ( 2/7/95 modified by X.Li:
 *          q is a vector indexed by 0:n-1 such that vertex i is the
 *          q[i]-th vertex in a postorder numbering of the tree.
 *          That is, this is the inverse of the previous q. )
 *
 *	In the child structure, lower-numbered children are represented
 *	first, so that a tree which is already numbered in postorder
 *	will not have its order changed.
 *
 *  Written by John Gilbert, Xerox, 10 Dec 1990.
 *  Based on code written by John Gilbert at CMI in 1987.
 * </pre>
 */

// static int_t	*first_kid, *next_kid;	/* Linked list of children.	*/
// static int_t	*post, postnum;

static
	/*
	 * Depth-first search from vertex v.
	 */
	void
	etdfs(
		int_t v,
		int_t first_kid[],
		int_t next_kid[],
		int_t post[],
		int_t *postnum)
{
	int w;

	for (w = first_kid[v]; w != -1; w = next_kid[w])
	{
		etdfs(w, first_kid, next_kid, post, postnum);
	}
	/* post[postnum++] = v; in Matlab */
	post[v] = (*postnum)++; /* Modified by X. Li on 08/10/07 */
}

static
	/*
	 * Depth-first search from vertex n.
	 * No recursion.
	 */
	void
	nr_etdfs(int_t n, int_t *parent,
			 int_t *first_kid, int_t *next_kid,
			 int_t *post, int_t postnum)
{
	int_t current = n, first, next;

	while (postnum != n)
	{

		/* no kid for the current node */
		first = first_kid[current];

		/* no first kid for the current node */
		if (first == -1)
		{

			/* numbering this node because it has no kid */
			post[current] = postnum++;

			/* looking for the next kid */
			next = next_kid[current];

			while (next == -1)
			{

				/* no more kids : back to the parent node */
				current = parent[current];

				/* numbering the parent node */
				post[current] = postnum++;

				/* get the next kid */
				next = next_kid[current];
			}

			/* stopping criterion */
			if (postnum == n + 1)
				return;

			/* updating current node */
			current = next;
		}
		/* updating current node */
		else
		{
			current = first;
		}
	}
}

/*
 * Post order a tree
 */
int_t *TreePostorder_dist(
	int_t n,
	int_t *parent)
{
	int_t v, dad;
	int_t *first_kid, *next_kid, *post, postnum;

	/* Allocate storage for working arrays and results	*/
	if (!(first_kid = mxCallocInt(n + 1)))
		ABORT("mxCallocInt fails for first_kid[]");
	if (!(next_kid = mxCallocInt(n + 1)))
		ABORT("mxCallocInt fails for next_kid[]");
	if (!(post = mxCallocInt(n + 1)))
		ABORT("mxCallocInt fails for post[]");

	/* Set up structure describing children */
	for (v = 0; v <= n; first_kid[v++] = -1)
		;
	for (v = n - 1; v >= 0; v--)
	{
		dad = parent[v];
		next_kid[v] = first_kid[dad];
		first_kid[dad] = v;
	}

	/* Depth-first search from dummy root vertex #n */
	postnum = 0;
#if 0
	/* recursion */
	etdfs (n, first_kid, next_kid, post, &postnum);
#else
	/* no recursion */
	nr_etdfs(n, parent, first_kid, next_kid, post, postnum);
#endif

	SUPERLU_FREE(first_kid);
	SUPERLU_FREE(next_kid);
	return post;
}
