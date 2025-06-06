/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief SuperLU grid utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * February 8, 2019  version 6.1.1
 * October 5, 2021
 * </pre>
 */

#include "superlu_ddefs.h"

#if 0 // obsolete
/* Define global variables */
MPI_Datatype SuperLU_MPI_DOUBLE_COMPLEX = MPI_DATATYPE_NULL;
#endif

/*! \brief All processes in the MPI communicator must call this routine.
 *
 *  On output, if a process is not in the SuperLU group, the following
 *  values are assigned to it:
 *      grid->comm = MPI_COMM_NULL
 *      grid->iam = -1
 */
void superlu_gridinit(MPI_Comm Bcomm, /* The base communicator upon which
                     the new grid is formed. */
                      int nprow, int npcol, gridinfo_t *grid)
{
    int Np = nprow * npcol;
    int *usermap;
    int i, j, info;

    /* Make a list of the processes in the new communicator. */
    usermap = SUPERLU_MALLOC(Np * sizeof(int));
    for (j = 0; j < npcol; ++j)
        for (i = 0; i < nprow; ++i)
            usermap[j * nprow + i] = i * npcol + j;

    /* Check MPI environment initialization. */
    MPI_Initialized(&info);
    if (!info)
        ABORT("C main program must explicitly call MPI_Init()");

    MPI_Comm_size(Bcomm, &info);
    if (info < Np)
    {
        printf("Number of processes %d is smaller than NPROW * NPCOL %d", info, Np);
        exit(-1);
    }

    superlu_gridmap(Bcomm, nprow, npcol, usermap, nprow, grid);

    SUPERLU_FREE(usermap);

#ifdef GPU_ACC
    /* Binding each MPI to a GPU device */
    char *ttemp;
    ttemp = getenv("SUPERLU_BIND_MPI_GPU");

    if (ttemp)
    {
        int devs, rank;
        MPI_Comm_rank(Bcomm, &rank);                                      // MPI_COMM_WORLD??
        gpuGetDeviceCount(&devs);                                         // Returns the number of compute-capable devices
        int device_id = (int)(rank / get_mpi_process_per_gpu()) % (devs); // YL: allow multiple MPIs per GPU
        gpuSetDevice(device_id);                                          // Set device to be used for GPU executions

        int get_cur_dev;
        gpuGetDevice(&get_cur_dev);
        printf("** MPI rank %d, gpu=%d **\n", rank, get_cur_dev);
        fflush(stdout);
    }
#endif
}

/*! \brief All processes in the MPI communicator must call this routine.
 *
 *  On output, if a process is not in the SuperLU group, the following
 *  values are assigned to it:
 *      grid->comm = MPI_COMM_NULL
 *      grid->iam = -1
 */
void superlu_gridmap(
    MPI_Comm Bcomm, /* The base communicator upon which
           the new grid is formed. */
    int nprow,
    int npcol,
    int usermap[], /* usermap(i,j) holds the process
            number to be placed in {i,j} of
            the process grid.  */
    int ldumap,    /* The leading dimension of the
            2D array usermap[].  */
    gridinfo_t *grid)
{
    MPI_Group mpi_base_group, superlu_grp;
    int Np = nprow * npcol, mycol, myrow;
    int *pranks;
    int i, j, info;

#if 0 // older MPI doesn't support complex in C    
    /* Create datatype in C for MPI complex. */
    if ( SuperLU_MPI_DOUBLE_COMPLEX == MPI_DATATYPE_NULL ) {
	MPI_Type_contiguous( 2, MPI_DOUBLE, &SuperLU_MPI_DOUBLE_COMPLEX );
	MPI_Type_commit( &SuperLU_MPI_DOUBLE_COMPLEX );
    }
#endif

    /* Check MPI environment initialization. */
    MPI_Initialized(&info);
    if (!info)
        ABORT("C main program must explicitly call MPI_Init()");

    grid->nprow = nprow;
    grid->npcol = npcol;

    /* Make a list of the processes in the new communicator. */
    pranks = (int *)SUPERLU_MALLOC(Np * sizeof(int));
    for (j = 0; j < npcol; ++j)
        for (i = 0; i < nprow; ++i)
            pranks[i * npcol + j] = usermap[j * ldumap + i];

    /*
     * Form MPI communicator for all.
     */
    /* Get the group underlying Bcomm. */
    MPI_Comm_group(Bcomm, &mpi_base_group);
    /* Create the new group. */
    MPI_Group_incl(mpi_base_group, Np, pranks, &superlu_grp);
    /* Create the new communicator. */
    /* NOTE: The call is to be executed by all processes in Bcomm,
       even if they do not belong in the new group -- superlu_grp.
       The function returns MPI_COMM_NULL to processes that are not in superlu_grp. */
    MPI_Comm_create(Bcomm, superlu_grp, &grid->comm);

    /* Bail out if I am not in the group "superlu_grp". */
    if (grid->comm == MPI_COMM_NULL)
    {
        // grid->comm = Bcomm;  do not need to reassign to a valid communicator
        grid->iam = -1;
        // SUPERLU_FREE(pranks);
        // return;
        goto gridmap_out;
    }

    MPI_Comm_rank(grid->comm, &(grid->iam));
    myrow = grid->iam / npcol;
    mycol = grid->iam % npcol;

    /*
     * Form MPI communicator for myrow, scope = COMM_ROW.
     */
#if 0
    for (i = 0; i < npcol; ++i) pranks[i] = myrow*npcol + i;
    MPI_Comm_group( grid->comm, &superlu_grp );          /* Find all's group */
    MPI_Group_incl( superlu_grp, npcol, pranks, &grp );  /* Form new group */
    MPI_Comm_create( grid->comm, grp, &grid->rscp.comm );/* Create new comm */
#else
    MPI_Comm_split(grid->comm, myrow, mycol, &(grid->rscp.comm));
#endif

    /*
     * Form MPI communicator for mycol, scope = COMM_COLUMN.
     */
#if 0
    for (i = 0; i < nprow; ++i) pranks[i] = i*npcol + mycol;
    MPI_Group_incl( superlu_grp, nprow, pranks, &grp );  /* Form new group */
    MPI_Comm_create( grid->comm, grp, &grid->cscp.comm );/* Create new comm */
#else
    MPI_Comm_split(grid->comm, mycol, myrow, &(grid->cscp.comm));
#endif

    grid->rscp.Np = npcol;
    grid->rscp.Iam = mycol;
    grid->cscp.Np = nprow;
    grid->cscp.Iam = myrow;

#if 0
    {
	int tag_ub;
	if ( !grid->iam ) {
	    MPI_Comm_get_attr(Bcomm, MPI_TAG_UB, &tag_ub, &info);
	    printf("MPI_TAG_UB %d\n", tag_ub);
	    /* returns 4295677672
	       In reality it is restricted to no greater than 16384. */
	}
	exit(0);
    }
#endif

gridmap_out:
    SUPERLU_FREE(pranks);
    MPI_Group_free(&superlu_grp);
    MPI_Group_free(&mpi_base_group);

} /* superlu_gridmap */

void superlu_gridexit(gridinfo_t *grid)
{
    if (grid->comm != MPI_COMM_NULL)
    {
        /* Marks the communicator objects for deallocation. */
        MPI_Comm_free(&grid->rscp.comm);
        MPI_Comm_free(&grid->cscp.comm);
        MPI_Comm_free(&grid->comm);
    }
#if 0    
    if ( SuperLU_MPI_DOUBLE_COMPLEX != MPI_DATATYPE_NULL ) {
	MPI_Type_free( &SuperLU_MPI_DOUBLE_COMPLEX );
	SuperLU_MPI_DOUBLE_COMPLEX = MPI_DATATYPE_NULL; /* some MPI system does not set this
							   to be NULL after Type_free */
    }
#endif
}
