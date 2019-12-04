/*
 * knnring_mpi_sync.c
 *
 *  Created on: Nov 24, 2019
 *      Author: Lambis
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "knnring.h"

knnresult distrAllkNN(double * X, int n, int d, int k) {

    knnresult knn;

    // Allocating memory for the knnresult.
    knn.nidx = (int *)malloc(n*k*sizeof(int));
    knn.ndist = (double *)malloc(n*k*sizeof(double));

    // Variables for time measurements.
    double start, end;

    // Get the number of processes.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process.
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Set MPI communication tag and status.
    int tag = 1;
    MPI_Status status;

    // Allocating memory for corpus points.
    double* Y = (double *)malloc(n*d * sizeof(double));

    // Allocating memory for points to be received.
    double* Y_temp = (double *)malloc(n*d * sizeof(double));

    // For the first iteration, the corpus points match the query points.
    memcpy(Y, X, n*d * sizeof(double));

    start = MPI_Wtime();

    // First iteration call. Each node calculates distances between its own points.
    knn = kNN(Y, X, n, n, d, k);

    end = MPI_Wtime();
    printf("Node %d: Calculation #0 took %f seconds.\n", world_rank, end-start);

    // Correcting the corpus points' indexes after first iteration.
    for (int i=0; i<n*k; i++)
        knn.nidx[i] += ( (world_rank-1+world_size) % world_size) * n;

    for (int i=1; i<world_size; i++) {

        start = MPI_Wtime();

        // Nodes with even or 0 rank send and nodes with odd rank receive.
        if (world_rank % 2 == 0)
            MPI_Send(Y, n*d, MPI_DOUBLE, (world_rank+1) % world_size, tag, MPI_COMM_WORLD);
        else
            MPI_Recv(Y_temp, n*d, MPI_DOUBLE, (world_rank-1+world_size) % world_size, tag, MPI_COMM_WORLD, &status);

        // Nodes with odd rank send and nodes with even or 0 rank receive.
        if (world_rank % 2 != 0)
            MPI_Send(Y, n*d, MPI_DOUBLE, (world_rank+1) % world_size, tag, MPI_COMM_WORLD);
        else
            MPI_Recv(Y_temp, n*d, MPI_DOUBLE, (world_rank-1+world_size) % world_size, tag, MPI_COMM_WORLD, &status);

        end = MPI_Wtime();
        printf("Node %d: Communication #%d took %f seconds.\n", world_rank, i-1, end-start);

        // Copying received array into corpus array.
        memcpy(Y, Y_temp, n*d * sizeof(double));

        start = MPI_Wtime();

        // Calculating knn between query points and new corpus points.
        knnresult knn_temp = kNN(Y, X, n, n, d, k);

        // Checking if a shorter distance is present.
        for (int l=0; l<knn.m; l++) {

            for (int j=0; j<k; j++) {

                // There is no shorter distance.
                if (knn_temp.ndist[l*k+j] >= knn.ndist[l*k+(k-1)]) {
                    break;
                }
                // There is a shorter distance. Passing it to the final knnresult struct and correcting its corpus point's index.
                else {
                    knn.ndist[l*k+(k-1)] = knn_temp.ndist[l*k+j];
                    knn.nidx[l*k+(k-1)] = knn_temp.nidx[l*k+j] + ( ( ( (world_rank-1+world_size) % world_size) - i+world_size) % world_size ) * n;

                    // Sorting again the distances in the final knnresult struct.
                    quickSort(&knn.ndist[l*k], 0, k-1, &knn.nidx[l*k]);
                }

            }

        }

        end = MPI_Wtime();
        printf("Node %d: Calculation #%d took %f seconds.\n", world_rank, i, end-start);

    }

    // Cleanup.
    free(Y_temp);
    free(Y);

    return knn;
}
