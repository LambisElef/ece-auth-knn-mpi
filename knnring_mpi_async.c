/*
 * knnring_mpi_async.c
 *
 *  Created on: Nov 26, 2019
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

    // Set MPI communication tag, request and status.
    int tag = 1;
    MPI_Request send_request, recv_request;
    MPI_Status status;

    // Allocating memory for corpus points.
    double* Y = (double *)malloc(n*d * sizeof(double));

    // Allocating memory for points to be received.
    double* Y_temp = (double *)malloc(n*d * sizeof(double));

    // For the first iteration, the corpus points match the query points.
    memcpy(Y, X, n*d * sizeof(double));

    // Sending and receiving points for the next iteration.
    MPI_Isend(Y, n*d, MPI_DOUBLE, (world_rank+1) % world_size, tag, MPI_COMM_WORLD, &send_request);
    MPI_Irecv(Y_temp, n*d, MPI_DOUBLE, (world_rank-1+world_size) % world_size, tag, MPI_COMM_WORLD, &recv_request);

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

        // Wait if previous transfer isn't finished yet.
        MPI_Wait(&send_request, &status);
        MPI_Wait(&recv_request, &status);

        end = MPI_Wtime();
        printf("Node %d: Communication #%d stalled the execution for %f seconds.\n", world_rank, i-1, end-start);

        // Copying received array into corpus array.
        memcpy(Y, Y_temp, n*d * sizeof(double));

        // No further communication needed if i reaches world_size-1.
        if (i != world_size-1) {
            // Sending and receiving points for the next iteration.
            MPI_Isend(Y, n*d, MPI_DOUBLE, (world_rank+1) % world_size, tag, MPI_COMM_WORLD, &send_request);
            MPI_Irecv(Y_temp, n*d, MPI_DOUBLE, (world_rank-1+world_size) % world_size, tag, MPI_COMM_WORLD, &recv_request);
        }

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

    // Variables for finding the total minimun and maximum distances of the knn neighbors.
    double node_minDist, node_maxDist, total_minDist, total_maxDist;

    // Assigning the first point's minimum and maximum distances as nodes minimum and maximum distances excluding the zero distance from its self.
    node_minDist = knn.ndist[1];
    node_maxDist = knn.ndist[k-1];

    for (int i=1; i<n; i++) {
        if (knn.ndist[i*k+1] < node_minDist)
	    node_minDist = knn.ndist[i*k+1];
	if(knn.ndist[i*k + k-1] > node_maxDist)
	    node_maxDist = knn.ndist[i*k + k-1];
    }

    // MPI Reduction and saving minimum distance to total_minDist and maximum distance to total_maxDist.
    MPI_Reduce(&node_minDist, &total_minDist, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&node_maxDist, &total_maxDist, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return knn;
}
