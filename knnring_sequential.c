/*
 * knnring_sequential.c
 *
 *  Created on: Nov 21, 2019
 *      Author: Lambis
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "knnring.h"

knnresult kNN(double * X, double * Y, int n, int m, int d, int k) {

	knnresult knn;

    // Allocating memory for the knnresult.
	knn.nidx = (int *)malloc(m*k*sizeof(int));
	knn.ndist = (double *)malloc(m*k*sizeof(double));

	// Passing m and k values to knnresult.
	knn.m = m;
	knn.k = k;

    // Allocating memory for the distances array.
	double *D = (double *)malloc(m*n*sizeof(double));

	// Calculation of sum(X.^2,2).
	double *a = (double *)malloc(n*sizeof(double));
	for (int i=0; i<n; i++) {
		a[i] = cblas_dnrm2(d, &X[i*d], 1);
		a[i] = a[i]*a[i];
	}

    // Calculation of sum(Y.^2,2).
	double *b = (double *)malloc(m*sizeof(double));
	for (int i=0; i<m; i++) {
		b[i] = cblas_dnrm2(d, &Y[i*d], 1);
		b[i] = b[i]*b[i];
	}

    // Calculation of -2*X*Y.' multiplication.
    double *c = (double *)malloc(n*m*sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, c, m);

	// Adding sum(X.^2,2) sum(Y.^2,2).' to the c array.
    for (int i=0; i<n; i++)
		for(int j=0; j<m; j++)
			c[i*m+j] += a[i] + b[j];

    // Cleanup.
    free(a);
    free(b);

    // Applying elementwise square root to the c array.
	for (int i=0; i<n; i++)
		for(int j=0; j<m; j++)
			c[i*m+j] = sqrt(fabs(c[i*m+j]));

    // Transposing c array and storing it in D.
    cblas_domatcopy(CblasRowMajor, CblasTrans, n, m, 1, c, m, D, n);

    // Cleanup.
    free(c);

    // Setup of idx number for corpus points.
	int *idx = (int *)malloc(m*n*sizeof(int));
	for (int i=0; i<m; i++)
        for (int j=0; j<n; j++)
            idx[i*n+j]= j;

    // Sorting of all distances.
	for (int i=0; i<m; i++)
		quickSort(D, i*n, ((i+1)*n)-1, idx);

    // Passing the k nearest neighbors' indexes and distances to knnresult.
	for (int i=0; i<m; i++) {
		for (int j=0; j<k; j++) {
			knn.nidx[i*k+j] = idx[i*n+j];
			knn.ndist[i*k+j] = D[i*n+j];
		}
	}

	return knn;

}

// A utility function to swap two elements.
void swap(void* a, void* b, size_t s) {
    void* tmp = malloc(s);
    memcpy(tmp, a, s);
    memcpy(a, b, s);
    memcpy(b, tmp, s);
    free(tmp);
}

/* This function takes last element as pivot, places the pivot element at its correct position in sorted array
 * and places all smaller (smaller than pivot) to left of pivot and all greater elements to right of pivot */
int partition(double *arr, int low, int high, int *idx) {
    double pivot = arr[high];    		// Pivot
    int i = (low - 1);  			// Index of smaller element

    for (int j=low; j<=high-1; j++) {
        // If current element is smaller than the pivot
        if (arr[j] < pivot) {
            i++;    				// Increment index of smaller element
            swap(&arr[i], &arr[j], sizeof(double));
            swap(&idx[i], &idx[j], sizeof(int));
        }
    }
    swap(&arr[i + 1], &arr[high], sizeof(double));
    swap(&idx[i + 1], &idx[high], sizeof(int));
    return (i + 1);
}

// The main function that implements QuickSort arr[] --> Array to be sorted, low  --> Starting index, high  --> Ending index
void quickSort(double *arr, int low, int high, int *idx) {
    if (low < high) {
        // pi is partitioning index, arr[pi] is now at right place
        int pi = partition(arr, low, high, idx);

        // Separately sort elements before partition and after partition
        quickSort(arr, low, pi - 1, idx);
        quickSort(arr, pi + 1, high, idx);
    }
}
