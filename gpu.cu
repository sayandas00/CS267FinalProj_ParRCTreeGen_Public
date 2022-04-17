#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int edge_blks;
int vertex_blks;
int rowLen;
int numBins;

// CPU/GPU version of various arrays - correspond to steps 1/2/3 of section slides
unsigned int* gpu_binCounts;
int* gpu_prefixSum;
int* gpu_sortedParts;

void init_process(edge_t* edges, int num_vertices, int num_edges) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // edges should live in gpu memory!

    edge_blks = (num_edges + NUM_THREADS - 1) / NUM_THREADS;
    vertex_blks = (num_vertices + NUM_THREADS - 1) / NUM_THREADS;

    cudaError_t err;
    err = cudaMalloc((void**) &gpu_binCounts, numBins*sizeof(unsigned int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMemset(gpu_binCounts, 0, numBins*sizeof(unsigned int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_prefixSum, (numBins+1)*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_sortedParts, num_parts*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }

}


void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges) {
    // edges live in GPU memory


}


