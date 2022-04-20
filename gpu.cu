#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int edge_blks;
int vertex_blks;
int rcTree_blks;

int edgesAllocated;

// GPU version of edge adjacency lists
unsigned int* gpu_degCounts;
int* gpu_degPrefixSum;
int* gpu_edgeAdjList;

// GPU version of new rc tree edge and vertex list
rcTreeNode_t* gpu_rcTreeNodes; //original vertices first, then original edges, then clusters
edge_t* gpu_rcTreeEdges;
int rcTreeVertices; // any node entry >= to this number is unallocated
int lenRCTreeArrays;


// use GPU to initialize rcTreeEdges and rcTreeNodes
__global__ void init_rcTreeArrays(int len, int num_vertices, int num_edges) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len)
        return;
    // init edge list
    gpu_rcTreeEdges[tid].valid = false;
    gpu_rcTreeEdges[tid].id = eid + 1;
    gpu_rcTreeEdges[tid].weight = -1;
    gpu_rcTreeEdges[tid].vertex_1 = -1;
    gpu_rcTreeEdges[tid].vertex_2 = -1;

    if (tid >= num_vertices) {
        // original edges
        gpu_rcTreeNodes[tid].cluster_degree = -1;
        gpu_rcTreeNodes[tid].rep_vertex = -1;
        gpu_rcTreeNodes[tid].vertex_id = -1;
        gpu_rcTreeNodes[tid].vertex_id = -1;
        gpu_rcTreeNodes[tid].bound_vertex_1 = -1;
        gpu_rcTreeNodes[tid].bound_vertex_2 = -1;
        gpu_rcTreeNodes[tid].edge_id = tid - num_vertices + 1;
    } else {
        // original vertices
        gpu_rcTreeNodes[tid].cluster_degree = -1;
        gpu_rcTreeNodes[tid].rep_vertex = -1;
        gpu_rcTreeNodes[tid].vertex_id = tid + 1;
        gpu_rcTreeNodes[tid].edge_id = -1;
        gpu_rcTreeNodes[tid].bound_vertex_1 = -1;
        gpu_rcTreeNodes[tid].bound_vertex_2 = -1;
    }
}



void init_process(edge_t* edges, int num_vertices, int num_edges) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // edges should live in gpu memory!

    edge_blks = ((2*num_edges) + NUM_THREADS - 1) / NUM_THREADS;
    vertex_blks = (num_vertices + NUM_THREADS - 1) / NUM_THREADS;
    lenRCTreeArrays = 2*num_vertices + num_edges;
    rcTree_blks = (lenRCTreeArrays + NUM_THREADS - 1) / NUM_THREADS;

    edgesAllocated = num_edges;
    rcTreeVertices = num_vertices + num_edges;
    cudaError_t err;
    
    err = cudaMalloc((void**) &gpu_degCounts, num_vertices*sizeof(unsigned int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMemset(gpu_degCounts, 0, num_vertices*sizeof(unsigned int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_degPrefixSum, (num_vertices+1)*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    // edges inserted twice into edgeAdjList, once for each vertex connected to
    err = cudaMalloc((void**) &gpu_edgeAdjList, num_edges*2*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_rcTreeNodes, lenRCTreeArrays*sizeof(rcTreeNode_t));
    if(err) {
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_rcTreeEdges, lenRCTreeArrays*sizeof(edge_t));
    if(err) {
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    // need gpu function to initialize gpu_rcTreeNodes and gpu_rcTreeEdges properly
    init_rcTreeArrays<<<rcTree_blks, NUM_THREADS>>>(lenRCTreeArrays, num_vertices, num_edges);

    // might need a synchronize before exiting to ensure that process initialization has completed before exiting
    cudaDeviceSynchronize();
}


void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges) {
    // edges live in GPU memory

    // 1. parallelize by edge -> count vertex degrees

    // 2. parallelize by vertex -> read degree of self -> rake/compress

}


