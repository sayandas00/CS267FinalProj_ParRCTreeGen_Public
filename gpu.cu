#include "common.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <iostream>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int edge_blks;
int vertex_blks;
int rcTree_blks;

int* gpu_edgesAllocated;

// GPU version of edge adjacency lists
unsigned int* gpu_degCounts;
int* gpu_degPrefixSum;
int* gpu_edgeAdjList;

// GPU version new rc tree edge and vertex numbers
int* num_rcTreeVertices; // any node entry >= to this number is unallocated
int lenRCTreeArrays;

// Citation: curand setup code https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand
curandState* gpu_randStates;
float* gpu_randValues;
int* lubyConsiderNodes;

__global__ void count_degree(edge_t* edges, int len, unsigned int* degCounts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len)
        return;
    if (edges[tid].valid) {
        edges[tid].marked = 0;
        int vertex_1 = edges[tid].vertex_1;
        int vertex_2 = edges[tid].vertex_2;
        atomicAdd(&degCounts[vertex_1 - 1], 1);
        atomicAdd(&degCounts[vertex_2 - 1], 1);
    }
}

// Citation: curand setup code https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand
__global__ void gpu_random(int num_vertices, curandState *states, int offset, float* randValues) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_vertices) {
        return;
    }
	int seed = tid + offset; // different seed per thread
    curand_init(seed, tid, 0, &states[tid]);  // 	Initialize CURAND
    atomicExch(&randValues[tid], 0);
}


// use GPU to initialize edges
__global__ void init_emptyEdges(edge_t* edges, int len, int num_edges) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ((tid >= len) || (tid < num_edges))
        return;
    // init edge list
    edges[tid].vertex_1 = -1;
    edges[tid].vertex_2 = -1;
    edges[tid].weight = -1;
    edges[tid].valid = false;
    edges[tid].id = tid + 1;
    edges[tid].marked = 0;
}


// use GPU to initialize rcTreeEdges and rcTreeNodes
__global__ void init_rcTreeArrays(int len, int num_vertices, int num_edges, edge_t* rcTreeEdges, rcTreeNode_t* rcTreeNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len)
        return;
    // init edge list
    rcTreeEdges[tid].valid = false;
    rcTreeEdges[tid].marked = 0;
    rcTreeEdges[tid].id = tid + 1;
    rcTreeEdges[tid].weight = -1;
    rcTreeEdges[tid].vertex_1 = -1;
    rcTreeEdges[tid].vertex_2 = -1;

    if ((tid >= num_vertices) && (tid < num_vertices + num_edges)) {
        // original edges
        rcTreeNodes[tid].cluster_degree = -1;
        rcTreeNodes[tid].rep_vertex = -1;
        rcTreeNodes[tid].vertex_id = -1;
        rcTreeNodes[tid].vertex_id = -1;
        rcTreeNodes[tid].bound_vertex_1 = -1;
        rcTreeNodes[tid].bound_vertex_2 = -1;
        rcTreeNodes[tid].edge_id = tid - num_vertices + 1;
    } else if (tid < num_vertices) {
        // original vertices
        rcTreeNodes[tid].cluster_degree = -1;
        rcTreeNodes[tid].rep_vertex = -1;
        rcTreeNodes[tid].vertex_id = tid + 1;
        rcTreeNodes[tid].edge_id = -1;
        rcTreeNodes[tid].bound_vertex_1 = -1;
        rcTreeNodes[tid].bound_vertex_2 = -1;
    } else {
        rcTreeNodes[tid].cluster_degree = -1;
        rcTreeNodes[tid].rep_vertex = -1;
        rcTreeNodes[tid].vertex_id = -1;
        rcTreeNodes[tid].edge_id = -1;
        rcTreeNodes[tid].bound_vertex_1 = -1;
        rcTreeNodes[tid].bound_vertex_2 = -1;
    }
}

__global__ void build_adjList(edge_t* edges, int len, int* edgeAdjList, int* degPrefixSum, unsigned int* degCounts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len)
        return;
    if (edges[tid].valid) {
        int vertex_1_posn = edges[tid].vertex_1 - 1;
        int vertex_2_posn = edges[tid].vertex_2 - 1;
        // insert edge id into edgeAdjList for vertex 1
        int offset = atomicSub(&degCounts[vertex_1_posn], 1) - 1;
        edgeAdjList[degPrefixSum[vertex_1_posn] + offset] = edges[tid].id;
        // insert edge id into edgeAdjList for vertex 2
        offset = atomicSub(&degCounts[vertex_2_posn], 1) - 1;
        edgeAdjList[degPrefixSum[vertex_2_posn] + offset] = edges[tid].id;
    }
}

// determine randomValue for IS determination
__global__ void genRandValues(edge_t* edges, int num_vertices, int* edgeAdjList, int* degPrefixSum, int root_vertex, curandState* randStates, float* randValues, int* numLubyNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_vertices)
        return;
    // check degree of vertex to see if rake or compress must be performed
    int deg = degPrefixSum[tid + 1] - degPrefixSum[tid];
    if (deg == 0) {
        return;
    } else if (atomicAdd(&randValues[tid], 0) == 2) {
        return;
    } else if (tid == root_vertex - 1) {
        atomicExch(&randValues[tid], 2);
        atomicSub(numLubyNodes, 1);
    } else if (deg == 1) {
        int edge_id = edgeAdjList[degPrefixSum[tid]];
        int neighbor_id = edges[edge_id - 1].vertex_1;
        if (neighbor_id == tid + 1) {
            neighbor_id = edges[edge_id - 1].vertex_2;
        }
        int neighbor_deg = degPrefixSum[neighbor_id] - degPrefixSum[neighbor_id - 1];
        if ((neighbor_deg == 1) && (neighbor_id > tid) && (neighbor_id != root_vertex)) {
            atomicExch(&randValues[tid], 2);
            atomicSub(numLubyNodes, 1);
            return;
        }
        atomicExch(&randValues[tid], -1);
        // remove both this node from luby set
        atomicSub(numLubyNodes, 1);
    } else if (deg == 2) {
        // check neighbor vertices to see if they are both not degree 1
        int edge_id_1 = edgeAdjList[degPrefixSum[tid]];
        int neighbor_id_1 = edges[edge_id_1 - 1].vertex_1;
        if (neighbor_id_1 == tid + 1) {
            neighbor_id_1 = edges[edge_id_1 - 1].vertex_2;
        }
        int neighbor_deg = degPrefixSum[neighbor_id_1] - degPrefixSum[neighbor_id_1 - 1];
        if ((neighbor_deg == 1) && (neighbor_id_1 != root_vertex)) {
            atomicExch(&randValues[tid], 2);
            atomicSub(numLubyNodes, 1);
            return;
        }
        int edge_id_2 = edgeAdjList[degPrefixSum[tid] + 1];
        int neighbor_id_2 = edges[edge_id_2 - 1].vertex_1;
        if (neighbor_id_2 == tid + 1) {
            neighbor_id_2 = edges[edge_id_2 - 1].vertex_2;
        }
        neighbor_deg = degPrefixSum[neighbor_id_2] - degPrefixSum[neighbor_id_2 - 1];
        if ((neighbor_deg == 1) && (neighbor_id_2 != root_vertex)) {
            atomicExch(&randValues[tid], 2);
            atomicSub(numLubyNodes, 1);
            return;
        }
        atomicExch(&randValues[tid], curand_uniform(&randStates[tid]));
    } else {
        atomicExch(&randValues[tid], 2);
        atomicSub(numLubyNodes, 1);
    }
}

// determine who to add to maximal IS among compress eligible nodes
__global__ void addToMIS(edge_t* edges, int num_vertices, int* edgeAdjList, int* degPrefixSum, float* randValues, int* numLubyNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_vertices)
        return;
    float currRandValue = atomicAdd(&randValues[tid], 0);
    if ((currRandValue == -1) || (currRandValue == 2)) {
        return;
    }
    // min over two neighbors
    int edge_id_1 = edgeAdjList[degPrefixSum[tid]];
    int neighbor_id_1 = edges[edge_id_1 - 1].vertex_1;
    if (neighbor_id_1 == tid + 1) {
        neighbor_id_1 = edges[edge_id_1 - 1].vertex_2;
    }
    int edge_id_2 = edgeAdjList[degPrefixSum[tid] + 1];
    int neighbor_id_2 = edges[edge_id_2 - 1].vertex_1;
    if (neighbor_id_2 == tid + 1) {
        neighbor_id_2 = edges[edge_id_2 - 1].vertex_2;
    }
    float neighbor_1_randValue = atomicAdd(&randValues[neighbor_id_1 - 1], 0);
    float neighbor_2_randValue = atomicAdd(&randValues[neighbor_id_2 - 1], 0);
    if ((currRandValue >= neighbor_1_randValue) || (currRandValue >= neighbor_1_randValue) ) {
        return;
    }
    // minimum of neighbors, update numLubyNodes
    atomicExch(&randValues[tid], -1);
    atomicExch(&randValues[neighbor_id_1 - 1], 2);
    atomicExch(&randValues[neighbor_id_2 - 1], 2);
    atomicSub(numLubyNodes, 3);
}



// only call after degree check, num_edges and num_vertices are of the original base graph
__device__ void rake(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int tid, int iter, unsigned int* degCounts, int root_vertex) {
    // check degree of neighbor, need to account for base case of 2 1 degree vertices
    // larger vertex id will rake
    int edge_id = edgeAdjList[degPrefixSum[tid]];
    int neighbor_id = edges[edge_id - 1].vertex_1;
    if (neighbor_id == tid + 1) {
        neighbor_id = edges[edge_id - 1].vertex_2;
    }

    // mark edge unvalid and get vertex id of neighbor
    edges[edge_id - 1].valid = false;

    // update degCount for neighbor since they are losing an edge after the rake
    atomicSub(&degCounts[neighbor_id - 1], 1);

    // get new rcTreeCluster
    int newRCTreeClust = atomicAdd(numRCTreeVertices, 1);
    rcTreeNodes[newRCTreeClust].cluster_degree = 1;
    rcTreeNodes[newRCTreeClust].rep_vertex = tid + 1;
    rcTreeNodes[newRCTreeClust].bound_vertex_1 = neighbor_id;
    rcTreeNodes[newRCTreeClust].bound_vertex_2 = -1;
    rcTreeNodes[newRCTreeClust].edge_id = -1;
    rcTreeNodes[newRCTreeClust].vertex_id = -1;
    // add new edge in rcTree connecting vertex to cluster
    rcTreeEdges[tid].vertex_1 = tid + 1;
    rcTreeEdges[tid].vertex_2 = newRCTreeClust + 1;
    rcTreeEdges[tid].weight = 1;
    rcTreeEdges[tid].id = tid + 1;
    rcTreeEdges[tid].marked = 0;
    rcTreeEdges[tid].valid = true;
    rcTreeEdges[tid].iter_added = iter;
    // add new edge in rcTree connecting original edge to cluster
    if (edge_id > num_edges) {
        // if not original edge, return
        return;
    }
    rcTreeEdges[edge_id - 1 + num_vertices].vertex_1 = edge_id + num_vertices;
    rcTreeEdges[edge_id - 1 + num_vertices].vertex_2 = newRCTreeClust + 1;
    rcTreeEdges[edge_id - 1 + num_vertices].weight = 1;
    rcTreeEdges[edge_id - 1 + num_vertices].id = edge_id + num_vertices;
    rcTreeEdges[edge_id - 1 + num_vertices].marked = 0;
    rcTreeEdges[edge_id - 1 + num_vertices].valid = true;
}

// only call on vertices with degree = 2
// 0 if success, != 0 if not success
__device__ int compress(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int tid, int* edgeAllocd, int iter, int root_vertex) {

    // check neighbor vertices to see if they are both not degree 1
    int edge_id_1 = edgeAdjList[degPrefixSum[tid]];
    int neighbor_id_1 = edges[edge_id_1 - 1].vertex_1;
    if (neighbor_id_1 == tid + 1) {
        neighbor_id_1 = edges[edge_id_1 - 1].vertex_2;
    }
    int edge_id_2 = edgeAdjList[degPrefixSum[tid] + 1];
    int neighbor_id_2 = edges[edge_id_2 - 1].vertex_1;
    if (neighbor_id_2 == tid + 1) {
        neighbor_id_2 = edges[edge_id_2 - 1].vertex_2;
    }

    // invalidate edges
    edges[edge_id_1 - 1].valid = false;
    edges[edge_id_2 - 1].valid = false;
    // add a new edge to the base graph
    int new_edge_posn = atomicAdd(edgeAllocd, 1);
    edges[new_edge_posn].valid = true;
    edges[new_edge_posn].marked = 0;
    edges[new_edge_posn].id = new_edge_posn + 1;
    edges[new_edge_posn].weight = 0;
    edges[new_edge_posn].vertex_1 = neighbor_id_1;
    edges[new_edge_posn].vertex_2 = neighbor_id_2;
    // get new rcTreeCluster
    int newRCTreeClust = atomicAdd(numRCTreeVertices, 1);
    rcTreeNodes[newRCTreeClust].cluster_degree = 2;
    rcTreeNodes[newRCTreeClust].rep_vertex = tid + 1;
    rcTreeNodes[newRCTreeClust].bound_vertex_1 = neighbor_id_1;
    rcTreeNodes[newRCTreeClust].bound_vertex_2 = neighbor_id_2;
    rcTreeNodes[newRCTreeClust].edge_id = -1;
    rcTreeNodes[newRCTreeClust].vertex_id = -1;
    // add new edge in rcTree connecting vertex to rcTreeCluster
    rcTreeEdges[tid].vertex_1 = tid + 1;
    rcTreeEdges[tid].vertex_2 = newRCTreeClust + 1;
    rcTreeEdges[tid].weight = 1;
    rcTreeEdges[tid].id = tid + 1;
    rcTreeEdges[tid].marked = 0;
    rcTreeEdges[tid].valid = true;
    rcTreeEdges[tid].iter_added = iter;
    // add new edges in rcTree connecting original edges to cluster
    if (edge_id_1 <= num_edges) {
        // add if original edge
        rcTreeEdges[edge_id_1 - 1 + num_vertices].vertex_1 = edge_id_1 + num_vertices;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].vertex_2 = newRCTreeClust + 1;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].weight = 1;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].id = edge_id_1 + num_vertices;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].marked = 0;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].valid = true;
    }
    if (edge_id_2 <= num_edges) {
        // add if original edge
        rcTreeEdges[edge_id_2 - 1 + num_vertices].vertex_1 = edge_id_2 + num_vertices;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].vertex_2 = newRCTreeClust + 1;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].weight = 1;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].id = edge_id_2 + num_vertices;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].marked = 0;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].valid = true;
    }
    return 0;
}


__global__ void rakeCompress(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int* edgeAllocd, int iter, unsigned int* degCounts, int root_vertex, float* randValues) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_vertices)
        return;
    // check degree of vertex to see if rake or compress must be performed
    int deg = degPrefixSum[tid + 1] - degPrefixSum[tid];
    if (deg == 0) {
        // check base case of rake
        if (!rcTreeEdges[tid].valid) {
            // get new rcTreeCluster
            int newRCTreeClust = atomicAdd(numRCTreeVertices, 1);
            rcTreeNodes[newRCTreeClust].cluster_degree = 0;
            rcTreeNodes[newRCTreeClust].rep_vertex = tid + 1;
            rcTreeNodes[newRCTreeClust].bound_vertex_1 = -1;
            rcTreeNodes[newRCTreeClust].bound_vertex_2 = -1;
            rcTreeNodes[newRCTreeClust].edge_id = -1;
            rcTreeNodes[newRCTreeClust].vertex_id = -1;
            // add new edge in rcTree connecting vertex to cluster
            rcTreeEdges[tid].vertex_1 = tid + 1;
            rcTreeEdges[tid].vertex_2 = newRCTreeClust + 1;
            rcTreeEdges[tid].weight = 1;
            rcTreeEdges[tid].id = tid + 1;
            rcTreeEdges[tid].marked = 0;
            rcTreeEdges[tid].valid = true;
            rcTreeEdges[tid].iter_added = iter;
        }
        return;
    }
    if (atomicAdd(&randValues[tid], 0) != -1) {
        // for vertices with non-zero degree who did not rake or compress, we need to update degCounts
        atomicAdd(&degCounts[tid], deg);
        atomicExch(&randValues[tid], 0);
        return;
    }
    // past this point eligible to rake
    if (deg == 1) {
        // rake
        rake(edges, num_vertices, num_edges, numRCTreeVertices, rcTreeNodes, rcTreeEdges, edgeAdjList, degPrefixSum, tid, iter, degCounts, root_vertex);
    } else if (deg == 2) {
        // compress
        compress(edges, num_vertices, num_edges, numRCTreeVertices, rcTreeNodes, rcTreeEdges, edgeAdjList, degPrefixSum, tid, edgeAllocd, iter, root_vertex);
    } 
}

__global__ void updateClusterEdges(int rcTreeArrayLen, edge_t* rcTreeEdges, rcTreeNode_t* rcTreeNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= rcTreeArrayLen)
        return;
    // check edge not assigned yet 
    if (rcTreeEdges[tid].valid) {
        return;
    }
    // at the end, all original vertices must be attached to a cluster
    // get representative vertices
    if (rcTreeNodes[tid].cluster_degree == 1) {
        // 1 bound vertex
        int bound_vertex = rcTreeNodes[tid].bound_vertex_1;
        // attach this cluster to same cluster as boundary vertex attached to
        rcTreeEdges[tid].vertex_1 = tid + 1;
        rcTreeEdges[tid].vertex_2 = rcTreeEdges[bound_vertex - 1].vertex_2;
        rcTreeEdges[tid].weight = 1;
        rcTreeEdges[tid].id = tid + 1;
        rcTreeEdges[tid].marked = 0;
        rcTreeEdges[tid].valid = true;
    } else if (rcTreeNodes[tid].cluster_degree == 2) {
        // 2 bound vertices, figure out which one contracted first
        int bound_vertex = rcTreeNodes[tid].bound_vertex_1;
        if (rcTreeEdges[rcTreeNodes[tid].bound_vertex_2 - 1].iter_added < rcTreeEdges[bound_vertex - 1].iter_added) {
            bound_vertex = rcTreeNodes[tid].bound_vertex_2;
        }
        // attach this cluster to same cluster as boundary vertex attached to
        rcTreeEdges[tid].vertex_1 = tid + 1;
        rcTreeEdges[tid].vertex_2 = rcTreeEdges[bound_vertex - 1].vertex_2;
        rcTreeEdges[tid].weight = 1;
        rcTreeEdges[tid].id = tid + 1;
        rcTreeEdges[tid].marked = 0;
        rcTreeEdges[tid].valid = true;
    }
}

void init_process(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int seed_offset) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // edges should live in gpu memory!

    edge_blks = ((2*num_edges) + NUM_THREADS - 1) / NUM_THREADS;
    vertex_blks = (num_vertices + NUM_THREADS - 1) / NUM_THREADS;
    lenRCTreeArrays = 2*num_vertices + num_edges;
    rcTree_blks = (lenRCTreeArrays + NUM_THREADS - 1) / NUM_THREADS;

    cudaError_t err;
    err = cudaMallocManaged((void**) &num_rcTreeVertices, sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    *num_rcTreeVertices = num_vertices + num_edges;
    err = cudaMalloc((void**) &gpu_edgesAllocated, sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMemcpy(gpu_edgesAllocated, &num_edges, sizeof(int), cudaMemcpyHostToDevice);
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
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
    // allocate state for all threads
    // Citation: https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand
    err = cudaMalloc((void**) &gpu_randStates, num_vertices*sizeof(curandState));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_randValues, num_vertices*sizeof(float));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    // luby Alg C value
    err = cudaMallocManaged((void**) &lubyConsiderNodes, sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }

    // initialize random state
    // Citation: https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand
    gpu_random<<<vertex_blks, NUM_THREADS>>>(num_vertices, gpu_randStates, seed_offset, gpu_randValues);

    // init emptyEdges in case of future compress forming edges
    init_emptyEdges<<<edge_blks, NUM_THREADS>>>(edges, num_edges*2, num_edges);

    // need gpu function to initialize gpu_rcTreeNodes and gpu_rcTreeEdges properly
    init_rcTreeArrays<<<rcTree_blks, NUM_THREADS>>>(lenRCTreeArrays, num_vertices, num_edges, rcTreeEdges, rcTreeNodes);

    // might need a synchronize before exiting to ensure that process initialization has completed before exiting
    cudaDeviceSynchronize();
}

// pass root_vertex = -1 if unrooted
void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int root_vertex) {
    // edges live in GPU memory
    // parallelize by edge -> count vertex degrees
    count_degree<<<edge_blks, NUM_THREADS>>>(edges, num_edges*2, gpu_degCounts);
    // synchronize before cpu read at top of loop
    cudaDeviceSynchronize();
    int iter = 0;
    while (*num_rcTreeVertices != num_edges + 2*num_vertices) {
        // exclude root from 
        *lubyConsiderNodes = num_edges + 2*num_vertices - *num_rcTreeVertices;

        // 1. prefix sum degrees
        thrust::exclusive_scan(thrust::device, gpu_degCounts, gpu_degCounts+num_vertices+1, gpu_degPrefixSum);

        // 2. build adjacency list
        build_adjList<<<edge_blks, NUM_THREADS>>>(edges, 2*num_edges, gpu_edgeAdjList, gpu_degPrefixSum, gpu_degCounts);

        // we don't need to synchronize here for lubyConsiderNodes
        // 3. lubyMIS algorithm
        while (*lubyConsiderNodes != 0) {

            // gen random values for vertices of degree 2 eligible to compress
            genRandValues<<<vertex_blks, NUM_THREADS>>>(edges, num_vertices, gpu_edgeAdjList, gpu_degPrefixSum, root_vertex, gpu_randStates, gpu_randValues, lubyConsiderNodes);
            
            // figure out who gets to compress
            addToMIS<<<vertex_blks, NUM_THREADS>>>(edges, num_vertices, gpu_edgeAdjList, gpu_degPrefixSum, gpu_randValues, lubyConsiderNodes);
            
            // synchronize before cpu read at top of loop
            cudaDeviceSynchronize();
        }
        

        // 4. parallelize RC step, count deg for next iteration, reset randVals
        rakeCompress<<<vertex_blks, NUM_THREADS>>>(edges, num_vertices, num_edges, num_rcTreeVertices, rcTreeNodes, rcTreeEdges, gpu_edgeAdjList, gpu_degPrefixSum, gpu_edgesAllocated, iter, gpu_degCounts, root_vertex, gpu_randValues);

        // synchronize before cpu read at top of loop
        cudaDeviceSynchronize();
        iter += 1;
    }

    // parallelize RC Tree cluster node add to RC Tree
    updateClusterEdges<<<rcTree_blks, NUM_THREADS>>>(lenRCTreeArrays, rcTreeEdges, rcTreeNodes);
}


