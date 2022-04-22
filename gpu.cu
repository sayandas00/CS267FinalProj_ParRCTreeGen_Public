#include "common.h"
#include <cuda.h>
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

__global__ void count_degree(edge_t* edges, int len, unsigned int* degCounts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len)
        return;
    if (edges[tid].valid) {
        edges[tid].marked = 0;
        int vertex_1 = edges[tid].vertex_1;
        int vertex_2 = edges[tid].vertex_2;
        degCounts[vertex_1 - 1] = 1;
        degCounts[vertex_2 - 1] = 1;
    }
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

// only call after degree check, num_edges and num_vertices are of the original base graph
__device__ void rake(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int tid) {
    // check degree of neighbor, need to account for base case of 2 1 degree vertices
    // larger vertex id will rake
    int edge_id = edgeAdjList[degPrefixSum[tid]];
    int neighbor_id = edges[edge_id - 1].vertex_1;
    if (neighbor_id == tid + 1) {
        neighbor_id = edges[edge_id - 1].vertex_2;
    }
    int neighbor_deg = degPrefixSum[neighbor_id] - degPrefixSum[neighbor_id - 1];
    if ((neighbor_deg == 1) && (neighbor_id > tid)) {
        return;
    }
    // mark edge unvalid and get vertex id of neighbor
    edges[edge_id - 1].valid = false;

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
    rcTreeEdges[tid].vertex_2 = newRCTreeClust;
    rcTreeEdges[tid].weight = 1;
    rcTreeEdges[tid].id = tid + 1;
    rcTreeEdges[tid].marked = 0;
    rcTreeEdges[tid].valid = true;
    // add new edge in rcTree connecting original edge to cluster
    if (edge_id > num_edges) {
        // if not original edge, return
        return;
    }
    rcTreeEdges[edge_id - 1 + num_vertices].vertex_1 = edge_id + num_vertices;
    rcTreeEdges[edge_id - 1 + num_vertices].vertex_2 = newRCTreeClust;
    rcTreeEdges[edge_id - 1 + num_vertices].weight = 1;
    rcTreeEdges[edge_id - 1 + num_vertices].id = edge_id + num_vertices;
    rcTreeEdges[edge_id - 1 + num_vertices].marked = 0;
    rcTreeEdges[edge_id - 1 + num_vertices].valid = true;

}

// only call on vertices with degree = 2
__device__ void compress(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int tid, int* edgeAllocd) {
    // check neighbor vertices to see if they are both not degree 1
    int edge_id_1 = edgeAdjList[degPrefixSum[tid]];
    int neighbor_id_1 = edges[edge_id_1 - 1].vertex_1;
    if (neighbor_id_1 == tid + 1) {
        neighbor_id_1 = edges[edge_id_1 - 1].vertex_2;
    }
    int neighbor_deg = degPrefixSum[neighbor_id_1] - degPrefixSum[neighbor_id_1 - 1];
    if ((neighbor_deg == 1)) {
        return;
    }
    int edge_id_2 = edgeAdjList[degPrefixSum[tid] + 1];
    int neighbor_id_2 = edges[edge_id_2 - 1].vertex_1;
    if (neighbor_id_2 == tid + 1) {
        neighbor_id_2 = edges[edge_id_2 - 1].vertex_2;
    }
    neighbor_deg = degPrefixSum[neighbor_id_2] - degPrefixSum[neighbor_id_2 - 1];
    if ((neighbor_deg == 1)) {
        return;
    }
    // for simplicity we always grab edges we might prune, grab lower id first
    // ensures independent set
    int marked;
    if (edge_id_1 < edge_id_2) {
        marked = atomicAdd(&edges[edge_id_1 - 1].marked, 1);
        if (marked != 0) {
            atomicSub(&edges[edge_id_1 -1].marked, 1);
            return;   
        }
        marked = atomicAdd(&edges[edge_id_2 - 1].marked, 1);
        if (marked != 0) {
            atomicSub(&edges[edge_id_2 - 1].marked, 1);
            atomicSub(&edges[edge_id_1 - 1].marked, 1);
            return;
        }
    } else {
        marked = atomicAdd(&edges[edge_id_2 - 1].marked, 1);
        if (marked != 0) {
            atomicSub(&edges[edge_id_2 - 1].marked, 1);
            return;   
        }
        marked = atomicAdd(&edges[edge_id_1 - 1].marked, 1);
        if (marked != 0) {
            atomicSub(&edges[edge_id_1 - 1].marked, 1);
            atomicSub(&edges[edge_id_2 - 1].marked, 1);
            return;
        }   
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
    rcTreeEdges[tid].vertex_2 = newRCTreeClust;
    rcTreeEdges[tid].weight = 1;
    rcTreeEdges[tid].id = tid + 1;
    rcTreeEdges[tid].marked = 0;
    rcTreeEdges[tid].valid = true;
    // add new edges in rcTree connecting original edges to cluster
    if (edge_id_1 <= num_edges) {
        // add if original edge
        rcTreeEdges[edge_id_1 - 1 + num_vertices].vertex_1 = edge_id_1 + num_vertices;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].vertex_2 = newRCTreeClust;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].weight = 1;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].id = edge_id_1 + num_vertices;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].marked = 0;
        rcTreeEdges[edge_id_1 - 1 + num_vertices].valid = true;
    }
    if (edge_id_2 <= num_edges) {
        // add if original edge
        rcTreeEdges[edge_id_2 - 1 + num_vertices].vertex_1 = edge_id_2 + num_vertices;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].vertex_2 = newRCTreeClust;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].weight = 1;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].id = edge_id_2 + num_vertices;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].marked = 0;
        rcTreeEdges[edge_id_2 - 1 + num_vertices].valid = true;
    }
}

__global__ void rakeCompress(edge_t* edges, int num_vertices, int num_edges, int* numRCTreeVertices, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges, int* edgeAdjList, int* degPrefixSum, int* edgeAllocd) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_vertices)
        return;
    // check degree of vertex to see if rake or compress must be performed
    int deg = degPrefixSum[tid + 1] - degPrefixSum[tid];
    if (deg == 1) {
        // rake
        rake(edges, num_vertices, num_edges, numRCTreeVertices, rcTreeNodes, rcTreeEdges, edgeAdjList, degPrefixSum, tid);
    } else if (deg == 2) {
        // compress
        compress(edges, num_vertices, num_edges, numRCTreeVertices, rcTreeNodes, rcTreeEdges, edgeAdjList, degPrefixSum, tid, edgeAllocd);
    } else if (deg == 0) {
        // check base case of rake
        int remaining_vertices = 2*num_vertices + num_edges - atomicAdd(numRCTreeVertices, 0);
        if (remaining_vertices == 1) {
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
            rcTreeEdges[tid].vertex_2 = newRCTreeClust;
            rcTreeEdges[tid].weight = 1;
            rcTreeEdges[tid].id = tid + 1;
            rcTreeEdges[tid].marked = 0;
            rcTreeEdges[tid].valid = true;
        }
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
    // get representative vertices
    if (rcTreeNodes[tid].cluster_degree == 1) {
        // 1 bound vertex
        int bound_vertex = rcTreeNodes[tid].bound_vertex_1;
        if (rcTreeEdges[bound_vertex - 1].valid) {
            // attach this cluster to same cluster as boundary vertex attached to
            rcTreeEdges[tid].vertex_1 = tid + 1;
            rcTreeEdges[tid].vertex_2 = rcTreeEdges[bound_vertex - 1].vertex_2;
            rcTreeEdges[tid].weight = 1;
            rcTreeEdges[tid].id = tid + 1;
            rcTreeEdges[tid].marked = 0;
            rcTreeEdges[tid].valid = true;
        }
    } else if (rcTreeNodes[tid].cluster_degree == 2) {
        // 2 bound vertices
        //check the first one
        int bound_vertex = rcTreeNodes[tid].bound_vertex_1;
        if (rcTreeEdges[bound_vertex - 1].valid) {
            // attach this cluster to same cluster as boundary vertex attached to
            rcTreeEdges[tid].vertex_1 = tid + 1;
            rcTreeEdges[tid].vertex_2 = rcTreeEdges[bound_vertex - 1].vertex_2;
            rcTreeEdges[tid].weight = 1;
            rcTreeEdges[tid].id = tid + 1;
            rcTreeEdges[tid].marked = 0;
            rcTreeEdges[tid].valid = true;
            return;
        }
        // check the second one
        bound_vertex = rcTreeNodes[tid].bound_vertex_2;
        if (rcTreeEdges[bound_vertex - 1].valid) {
            // attach this cluster to same cluster as boundary vertex attached to
            rcTreeEdges[tid].vertex_1 = tid + 1;
            rcTreeEdges[tid].vertex_2 = rcTreeEdges[bound_vertex - 1].vertex_2;
            rcTreeEdges[tid].weight = 1;
            rcTreeEdges[tid].id = tid + 1;
            rcTreeEdges[tid].marked = 0;
            rcTreeEdges[tid].valid = true;
            return;
        }
    }

}

void init_process(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges) {
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
    err = cudaMemset(gpu_edgesAllocated, num_edges, sizeof(int));
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
    // need gpu function to initialize gpu_rcTreeNodes and gpu_rcTreeEdges properly
    init_rcTreeArrays<<<rcTree_blks, NUM_THREADS>>>(lenRCTreeArrays, num_vertices, num_edges, rcTreeEdges, rcTreeNodes);

    // might need a synchronize before exiting to ensure that process initialization has completed before exiting
    cudaDeviceSynchronize();
}


void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges) {
    // edges live in GPU memory
    int cnt = 0;
    while (*num_rcTreeVertices != num_edges + 2*num_vertices) {

        // 1. parallelize by edge -> count vertex degrees
        count_degree<<<edge_blks, NUM_THREADS>>>(edges, num_edges*2, gpu_degCounts);

        // 2. prefix sum degrees
        thrust::exclusive_scan(thrust::device, gpu_degCounts, gpu_degCounts+num_vertices+1, gpu_degPrefixSum);

        // 3. build adjacency list
        build_adjList<<<edge_blks, NUM_THREADS>>>(edges, 2*num_edges, gpu_edgeAdjList, gpu_degPrefixSum, gpu_degCounts);

        // 4. parallelize RC step
        rakeCompress<<<vertex_blks, NUM_THREADS>>>(edges, num_vertices, num_edges, num_rcTreeVertices, rcTreeNodes, rcTreeEdges, gpu_edgeAdjList, gpu_degPrefixSum, gpu_edgesAllocated);

        // synch needed?
        // 5. parallelize RC Tree cluster node add to RC Tree
        updateClusterEdges<<<rcTree_blks, NUM_THREADS>>>(lenRCTreeArrays, rcTreeEdges, rcTreeNodes);
        cnt += 1;
        if (cnt > 0) {
            cudaDeviceSynchronize();
            return;
        }
        
    }


    


}


