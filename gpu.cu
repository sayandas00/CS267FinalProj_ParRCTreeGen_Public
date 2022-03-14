#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int rowLen;
int numBins;

// CPU/GPU version of various arrays - correspond to steps 1/2/3 of section slides
unsigned int* cpu_binCounts;
unsigned int* gpu_binCounts;
int* cpu_prefixSum;
int* gpu_prefixSum;
int* cpu_sortedParts;
int* gpu_sortedParts;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    atomicAdd(&(particle.ax), coef * dx);
    atomicAdd(&(particle.ay), coef * dy);
    atomicAdd(&(neighbor.ax), -1*coef * dx);
    atomicAdd(&(neighbor.ay), -1*coef * dy);
}

__global__ void count_bins(particle_t* particles, int num_parts, unsigned int * binCounts, int rowLen) {
    // tid = particle idx
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // Reset the particle acceleration at the beginning of each step for each particle
    particles[tid].ax = particles[tid].ay = 0;
    int particleBin1d = int(floor(particles[tid].x / cutoff))*rowLen + int(floor(particles[tid].y / cutoff));
    atomicAdd(&binCounts[particleBin1d], 1);
}

__global__ void sort_parts(particle_t* particles, int num_parts, unsigned int * binCounts, 
                                    int * prefixSum, int * sortedParts, int rowLen, int numBins) {
    // tid = particle idx
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int particleBin1d = int(floor(particles[tid].x / cutoff))*rowLen + int(floor(particles[tid].y / cutoff));
    int binIdx = atomicSub(&binCounts[particleBin1d], 1);
    sortedParts[prefixSum[particleBin1d] + binIdx - 1] = tid;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int * prefixSum, int * sortedParts, int rowLen, int numBins) {
    // tid = bin id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numBins)
        return;

    // Empty bin
    if (prefixSum[tid] == prefixSum[tid+1])
        return;

    // Here, tid = bin index
    // For ease of visualization, first convert to 2d indices
    int x = tid/rowLen;
    int y = tid%rowLen;

    // Self 
    for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
        for (int j = i+1; j < prefixSum[tid+1]; j++) {
            apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
        }
    }
    // right
    if (y != rowLen - 1) {
        int rightBin = x*rowLen + (y+1);
        for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
            for (int j = prefixSum[rightBin]; j < prefixSum[rightBin+1]; j++) {
                apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
            }
        }
    }
    //upper
    if(x != rowLen - 1) {
        // upper left
        if(y != 0){
            int upperLeftBin = (x+1)*rowLen + (y-1);
            for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
                for (int j = prefixSum[upperLeftBin]; j < prefixSum[upperLeftBin+1]; j++) {
                    apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
                }
            }
        }
        // upper
        int upperBin = (x+1)*rowLen + y;
        for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
            for (int j = prefixSum[upperBin]; j < prefixSum[upperBin+1]; j++) {
                apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
            }
        }
        // upper right
        if(y != rowLen - 1){
            int upperRightBin = (x+1)*rowLen + (y+1);
            for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
                for (int j = prefixSum[upperRightBin]; j < prefixSum[upperRightBin+1]; j++) {
                    apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    rowLen = floor(size / cutoff) + 1;
    numBins = rowLen*rowLen;

    cpu_binCounts = new unsigned int[numBins];
    cpu_prefixSum = new int[numBins+1];
    memset(cpu_binCounts, 0, numBins*sizeof(int));
    memset(cpu_prefixSum, 0, (numBins+1)*sizeof(int));

    cudaError_t err;

    err = cudaMalloc((void**) &gpu_binCounts, numBins*sizeof(unsigned int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMemcpy(gpu_binCounts, cpu_binCounts, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMalloc((void**) &gpu_prefixSum, (numBins+1)*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    err = cudaMemcpy(gpu_prefixSum, cpu_prefixSum, (numBins+1)*sizeof(int), cudaMemcpyHostToDevice);
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }    
    err = cudaMalloc((void**) &gpu_sortedParts, num_parts*sizeof(int));
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    
    // Create array of particles sorted by binID at each step

    // 1: Count number of particles per bin 
    count_bins<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_binCounts, rowLen);
    cudaError_t err = cudaMemcpy(cpu_binCounts, gpu_binCounts, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    // 2: Prefix Sum
    thrust::exclusive_scan(cpu_binCounts, cpu_binCounts+numBins, cpu_prefixSum);
    err = cudaMemcpy(gpu_prefixSum, cpu_prefixSum, (numBins+1)*sizeof(int), cudaMemcpyHostToDevice);
    if(err){
        std::cout << cudaGetErrorName(err) << std::endl;
    }
    // 3: Sort particles indices by order of bins
    sort_parts<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_binCounts, gpu_prefixSum, gpu_sortedParts, rowLen, numBins);


    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_prefixSum, gpu_sortedParts, rowLen, numBins);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
