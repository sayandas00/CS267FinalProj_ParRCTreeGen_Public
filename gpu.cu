#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int rowLen;
int numBins;
int iter_count = 1;

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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int particleBin1d = int(floor(particles[tid].x / cutoff)) + int(floor(particles[tid].y / cutoff))*rowLen;
    //std::atomicAdd(binCounts[particleBin1d], 1); //Array of atomics, so the increment is also atomic
    atomicInc(&binCounts[particleBin1d], 1);

}

__global__ void sort_parts(particle_t* particles, int num_parts, unsigned int * binCounts, 
                                    int * prefixSum, int * sortedParts, int rowLen, int numBins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int particleBin1d = int(floor(particles[tid].x / cutoff)) + int(floor(particles[tid].y / cutoff))*rowLen;
    int binIdx = atomicSub(&binCounts[particleBin1d], 1);
    sortedParts[prefixSum[particleBin1d] + binIdx - 1] = tid;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int * prefixSum, int * sortedParts, int rowLen, int numBins) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numBins)
        return;

    // Empty bin
    if (prefixSum[tid] == prefixSum[tid+1])
        return;

    // Here, tid = bin index
    // For ease of visualization, first convert to 2d indices
    int x = int(floor(particles[tid].x / cutoff));
    int y = int(floor(particles[tid].y / cutoff));

    // Self 
    for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
        for (int j = i+1; j < prefixSum[tid+1]; j++) {
            apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
        }
    }
    // right
    if (x != rowLen - 1) {
        int rightBin = y*rowLen + (x+1);
        for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
            for (int j = prefixSum[rightBin]; j < prefixSum[rightBin+1]; j++) {
                apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
            }
        }
    }
    //upper
    if(y != rowLen - 1) {
        // upper left
        if(x != 0){
            int upperLeftBin = (y+1)*rowLen + (x-1);
            for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
                for (int j = prefixSum[upperLeftBin]; j < prefixSum[upperLeftBin+1]; j++) {
                    apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
                }
            }
        }
        // upper
        int upperBin = (y+1)*rowLen + x;
        for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
            for (int j = prefixSum[upperBin]; j < prefixSum[upperBin+1]; j++) {
                apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
            }
        }
        // upper right
        if(x != rowLen - 1){
            int upperRightBin = (y+1)*rowLen + (x+1);
            for (int i = prefixSum[tid]; i < prefixSum[tid+1]; i++){
                for (int j = prefixSum[upperRightBin]; j < prefixSum[upperRightBin+1]; j++) {
                    apply_force_gpu(particles[sortedParts[i]], particles[sortedParts[j]]);
                }
            }
        }
    }

    // particles[tid].ax = particles[tid].ay = 0;
    // for (int j = 0; j < num_parts; j++)
    //     apply_force_gpu(particles[tid], particles[j]);
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

    int cpu_rowLen = floor(size / cutoff) + 1;
    int cpu_numBins = cpu_rowLen*cpu_rowLen;
    rowLen = cpu_rowLen;
    numBins = cpu_numBins;

    cpu_binCounts = new unsigned int[numBins];
    cpu_prefixSum = new int[numBins+1];
    int cpu_sortedParts[num_parts];
    memset(cpu_binCounts, 0, numBins*sizeof(int));
    memset(cpu_prefixSum, 0, (numBins+1)*sizeof(int));
    memset(cpu_sortedParts, 0, num_parts*sizeof(int));

    // cudaMemcpyFromSymbol(&rowLen, cpu_binCounts, sizeof(int), 0, cudaMemcpyDeviceToHost);
    // cudaMemcpyFromSymbol(&cpu_numBins, numBins, sizeof(int), 0, cudaMemcpyDeviceToHost);

    cudaMalloc((void**) &gpu_binCounts, numBins*sizeof(unsigned int));
    cudaMemcpy(gpu_binCounts, cpu_binCounts, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &gpu_prefixSum, (numBins+1)*sizeof(int));
    cudaMemcpy(gpu_prefixSum, cpu_prefixSum, (numBins+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &gpu_sortedParts, num_parts*sizeof(int));

    // std::cout << cudaGetErrorName(cudaMalloc((void**) &gpu_binCounts, numBins*sizeof(unsigned int))) << std::endl;
    // std::cout << cudaGetErrorName(cudaMemcpy(gpu_binCounts, cpu_binCounts, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice)) << std::endl;
    // std::cout << cudaGetErrorName(cudaMalloc((void**) &gpu_prefixSum, (numBins+1)*sizeof(int))) << std::endl;
    // std::cout << cudaGetErrorName(cudaMemcpy(gpu_prefixSum, cpu_prefixSum, (numBins+1)*sizeof(int), cudaMemcpyHostToDevice)) << std::endl;
    // std::cout << cudaGetErrorName(cudaMalloc((void**) &gpu_sortedParts, num_parts*sizeof(int))) << std::endl;
    // // std::cout << cudaGetErrorName(cudaMemcpy(gpu_prefixSum, cpu_prefixSum, numBins*sizeof(int), cudaMemcpyHostToDevice)) << std::endl;

    // std::cout << numBins << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    
    // Create array of particles sorted by binID at each step
    if(!iter_count){
        particle_t * temp = new particle_t[num_parts];
        cudaMemcpy(temp, parts, num_parts*sizeof(particle_t), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 10; i++){
            std::cout << temp[i].x << ", ";
        }
        std::cout << std::endl;

        for(int i = 0; i < num_parts; i++){
            int particleBin1d = int(floor(temp[i].x / cutoff)) + int(floor(temp[i].y / cutoff))*rowLen;
            //std::cout << particleBin1d << ", ";
            if( particleBin1d < 150)
                std::cout<< "True: " << particleBin1d << ", ";
        }
        std::cout << std::endl;
        delete[] temp;
    }
    // // 1: Count number of particles per bin 
    count_bins<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_binCounts, rowLen);
    cudaError_t err = cudaMemcpy(cpu_binCounts, gpu_binCounts, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // // 2: Prefix Sum
    if(!iter_count){
    std::cout << cudaGetErrorName(err) << std::endl;
    for(int i = 140; i < 140+25; i++){
        std::cout << cpu_binCounts[i] << ", ";
    }
    std::cout << std::endl;
    }
    thrust::exclusive_scan(cpu_binCounts, cpu_binCounts+numBins, cpu_prefixSum);
    if(!iter_count){
    for(int i = 140; i < 140+25; i++){
        std::cout << cpu_prefixSum[i] << ", ";
    }
    std::cout << std::endl;
    }
    cudaMemcpy(gpu_prefixSum, cpu_prefixSum, (numBins+1)*sizeof(int), cudaMemcpyHostToDevice);
    // // 3: Sort particles indices by order of bins
    sort_parts<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_binCounts, gpu_prefixSum, gpu_sortedParts, rowLen, numBins);
    //cudaMemcpy(cpu_binCounts, gpu_binCounts, numBins*sizeof(int), cudaMemcpyDeviceToHost);
    if(!iter_count){
        particle_t * temp = new particle_t[num_parts];
        cudaMemcpy(temp, parts, num_parts*sizeof(particle_t), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < 10; i++){
        //     std::cout << cpu[temp[i].x << ", ";
        // }
        // std::cout << std::endl;

        for(int i = 0; i < 30; i++){
            int particleBin1d = int(floor(temp[i].x / cutoff)) + int(floor(temp[i].y / cutoff))*rowLen;
            std::cout << cpu_prefixSum[particleBin1d] << ", ";
            // if( particleBin1d < 150)
            //     std::cout<< "True: " << particleBin1d << ", ";
        }
        std::cout << std::endl;
        delete[] temp;
    }
    if(!iter_count){
        int * temp2 = new int[num_parts];

        for(int i = 140; i < 175 ; i++){
            std::cout << cpu_binCounts[i] << ", ";
        }
        std::cout << std::endl;
        cudaMemcpy(cpu_binCounts, gpu_binCounts, numBins*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp2, gpu_sortedParts, num_parts*sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i < 50; i++){
            std::cout << temp2[i] << ", ";
        }
        std::cout << std::endl;
        for(int i = 140; i < 175 ; i++){
            std::cout << cpu_binCounts[i] << ", ";
        }
        std::cout << std::endl;
        delete[] temp2;
    }

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_prefixSum, gpu_sortedParts, rowLen, numBins);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
