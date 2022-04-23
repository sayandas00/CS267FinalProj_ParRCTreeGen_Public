#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        // fsave << parts[i].x << " " << parts[i].y << std::endl;
    }

    fsave << std::endl;
}


// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// assumes rcTreeEdges at each index vertex 1 is the corresponding index of the rcTreeNodes
// also assumes that higher up clusters are allocated at a higher index
int height_rcTree(edge_t* rcTreeEdges, int num_vertices) {
    int* heights = new int[num_vertices];
    for (int i = 0; i < num_vertices; i++) {
        heights[i] = 0;
    }
    for (int i = 0; i < num_vertices; i++) {
        if (rcTreeEdges[i].valid) {
            int other_node_id = rcTreeEdges[i].vertex_2;
            heights[other_node_id - 1] = std::max(heights[other_node_id - 1], heights[i] + 1);
        }
    }
    int maxHeight = heights[num_vertices - 1];
    free(heights);
    return maxHeight;
}


// ==============
// Main Function
// ==============

int main(int argc, char** argv) {

    bool debug = true;
  
    // check that a file name is specified
    if (argc != 2) {
        std::cout << "Need to specify 1 file to read edge list from" << '\n';
        return 0;
    }
    // initialize graph variables
    int num_edges = 0;
    int num_vertices = 0;
    edge_t* edges;

    // Citation from https://www.cplusplus.com/doc/tutorial/files/ for reading from a file
    // read edge list from text file, assuming well formatted text file
    // first line of file: num_vertices num_edges
    // rest of lines: vertex_1 vertex_2 edge_weight
    std::string str_line;
    std::ifstream myfile (argv[1]);
    int line_cnt = 0;
    if (myfile.is_open()) {
        while ( getline (myfile, str_line) ) {
            //This conversion is only safe in C++ 11 and onward!!!
            char* line = &*str_line.begin();
            line_cnt += 1;
            if (line_cnt > num_edges + 1) {
                std::cout << "File incorrectly formatted, too many edges" << '\n';
                myfile.close();
                return 0;
            }
            if (line_cnt == 1) {
                // Citation from https://www.javatpoint.com/how-to-split-strings-in-cpp
                // for parsing and splitting strings
                char* curr_ptr = std::strtok(line, " ");
                if (curr_ptr == NULL) {
                    std::cout << "File incorrectly formatted, no num_vertices given" << '\n';
                    myfile.close();
                    return 0;
                }
                num_vertices = std::stoi(curr_ptr);
                curr_ptr = std::strtok(NULL, " ");
                if (curr_ptr == NULL) {
                    std::cout << "File incorrectly formatted, no num_edges given" << '\n';
                    myfile.close();
                    return 0;
                }
                num_edges = std::stoi(curr_ptr);
                curr_ptr = std::strtok(NULL, " ");
                if (curr_ptr != NULL) {
                    std::cout << "File incorrectly formatted" << '\n';
                    myfile.close();
                    return 0;
                }
                if (num_edges == 0) {
                    std::cout << "No edge graph, return" << '\n';
                    return 0;
                }
                // allocate array for edges
                edges = new edge_t[2 * num_edges];
            } else {
                // Citation from https://www.javatpoint.com/how-to-split-strings-in-cpp
                // for parsing and splitting strings
                int edge_posn = line_cnt - 2;
                char* curr_ptr = std::strtok(line, " ");
                if (curr_ptr == NULL) {
                    std::cout << "File incorrectly formatted, no vertex_1 given" << '\n';
                    myfile.close();
                    return 0;
                }
                edges[edge_posn].vertex_1 = std::stoi(curr_ptr);
                curr_ptr = std::strtok(NULL, " ");
                if (curr_ptr == NULL) {
                    std::cout << "File incorrectly formatted, no vertex_2 given" << '\n';
                    myfile.close();
                    return 0;
                }
                edges[edge_posn].vertex_2 = std::stoi(curr_ptr);
                curr_ptr = std::strtok(NULL, " ");
                if (curr_ptr == NULL) {
                    std::cout << "File incorrectly formatted, no vertex_2 given" << '\n';
                    myfile.close();
                    return 0;
                }
                edges[edge_posn].weight = std::stod(curr_ptr);
                curr_ptr = std::strtok(NULL, " ");
                if (curr_ptr != NULL) {
                    std::cout << "File incorrectly formatted" << '\n';
                    myfile.close();
                    return 0;
                }
                edges[edge_posn].id = edge_posn + 1;
                edges[edge_posn].valid = true;
                edges[edge_posn].marked = 0;
            }
        }
        myfile.close();
        if (line_cnt != num_edges + 1) {
            std::cout << "File incorrectly formatted, too few edges specified" << '\n';
            return 0;
        }
    }
    else 
    {
        std::cout << "Unable to open file";
        return 0;
    }
    if (debug) {
        std::cout << "Num Vertices: " << num_vertices << " Num Edges: " << num_edges << '\n';
        for (int i = 0; i < num_edges; i++) {
            std::cout << "Edge: " << i + 1 << " vertex_1: " << edges[i].vertex_1 << " vertex_2: " << edges[i].vertex_2 << " weight: " << edges[i].weight << '\n';
        }
    }

    // start timing Citation: CS267 Spring 2022 HW23
    auto start_time = std::chrono::steady_clock::now();

    // fill the rest of the edge list with zeros
    for (int i = num_edges; i < 2*num_edges; i++) {
        edges[i].vertex_1 = -1;
        edges[i].vertex_2 = -1;
        edges[i].weight = -1;
        edges[i].valid = false;
        edges[i].id = i + 1;
        edges[i].marked = 0;
    }

    edge_t* edges_gpu;
    cudaMalloc((void**)&edges_gpu, 2*num_edges * sizeof(edge_t));
    cudaMemcpy(edges_gpu, edges, 2*num_edges * sizeof(edge_t), cudaMemcpyHostToDevice);

    rcTreeNode_t* gpu_rcTreeNodes;
    int lenRCTreeArrays = 2*num_vertices + num_edges;
    edge_t* gpu_rcTreeEdges;
    rcTreeNode_t* cpu_rcTreeNodes = new rcTreeNode_t[lenRCTreeArrays];
    edge_t* cpu_rcTreeEdges = new edge_t[lenRCTreeArrays];
    cudaMalloc((void**) &gpu_rcTreeNodes, lenRCTreeArrays*sizeof(rcTreeNode_t));
    cudaMalloc((void**) &gpu_rcTreeEdges, lenRCTreeArrays*sizeof(edge_t));

    init_process(edges_gpu, num_vertices, num_edges, gpu_rcTreeNodes, gpu_rcTreeEdges);
    rc_tree_gen(edges_gpu, num_vertices, num_edges, gpu_rcTreeNodes, gpu_rcTreeEdges);

    cudaDeviceSynchronize();

    // Timing Code Citation: CS267 Spring 2022 HW23
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    std::cout << "Simulation Time = " << seconds << " seconds for " << num_vertices << " vertices.\n";

    // copy from gpu memory to cpu memory
    cudaMemcpy(cpu_rcTreeNodes, gpu_rcTreeNodes, lenRCTreeArrays*sizeof(rcTreeNode_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_rcTreeEdges, gpu_rcTreeEdges, lenRCTreeArrays*sizeof(edge_t), cudaMemcpyDeviceToHost);
    if (debug) {
        std::cout << "RCTree Edges" << std::endl;
        for (int i = 0; i < lenRCTreeArrays; i++) {
            if (cpu_rcTreeEdges[i].valid) {
                std::cout << "Edge: " << i + 1 << " vertex_1: " << cpu_rcTreeEdges[i].vertex_1 << " vertex_2: " << cpu_rcTreeEdges[i].vertex_2 << " weight: " << cpu_rcTreeEdges[i].weight << '\n';
            }
        }
        std::cout << "RCTree Nodes" << std::endl;
        for (int i = 0; i < lenRCTreeArrays; i++) {
            std::cout << "Node: " << i + 1;
            if (cpu_rcTreeNodes[i].cluster_degree != -1) {
                std::cout << " Cluster_degree: " << cpu_rcTreeNodes[i].cluster_degree;
            }
            if (cpu_rcTreeNodes[i].rep_vertex != -1) {
                std::cout << " Rep_vertex: " << cpu_rcTreeNodes[i].rep_vertex;
            }
            if (cpu_rcTreeNodes[i].bound_vertex_1 != -1) {
                std::cout << " Bound_vertex_1: " << cpu_rcTreeNodes[i].bound_vertex_1;
            }
            if (cpu_rcTreeNodes[i].bound_vertex_2 != -1) {
                std::cout << " Bound_vertex_2: " << cpu_rcTreeNodes[i].bound_vertex_2;
            }
            if (cpu_rcTreeNodes[i].edge_id != -1) {
                std::cout << " Edge_id: " << cpu_rcTreeNodes[i].edge_id;
            }
            if (cpu_rcTreeNodes[i].vertex_id != -1) {
                std::cout << " Vertex_id: " << cpu_rcTreeNodes[i].vertex_id;
            }
            std::cout << std::endl;
        }
    }
    int height = height_rcTree(cpu_rcTreeEdges, lenRCTreeArrays);
    std::cout << "Height of rcTree: " << height << std::endl;

    free(cpu_rcTreeNodes);
    free(cpu_rcTreeEdges);
    free(edges);
    cudaFree(gpu_rcTreeNodes);
    cudaFree(gpu_rcTreeEdges);
}
