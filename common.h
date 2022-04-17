#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants

// Edge Data Structure
typedef struct edge_t {
    int vertex_1;  // 1st vertex id
    int vertex_2;  // 2nd vertex id
    double weight; // weight of the edge
} edge_t;

// Vertex Data Structure
typedef struct vertex_t {
    int id;
    int degree;
} vertex_t;

// Routines
void init_process(edge_t* edges, int num_vertices, int num_edges);
void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges);

#endif
