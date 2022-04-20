#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants

// Edge Data Structure
typedef struct edge_t {
    int vertex_1;  // 1st vertex id
    int vertex_2;  // 2nd vertex id
    double weight; // weight of the edge
    int id; // id of edge
    bool valid; // edge valid or not, useful for rc tree generation
} edge_t;

// Vertex Data Structure
typedef struct vertex_t {
    int id; // id of degree
    int degree; // degree of edge, might mutate
} vertex_t;

// RC tree node data structure
typedef struct rc_tree_node_t {
    int cluster_degree; // nullary, unary, binary
    int rep_vertex; // id of representative vertex

    bool has_edge;
    bool has_vertex;
    int edge_id;
    int vertex_id;
} rcTreeNode_t;


// Routines
void init_process(edge_t* edges, int num_vertices, int num_edges);
void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges);

#endif
