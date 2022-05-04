#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants

// Edge Data Structure, self edges are not supported currently
typedef struct edge_t {
    int vertex_1;  // 1st vertex id
    int vertex_2;  // 2nd vertex id, use as the parent vertex in rcTree
    double weight; // weight of the edge
    int id; // id of edge, assume = posn + 1
    bool valid; // edge valid or not, useful for rc tree generation
    int marked; // edge marked for IS or not, !=0 means someone already grabbed it
    int iter_added; // helpful to determine which boundary vertex contracted first, only init in rcTreeEdges for edges corresponding to original vertices
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

    int bound_vertex_1; // id of boundary vertex, set to -1 if none
    int bound_vertex_2; // id of boundary vertex, set to -1 if none

    int edge_id; // id of edge, set to -1 if none
    int vertex_id; // id of vertex, set to -1 if none
} rcTreeNode_t;


// Routines
void init_process(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges);
void rc_tree_gen(edge_t* edges, int num_vertices, int num_edges, rcTreeNode_t* rcTreeNodes, edge_t* rcTreeEdges);

#endif
