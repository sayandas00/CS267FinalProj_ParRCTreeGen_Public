#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << std::endl;
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

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {

    bool debug = true;
  
    // check that a file name is specified
    if (argc != 2) {
        cout << "Need to specify 1 file to read edge list from" << '\n';
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
    string line;
    ifstream myfile (argv[1]);
    int line_cnt = 0;
    if (myfile.is_open())
      {
        while ( getline (myfile, line) )
        {
          line_cnt += 1;
          if (line_cnt > num_edges + 1) {
            cout << "File incorrectly formatted, too many edges" << '\n';
            myfile.close();
            return 0;
          }
          if (line_cnt == 1) {
              // Citation from https://www.javatpoint.com/how-to-split-strings-in-cpp
              // for parsing and splitting strings
              char* curr_ptr = strtok(line, " ");
              if (curr_ptr == NULL) {
                cout << "File incorrectly formatted, no num_vertices given" << '\n';
                myfile.close();
                return 0;
              }
              num_vertices = atoi(curr_ptr);
              curr_ptr = strtok(NULL, " ");
              if (curr_ptr == NULL) {
                cout << "File incorrectly formatted, no num_edges given" << '\n';
                myfile.close();
                return 0;
              }
              num_edges = atoi(curr_ptr);
              curr_ptr = strtok(NULL, " ");
              if (curr_ptr != NULL) {
                cout << "File incorrectly formatted" << '\n';
                myfile.close();
                return 0;
              }
              if (num_edges == 0) {
                cout << "No edge graph, return" << '\n';
                return 0;
              }
              // allocate array for edges
              edges = new edge_t[num_edges];
          } else {
              // Citation from https://www.javatpoint.com/how-to-split-strings-in-cpp
              // for parsing and splitting strings
              int edge_posn = line_cnt - 2;
              char* curr_ptr = strtok(line, " ");
              if (curr_ptr == NULL) {
                cout << "File incorrectly formatted, no vertex_1 given" << '\n';
                myfile.close();
                return 0;
              }
              edges[edge_posn].vertex_1 = atoi(curr_ptr);
              curr_ptr = strtok(NULL, " ");
              if (curr_ptr == NULL) {
                cout << "File incorrectly formatted, no vertex_2 given" << '\n';
                myfile.close();
                return 0;
              }
              edges[edge_posn].vertex_2 = atoi(curr_ptr);
              if (curr_ptr == NULL) {
                cout << "File incorrectly formatted, no vertex_2 given" << '\n';
                myfile.close();
                return 0;
              }
              edges[edge_posn].weight = atof(curr_ptr);
              if (curr_ptr != NULL) {
                cout << "File incorrectly formatted" << '\n';
                myfile.close();
                return 0;
              }
          }
        }
        myfile.close();
        if (line_cnt != num_edges + 1) {
            cout << "File incorrectly formatted, too few edges specified" << '\n';
            return 0;
        }
    }

    else 
    {
        cout << "Unable to open file";
        return 0;
    }
    if (debug) {
        cout << "Num Vertices: " << num_vertices << " Num Edges: " << num_edges << '\n';
        for (int i = 0; i < num_edges; i++) {
            cout << "Edge: " << i << " vertex_1: " << edges[i].vertex_1 << " vertex_2: " << edges[i].vertex_2 << " weight: " << edges[i].weight << '\n';
        }
    }
    free(edges);
}
