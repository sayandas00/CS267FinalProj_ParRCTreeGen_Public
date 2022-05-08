# Implementing Parallel RC Tree Generation for GPU
## Sayan Das (sayan_g_das@berkeley.edu) and Aditya Kumar (aditya_kumar@berkeley.edu)

RC trees are commonly used to support efficient updates and queries on trees.
Here we implement a few parallel RC tree generation algorithms for bounded degree (3), undirected trees, both rooted and unrooted.

## Input
A text file where the first line is the number of vertices, then a space, then the number of edges in the tree.
Vertex ids should be integers in the inclusive range (1, number of vertices)
The next lines should be the edges in the tree, each edge their own line, described as vertex_1 vertex_2 edge_weight
At the end of the file, on a new line, if the tree is rooted, write the vertex id of the root. If unrooted, you don't include this line.

Example: for a chain graph 1<->2<->3 rooted on 2, where all edges have weight 2, the input file would be: <br>
3 2<br>
1 2 2<br>
2 3 2<br>
2<br>

Example: see anderson_tree.txt for how to represent an unrooted tree in Figure 1 of Anderson and Blelloch's Parallel Minimum Cuts in O(m log^2 (n)) Work and Low Depth.

## Output
Program will print to stdout the edges provided in the text file, then describe the RC Tree edges, then describe the RC Tree nodes.
Cluster nodes in the RC Tree will have representative vertices, while nodes corresponding to original graph edges and vertices will only print those fields respectively.

Example: see rctree_anderson_tree.out for a sample output of the RC tree generated from the input anderson_tree.txt

## How to run the program on Bridges2
Program should only be run on the rootedTree, lubyMIS, and randCompress branches.
The following commands are for the rootedTree branch. See variants for the other branches. <br>
Use the following italicized commands to ensure that the environment is built correctly: <br>
finalProj>*module load cuda* <br>
finalProj>*mkdir build* <br>
finalProj>*cd build* <br>
finalProj/build>*cmake -DCMAKE_BUILD_TYPE=Release ..* <br>
finalProj/build>*make* <br>
Grab an interactive node with: <br>
finalProj/build>*salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive* <br>
To run on an input file and pipe stdout to a new file:
finalProj/build>*./gpu input.txt > output.out

## Variants for determining independent set
On the rootedTree branch, we have the edge grabbing independent set approach to determine which nodes get to compress/rake.<br> <br>
On the lubyMIS branch, the independent set is generated using Luby's maximal independent set algorithm from Luby's A Simple Parallel Algorithm for the Maximal Independent Set Problem.
The only difference in running the program is to replace the previous *./gpu ....* command with *./gpu input.txt -s seed_num > output.out*
On the randCompress branch, the independent set is generated using one iteration of Luby's maximal independent set algorithm.
The only difference in running the program is to replace the original *./gpu ....* command with *./gpu input.txt -s seed_num > output.out*

## References
https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand for curand setup code <br>
CS267 Spring 2022 HW2-3 Writers for command line parsing code, timing code, cmake file, repo skeleton, and build + run commands <br>
https://www.javatpoint.com/how-to-split-strings-in-cpp for parsing input file code. <br>
https://www.cplusplus.com/doc/tutorial/files/ for reading input file code. <br>
Luby's A Simple Parallel Algorithm for the Maximal Independent Set Problem for the lubyMIS and randCompress approaches to determining an independent set of nodes to contract. <br>
Anderson and Blelloch's Parallel Minimum Cuts in O(m log^2 (n)) Work and Low Depth for the anderson_tree tree. <br>
Acar, et al.'s Parallel Batch-dynamic Trees via Change Propagation and Miller and Reif's Parallel Tree Contraction and Its Application for rake and compress rules.
