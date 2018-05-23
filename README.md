# PIHC_TSP
Sequential and GPU-based parallel implementation of Traveling Salesman Problem (TSP) 
Codes are categorized into two types, sequential and parallel. Sequential is implemented using C language and which is available in "sequential" folder.  Parallel codes are implemented using CUDA and which is available in "parallel" folder. The parallel folder contains three code files namely PIHC, DM, and TDM.  DM and TDM use TPN strategy. Different initial solution construction techniques and thread mapping strategies are available in PIHC code.

For parallel:
Compilation command: nvcc PIHC.cu -arch=sm_35 -o pihc
Execution command: ./pihc ../instances/kroA200.tsp 

prerequisite:
1. In CUDA code, we have used a 64-bit atomicMin() function that supports a GPU device which has computing capability 3.5 and higher. 
2. To run Christofides' algorithm, you have to use blossom V tool which is available at link: http://pub.ist.ac.at/~vnk/software.html

For sequential:
Compilation comand: gcc IHC.c -lm -o ihc
Execution command: ./pihc ../instances/kroA200.tsp 

The optimal solutions of TSPLIB instances are available at the link: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html

