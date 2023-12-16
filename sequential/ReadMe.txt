We highly recommened Nvidia Platform to run and profile the code.

To run the sequential C code, please keep the Kmeans-sequential.c and dataset-1000000 in the same folder.

use the terminal to complile and run code

the following are the command for compliling, running, and profiling:

nvc -fast -o Kmeans-sequential -Mprof=ccff -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include Kmeans-sequential.c && echo "Compilation Successful!" && ./Kmeans-sequential
nsys profile -t nvtx --stats=true --force-overwrite true -o Kmeans-sequential.c ./Kmeans-sequential

After compilation the algorithm will generate and output new cluster file and new centroid file.

to change the number of clusters you can change the value of k in the main function.
