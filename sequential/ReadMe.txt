We highly recommened Nvidia Platform to run and profile the code.

To run the code, please keep the Kmeans-OpennAcc-managed and/or Kmeans-OpenAcc-Structured and dataset-1000000 in the same folder.

use the terminal to complile and run code

the following are the command for compliling, running, and profiling:

1- Managed Memory:

nvc -fast -ta=tesla:managed -Minfo=accel -o Kmeans-OpenAcc-managed -Mprof=ccff -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include Kmeans-OpenAcc-managed.c && echo "Compilation Successful!" && ./Kmeans-OpenAcc-managed

nsys profile -t nvtx --stats=true --force-overwrite true -o Kmeans-OpenAcc-managed.c ./Kmeans-OpenAcc-managed


2- Structured Data Model

nvc -fast -ta=tesla: -Minfo=accel -o Kmeans-OpenAcc-Structured -Mprof=ccff -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include Kmeans-OpenAcc-Structured.c && echo "Compilation Successful!" && ./Kmeans-OpenAcc-Structured

nsys profile -t nvtx --stats=true --force-overwrite true -o Kmeans-OpenAcc-Structured.c ./Kmeans-OpenAcc-Structured

