# kmeans_gpu
Accelerated k-means algorithm using GPU.

intro:
This repo contains the source code for k-means GPU acceleration project. This project aims to implement GPU acceleraion (using 3 methods) on the algorithm, and compare the results.

objectives:
1- accelerate k-means algorithm using: 1) OpenACC, 2) CUDA C/C++, 3) and Python Numba.
2- compare and study the perforamce of the three implementaion, using the sequential performance as a benchmark.


inlcuded in this repo:
1- sequential c implementaion of the k-means algorithm
2- sequential python implementaion of the k-means algorithm
3- GPU Accelerated c implementioan of the k-means algorithm using OpenACC
4- GPU Accelerated c implementioan of the k-means algorithm using CUDA C
5- GPU Accelerated python implementioan of the k-means algorithm using NUMBA CUDA
6- datasets (for both c and python codes)
