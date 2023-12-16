#include <assert.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvtx3/nvToolsExt.h>

using namespace std;

// stopping criteria either reach MAX_ITER or delta < THRESHOLD
#define MAX_ITER 100
#define THRESHOLD 1e-6

// float* centroids_global;

/*----------------------------
    START function decleration
----------------------------*/
void dataset_in(char* dataset_filename, int* N, int** data_points);
void clusters_out(const char* cluster_filename, int N, int* cluster_points);
void centroids_out(const char* centroid_filename, int K, int num_iterations, float** centroids);
void print_final_centroids(int K, float* centroids, int num_iterations);

__global__
void kmeans_init
(
    int N,
    int K,
    int* data_points,
    int* cluster_points,
    float* centroids
);

__global__
void kmeans_execution
(
    int N,
    int K,
    int* data_points,
    int* num_iterations,
    int* cluster_points,
    float* centroids,
    int* cluster_index,
    int* cluster_count,
    float* cluster_sum
);

__global__
void calculate_distance
(
    int N,
    int K,
    int* data_points,
    int* num_iterations,
    float* centroids,
    int* cluster_index,
    int* cluster_count,
    float* cluster_sum,
    int iter_counter
);

__global__
void kmeans_print(int* data_points, int N);
/*----------------------------
    END function decleration
----------------------------*/

/*-------------------------
    START main
-------------------------*/
int main(int argc, char* argv[])
{
    /*------------------------
        INPUTS
    ------------------------*/
    int N;					        // no. of data points (prompt)
    int K;					        // no. of clusters to be formed (prompt)

    // Host array
    int* data_points;		        // data points (from file)

    // Device array
    int* d_data_points;             // data point array on Device (from Host)
    /*------------------------
        OUTPUTS
    ------------------------*/
    // Host variable
    int num_iterations;             // no of iterations performed by algo (variable)
    // device pointer to above
    // int* d_num_iterations;           // calculate on Device then (to Host)

    // Host arrays
    int* cluster_points;	        // clustered data points (to file)
    float* centroids;   	        // final centroids (to file)

    // Device arrays
    int* d_cluster_points;          // clusterd data points (to host)
    float* d_centroids;             // centroids in each iteration (to host)

    // Managed device arrays
    int* d_cluster_index;           // index for each cluster
    int* d_cluster_count;           // the number of points in that cluster
    float* d_cluster_sum;           // the sum of all points in that cluster
    /*------------------------
        FILE NAMES
    ------------------------*/
    char dataset_filename[100] = ""; // input filename (runtime or prompt)
    const char* dataset_prefix = "dataset-";
    const char* cluster_filename = "cluster_CUDA_dataset.txt";
    const char* centroid_filename = "centroid_CUDA_dataset.txt";

    /*------------------------
        CUDA Variables
    ------------------------*/
    cudaError_t initError;
    cudaError_t execError;

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    cout << "---CUDA Properties---";
    cout << "\nDevice name: " << props.name;
    cout << "\nNumber of SMs: " << props.multiProcessorCount;
    cout << "\nMax blocks per SM: " << props.maxBlocksPerMultiProcessor;
    cout << "\nMax threads per SM: " << props.maxThreadsPerMultiProcessor;
    cout << "\nMax threads per block: " << props.maxThreadsPerBlock;
    cout << "\nComputer Compute Mode: " << props.computeMode;
    cout << "\nCompute Major: " << props.major;
    cout << "\nCompute Minor: " << props.minor;


    // checking for runtime arguments. get them if not set properly
    if (argc != 3) {
        // get number of clusters (K)
        cout << "[main()] - Enter No. of Clusters: ";
        cin >> K;
        // get number of datapoints (N)
        // it is also in the dataset file header
        cout << "[main()] - Enter Num. of Data Points: ";
        cin >> N;

        char file_index_char[10];
        snprintf(file_index_char, 10, "%d", N); // convert to string for strcat

        // generate appropriate file names
        strcat(dataset_filename, dataset_prefix);
        strcat(dataset_filename, file_index_char); // concatenate num. of datapoints
        strcat(dataset_filename, ".txt");
    }

    // if proper runtime arguments passed
    else {
        K = atoi(argv[1]);
        dataset_prefix = argv[2];
    }

    // read input data
    nvtxRangePushA("[main()] - dataset_in");
    dataset_in(dataset_filename, &N, &data_points);
    nvtxRangePop();
    cout << "[main()] - Finished reading input data\n";

    /*------------------------
        MEMORY MANAGEMENT
    ------------------------*/
    /*----- Memory Related variables -----*/
    size_t index_size = N * sizeof(int);
    size_t data_size = N * 3 * sizeof(int);
    size_t cluster_size = N * 4 * sizeof(int);
    size_t count_size = K * sizeof(int);
    size_t max_centroid_size = (MAX_ITER + 1) * 3;
    size_t centroids_size;

    /*----- Host variables -----*/
    // Allocating 4 units of space for each data point (x,y,z,cluster_index):
    cluster_points = (int*)malloc(cluster_size);

    // centroids will have space allocated after calculation
    // centroids = (float*) calloc(centroid_size, sizeof(float));


    /*----- Device variables  -----*/
    cudaMalloc(&d_num_iterations, 1*sizeof(int));
    cudaMalloc(&d_data_points, data_size);
    cudaMalloc(&d_cluster_points, cluster_size);
    cudaMalloc(&d_centroids, max_centroid_size);

    // Auxllary arrays for kmeans_execution()
    cudaMalloc(&d_cluster_index, index_size);
    cudaMalloc(&d_cluster_count, count_size);
    cudaMalloc(&d_cluster_sum, 3 * count_size);
    printf("[main()] - Allocated memory for device arrays\n");

    cudaMemcpy(d_data_points, data_points, data_size, cudaMemcpyHostToDevice);

    // initializing arrays
    nvtxRangePushA("[main()] - kmeans_init");
    kmeans_init<<<1, 1>>>
    (
        N,
        K,
        d_data_points,
        d_cluster_points,
        d_centroids
    );
    nvtxRangePop();

    initError = cudaGetLastError();

    if(initError == cudaSuccess) {
        cout << "[main()] - K-Means arrays initialized\n";
    } else {
        cout << "[main()] - K-Means init error\n";
    }

    // Executing k-means sequential:
    nvtxRangePushA("[kmeans_init()] - KmeansExecution");
    kmeans_execution<<<1, 1>>>
    (
        N,
        K,
        d_data_points,
        d_num_iterations,
        d_cluster_points,
        d_centroids,
        d_cluster_index,
        d_cluster_count,
        d_cluster_sum
    );
    nvtxRangePop();

    execError = cudaGetLastError();

    if(execError == cudaSuccess) {
        cout << "[main()] - Finished K-means execution\n";
    } else {
        cout << "[main()] - K-Means execution error\n";
    }

    // Memory transfer to Host
    // Copy actual number of iterations to allocate centroid memory
    cudaMemcpy(&num_iterations, d_num_iterations, sizeof(int), cudaMemcpyDeviceToHost);

    // allocate space according to runtime number of iterations
    centroids_size = (num_iterations + 1) * K * 3;
    centroids = (float*)calloc(centroids_size, sizeof(float));

    // copy output arrays to host
    cudaMemcpy(cluster_points, d_cluster_points, cluster_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);

    printf("[main()] - Number of iterations:%d\n", num_iterations);
    printf("[main()] - Centroids_size:%d\n", (int)centroids_size);

    // write outputs to files
    nvtxRangePushA("[main()] - clusters_out");
    clusters_out(cluster_filename, N, cluster_points);
    nvtxRangePop();

    nvtxRangePushA("[main()] - centroids_out");
    centroids_out(centroid_filename, K, num_iterations, &centroids);
    nvtxRangePop();

    print_final_centroids(K, centroids, num_iterations);

    // free memory
    cudaFree(data_points);
    cudaFree(cluster_points);
    cudaFree(centroids);
    cudaFree(d_cluster_index);
    cudaFree(d_cluster_sum);
    cudaFree(d_cluster_count);

    return 0;
}
/*-------------------------
    END main
-------------------------*/

/*
    DATASET_IN function implementation

    inputs: char* dataset_filename, int N
    outputs: int **data_points

    -----------------------------------
    reads dataset points from input file and creates array
    containing data points, using 3 index per data point, ie
    -----------------------------------------------------
    | pt1_x | pt1_y | pt1_z | pt2_x | pt2_y | pt2_z | ...
    -----------------------------------------------------
*/
void dataset_in(char* dataset_filename, int* N, int** data_points)
{
    cout << "[dataset_in()] - Reading input data from" << dataset_filename << "\n";
    FILE* fin = fopen(dataset_filename, "r");

    // read number of points
    fscanf(fin, "%d", N);

    // allocate memory for the data_points
    size_t data_size = (*N) * 3 * sizeof(int);
    *data_points = (int*)malloc(data_size);
    //cudaMallocManaged(&data_points, data_size);

    // read the data_points
    for (int i = 0; i < (*N) * 3; i++)
    {
        fscanf(fin, "%d", (*data_points + i));
    }

    fclose(fin);
}
/*
    CLUSTERS_OUT function implementation

    inputs:  int N, int *cluster_points
    outputs: char* cluster_filename

    -----------------------------------
    given a dataset array and no. of datapoints,
    the array is written to a .txt file inside the outputs folder.
    this function appends the cluser no. to each point in the same line inside the file
*/
void clusters_out(const char* cluster_filename, int N, int* cluster_points)
{
    cout << "[clusters_out()] - Writing to " << cluster_filename << "cluster file\n";
    FILE* fout = fopen(cluster_filename, "w");
    for (int i = 0; i < N; i++)
    {
        fprintf(fout, "%d %d %d %d\n",
            cluster_points[i * 4],
            cluster_points[i * 4 + 1],
            cluster_points[i * 4 + 2],
            cluster_points[i * 4 + 3]
        );
    }

    fclose(fout);
    cout << "[clusters_out()] - Clustered points output file " << cluster_filename << " saved\n";
}
/*
    CENTROIDS_OUT function implementation

    inputs:  int K, int num_iterations, int *cluster_points
    outputs: char* centroid_filename

    -----------------------------------
    given a dataset array and no. of datapoints,
    the array is written to a .txt file inside the outputs folder.
    this function appends the cluser no. to each point in the same line inside the file
*/
void centroids_out(const char* centroid_filename, int K, int num_iterations, float** centroids)
{
    cout << "[centroids_out()] - Writing to " << centroid_filename << " centroid file\n";
    FILE* fout = fopen(centroid_filename, "w");
    for (int i = 0; i < num_iterations + 1; i++)
    { //ith iteration
        fprintf(fout, "---Iteration#%d---\n", i+1);
        for (int j = 0; j < K; j++)
        { //jth centroid of ith iteration
            fprintf(fout, "%.4f %.4f %.4f\n",
                (*centroids)[(i * K + j) * 3],	 // x-coordinate
                (*centroids)[(i * K + j) * 3 + 1], // y-coordinate
                (*centroids)[(i * K + j) * 3 + 2]  // z-coordinate
            );
        }
        
    }

    fclose(fout);
    cout << "[centroids_out()] - Cluster Centroid point output file " << centroid_filename << " saved\n";
}

// Printing final centroids:
void print_final_centroids(int K, float* centroids, int num_iterations) {
    printf(
        "[print_final_centroids()] - Centroid #%d:\t\tx= %.2f\ty= %.2f\tz= %.2f\n",
        K,
        centroids[(num_iterations * (K-1)) * 3],
        centroids[(num_iterations * (K-1)) * 3 + 1],
        centroids[(num_iterations * (K-1)) * 3 + 2]
    );
}
/*
    KMEANS_INIT function implementation

    inputs:
        - Number of points (N)
        - Number of clusters (K)
        - data points
    outputs:
        - number of iterations
        - clustered data points
        - centroids

    -----------------------------------
    this function initializes global process variables and prints:
    -- inital centroids
    -- no. of iterations
    -- centroid size
    -- final centroids

    outputs to be used in printing to terminal and output files
    ----------------------------------------------------------------------------
    this is where the dataset points are group into K clusters.
    this function appends the new centroid after each iteration into its array.
    this is the time section of the program which consumes the most time.
    we will focus on parallelizing and optimizing this code using CUDA C/C++
    ----------------------------------------------------------------------------
*/
__global__
void kmeans_init
(
    int N,
    int K,
    int* data_points,
    int* cluster_points,
    float* centroids
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockIdx.x * blockDim.x;
    // Assigning first K points to be initial centroids
    for (int i = idx; i < K; i += stride)
    {
        centroids[i * 3] = data_points[i * 3];
        centroids[i * 3 + 1] = data_points[i * 3 + 1];
        centroids[i * 3 + 2] = data_points[i * 3 + 2];
    }
    // Printing initial centroids:
    for (int i = idx; i < K; i += stride)
    {
        printf("[kmeans_init()] - Initial centroid #%d:\tx= %.2f\ty= %.2f\tz= %.2f\n",
            i + 1,
            centroids[i * 3],
            centroids[i * 3 + 1],
            centroids[i * 3 + 2]
        );
    }
}
/*
    KMEANS_EXECUTION helper function implementation

    inputs: none
    outputs: none

    -----------------------------------
    this it the function which requires parallelism
*/
void kmeans_execution
(
    int N,
    int K,
    int* data_points,
    int* num_iterations,
    int* cluster_points,
    float* centroids,
    int* cluster_index,
    int* cluster_count,
    float* cluster_sum
)
{
    int centroid_index, previous_centroid_index;
    int iter_counter = 0;
    double delta_x, delta_y, delta_z, delta = THRESHOLD + 1;
    double temp_delta = 0.0;
    double x, y, z;
    double min_dist, current_dist;

    /*------------------------
        MEMORY MANAGEMENT
    ------------------------*/
    /*----- Memory Related variables -----*/
    size_t index_size = N * sizeof(int);
    size_t count_size = K * sizeof(int);
    
    /*-----  Host Arrays -----*/
    // index for each cluster
    // int* cluster_index = (int *)malloc(index_size);

    // // the number of points in that cluster
    // int* cluster_count = (int *)malloc(count_size);

    // // the sum of all points in that cluster
    // float* cluster_sum = (float *)malloc(3*count_size);

    /*-----  Device Arrays -----*/
    int* d_cluster_index;           // index for each cluster
    int* d_cluster_count;           // the number of points in that cluster
    float* d_cluster_sum;           // the sum of all points in that cluster

    // cudaMallocManaged(&cluster_index, index_size);
    // cudaMallocManaged(&cluster_count, count_size);
    // cudaMallocManaged(&cluster_sum, 3 * count_size);

    printf("[kmeans_execution()] - Starting K-Means execution\n");

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while ((delta > THRESHOLD) && (iter_counter < MAX_ITER)) //+1 is for the last assignment to cluster centroids (from previous iter)
    {
        /*----- Loop 1 -----*/
        // reset clusters
        for (int i = 0; i < K * 3; i++)
        {
            cluster_sum[i] = 0.0;
        }
        /*----- Loop 2 -----*/
        // reset cluster_count
        for (int i = 0; i < K; i++)
        {
            cluster_count[i] = 0;
        }

        /*----- Loop 3 -----*/
        //for each data point
        calculate_distance<<<1,1>>>
        (
            N,
            K,
            d_data_points,
            num_iterations,
            centroids,
            cluster_index,
            cluster_count,
            cluster_sum,
            iter_counter
        );
        cudaDeviceSynchronize();


        // for (int i = idx; i < N; i += stride)
        // {
        //     // assign these points to their nearest cluster
        //     min_dist = DBL_MAX;
        //     // for each cluster
        //     for (int j = 0; j < K; j++)
        //     {
        //         // calculate centroid x,y,z distance from data point x,y,z
        //         x = centroids[(iter_counter * K + j) * 3] - (float)data_points[i * 3];
        //         y = centroids[(iter_counter * K + j) * 3 + 1] - (float)data_points[i * 3 + 1];
        //         z = centroids[(iter_counter * K + j) * 3 + 2] - (float)data_points[i * 3 + 2];

        //         current_dist = x * x + y * y + z * z;
        //         if (current_dist < min_dist)
        //         {
        //             min_dist = current_dist;
        //             cluster_index[i] = j;
        //         }
        //     }
        //     // add to cluster count
        //     cluster_count[cluster_index[i]] += 1;
        //     //add to local cluster coordinates
        //     cluster_sum[cluster_index[i] * 3] += (float)data_points[i * 3];
        //     cluster_sum[cluster_index[i] * 3 + 1] += (float)data_points[i * 3 + 1];
        //     cluster_sum[cluster_index[i] * 3 + 2] += (float)data_points[i * 3 + 2];
        // }

        /*----- Loop 4 -----*/
        // write cluster to next iteration of centroids
        for (int i = 0; i < K; i++)
        {
            assert(cluster_count[i] != 0);
            centroids[((iter_counter + 1) * K + i) * 3] = cluster_sum[i * 3] / cluster_count[i];
            centroids[((iter_counter + 1) * K + i) * 3 + 1] = cluster_sum[i * 3 + 1] / cluster_count[i];
            centroids[((iter_counter + 1) * K + i) * 3 + 2] = cluster_sum[i * 3 + 2] / cluster_count[i];
        }


        /*----- Loop 5 -----*/
        // Convergence check: Sum of L2-norms over every cluster
        temp_delta = 0.0;

        for (int i = 0; i < K; i++)
        {
            // get the centroid and previous centroid index
            centroid_index = ((iter_counter + 1) * K + i) * 3;
            previous_centroid_index = (iter_counter * K + i) * 3;

            // calculate the delta values between the current centroid and the previous one
            delta_x = centroids[centroid_index] - centroids[previous_centroid_index];
            delta_y = centroids[centroid_index + 1] - centroids[previous_centroid_index + 1];
            delta_z = centroids[centroid_index + 2] - centroids[previous_centroid_index + 2];

            temp_delta += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
        }
        delta = temp_delta;
        iter_counter++;
    }
    
    (*num_iterations) = iter_counter;

    /*----- Loop 6 -----*/
    // assign points to their final choice of clusters
    for (int i = 0; i < N; i++)
    {
        //assign points to clusters
        cluster_points[i * 4] = data_points[i * 3];
        cluster_points[i * 4 + 1] = data_points[i * 3 + 1];
        cluster_points[i * 4 + 2] = data_points[i * 3 + 2];
        cluster_points[i * 4 + 3] = cluster_index[i];
        assert(cluster_index[i] >= 0 && cluster_index[i] < K);
    }
}

// __global__
// void kmeans_print(int* data_points, int N) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = idx; i < N; i += stride) {
//         printf("x=%d\t\ty=%d\t\tz=%d\n", data_points[i], data_points[i + 1], data_points[i + 2]);
//     }
// }

// __global__
// void calculate_distance
// (
//     int N,
//     int K,
//     int* data_points,
//     int* num_iterations,
//     float* centroids,
//     int* cluster_index,
//     int* cluster_count,
//     float* cluster_sum,
//     int iter_counter
// )
// {
//     int x, y, z;
    
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int stride = blockDim.x * gridDim.x;
//     double min_dist, current_dist;
//     for (int i = idx; i < N; i += stride)
//         {
//             // assign these points to their nearest cluster
//             min_dist = DBL_MAX;
//             // for each cluster
//             for (int j = 0; j < K; j++)
//             {
//                 // calculate centroid x,y,z distance from data point x,y,z
//                 x = centroids[(iter_counter * K + j) * 3] - (float)data_points[i * 3];
//                 y = centroids[(iter_counter * K + j) * 3 + 1] - (float)data_points[i * 3 + 1];
//                 z = centroids[(iter_counter * K + j) * 3 + 2] - (float)data_points[i * 3 + 2];

//                 current_dist = x * x + y * y + z * z;
//                 if (current_dist < min_dist)
//                 {
//                     min_dist = current_dist;
//                     cluster_index[i] = j;
//                 }
//             }
//             // add to cluster count
//             cluster_count[cluster_index[i]] += 1;
//             //add to local cluster coordinates
//             cluster_sum[cluster_index[i] * 3] += (float)data_points[i * 3];
//             cluster_sum[cluster_index[i] * 3 + 1] += (float)data_points[i * 3 + 1];
//             cluster_sum[cluster_index[i] * 3 + 2] += (float)data_points[i * 3 + 2];
//         }
// }