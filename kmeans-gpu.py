from math import ceil, sqrt
from numba import int32, float32, cuda

NUMBER_OF_POINTS = 10000000
# Number of Clusters
NUMBER_OF_CLUSTERS = 100
# Number of iterations of k-means algorithm
ITERATIONS = 15

# Number of repetitions
REPEAT = 10

# number of GPU block & threads
num_blocks = 256
num_threads = 1024


'''
    calculate distance between point and cluster mean (called from device)
'''
@cuda.jit('float32(float32, float32, float32, float32)', device='gpu')
def distance(point_x, point_y, mean_x, mean_y):
    dx = point_x - mean_x
    dy = point_y - mean_y
    return sqrt(dx * dx + dy * dy)

'''
   Group points into clusters
'''
@cuda.jit('void(float32[:,:], int32[:], '
          'float32[:,:], '
          'int32, int32)',
          target='gpu')
def groupByCluster(Points, Points_Cluster,
                   Clusters,
                   num_points, num_clusters):

    # get global index of thread
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # check that index is less than number of points
    if idx < num_points:
        # initialize minimum distance to -1
        min_distance = -1

        for i in range(num_clusters):
            # calculate the distance between point and cluster mean
            my_distance = distance(Points[idx, 0], Points[idx, 1], Clusters[i, 0], Clusters[i, 1])
            # update min-distance
            if min_distance > my_distance or min_distance == -1:
                min_distance = my_distance
                Points_Cluster[idx] = i


'''
    Calculate cluster sums
'''
@cuda.jit('void(float32[:,:], int32[:], '
          'float32[:,:], int32[:], '
          'int32, int32)',
          target='gpu')
def calCentroidsSum(Points, Points_Cluster,
                    Clusters_Sums, Cluster_Num_Points,
                    num_points, num_clusters):
    # define shared arrays for threads in each block
    s_Cluster_Num_Points = cuda.shared.array(shape=(num_threads, 2), dtype=float32)
    s_Cluster_Sums = cuda.shared.array(shape=num_threads, dtype=int32)

    tdx = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Initialize Cluster_Sums array
    if idx < num_clusters:
        Clusters_Sums[idx, 0] = 0.0
        Clusters_Sums[idx, 1] = 0.0
        Cluster_Num_Points[idx] = 0

    # Initialize s_Cluster_Sums array (shared memory)
    if tdx < num_clusters:
        s_Cluster_Num_Points[tdx, 0] = 0.0
        s_Cluster_Num_Points[tdx, 1] = 0.0
        s_Cluster_Sums[tdx] = 0.0

    cuda.syncthreads()

    # atomic add is used to avoid errors
    # Calculate x_sum and y_sum and num_points for each cluster (shared memory)
    if idx < num_points:
        i = Points_Cluster[idx]
        cuda.atomic.add(s_Cluster_Num_Points[i], 0, Points[idx, 0]);
        cuda.atomic.add(s_Cluster_Num_Points[i], 1, Points[idx, 1]);
        cuda.atomic.add(s_Cluster_Sums, i, 1);

    cuda.syncthreads()

    # collect data from all blocks (shared memory) into Cluster_Sums (device memory)
    if tdx < num_clusters:
        cuda.atomic.add(Clusters_Sums[tdx], 0, s_Cluster_Num_Points[tdx, 0]);
        cuda.atomic.add(Clusters_Sums[tdx], 1, s_Cluster_Num_Points[tdx, 1]);
        cuda.atomic.add(Cluster_Num_Points, tdx, s_Cluster_Sums[tdx]);


'''
    Calculate cluster mean
'''
@cuda.jit('void(float32[:,:], float32[:,:], int32[:], '
          'int32)',
          target='gpu')
def updateCentroids(Clusters, Clusters_Sums, Cluster_Num_Points,
                    num_clusters):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx < num_clusters:
        # calculate mean for clusters whose sums have been calculated (avoid division by zero)
        if Cluster_Num_Points[idx] > 0:
            Clusters[idx, 0] = Clusters_Sums[idx, 0] / Cluster_Num_Points[idx]
            Clusters[idx, 1] = Clusters_Sums[idx, 1] / Cluster_Num_Points[idx]

'''
    Calculate cluster mean
'''
@cuda.jit('void(float32[:,:], float32[:,:], int32[:], '
          'int32)',
          target='gpu')
def updateCentroids(Clusters, Clusters_Sums, Cluster_Num_Points,
                    num_clusters):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx < num_clusters:
        # calculate mean for clusters whose sums have been calculated (avoid division by zero)
        if Cluster_Num_Points[idx] > 0:
            Clusters[idx, 0] = Clusters_Sums[idx, 0] / Cluster_Num_Points[idx]
            Clusters[idx, 1] = Clusters_Sums[idx, 1] / Cluster_Num_Points[idx]


'''
    Kmeans handler function
'''
def kmeans(Points, Points_Cluster,
           Clusters, Clusters_Sums, Cluster_Num_Points,
           num_points, num_clusters):

    # allocate & transfer data from host to device
    d_Points = cuda.to_device(Points)
    d_Points_Cluster = cuda.to_device(Points_Cluster)
    d_Clusters = cuda.to_device(Clusters)
    d_Clusters_Sums = cuda.to_device(Clusters_Sums)
    d_Cluster_Num_Points = cuda.to_device(Cluster_Num_Points)

    # run algorithm for multiple iterations
    for i in range(ITERATIONS):
        # group points into clusters
        groupByCluster[num_blocks, num_threads](
            d_Points, d_Points_Cluster,
            d_Clusters,
            num_points, num_clusters
        )
        cuda.synchronize()

        # calculate cluster sums
        calCentroidsSum[num_blocks, num_threads](
            d_Points, d_Points_Cluster,
            d_Clusters_Sums, d_Cluster_Num_Points,
            NUMBER_OF_POINTS, NUMBER_OF_CLUSTERS
        )
        cuda.synchronize()

        # calculate cluster means
        updateCentroids[num_blocks, num_threads](
            d_Clusters, d_Clusters_Sums, d_Cluster_Num_Points,
            NUMBER_OF_CLUSTERS
        )
        cuda.synchronize()

    # copy data back from device to host
    Clusters = d_Clusters.copy_to_host()
    Clusters_Sums = d_Clusters_Sums.copy_to_host()
    Cluster_Num_Points = d_Cluster_Num_Points.copy_to_host()

    return Clusters, Clusters_Sums, Cluster_Num_Points

from time import time
import numpy
import json




'''
    Print clusters info
'''
def printCentroid(Clusters, Clusters_Sums, Cluster_Num_Points):
    for i in range(NUMBER_OF_CLUSTERS):
        print("[x={:4f}, y={:4f}, x_sum={:4f}, y_sum={:4f}, num_points={:d}]".format(
            Clusters[i, 0], Clusters[i, 1], Clusters_Sums[i, 0], Clusters_Sums[i, 1], Cluster_Num_Points[i])
        )

    print('--------------------------------------------------')


'''
  Run algorithm and record time for a number of repeatitions
'''
def runKmeans(Points, Points_Clusters,
              Clusters, Clusters_Sums, Cluster_Num_Points):

    # record start time
    start = time()

    for i in range(REPEAT):
        # initialize clusters
        for i1 in range(NUMBER_OF_CLUSTERS):
            Clusters[i1, 0] = Points[i1, 0]
            Clusters[i1, 1] = Points[i1, 1]

        Clusters, Clusters_Sums, Cluster_Num_Points = kmeans(
            Points, Points_Clusters,
            Clusters, Clusters_Sums, Cluster_Num_Points,
            NUMBER_OF_POINTS, NUMBER_OF_CLUSTERS
        )

        if i + 1 == REPEAT:
            printCentroid(Clusters, Clusters_Sums, Cluster_Num_Points)

    # record end time and calculate average execution time for all repeatitions
    end = time()
    total = (end - start) * 1000 / REPEAT
    
    print("Iterations: {:d}".format(ITERATIONS))
    print("Average Time: {:.4f} ms".format(total))


def main():
    # read points from json file
    with open("../points.json") as f:
        listPoints = list(map(lambda x: (x[0], x[1]), json.loads(f.read())))

    # create arrays
    Points = numpy.ones((NUMBER_OF_POINTS, 2), dtype=numpy.float32)
    Points_Clusters = numpy.arange(NUMBER_OF_POINTS, dtype=numpy.int32)

    Clusters = numpy.ones((NUMBER_OF_CLUSTERS, 2), dtype=numpy.float32)
    Clusters_Sums = numpy.ones((NUMBER_OF_CLUSTERS, 2), dtype=numpy.float32)
    Cluster_Num_Points = numpy.arange(NUMBER_OF_CLUSTERS, dtype=numpy.int32)

    # Initialize Points
    for i, d in enumerate(listPoints):
        Points[i, 0] = d[0]
        Points[i, 1] = d[1]

    runKmeans(Points, Points_Clusters,
              Clusters, Clusters_Sums, Cluster_Num_Points)


if __name__ == '__main__':
    main()
