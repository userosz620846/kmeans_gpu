from math import sqrt
from time import time
import json


NUMBER_OF_POINTS = 100000

# Number of Clusters
NUMBER_OF_CENTROIDS = 10


REPEAT = 10

# Number of iterations of k-means algorithm
ITERATIONS = 15


'''
    Point object
'''
class Point(object):
    x = None
    y = None
    cluster = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cluster = -1


'''
    Cluster object
'''
class Centroids(object):
    x = None
    y = None
    x_sum = None
    y_sum = None
    num_points = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_sum = 0
        self.y_sum = 0
        self.num_points = 0


'''
    Generate clusters from list of points
'''
def getListCentroids(listPoints):
    listCentroids = list()
    for d in listPoints:
        listCentroids.append(Centroids(d.x, d.y))

    return listCentroids


'''
    Calculate distance from point to cluster mean
'''
def distance(point, centroid):
    dx = point.x - centroid.x
    dy = point.y - centroid.y
    return sqrt(dx * dx + dy * dy)


'''
    Print cluster
'''
def printCentroid(listCentroids):
    for d in listCentroids:
        print("[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
            d.x, d.y, d.x_sum, d.y_sum, d.num_points)
        )

    print('--------------------------------------------------\n')


'''
    Group points into clusters
'''
def groupByCluster(listPoints, listCentroids):
    for i0, _ in enumerate(listPoints):
        # Initialize the shortest distance
        minor_distance = -1

        for i1, centroid in enumerate(listCentroids):
            # Calculate distance between current cluster mean and point
            my_distance = distance(listPoints[i0], centroid)
            # update shortest distance and group points into clusters
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                listPoints[i0].cluster = i1
    return listPoints


'''
    Calculate cluster sums
'''
def calCentroidsSum(listPoints, listCentroids):
    # Initialization
    for i in range(NUMBER_OF_CENTROIDS):
        listCentroids[i].x_sum = 0
        listCentroids[i].y_sum = 0
        listCentroids[i].num_points = 0

    # Calculate x_sum and y_sum and num_points for each cluster 
    for point in listPoints:
        i = point.cluster
        listCentroids[i].x_sum += point.x
        listCentroids[i].y_sum += point.y
        listCentroids[i].num_points += 1

    return listCentroids


'''
    Calculate the cluster mean
'''
def updateCentroids(listCentroids):
    # devide x_sum and y_sum by number of points to get mean point
    for i, centroid in enumerate(listCentroids):
        listCentroids[i].x = centroid.x_sum / centroid.num_points
        listCentroids[i].y = centroid.y_sum / centroid.num_points
    return listCentroids


'''
Kmeans handler function
'''
def kmeans(listPoints, listCentroids):
    for i in range(ITERATIONS):
        listPoints = groupByCluster(listPoints, listCentroids)
        listCentroids = calCentroidsSum(listPoints, listCentroids)
        listCentroids = updateCentroids(listCentroids)

    return listCentroids


'''
    Run algorithm and record time for a number of repeatitions
'''
def runKmeans(listPoints):
    # record start time
    start = time()

    listCentroids = None
    for i in range(REPEAT):
        listCentroids = getListCentroids(listPoints[:NUMBER_OF_CENTROIDS])
        listCentroids = kmeans(listPoints, listCentroids)
        if i+1 == REPEAT:
            printCentroid(listCentroids)

    # record end time and calculate average execution time
    end = time()
    total = (end - start) * 1000 / REPEAT

    print("Iterations: {:d}".format(ITERATIONS))
    print("Average Time: {:.4f} ms".format(total))



if __name__ == "__main__":
    # read points from json file
    with open("C:/Users/goss4/Desktop/school/T3/COE 506 - GPU Programming/project/kmeans-master/points.json") as f:
        listPoints = list(map(lambda x: Point(x[0], x[1]), json.loads(f.read())))

    runKmeans(listPoints)
