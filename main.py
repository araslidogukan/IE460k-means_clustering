import pandas as pd
import numpy as np
from math import inf
import time
from kmeans import my_kmeans
from bisecting_kmeans import my_bisecting_kmeans

df = pd.read_excel('data_directory.xlsx',header=None) #Assuming data is in the correct form with attribute values on columns and every row is a different data point
dermat = df.values

for num_of_clusters in range(3,11):
    print('When K=',num_of_clusters)
    #Kmeans
    start = time.time()
    sse_a_list = []
    time_a_list = []
    best_sse_a = inf
    best_cluster_sizes_a = [0] * num_of_clusters
    cluster_sizes = [0] * num_of_clusters
    for iter in range(500):
        SSE_list, cluster_assignments = my_kmeans(dermat,num_of_clusters)
        SSE = float(np.sum(SSE_list))
        for c in range(num_of_clusters):
            cluster_sizes[c] = int(np.sum(cluster_assignments==c))
        if SSE < best_sse_a:
            best_sse_a = SSE
            best_cluster_sizes_a = cluster_sizes
    end = time.time()
    sec = end - start
    print('K means results:',best_sse_a,best_cluster_sizes_a,sec)
    sse_a_list.append(best_sse_a)
    time_a_list.append(sec)

    #Bisecting Kmeans with 50 calls of k-means at every bisection
    start = time.time()
    time_b50_list = []
    SSE_list, cluster_assignments_b50 = my_bisecting_kmeans(dermat,num_of_clusters,50)
    cluster_sizes_b50 = [0] * num_of_clusters
    sse_b50_list = []
    SSE_b50 = float(np.sum(SSE_list))
    for c in range(num_of_clusters):
        cluster_sizes_b50[c] = int(np.sum(cluster_assignments_b50==c))
    end = time.time()
    sec = end - start
    print('Bisecting with 50 results:',SSE_b50,cluster_sizes_b50,sec)
    sse_b50_list.append(SSE_b50)
    time_b50_list.append(sec)

    #Bisecting Kmeans with 100 calls of k-means at every bisection
    start = time.time()
    time_b100_list = []
    SSE_list, cluster_assignments_b100 = my_bisecting_kmeans(dermat,num_of_clusters,100)
    cluster_sizes_b100 = [0] * num_of_clusters
    SSE_b100 = float(np.sum(SSE_list))
    sse_b100_list = []
    for c in range(num_of_clusters):
        cluster_sizes_b100[c] = int(np.sum(cluster_assignments_b100==c))
    end = time.time()
    sec = end - start
    print('Bisecting with 100 results:',SSE_b100,cluster_sizes_b100,sec)
    sse_b100_list.append(SSE_b100)
    time_b100_list.append(sec)
