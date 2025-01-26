#Dogukan Arasli 2517522
import numpy as np
from math import inf
from kmeans_ID2517522 import my_kmeans
def my_bisecting_kmeans(whole_data,k,num_of_kmeans_iters):
    #Initialize the outputs as empty arrays
    SSEs = np.zeros(k)
    assignments = np.zeros(whole_data.shape[0])

    #Initialize partitions of the whole set for the start
    data = whole_data.copy()
    max_index = 0

    for i in range(k-1):
        best_sse = inf
        best_bisection = [0] * 2
        for a in range(num_of_kmeans_iters):
            SSE_list, bisection = my_kmeans(data,2)
            SSE = float(np.sum(SSE_list))
            if SSE < best_sse:
                best_sse = SSE
                best_bisection = bisection.copy()

        #Change the assignment indices from 0 and 1 to corresponding ones
        update = best_bisection.copy()
        update[best_bisection==0] = max_index
        update[best_bisection==1] = i+1

        #Change the cumulative variables
        SSEs[[max_index,i+1]] = best_sse
        assignments[assignments==max_index] = update

        max_index = SSEs.argmax() #Calculate the cluster with biggest SSE from its center
        data = whole_data[assignments==max_index,:].copy() #Slice the data to obtain corresponding cluster:
        
    return SSEs, assignments