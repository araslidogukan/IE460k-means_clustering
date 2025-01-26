import numpy as np
def my_kmeans(data,k):
    num_of_rows = data.shape[0]  #Store the number of rows
    num_of_attrs = data.shape[1] #Store the number of columns
    
    #Initialize Centers
    means = np.mean(data,axis=0) 
    cov = np.cov(data,rowvar=False)
    centers = np.random.multivariate_normal(means,cov,k).T.reshape(1,num_of_attrs,k)
    centers_new = centers.copy()

    data = data.reshape(num_of_rows,num_of_attrs,1) #reshape data into 3d to use broadcasting

    is_updated = True #flag for convergence
    while is_updated:
        distances = np.linalg.norm(data - centers,axis=1) #Euclidian Distance
        assignments = distances.argmin(axis=1) #Assignments to closest centers
        for i in range(k):
            if len(data[assignments == i,:,:]) == 0 : assignments[distances[:,i].argmin(axis=0)] = i #Assign closest points to empty cluster centers
            centers_new[:,:,i] = np.mean(data[assignments == i,:,:],axis=0).T #Average to calculate centers

        if np.all(centers == centers_new):
            is_updated = False #if centers are not updated set the flag
        centers = centers_new.copy() #update centers
   
    SSE = np.zeros(k)
    for i in range(k):
        SSE[i] = float(np.sum((data[assignments==i,:]-centers[:,:,i:i+1])**2))

    return SSE,assignments
