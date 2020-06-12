
from numpy.matlib import repmat
import numpy as np

def Kmeans(points,centers,max_iter=10,stationary=[]):
    """
    Refine the centers using the Kmeans algorithm

    Parameters
    ----------
    points : list or np.array
        The points to be estimated using the centers
    centers : list or np.array
        The centers. 
    stationary: A list of indices of centers that are not to be moved.
    max_iter : int, optional
        DESCRIPTION. the maximal number of iterations. 
        Stop earlier if reached a statiobary point. The default is 10.

    Returns
    -------
    centers_list : list
        an list of updated centers.
    cost: float,
        mean distance to closest center
    """
    
    if type(points)!=np.ndarray:
        points = np.stack(points)
    if type(centers) !=np.ndarray:
        centers=np.stack(centers)
    last_cost=-1
    for _iter in range(5):
        nearest=[]; dists=[]
        for i in range(points.shape[0]):
           P=points[i,:]
           Prep=repmat(P,centers.shape[0],1)
           diff=(centers - Prep)**2
           dist=np.sum(diff,axis=1)
           nearest.append(np.argmin(dist))
           dists.append(np.min(dist))
        dists=np.array(dists)
        nearest=np.array(nearest)
        cost=np.mean(dists)
        #print('iter=',iter,'cost=',cost)
        if cost==last_cost:
            break
        else:
            last_cost=cost
        for i in range(centers.shape[0]):
            if i in stationary:
                continue
            closest=points[nearest==i,:]
            # print("i=%d, closest.no=%d"%(i,closest.shape[0]))
            if closest.shape[0]>0:
                centers[i,:] = np.mean(closest,axis=0)
    centers_list=[centers[i,:] for i in range(centers.shape[0])]
    return cost,centers_list