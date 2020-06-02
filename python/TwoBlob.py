"""
Code to work with the two-blob example from Alex
"""

import random
import numpy as np
import pandas as pd
import pylab as py
from CoverTree import ElaboratedTreeNode, gen_scatter

def twoBlob():
    data = pd.read_csv('../../Data/twoBlob.csv',header=None)
    print(data.shape)
    np.random.shuffle(data.values)
    #data=data.iloc[:1000,:]
    center=np.mean(data,axis=0)
    T=ElaboratedTreeNode(center,radius=4,path=(),alpha=0.01,thr=0.99)
    print(T._str_to_level(1))
    
    T.state.set_state('seed')

    for i in range(1,data.shape[0]):
        point=np.array(data.iloc[i,:])
        T.insert(point)
    
    print(T._str_to_level(2))
    Nodes = T.collect_nodes()
    C=[]
    for node in Nodes:
        if node.state.get_state() in ['seed','passThrough']:
            n=node.no_of_children()
            d=np.log(n+1)/np.log(2.)
            if n<10:
                C.append((node.center,d,node.path)) 
    centers=np.array([c[0] for c in C])
    D=[c[1] for c in C]
    max_d=np.max(D)
    S=[(max_d+3-d)*50 for d in D]
    gen_scatter(T,data,level=0)
    py.scatter(centers[:,0],centers[:,1],c=D,s=S)
    print('Finished TwoBlob')
