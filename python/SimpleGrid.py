import numpy as np
import json as js

def neighbors(i,j,nrow,ncol):
    N=[]
    if i > 0:
        N.append((i-1,j))
    if j>0:
        N.append((i,j-1))
    # if i>0 and j>0:
    #     N.append((i-1,j-1))
    # if i>0 and j<ncol-1:
    #     N.append((i-1,j+1))
    return N

def grid(nrow,ncol):
    vertices=[]
    edges=[]
    for i in range(nrow):
        for j in range(ncol):
            vertices.append({'id':'%d,%d'%(i,j)})
            N=neighbors(i, j, nrow, ncol)
            for (i2,j2) in N:
                edges.append({"source":'%d,%d'%(i,j), "target":'%d,%d'%(i2,j2)})
    Dict={'directed':False,
          'nodes':vertices,
          "edges":edges}
    return Dict
    

if __name__=="__main__":
    for (nrow,ncol) in [(3,5),(5,7),(10,11)]:
        Dict=grid(nrow,ncol)
    
        with open('grid(%d,%d).json'%(nrow,ncol), 'w') as json_file:
            js.dump(Dict,json_file)
        print('done with (%d,%d):'%(nrow,ncol))
    