"""
Translate a distance graph into json format for Owen's applet'
Json format:
{
 "directed": boolean
 "nodes": [{"id": "string1"}, ...]
 "edges": [{"source": "stringA", "target": "stringB", "weight": float, "distance": float}, ...]
}

directed tells whether or not the graph is directed.
nodes is a list of objects with string ids.
edges is a list of edge objects which specify a source and a target node id.
weight is an optional parameter and defaults to 1.
distance is an optional parameter and defaults to 30.    
"""

import numpy as np
import json as js
def _diff(x,y):
        return (x-y)**2

def generate_json(edges,filename,weight=1.):

    Dict={'directed':False}

    segments=edges['segments']
    
    new_pairs=edges['new_pairs']
    n=edges['nodes_no']  
    Dict['nodes']=[{'id':str(i)} for i in range(n)]
    
    edges=[]
    for i in range(len(new_pairs[0])):
        source = new_pairs[0][i]
        target = new_pairs[1][i]
        G=segments[i]
        distance=np.sqrt(np.sum((G[0,:]-G[1,:])**2))
        edge = {'source':str(source),
                'target':str(target),
                'weight':weight,
                'distance':distance
                }
        edges.append(edge)
                
    Dict['edges']=edges
    
    with open(filename, 'w') as json_file:
            js.dump(Dict,json_file)
    print('done generating',filename)
 
    
    
    