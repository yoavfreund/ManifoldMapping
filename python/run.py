#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:12:43 2020

@author: yoavfreund
"""


from  Generators import DataGenerator, Compute_edges
from CoverTrees import ElaboratedTreeNode
import numpy as np
import pylab as pl

from matplotlib.patches import Circle
from matplotlib import collections  as mc


def FlexCircle( xy, radius, color="lightsteelblue", facecolor="none", alpha=1, ax=None ):
    """ add a circle to ax= or current axes
    """
        # from .../pylab_examples/ellipse_demo.py
    e = Circle( xy=xy, radius=radius )
    if ax is None:
        ax = pl.gca()  # ax = subplot( 1,1,1 )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_edgecolor( color )
    e.set_facecolor( facecolor )  # "none" not None
    e.set_alpha( alpha )
    return e
    

def toColor(i):
    """ implement a color cyces """
    colors=['c', 'm', 'y', 'r','k','g']
    nc=len(colors)
    return(colors[i % nc])
 
Markers=['X','P','1','2','3','4','>','<','v','^','.']
def plot_graph(T,data,max_depth=-1,debug=False):
#    lc = mc.LineCollection(segments, colors='r', linewidths=0.1)
    nodes=T.collect_nodes(depth=max_depth)
    if debug:
        print('in plot_graph, number of nodes=',len(nodes),'max_depth=',max_depth)
    X=np.stack([x.center for x in nodes])
    depth=np.array([len(x.path) for x in nodes])
    radius = np.array([x.radius for x in nodes])
    parent_state=[]
    for node in nodes:
        if len(node.path) == 0:  # if root node
            parent_state.append('passThrough')
        else:
            parent_state.append(node.parent.state.get_state())
    
    if debug:
        print('plot_graph, no. of nodes=',len(nodes),'depth min/max=',np.min(depth),np.max(depth))
        
    fig, ax = pl.subplots(figsize=[12,12])
    
    _colors=[toColor(d) for d in depth]
    sizes=[20*(4-d)**2 for d in depth]
    ax.scatter(X[:,0],X[:,1],marker='o',facecolors='none',linewidths=5,edgecolors=_colors,s=sizes,alpha=0.5);
    ax.scatter(data[:,0],data[:,1],marker='.',c='b',s=1);

    md=np.max(depth)
    if max_depth!=-1 and max_depth<md:
        md=max_depth
    edges=[]
    for d in range(md):
        E=X[depth==d,:]
        first=np.nonzero(depth==d)[0][0]
        edges.append({'depth':d,
                      'nodes':E,
                      'radius':radius[first]
                      })
    for d in range(1,md):   
        E=edges[d]
        S=Compute_edges(E['nodes'],2.2*E['radius'])
        
        lc = mc.LineCollection(S['segments'], colors=toColor(d), linewidths=1)
        ax.add_collection(lc)
        
# =============================================================================
#  plot circles
#
#     patches = []
#     for i in range(radius.shape[0]):
#         ps=parent_state[i]
#         if not ps is None and not ps in ['passThrough','refine']:
#             continue
#         if depth[i]>max_depth:
#             continue
#         circle = FlexCircle((X[i,0], X[i,1]), radius[i],alpha=1,ax=ax,color=_colors[i])
#         patches.append(circle)
#     
#     p = PatchCollection(patches)
#     p.set_array(np.array(depth))
#     ax.add_collection(p)
# 
# =============================================================================
    #pl.xlim([-1,2])
    #pl.ylim([-1,2])
    #ax.add_collection(lc)
    #ax.set_xlim([0,1])

print('start')
generator=DataGenerator()

data=generator.Sample(n=10000)
center=np.mean(data,axis=0)
T=ElaboratedTreeNode(center,radius=1,path=(),alpha=0.01,thr=0.99)
T.state.set_state('seed')

depth=5

#T.set_debug()
T.set_max_depth(depth)

for i in range(1,data.shape[0]):
    print('\r',i,end='')
    point=np.array(data[i,:])
    T.insert(point)

print()
print('#-'*50)    

print(T._str_to_level(depth))
plot_graph(T,data,depth,debug=True)



