"""
Generate and manage cover trees.

Minimal description
"""

import random
import numpy as np
import pandas as pd
import pylab as py
from Kmeans import Kmeans

#%%
import sys
_MAX = sys.float_info.max

def _dist(x,y):
    """
    Compute the distance between two points. Euclidean distance between 1D numpy arrays.

    Parameters
    ----------
    x: numpy array
        First Point
    y: numpy array
        Second Point 

    Returns
    -------
    float
    Euclidean distance

    """
    return np.sqrt(np.sum((x-y)**2))
    
class CoverTreeNode(object):
    
    def __init__(self,center,radius,path):
        """
        Defines a node in a covertree

        Parameters
        ----------        
        center: point  
            The center of the region for this node
        radius: positive float
            Only points whose distance from center is at most radius are included in the node
        path:   a list of CoverTrees
            A sequence of nodes that defines the tree path leading to this node.

        Returns
        -------
        None.

        """
        self.center=center
        self.radius=radius
        self.counter=1  # number of points covered by this tree
        self.too_far=0  # count how many points were too far (should only be non-zero at root
        self.path=path
        self.children = []
    
    def dist_to_x(self,x):
        return _dist(x,self.center)

    def covers(self,x):
        return self.dist_to_x(x) < self.radius
        
    def no_of_children(self):
        return len(self.children)

    def get_level(self):
        return len(self.path)
    
    def find_path(self,x):
        """
        Find the path in the current tree for the point x.

        Parameters
        ----------
        x : point
            point for which we want to know the tree path.

        Returns
        -------
        list of CoverTreeNode s
            The list of nodes that contain the point x

        """
        d= _dist(x,self.center)
        #print(str(self.path),d)
        if d>self.radius:
             return None
        if len(self.children)==0:
            return [self]
        else:
            for child in self.children:
                child_path = child.find_path(x)
                if child_path is None:
                    return [self]
                else:
                    return [self]+child_path
            return [self]

    def insert(self,x):
        """
        Add a new node that contains a given point.  
        Fails if node is outside of the ballof the root.

        Parameters
        ----------
        x : point
            The point to seed the new node

        Returns
        -------
        bool
            Success/Failure 

        """
        path=self.find_path(x)
        if path is None:
            return False

        #found a non-trivial path
        leaf=path[-1]
        new=CoverTreeNode(x,leaf.radius/2,leaf.path+(leaf.no_of_children(),))
        leaf.children.append(new)
        for node in path:
            node.counter +=1
        return True
                    
    def collect_centers(self):
        """
        Collect all of the centers defined by a tree.

        Returns
        -------
        C : list
            returns a list where each element is a center, followed by the level of the center
        """
        C=[(self.center,len(self.path))]
        if len(self.children)>0:
            for child in self.children:
                C = C+ child.collect_centers()
        return C
           
    def collect_nodes(self):
        """
        Create list of nodes.
        
        Returns
        -------
        N : list
            returns a list of all nodes in the tree
        """
        N=[self]
        if len(self.children)>0:
            for child in self.children:
                N=N+child.collect_nodes()
        return N

    def __str__(self):
        """
        Create a string describing this node.

        Returns
        -------
        string
            description of the node

        """
        return str(self.path)+': r=%4.2f, no_child=%d, count=%d'%(self.radius,len(self.children),self.counter)

    def _str_to_level(self,max_level=0,print_leaves=False):
        """
        Create a string that descibes the nodes along the path to this node.

        Parameters
        ----------
        max_level : integer
            The maximal level  The default is 0 which correspond to unlimited level.
        print_initial : Boolean
            print all nodes, including leaves, The default is False (don't print leaves')

        Returns
        -------
        s : string
            string.

        """
        s=self.__str__()+'\n'
        if self.get_level() < max_level and len(self.children)>0:
            for i in range(len(self.children)):
                child=self.children[i]
                if child.state.get_state() != 'initial':
                    s+=child._str_to_level(max_level)
        return s    

class NodeState:
    """
       states = ('initial', # initial state, collect statistics
              'seed',    # collect centers
              'refine',  # refine centers
              'passThrough') # only collect statistics, advance children from 'initial' to 'seed'
    """
    states = ('initial', # initial state, collect statistics
              'seed',    # collect centers
              'refine',  # refine centers
              'passThrough') # only collect statistics, advance children from 'initial' to 'seed'
    def __init__(self):
        self.state = 'initial'
    def get_state(self):
        return self.state
    def set_state(self,state):
        assert(str(state) in NodeState.states)
        self.state=state
        
class ElaboratedTreeNode(CoverTreeNode):
    def __init__(self,center,radius,path,thr=0.9,alpha=0.1,max_children=10):
        """
        Initialize and elaboratedTreeNode

        Parameters
        ----------
        center, radius, path : 
            as defined in CoverTreeNode.__init__()
        thr : TYPE, optional
            The minimal estimated coverage of the node to allow it's children to grow. The default is 0.9.
        alpha : TYPE, optional
                The mixing coefficient for the estimator: estim=(1-alpha)*estim + alpha*new value
        Returns
        -------
        None.

        """
        super().__init__(center,radius,path)
        self.covered_fraction = 0
        self.state = NodeState()
        self.thr=thr
        self.alpha=alpha
        self.points=[]  # collects point the are covered by this node
        self.max_children=max_children
        self.cost=-1

    def find_closest_child(self,x):
        """
        Find the child of this node whose center is closest to x

        Parameters
        ----------
        x : point

        Returns
        -------
        closest_child: ElaboratedTreeNode
        the closest child
        
        _min_d: float
        distance from the closest child.
        """
        if self.no_of_children() ==0:
            return None,None
        _min_d = _MAX
        closest_child=None
        for child in self.children:
            _d = child.dist_to_x(x)
            if _d < _min_d:
                closest_child=child
                _min_d=_d
        assert(not closest_child is None), "find_closest_child failed, x=%f"%x+'node=\n'+str(self)
        return closest_child,_min_d

    def find_path(self,x):
        """ Find path in tree that corresponds to the point x

        :param x: the input point

        :returns: path from root to leaf
        :rtype: a list of nodes
        """

        if len(self.children)==0:
            return [self]
        else:
            closest_child,distance = self.find_closest_child(x)
            if closest_child is None:
                return [self]
            else:
                child_path = closest_child.find_path(x)
                return [self]+child_path
    
    def conditionally_add_child(self,x):
        """ Decide whether to add new child 
        :param x: 
        :returns: covered or init or filter-add or filter-discard
        :rtype: 

        """
        if self.no_of_children()==0:
            self.add_child(self.center)  # add parent center as child
            return 'init'
        _child,d = self.find_closest_child(x)
        assert(d != None)

        r=_child.radius
        if d <= r: 
            return 'covered'
        else:                   # if d>r far from center use modified kmeans++ rule
            P=min(1.0,((d-r)/r)**2)
            #print('d=%4.2f, r=%4.2f'%(d,r),end=' ')
            #print('adding point with P=%f'%P,end=', ')
            if random.random()<P:
                #print(self.path,' Success') 
                self.add_child(x)
                return 'filter-add'
            else:
                #print(self.path,' Fail')
                return 'filter-discard'
            

    def add_child(self,x):
        """ Add child to node

        :param x: 

        """
        new=ElaboratedTreeNode(x,self.radius/2,self.path+(self.no_of_children(),))
        self.children.append(new)
        
    def insert(self,x):
        """ insert an example into this node.
        :param x: 
        :returns: Flag indicating whether example was rejected.
        :rtype: Flag

        """
        if not self.covers(x):
            return False
        self.points.append(x)
        state = self.state.get_state()
        if state=='initial': # initial state, do nothing
            pass
        if state=='seed':                    
            self.points.append(x)
            add_status = self.conditionally_add_child(x);
            if add_status in ['init','covered']: 
                self.covered_fraction = (1-self.alpha)*self.covered_fraction + self.alpha
            else:
                self.covered_fraction = (1-self.alpha)*self.covered_fraction
            if self.covered_fraction>self.thr:
                print('node'+str(self.path)+
                      'finished seeding frac=%7.5f, no of points = %d, no of children=%2d'\
                      %(self.covered_fraction,len(self.points),self.no_of_children()),end=' ')
                self.state.set_state('refine')
                self.refine()
                print('cost = %7.5f'%self.cost)
                self.state.set_state('passThrough')
        if state=='passThrough':
            _child,d = self.find_closest_child(x)
            _child.insert(x)

    def refine(self):
        Centers=[_child.center for _child in self.children]
        self.cost,newCenters=Kmeans(self.points,Centers)
        push_through = self.no_of_children() <= self.max_children
        for i in range(len(self.children)):
            child=self.children[i]
            child.center = newCenters[i]
            if push_through: 
                child.state.set_state('seed')
        
    def __str__(self):
        return str(self.path)+': r=%4.2f, state= %s, no_child=%d, count=%d, cov_frac=%4.3f, cost=%4.3f'\
                %(self.radius,self.state.get_state(),len(self.children),len(self.points),self.covered_fraction,self.cost)

def gen_scatter(T,data,level=0):
    C=[]
    for i in range(data.shape[0]):
        point=np.array(data.iloc[i,:])
        #print(T.find_path(point))
        C.append(T.find_path(point)[-1].path[level])

    py.figure(figsize=[15,5])
    py.scatter(data[0],data[1],s=1,c=C,alpha=0.2)
    t='Level=%d, colors=%d'%(level,max(C)+1)
    py.title(t);

if __name__=='__main__':
    data = pd.read_csv('../../Data/twoBlob.csv',header=None)
    print(data.shape)
    np.random.shuffle(data.values)
    #data=data.iloc[:1000,:]
    center=np.mean(data,axis=0)
    T=ElaboratedTreeNode(center,radius=4,path=(),alpha=0.01,thr=0.99)
    print(T._str_to_level(1))
    
    T.state.set_state('seed')
    #%%
    for i in range(1,data.shape[0]):
#    for i in range(1,):
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
    print()
    
    