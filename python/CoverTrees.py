class CoverTree(object):
    def __init__(self,center,radius,path):
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
                    [self]
                else:
                    return [self]+child_path
            return [self]

    def insert(self,x):
        path=self.find_path(x)
        if path is None:
            return False

        #found a non-trivial path
        leaf=path[-1]
        new=CoverTree(x,leaf.radius/2,leaf.path+(leaf.no_of_children(),))
        leaf.children.append(new)
        for node in path:
            node.counter +=1
        return True
            
        
    def collect_centers(self):
        '''Collect all of the centers defined by a tree
        returns a list where each element is a center, followed by the level of the center'''
        C=[(self.center,len(self.path))]
        if len(self.children)>0:
            for child in self.children:
                C = C+ child.collect_centers()
        return C
            
    def collect_nodes(self):
        '''returns a list of all nodes in the tree'''
        N=[self]
        if len(self.children)>0:
            for child in self.children:
                N=N+child.collect_nodes()
        return N

    def __str__(self):
        return str(self.path)+': r=%4.2f, no_child=%d, count=%d'%(self.radius,len(self.children),self.counter)

    def _str_to_level(self,max_level):
        s=self.__str__()+'\n'
        if self.get_level() < max_level and len(self.children)>0:
            for i in range(len(self.children)):
                s+=self.children[i]._str_to_level(max_level)
        return s    

class NodeState:
    states = ('initial',\ # initial state, do nothing
              'seed',\    # collect centers
              'refine',\  # refine centers
              'passThrough') # only collect statistics, advance children from 'initial' to 'seed'
    def __init__(self):
        self.state = 'initial'
    def get_state(self):
        return self.state
    def set_state(self,state):
        assert(str(state) in states)
        self.state=state
    
    
    
class ElaboratedTree(CoverTree):
    def __init__(self,center,radius,path,mode='initial'):
        super().__init__(center,radius,path)
        self.covered_fraction = 0
        self.state = NodeState()

    def find_closest_child(self,x):
        """ find the child of this node that is closest to x
        :param x: 
        :returns: 
        :rtype: 
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
        assert(not closest_child is None)
        return closest_child,_min_d

    def find_path(self,x):
        """ Find path in tree that corresponds to the point x

        :param x: the input point

        :returns: path from root to leaf
        :rtype: a list of nodes
        """

        d= self.dist_to_x(x)
        #print(str(self.path),d)
        if d>self.radius:
             return None
        if len(self.children)==0:
            return [self]
        else:
            closest_child,distance = self.find_closest_child(x)
            if closest_child is None or not closest_child.covers(x):
                return [self]
            else:
                child_path = closest_child.find_path(x)
                print(child_path,closest_child)
                return [self]+child_path
    
    def conditionally_add_child(self,x):
        """ check altered kmeans++ condition and conditionally add point x
            Assumes mode of parent is such that addition is allowed
        :param x: 
        :returns: 0 if too close, 1 if far but did not add, 2 if add
        :rtype: 

        """

        _child,d = self.find_closest_child(x)
        if self.no_of_children()==0:
            self.add_child(self.center)  # add parent point as child
            return 2
        assert(d != None)
        r=self.radius/2
        print('conditionally_add_child d=%6.2f,r=%6.2f'%(d,r))
        d=min(d,2*r)
        if d is None or d<r: #Add only points that are at least radius/2 from closest center.
            return 0
        else:                   # if d>r far from center use modified kmeans++ rule
            P=min(1.0,((d-r)/r)**2)
            print('d=%4.2f, r=%4.2f'%(d,r),end=' ')
            print('adding point with P=%f'%P,end=', ')
            if random.random()<P:
                print(self.path,' Success') 
                self.add_child(x)
                return 2
            return 1
            

    def add_child(self,x):
        """ Add child to node

        :param x: 

        """
        new=ElaboratedTree(x,self.radius/2,self.path+(self.no_of_children(),))
        self.children.append(new)
        
    def insert(self,x):
        """ insert an example into this node.

        :param x: 
        :returns: Flag indicating whether example was rejected.
        :rtype: Flag

        """
        if not self.covers(x):
            return False
        state = self.state.get_state()
        if state=='initial': # initial state, do nothing
            pass
        if state=='seed':
            add_status = self.conditionally_add_child(x);
            if add_status==2:
                self.covered_fraction = (1-alpha)*self.covered_fraction + alpha
            elif add_status==0 or add_status==1:
                parent.covered_fraction = (1-alpha)*self.covered_fraction
            if self.covered>thr:
                print('node'+str(self.path)+\
                      'finished seeding frac=%7.5f, count= %d, siblings=%2d'\
                      %(self.covered_fraction,self.counter,self.no_of_children()))
                self.set_status('refine')
        if state=='refine':
            pass
        if state=='passThrough':
            pass

    def __str__(self):
        return str(self.path)+': r=%4.2f, no_child=%d, count=%d, cov_frac=%4.3f, punch_through=%1d'\
                %(self.radius,len(self.children),self.counter,self.covered_fraction,int(self.punch_through))

def gen_scatter(T,data,level=0):
    C=[]
    for i in range(data.shape[0]):
        point=np.array(data.iloc[i,:])
        C.append(T.find_path(point)[-1].path[level])

    figure(figsize=[15,5])
    scatter(data[0],data[1],s=1,c=C,alpha=0.2)
    t='Level=%d, colors=%d'%(level,max(C)+1)
    title(t);
