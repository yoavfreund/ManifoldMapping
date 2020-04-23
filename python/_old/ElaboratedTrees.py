class ElaboratedTree(CoverTree):
    def __init__(self,center,radius,path,mode='initial'):
        super().__init__(center,radius,path)
        self.covered_fraction = 0
        self.punch_through = False
        self.mode = mode         # 'initial' = node that has been started, but is not yet collecting
                                 # 'Collect' = collecting centers. Ends when covered fraction > thr
                                 # 'Tune' = Refine centers and collect statistics
                                 # 'PassThrough' = root node fixed: children can start collecting.


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
            if _d > child.radius:
                continue
            if _d < _min_d:
                closest_child=child
                _min_d=_d
        return closest_child,_min_d

    def find_path(self,x):
        """ Find path in tree that corresponds to the point x

        :param x: the input point

        :returns: path from root to leaf
        :rtype: a list of nodes
        """

        d= _dist(x,self.center)
        #print(str(self.path),d)
        if d>self.radius:
             return None
        if len(self.children)==0:
            return [self]
        else:
            closest_child,distance = self.find_losest(x)
            child_path = closest_child.find_path(x)
            if child_path is None:
                continue
            else:
                return [self]+child_path
            return [self]


    def conditionally_add_child(self,x):
        _child,d = find_closest_child(x)
        r=self.radius/2 
        if d<r: #Add only points that are at least radius/2 from closest center.
            return False
        else:                   # if >r/2 far from center use modified kmeans++ rule
            P=((d-r)/r)**2
            if random.random()<P:
                print('adding point with P=%f'%P)
                add_child(self,x)
                return True
            return False
            

    def add_child(self,x):
        """ Add child to node

        :param x: 

        """
        new=MuffledTree(x,self.radius/2,self.path+(self.no_of_children(),))
        self.children.append(new)

    def check_an_add(self,x):
        """ check altered kmeans++ condition and conditionally add point x
            Assumes mode of parent is such that addition is allowed
        :param x: 
        :returns: 
        :rtype: 

        """
        
    def insert(self,x):
        """ insert a new training example

        :param x: 
        :returns: flag indicating whether a non-trivial path was found. 
        :rtype: Flag

        """
        path=self.find_path(x)
        if path is None:
            return False

        #found a non-trivial path
        leaf=path[-1]
        is_root=len(path)==1
        if is_root:  #root node, if does not cover x, always adds x as a child
            leaf.conditionally_add_child(x)
        else:  #not root
            parent = path[-2]
            if parent.mode == 'Collect':
                parent.update_statistics(x)


            
            if parent.punch_through: #add new node
                leaf.add_child(x)
                if not parent.punch_through:
                    parent.covered_fraction = (1-alpha)*parent.covered_fraction
            else:     #don't add new node, instead, update parent statistics
                parent.covered_fraction = (1-alpha)*parent.covered_fraction + alpha
                if not parent.punch_through and parent.covered_fraction>thr:
                    print('node'+str(parent.path)+\
                          'punched through frac=%7.5f, count= %d, siblings=%2d'%(parent.covered_fraction,parent.counter,parent.no_of_children()))
                    parent.punch_through=True  # this is a latch, once the leaf is punched through it remains so forever
            
        for node in path:
            node.counter +=1
        return True

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
