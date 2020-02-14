
class CoverTree(object):
    def __init__(self,center,radius,path):
        self.center=center
        self.radius=radius
        self.counter=1  # number of points covered by this tree
        self.too_far=0  # count how many points were too far (should only be non-zero at root
        self.path=path
        self.children = []
    
    def dist_to_x(self,x):
        d=_dist(x,self.center)
        if d<self.radius:
            return d
        else:
            return _MAX
        
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
                    continue
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
