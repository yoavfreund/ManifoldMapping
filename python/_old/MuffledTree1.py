class MuffledTree(CoverTree):
    def __init__(self,center,radius,path):
        super().__init__(center,radius,path)
        self.covered_fraction = 0
        self.punch_through = False
    
    def insert(self,x):
        path=self.find_path(x)
        if path is None:
            return False

        #found a non-trivial path
        leaf=path[-1]
        is_root=len(path)==1
        if is_root:
            new=MuffledTree(x,leaf.radius/2,leaf.path+(leaf.no_of_children(),))
            leaf.children.append(new)
        else:  #not root
            parent = path[-2]
            if parent.punch_through: #add new node
                new=MuffledTree(x,leaf.radius/2,leaf.path+(leaf.no_of_children(),))
                leaf.children.append(new)
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
