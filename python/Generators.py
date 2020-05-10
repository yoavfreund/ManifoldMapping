import numpy.random as r
from numpy.linalg import norm
import numpy as np
import json as js

""" Compute the function that is the inverse of the integral of sqrt(1+(b*theta)**2) 
"""
from numpy import arcsinh, sqrt

class SwissToUniform:
    def __init__(self,min_theta=0,max_theta=5,resolution=0.01,b=1):
        self.b=b
        self.T=np.arange(min_theta-2*resolution,max_theta+2*resolution,resolution) # range of theta
        self.X=np.array([self.ThetatoX(yy) for yy in list(self.T)]) # table of X as a function of theta
        
    def getXRange(self):
        return self.X[0],self.X[-1]

    def ThetatoX(self,theta):
        b=self.b
        return (theta*sqrt((b*theta)**2 + 1) + arcsinh(b*theta)/b)/2.

    def XtoTheta(self,x):
        X=self.X
        l=X.shape[0]
        i=int(l/2.)
        step=l/4.
        while True:
            if X[i]>x:
                i=int(i-step)
            else:
                i=int(i+step)
            step/=2.
            if step<1.:
                break
        return self.T[i]

class DataGenerator:    
    
    def __init__(self,min_theta=1,max_theta=4,width=1.,b=1.,noise=0,dim=2):
        self.min_theta=min_theta
        self.max_theta = max_theta
        self.width=width
        self.b=b
        self.noise=noise
        self.dim=dim
               
    def distances(self,X,Z):
        """
        Compute pairwise distances between rows of X and rows of Z

        Parameters
        ----------
        X : numpy array shape= [:,dim]
        Z : numpy array shape= [:,dim]

        Returns
        -------
        Matrix of pairwise distances

        """
        assert Z.shape[1]==self.dim
        assert X.shape[1]==self.dim
 
        def _diff(x,y):
            return (x-y)**2

        # Ufunc are a way to vectorize python functions so that they execute efficiently on numpy arrays.
        ufunc_diff = np.frompyfunc(_diff,2,1)
        ufunc_sqrt=np.frompyfunc(lambda a:np.sqrt(a),1,1)
        
        
        D=None
        for d in range(self.dim):
            G=ufunc_diff.outer(X[:,d],Z[:,d])
            if D is None:
                D=G
            else:
                D+=G
    
        return ufunc_sqrt(D)

      
    def Sample(self,n=100):
        """Generate a uniform sample from the 2d square"""
        X=r.uniform(size=[n,self.dim])
        return X
        
    def epsilonNet(self,generator=None,n=100,epsilon=5,scale=1):
        """Generate an epsilon net that covers all but about 1/n of the square"""

        X=self.Sample(1)        
        while True:
            Z=self.Sample(n=n)
            #print('X=',X)
            #print('Z=',Z)
            _Dist=self.distances(X,Z) # compute matrix where entry i,j is the distance between center i and candidate j
            #print('Dist=',_Dist)
            min_dist=np.min(_Dist,axis=0)  # compute distance of each candidate to nearest center in X
            #print('min_dist=',min_dist)
            d_max=np.max(min_dist)
            #print('\r centers=%d, d_max= %5.3f, epsilon=%5.3f'%(X.shape[0],d_max,epsilon),end='')
            if d_max<epsilon:
                break
            else:
                i_max=np.argmax(min_dist)
                X=np.append(X,Z[i_max:i_max+1,:],axis=0) # accumulate furthest center
        return X
    
class SwissGenerator(DataGenerator):
    
    def __init__(self,min_theta=1,max_theta=4,width=1.,b=1.,resolution=0.01, **kwargs):
        self.min_theta=min_theta
        self.max_theta = max_theta
        self.width=width
        self.b=b
        super().__init__(**kwargs)
        self.dim=3
        self.toUniform=SwissToUniform(min_theta=min_theta,max_theta=max_theta,resolution=resolution,b=b)
        
    def Sample(self,n=100):
        x_min,x_max = self.toUniform.getXRange()
        X=r.uniform(size=[n,2])
       
        
        Out=np.zeros([n,3])
        T=np.zeros(n)
        Out[:,2] = X[:,1] * self.width
        
        X[:,0] = (X[:,0]*(x_max-x_min))+x_min
        for i in range(n):
            theta=self.toUniform.XtoTheta(X[i,0])
            T[i]=theta
            #print(theta)
            Out[i,0]=np.sin(theta)*(theta*self.b)
            Out[i,1]=np.cos(theta)*(theta*self.b)
        return Out,T
            
        
        
    
    
#####################    
    def Compute_edges(X,max_dist=0.2):
        """Compute the edges whose length is smaller than max_dist"""
    
        _Dist=self.distances(X,X)
        _Small = _Dist<max_dist
    
        pairs=np.nonzero(_Small)
        new_pairs=[[],[]]
        segments=[]
        for i in range(len(pairs[0])):
            c0=pairs[0][i]
            c1=pairs[1][i]
            if c0>c1:
                new_pairs[0].append(c0)
                new_pairs[1].append(c1)
                segment=[tuple(X[c0,:]),tuple(X[c1,:])]
                segments.append(segment)
        new_pairs=[np.array(new_pairs[0]),np.array(new_pairs[1])]
        print('pairs:%d,new_pairs:%d'%(pairs[0].shape[0],new_pairs[0].shape[0]))
        return segments,new_pairs
    
    def plot_graph(X,segments):
        lc = mc.LineCollection(segments, colors='r', linewidths=0.1)
        _min=np.min(X[:,0])
        _max=np.max(X[:,0])
        fig, ax = pl.subplots(figsize=[12*(_max-_min),12])
        ax.plot(X[:,0],X[:,1],'.');
        ax.add_collection(lc)
        #ax.set_xlim([0,1])
        
    def graph2json(X,pairs):
        vertices=[{'id':'%d'%i} for i in range(X.shape[0])]
        edges=[]
        for i in range(pairs[0].shape[0]):
            i1=pairs[0][i]
            i2=pairs[1][i]
            if i1<=i2:
                continue
            p1=X[i1,:]
            p2=X[i2,:]
            dist = norm(p1-p2)
            edges.append({"source":'%d'%i1, "target":'%d'%i2, "weight":1.0, "dist":'%5.2f'%dist})
    
        Dict={'directed':False,
              'nodes':vertices,
              "edges":edges}
        return Dict

generator=SwissGenerator(min_theta=0,max_theta=10)
i=0

X,theta=generator.Sample(n=100)
#scatter(X[:,0],X[:,1])

#X=generator.epsilonNet(n=4,epsilon=0.03)
print(X.shape)

# #_type="uniform"
# _type='epsilonCover'
# #_type='None'
# n=1000
# m=100
# epsilon=0.02
# max_dist=0.06
# scale=1/3.

# if _type=="uniform":
#     X=unifSample(n=n,scale=scale)
#     filename='Sample_n=%d_max_dist=%5.2f'%(n,max_dist)
# elif _type=='epsilonCover':
#     X=epsilonNet(n=m,epsilon=epsilon,scale=scale)
#     filename='Net_m=%d_epsilon=%5.2f_max_dist=%5.2f'%(m,epsilon,max_dist)
# else:
#     print('unrecognized type:',_type)
    
# filename = 'graph_jsons/'+filename+'.json'
# segments,pairs= Compute_edges(X,max_dist)

# plot_graph(X,segments)
# #pairs[0][:10],pairs[1][:10]

# #D=graph2json(X,pairs)

# #with open(filename, 'w') as json_file:
# #    js.dump(D,json_file)
# #    print('\n done with '+filename)
