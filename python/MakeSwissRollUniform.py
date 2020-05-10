
""" Compute the function that is the inverse of the integral of sqrt(1+(b*theta)**2) 
"""

from numpy import arcsinh, sqrt
import numpy as np

class SwissToUniform:
    def __init__(self,_from=0,_to=5,resolution=0.01,b=1):
        self.T=np.arange(_from-2*resolution,_to+2*resolution,resolution) # range of theta
        self.b=b
        self.X=np.array([self.ThetatoX(yy) for yy in list(self.T)]) # table of X as a function of theta

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
