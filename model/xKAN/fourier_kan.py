import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *

# code modified from https://github.com/GistNoesis/FourierKAN/

class FourierKANLayer(nn.Module):
    def __init__( self, inputdim, outdim, gridsize,addbias=True):
        super(FourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2,outdim,inputdim,gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize) ) )
        if self.addbias:
            self.bias  = torch.nn.Parameter(torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1))
        #This should be fused to avoid materializing memory
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y

class Fourier_KAN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size: int = 8,
        spline_order: int = 0, #  placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLayer(
                inputdim=in_dim, 
                outdim=out_dim,
                gridsize=grid_size,
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
