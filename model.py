import numpy as np
import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self,nsample):
        super(ConvModel,self).__init__()
        self.main=nn.Sequential(
            nn.Conv1d(nsample,)
        )
    def forward(self,x):
        # x: B,nsample,nx
        pass

class LinearSelect(nn.Module):
    def __init__(self,nsample,nx):
        super(LinearSelect,self).__init__()
        w=torch.randn([nx,nsample],dtype=torch.float32)
        self.weights=nn.Parameter(w,requires_grad=True)
        self.softmax=nn.Softmax(-1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,p=.05,temperature=1.0):
        # x: nx, nsample
        nx,nsample=x.size()
        # w=self.softmax(self.weights/temperature)
        w=self.sigmoid(self.weights)
        w=w/torch.sum(w,-1,True)
        y=(w*x).mean(-1)
        k=y[1:]-y[:-1]
        '''
        cnt=int(p*nx*nsample)
        id=torch.randint(0,nx,size=(cnt,2))
        i,j=id[:,0],id[:,1]
        overlap=(i==j)
        i[overlap]=0
        j[overlap]=nx-1
        k=(y[i]-y[j])/(i-j)'''
        # print(torch.max(w,-1)[0].size())
        return k,w

class LinearDirect(nn.Module):
    def __init__(self,nsample,nx):
        super(LinearDirect,self).__init__()
        mid_positions=torch.ones([2],dtype=torch.float32)*.5
        self.m=nn.Parameter(mid_positions,requires_grad=True)
        self.coeff=np.sqrt(2*3.1416)
        self.x=torch.linspace(0,1,nx,dtype=torch.float32).unsqueeze(-1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,std=1.0,p=.1):
        nx,nsample=x.size()
        exp_coeff=-2*std*std
        coeff=self.coeff*std
        m=self.sigmoid(self.m)
        m=self.x*(m[1]-m[0])+m[0]
        w=torch.exp(((x-m)**2)/exp_coeff)/coeff
        # print(self.m)
        # print(w)
        w=w/torch.sum(w,-1,True)
        y=(w*x).sum(-1)
        
        cnt=int(p*nx*nsample)
        id=torch.randint(0,nx,size=(cnt,2))
        i,j=id[:,0],id[:,1]
        overlap=(i==j)
        i[overlap]=0
        j[overlap]=nx-1
        k=(y[i]-y[j])/(i-j)

        return k,w