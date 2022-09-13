from nntrainer.trainer import get_optimizer_sheduler
from data import create_data
from model import LinearDirect
import torch
from nntrainer.config_utils import get_device
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def plot_scores(y,w):
    plt.clf()
    nx,nsample=y.size()
    for i in range(nx):
        c=[(torch.clip(sc,0,1),0,0) for sc in w[i]]
        plt.scatter([i for _ in range(nsample)],y[i],s=3,c=c)

class Stepper:
    def __init__(self,milestones,v):
        self.m=np.array(milestones)
        self.v=np.array(v)
    def step(self,s):
        idx=0
        for m in self.m:
            if s<m:
                return self.v[idx]
            idx+=1
        return self.v[-1]
    def __call__(self, *args,**kwargs):
        return self.step(*args,**kwargs)

if __name__=='__main__':
    vis_dir='vis/base'
    os.makedirs(vis_dir,exist_ok=True)
    nx,ny,nsample=200,15,10
    y,score,a,b=create_data(nx,ny,nsample)
    device=get_device()
    y=torch.tensor(y,dtype=torch.float32,device=device)
    model=LinearDirect(nsample,nx).to(device)
    opt,sch=get_optimizer_sheduler(
        model.parameters(),
        'adam',
        'multi_step',
        {'lr':.9},
        milestones=[500,10000,30000],
        gamma=.1
    )
    topweight_stepper=Stepper(
        [5000,10000,30000,40000],
        [.00001,.0005,.001,.005,0.01]
    )
    tmp=5.0
    for epoch in range(50000):
        opt.zero_grad()
        k,w=model(y,p=.9,std=tmp)
        l=k.std()
        if torch.isnan(l).any(): break
        loss=l+topweight_stepper(epoch)*(1-torch.max(w,-1)[0]).mean()
        loss.backward()
        opt.step()
        if epoch%200==0 and epoch <20000:
            tmp*=.95
            # print(f'New temperature={tmp}')
        if epoch%1000==0:
            print(epoch,loss.detach().cpu().item())
            plot_scores(y.detach().cpu(),w.detach().cpu())
            plt.savefig(f'{vis_dir}/w{epoch//1000}.png')
        if sch: sch.step()
    print(f'a={a}, b={b}, k={b-a}')
    print(f'ends={model.sigmoid(model.m)}, std={tmp}')
    plot_scores(y.detach().cpu(),w.detach().cpu())
    plt.savefig(f'{vis_dir}/w.png')