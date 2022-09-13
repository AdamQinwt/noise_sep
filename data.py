import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

def create_data(nx,ny,nsample):
    a,b=npr.rand(),npr.rand()
    y=npr.rand(nx,ny)
    y[:,0]=np.linspace(a,b,nx)
    for i in range(nx):
        npr.shuffle(y[i])
    y=y[:,:nsample]
    A=a-b
    B=1
    C=-a
    D=np.sqrt(A*A+B*B)
    A/=D
    B/=D
    C/=D
    upper=1# min(1-a,1-b,a,b)
    a_coeff=1/(upper**2)
    score=np.zeros_like(y)
    for i in range(nx):
        AxC=A*i/nx+C
        distance=np.abs(AxC+B*y[i])
        valid_dist=distance[distance<upper]
        score[i,distance<upper]=a_coeff*((valid_dist-upper)**2)
        score[i]=score[i]*nsample/1.6/score[i].sum()
    return y,score,a,b

if __name__=='__main__':
    nx,ny,nsample=100,20,10
    a,b=npr.rand(),npr.rand()# .2,.9
    y=npr.rand(nx,ny)
    y[:,0]=np.linspace(a,b,nx)
    print(y)
    for i in range(nx):
        npr.shuffle(y[i])
    y=y[:,:nsample]
    A=a-b
    B=1
    C=-a
    D=np.sqrt(A*A+B*B)
    A/=D
    B/=D
    C/=D
    upper=1# min(1-a,1-b,a,b)
    a_coeff=1/(upper**2)
    score=np.zeros_like(y)
    for i in range(nx):
        AxC=A*i/nx+C
        distance=np.abs(AxC+B*y[i])
        valid_dist=distance[distance<upper]
        score[i,distance<upper]=a_coeff*((valid_dist-upper)**2)
        score[i]=score[i]*nsample/1.6/score[i].sum()
        c=[(np.clip(sc,0,1),0,0) for sc in score[i]]
        plt.scatter([i for _ in range(nsample)],y[i],s=3,c=c)
    plt.show()