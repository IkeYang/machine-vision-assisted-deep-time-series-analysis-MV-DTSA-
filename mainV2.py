from trainMain import train
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import root_scalar
import numpy as np


def msfunction(x):
    return 1/h*(norm.cdf(x)-norm.cdf(-x))-2+2*norm.cdf(x)+x/h*np.sqrt(2/np.pi)*np.exp(-np.square(x)/2)

lookBackWindow=96
T=96
bs=16
data_path='ETTm2.csv'
h=200
MS =np.round(root_scalar(msfunction, bracket=[0, 10]).root,2)

#DeepLabMobIndependent VI-TSF-DeepLabV2-CI
#DeepLabMob  VI-TSF-DeepLabV2
#Unet VI-TSF-Unet
modelName='DeepLabMobIndependent'


epochs=50

train(h,lookBackWindow,T,bs,data_path,MS,modelName,epochs)









