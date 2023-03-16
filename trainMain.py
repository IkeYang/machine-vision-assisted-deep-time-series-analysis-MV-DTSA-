import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch.optim as optim
from loadData import Dataset_CustomVR,Dataset_ETTminVR,Dataset_ETThourVR
from model.model import modelDict

from test import test
import argparse
from utlize import  mkdir,EMD,GPUMemoryCheck
import torch.nn as nn
import datetime
import os
modelSizeDict={
            'exchange_rate.csv':4,
            'ETTm2.csv':4,
            'ETTm1.csv':4,
            'ETTh1.csv':4,
            'ETTh2.csv':4,
            'electricity.csv':30,
            'traffic.csv':75,
            'weather.csv':8,
            'national_illness.csv':4,
        }

def train(h,lookBackWindow,T,bs,data_path,MS,modelName,epochs,Norm_Insequence=True,modelAda=False
        ,dropout=0.1,TAP=3,TA=0.9,features='M',opt='Adam',num_workers=0,lr=2e-4,
          dNorm=1,weight_decay=0
         ):
    print(datetime.datetime.now())
    print('PID: ',os.getpid(),'***********')
    size=[lookBackWindow,0,T]
    modelSize=modelSizeDict[data_path]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lossType='EMD'
    parser = argparse.ArgumentParser(description='VI Forecasting')

    parser.add_argument('--data_path', type=str, required=False, default=data_path, help='data_path')
    parser.add_argument('--h', type=int, required=False, default=h, help='h')
    parser.add_argument('--size', required=False, default=size, help='size')
    parser.add_argument('--flag', required=False, default='train', help='flag')
    parser.add_argument('--features', required=False, default=features, help='features')
    parser.add_argument('--target', required=False, default='OT', help='target')
    parser.add_argument('--loss', required=False, default=lossType, help='loss')
    parser.add_argument('--maxScal', required=False, default=MS, help='maxScal')
    parser.add_argument('--weight_decay', required=False, default=weight_decay, help='weight_decay')
    parser.add_argument('--modelName', required=False, default=modelName, help='modelName')
    parser.add_argument('--dNorm', required=False, default=dNorm, help='dNorm')
    parser.add_argument('--bs', required=False, default=bs, help='bs')
    parser.add_argument('--maxTest', required=False, default=500, help='maxTest')
    parser.add_argument('--num_workers', required=False, default=num_workers, help='num_workers')
    parser.add_argument('--reducedChannelNumber', required=False, default=None, help='reducedChannelNumber')
    parser.add_argument('--TA', required=False, default=TA, help='TA')
    parser.add_argument('--TAP', required=False, default=TAP, help='TAP')
    parser.add_argument('--dropout', required=False, default=dropout, help='dropout')
    parser.add_argument('--Norm_Insequence', required=False, default=Norm_Insequence, help='Norm_Insequence')
    parser.add_argument('--modelSize', required=False, default=modelSize, help='modelSize')
    parser.add_argument('--modelAda', required=False, default=modelAda, help='modelAda')

    args = parser.parse_args()
    print(args)

    if 'ETTh' in data_path:
        trainDatas = Dataset_ETThourVR(args)
    elif 'ETTm' in data_path:
        trainDatas = Dataset_ETTminVR(args)
    else:
        trainDatas = Dataset_CustomVR(args)

    args.reducedChannelNumber=trainDatas.featureNumber
    batch_size = bs
    lr = lr
    modeldict=modelDict()
    model = modeldict[modelName](inchannel=trainDatas.featureNumber,T=trainDatas.seq_len+trainDatas.pred_len,args=args,DCNumber=None,
    out_channels=trainDatas.featureNumber,loss=args.loss)

    model.to(device)
    if opt=='Adam':
        optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    if opt=='SGD':
        optimizer = optim.SGD(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    dataloaderT = torch.utils.data.DataLoader(trainDatas, batch_size=batch_size,pin_memory=False,
                                              shuffle=True, num_workers=int(num_workers))

    onlyOut=True
    lossF=EMD(MSE=False,onlyOut=onlyOut,arg=args)
    for epoch in range(epochs):
        model.train()

        for i, (x, y, d) in enumerate(dataloaderT):

            if x.shape[0]==1:
                continue
            x = x.to(device)#bs,c,w,h
            y = y.to(device)
            d = d.to(device)
     
            
            optimizer.zero_grad()

            ypred = model(x)

            loss = lossF(ypred, d,y)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)


            optimizer.step()

            if (i) % int(len(dataloaderT) / 4) == 0:

                print('[%d/%d][%d/%d]\tLoss: %.8f\t  '
                      % (epoch, epochs, i, len(dataloaderT), loss))

        model.eval()
        args.flag = 'val'
        test(args, model=model, epoch=epoch)
        args.flag = 'test'
        test(args, model=model, epoch=epoch)












