import os
import numpy as np
import pandas as pd
import os
import torch
import sys
sys.path.append("..")
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utlize import time_features,moving_average
import matplotlib.pyplot as plt
import warnings
import time
import datetime
warnings.filterwarnings('ignore')
import tsaug
from einops import rearrange
import emd

def concentration(dataX):
    data=np.copy(dataX.flatten())
    data=np.abs(StandardScaler().fit_transform(data.reshape(-1, 1) ))
    concentration3=np.sum(data<3)/data.shape[0]
    concentration4=np.sum(data<4)/data.shape[0]
    concentration5=np.sum(data<5)/data.shape[0]
    concentration6=np.sum(data<6)/data.shape[0]
    concentration7=np.sum(data<7)/data.shape[0]
    concentration10=np.sum(data<10)/data.shape[0]
    print('concentration 3 4 5 6 7 10',
          concentration3,concentration4,concentration5,concentration6,concentration7,concentration10 )
class Dataset_VR(Dataset):
    def __init__(self, args):
        pass
    def featureNumberDict(self,dataPath):
        return {
            'exchange_rate.csv':8,
            'ETTm2.csv':7,
            'ETTm1.csv':7,
            'ETTh1.csv':7,
            'ETTh2.csv':7,
            'electricity.csv':321,
            'traffic.csv':862,
            'weather.csv':21,
            'national_illness.csv':7,
        }[dataPath]
    def __prepareD__(self):
        self.taskType = 'regression'
        self.D = np.zeros([self.h, self.h])
        for i in range(self.h):
            self.D[i, :i] = np.arange(1, i + 1)[::-1]
            self.D[i, i:] = np.arange(0, self.h - i)
        self.D = self.D ** self.Norm
        try:
            self.Norm_Insequence =  self.args.Norm_Insequence
        except:
            self.Norm_Insequence=False



    def data2Pixel(self, dataXIn, dataYIN):
        '''

        :param dataX: tin,pX dataY: whole,pY
        :return: imgX, (w,h,px)*C imgY, w,h,pY
        '''
        t1 = datetime.datetime.now()
        dataX = np.copy(dataXIn.T)
        dataY = np.copy(dataYIN.T)
        dataX[dataX > self.maxScal] = self.maxScal
        dataX[dataX < -self.maxScal] = -self.maxScal

        dataY[dataY > self.maxScal] = self.maxScal
        dataY[dataY < -self.maxScal] = -self.maxScal
        px = dataX.shape[0]
        py = dataY.shape[0]
        TY = dataY.shape[1]
        TX = dataX.shape[1]



        imgY0 = np.zeros([py, TY, self.h])

        maxstd = self.maxScal
        resolution = maxstd * 2 / (self.h - 1)
        indY = np.floor((dataY + maxstd) / resolution).astype('int16')

        aY = imgY0
        aY =aY.reshape(-1, self.h)
        aY[np.arange(TY * py), indY.astype('int16').flatten()] = 1
        imgY0= aY.reshape(py, TY, self.h)

        d= self.D[list(indY), :]

        imgX0=np.copy(imgY0)

        imgX0[:,TX:,:]=0

        return imgX0, imgY0, d##c,w,h d c,w,h




    def Pixel2data(self, imgX0, method='max'):
        if len(imgX0.shape) == 3:
            imgX0 = imgX0.unsqueeze(0)
        # bs,c,w,h imgX0
        bs, ch, w, h = imgX0.shape
        # res=np.zeros([bs,w,ch])
        try:
            imgX0 = imgX0.cpu().detach().numpy()
        except:
            pass
        if method == 'max':
            indx = np.argmax(imgX0, axis=-1)
        elif method == 'expection':
            imgX0 = imgX0 / np.sum(imgX0, axis=-1, keepdims=True)
            indNumber = np.arange(0, h)
            imgX0 *= indNumber
            indx = np.sum(imgX0, axis=-1)
        maxstd = self.maxScal
        resolution = maxstd * 2 / (self.h - 1)
        res = np.transpose(indx, (0, 2, 1)) * resolution - maxstd


        return res

    def __getitem__(self, index):




        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_xO = np.copy(self.data_x[s_begin:s_end])##no use
        seq_yO = np.copy(self.data_y[s_begin:r_end])

        std=np.std(seq_xO,axis=0).reshape(1,-1)+1e-7
        mu=np.mean(seq_xO,axis=0).reshape(1,-1)
        seq_x=(seq_xO-mu)/std
        seq_y=(seq_yO-mu)/std

        if self.flag=='train':
            if np.random.rand()<self.TAP:

                seq_y+=np.random.rand(1,seq_y.shape[1])-0.5
                seq_y+=np.random.randn(seq_y.shape[0],seq_y.shape[1])*0.05*6
                if  np.random.rand()<0.5:
                    seq_y=seq_y[::-1,:]
        x,y,d=self.data2Pixel(seq_x, seq_y)#c,w,h


        if 'train' not in self.flag:

            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), \
                   torch.from_numpy(d).float(), torch.from_numpy(seq_xO).float(), \
                   torch.from_numpy(seq_yO).float(),\
                   torch.from_numpy(mu).float(), torch.from_numpy(std).float(),


        else:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(d).float()

    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)







class Dataset_CustomVR(Dataset_VR):
    def __init__(self, args):
        # size [seq_len, label_len, pred_len]
        # info
        try:
            self.anomalyFlitter = args.anomalyFlitter
        except:
            self.anomalyFlitter=False
        size=args.size
        self.args=args

        flag=args.flag
        self.flag=args.flag
        h=args.h

        data_path=args.data_path
        features=args.features

        self.maxScal=args.maxScal
        target=args.target
        scale = True
        timeenc = 1
        freq = 'h'
        self.TA=args.TA
        self.TAP=args.TAP
        self.args=args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.h = h

        self.Norm = args.dNorm

        self.data_path = data_path
        self.__read_data__()
        self.__prepareD__()

    def __read_data__(self):

        df_raw = pd.read_csv(os.path.join('dataset',
                                          self.data_path))

        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''



        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)

        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.featureNumber=1



        if self.scale:

            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)
            data = self.scaler.transform(df_data.values)




        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


class Dataset_ETTminVR(Dataset_VR):
    def __init__(self, args):
        # size [seq_len, label_len, pred_len]
        # info
        size = args.size
        self.args = args

        self.anomalyFlitter = False

        flag = args.flag
        self.flag = args.flag
        h = args.h


        data_path = args.data_path
        self.TA=args.TA
        self.TAP=args.TAP
        target = args.target
        self.maxScal = args.maxScal
        target = args.target
        scale = True
        timeenc = 1
        freq = 't'
        self.args=args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features =  args.features

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.h = h

        self.Norm = args.dNorm

        self.data_path = data_path
        self.__read_data__()
        self.__prepareD__()


    def __read_data__(self):

        df_raw = pd.read_csv(os.path.join(r'dataset',
                                          self.data_path))

        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)

        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.featureNumber=1

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)

            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


class Dataset_ETThourVR(Dataset_VR):
    def __init__(self, args):
        # size [seq_len, label_len, pred_len]
        # info
        try:
            self.anomalyFlitter = args.anomalyFlitter
        except:
            self.anomalyFlitter=False
        self.TA=args.TA
        self.TAP=args.TAP
        size = args.size
        self.args = args

        flag = args.flag
        self.flag = args.flag
        h = args.h

        data_path = args.data_path
        features = 'S'
        target = args.target
        self.maxScal = args.maxScal
        target = args.target
        scale = True
        timeenc = 1
        freq = 'h'
        self.args=args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', ]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features =  args.features

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.h = h

        self.Norm = args.dNorm

        self.data_path = data_path
        self.__read_data__()
        self.__prepareD__()



    def __read_data__(self):

        df_raw = pd.read_csv(os.path.join('dataset',
                                          self.data_path))

        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)

        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.featureNumber=1

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)

            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]



