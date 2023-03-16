
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from loadData import Dataset_CustomVR,Dataset_ETTminVR,Dataset_ETThourVR
from model.model import modelDict
from utlize import  mkdir



def test(args,model=None,epoch=None):


    if 'ETTh' in args.data_path:
        valDatas = Dataset_ETThourVR(args)

    elif 'ETTm' in args.data_path:
        valDatas = Dataset_ETTminVR(args)

    else:
        valDatas = Dataset_CustomVR(args)


    print(epoch)



    device = next(model.parameters()).get_device()


    model.eval()  # switch to evaluation mode

    shuffle=False

    dataloaderV = torch.utils.data.DataLoader(valDatas, batch_size=args.bs,  # test data
                                              shuffle=shuffle, num_workers=int(args.num_workers))
    lossM=0
    lossE=0
    lossMAE=0
    lossMAE_E=0

    with torch.no_grad():
        c = 0

        for i, (x, y, d, xseq, yseq,mu,std) in enumerate(dataloaderV):

            c += 1
            x = x.to(device)
            y = y.to(device)

            yseq = yseq.detach().cpu().numpy()
            mu = mu.detach().cpu().numpy()
            std = std.detach().cpu().numpy()

            om = model(x)
            ypredMax = valDatas.Pixel2data(om, method='max')
            ypredExp = valDatas.Pixel2data(om, method='expection')


            ye = (ypredExp[:, args.size[0]:, :]*std+mu).reshape(-1, ypredExp.shape[-1]).reshape(
                ypredExp[:, args.size[0]:, :].shape)
            yp =(ypredMax[:, args.size[0]:, :]*std+mu).reshape(-1, ypredMax.shape[-1]).reshape(
                ypredMax[:, args.size[0]:, :].shape)
            yt =(yseq[:, args.size[0]:, :]).reshape(-1, yseq.shape[-1]).reshape(yseq[:, args.size[0]:, :].shape)

            lossM += np.mean(((yp - yt)) ** 2)
            lossMAE += np.mean(np.abs(yp - yt))

            lossE += np.mean((ye - yt) ** 2)
            lossMAE_E += np.mean(np.abs(ye - yt))



    print('MAX','MSE=',lossM/c,'RMSE=',np.sqrt(lossM/c),c)
    print('MAX','MAE=',lossMAE/c,'MSE_expection=',lossMAE_E/c)
    print('Expection','MSE=',lossE/c,'RMSE=',np.sqrt(lossE/c),c)
    print('Expection','MAE=',lossMAE_E/c,'RMSE=',np.sqrt(lossMAE_E/c),c)
    return lossM/c,lossMAE/c,lossE/c,lossMAE_E/c
















