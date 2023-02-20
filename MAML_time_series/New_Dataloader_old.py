import numpy as np
import pandas as pd
import os
import random
import utils.tools

from utils.tools import StandardScaler
from utils.tools import list_of_groups
from utils.tools import slip_window_data
from utils.timefeatures import time_features

root_path = './data/ETT/'
data_path = 'ETTh1.csv'
flag='train'
# size=None
# size = [400,100,100]
# size = [8,2,2]
features='M'
target='OT'
scale=True
inverse=True
timeenc=0
freq='h'
index=0

def new_dataloader(root_path ='./data/ETT/',flag='train',size = [8,2,2],features='M',
                  target='OT',scale=True,inverse=True,timeenc=0,freq='h',index=0):
    # init
    if size == None:
        seq_len = 24*4*4
        label_len = 24*4
        pred_len = 24*4
    else:
        seq_len = size[0]
        label_len = size[1]
        pred_len = size[2]

    ETT_hour_x,ETT_hour_y,ETT_hour_x_mark,ETT_hour_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='h')
    ETT_minute_x,ETT_minute_y,ETT_minute_x_mark,ETT_minute_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTm1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='t')
    ETT_hour_test_x,ETT_hour_test_y,ETT_hour_test_x_mark,ETT_hour_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', 
        flag='train', size=None,features='M', data_path='ETTh2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='h')
    ETT_minute_test_x,ETT_minute_test_y,ETT_minute_test_x_mark,ETT_minute_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/',                   flag='train', size=None,features='M', data_path='ETTm2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='t')
    
    meta_train = {
        "ETTh1": {
            "x_spt": ETT_hour_x[0:500],
            "y_spt": ETT_hour_y[0:500], 
            "x_qry": ETT_hour_x[500:2500],
            "y_qry": ETT_hour_y[500:2500]
            }, 
        "ETTm1": {
            "x_spt": ETT_minute_x[0:500], 
            "y_spt": ETT_minute_y[0:500], 
            "x_qry": ETT_minute_x[500:2500], 
            "y_qry": ETT_minute_y[500:2500]
            }
    }

    meta_test = {
        "ETTh2": {
            "x_spt": ETT_hour_test_x[0:500],
            "y_spt": ETT_hour_test_y[0:500], 
            "x_qry": ETT_hour_test_x[500:2500],
            "y_qry": ETT_hour_test_y[500:2500]
            }, 
        "ETTm2": {
            "x_spt": ETT_minute_test_x[0:500], 
            "y_spt": ETT_minute_test_y[0:500], 
            "x_qry": ETT_minute_test_x[500:2500], 
            "y_qry": ETT_minute_test_y[500:2500]
            }
    }
    print("------meta_train------")
    for key in meta_train.keys():
        meta_train[key]['x_spt'] = np.array(meta_train[key]['x_spt'])
        meta_train[key]['y_spt'] = np.array(meta_train[key]['y_spt'])
        meta_train[key]['x_qry'] = np.array(meta_train[key]['x_qry'])
        meta_train[key]['y_qry'] = np.array(meta_train[key]['y_qry'])
        print("{}: x_spt_shape {}, y_spt_shape {}, x_qry_shape {}, y_qry_shape {}".format(
            key, np.array(meta_train[key]['x_spt']).shape, np.array(meta_train[key]['y_spt']).shape, 
            np.array(meta_train[key]['x_qry']).shape, np.array(meta_train[key]['y_qry']).shape
        ))

    print("------meta_test------")
    for key in meta_test.keys():
        meta_test[key]['x_spt'] = np.array(meta_test[key]['x_spt'])
        meta_test[key]['y_spt'] = np.array(meta_test[key]['y_spt'])
        meta_test[key]['x_qry'] = np.array(meta_test[key]['x_qry'])
        meta_test[key]['y_qry'] = np.array(meta_test[key]['y_qry'])
        print("{}: x_spt_shape {}, y_spt_shape {}, x_qry_shape {}, y_qry_shape {}".format(
            key, np.array(meta_test[key]['x_spt']).shape, np.array(meta_test[key]['y_spt']).shape, 
            np.array(meta_test[key]['x_qry']).shape, np.array(meta_test[key]['y_qry']).shape
        ))

    return meta_train, meta_test
    
    
    

def Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', seq_len=400,label_len=100,pred_len=100,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h'):
    
    scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(root_path,data_path))
    
    type_map = {'train':0, 'val':1, 'test':2}
    set_type = type_map[flag]

    border1s = [0, 12*30*24 - seq_len, 12*30*24+4*30*24 - seq_len]
    border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    

    if features=='M' or features=='MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features=='S':
        df_data = df_raw[[target]]

    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values

    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    data_stamp = time_features(df_stamp, timeenc=timeenc, freq=freq)

    data_x = data[border1:border2]

    if inverse:
        data_y = df_data.values[border1:border2]
    else:
        data_y = data[border1:border2]

    # 把x和y分布好，注意这里是lstm版本的y，若改成informer的y要回tools.py改
    Data_x = []
    Data_y = []
    X_mark = []
    Y_mark = []
    # -640是为了取个整(doge)
    for i in range(0,len(data_x)-640):
        temp_x,temp_y,temp_x_mark,temp_y_mark = slip_window_data(data_x,data_y,data_stamp,index=i,
                                                                 inverse=False,seq_len=seq_len, label_len=label_len, pred_len=pred_len)
        Data_x.append(temp_x)
        Data_y.append(temp_y)
        X_mark.append(temp_x_mark)
        Y_mark.append(temp_y_mark)
    

    # 再多分一段，元学习需要四维数据,这里的2是为了跟github上的代码相符,而用pred_len则是可以随时改变size
    Data_x = list_of_groups(Data_x,pred_len)
    Data_y = list_of_groups(Data_y,pred_len)
    X_mark = list_of_groups(X_mark,pred_len)
    Y_mark = list_of_groups(Y_mark,pred_len)
    
    
    return Data_x,Data_y,X_mark,Y_mark



if __name__ == '__main__':
    meta_train, meta_test = new_dataloader()
    x_spt, y_spt, x_qry, y_qry = getBatchTask(meta_train)
    print('x_spt.shape = {}'.format(x_spt.shape))
    print('y_spt.shape = {}'.format(y_spt.shape))
    print('x_qry.shape = {}'.format(x_qry.shape))
    print('y_qry.shape = {}'.format(y_qry.shape))