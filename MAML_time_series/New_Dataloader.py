import numpy as np
import pandas as pd
import os
import random
import utils.tools

from utils.tools import StandardScaler
from utils.tools import list_of_groups
from utils.tools import slip_window_data
from utils.tools import get_highest_digit
from utils.timefeatures import time_features

root_path = './data/ETT/'
data_path = 'ETTh1.csv'
flag='train'
# size=None
# size = [400,100,100]
Size = [32,8,8]
features='M'
target='OT'
scale=True
inverse=True
timeenc=0
freq='h'
index=0

def new_dataloader(root_path ='./data/ETT/',flag='train',size = Size,features='M',
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

    ETT_hour1_x,ETT_hour1_y,ETT_hour1_x_mark,ETT_hour1_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='h')
    ETT_hour2_x,ETT_hour2_y,ETT_hour2_x_mark,ETT_hour2_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTh2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='h')    
    ETT_minute1_x,ETT_minute1_y,ETT_minute1_x_mark,ETT_minute1_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTm1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='t')
    ETT_minute2_x,ETT_minute2_y,ETT_minute2_x_mark,ETT_minute2_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', flag='train', size=None,
        features='M', data_path='ETTm2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,target='OT', scale=True, inverse=False,           timeenc=0, freq='t')
    
    ETT_hour1_test_x,ETT_hour1_test_y,ETT_hour1_test_x_mark,ETT_hour1_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', 
        flag='test', size=None,features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='h')
    ETT_hour2_test_x,ETT_hour2_test_y,ETT_hour2_test_x_mark,ETT_hour2_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', 
        flag='test', size=None,features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='h')    
    ETT_minute1_test_x,ETT_minute1_test_y,ETT_minute1_test_x_mark,ETT_minute1_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/',                   flag='test', size=None,features='M', data_path='ETTm1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='t')
    ETT_minute2_test_x,ETT_minute2_test_y,ETT_minute2_test_x_mark,ETT_minute2_test_y_mark = Dataset_ETT_hour(root_path='./data/ETT/',                   flag='test', size=None,features='M', data_path='ETTm2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='t')

    ETT_hour1_val_x,ETT_hour1_val_y,ETT_hour1_val_x_mark,ETT_hour1_val_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', 
        flag='val', size=None,features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='h')
    ETT_hour2_val_x,ETT_hour2_val_y,ETT_hour2_val_x_mark,ETT_hour2_val_y_mark = Dataset_ETT_hour(root_path='./data/ETT/', 
        flag='val', size=None,features='M', data_path='ETTh1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='h')
    ETT_minute1_val_x,ETT_minute1_val_y,ETT_minute1_val_x_mark,ETT_minute1_val_y_mark = Dataset_ETT_hour(root_path='./data/ETT/',                                     flag='val', size=None,features='M', data_path='ETTm1.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='t')
    ETT_minute2_val_x,ETT_minute2_val_y,ETT_minute2_val_x_mark,ETT_minute2_val_y_mark = Dataset_ETT_hour(root_path='./data/ETT/',                                     flag='val', size=None,features='M', data_path='ETTm2.csv', seq_len=seq_len,label_len=label_len,pred_len=pred_len,
        target='OT', scale=True, inverse=False, timeenc=0, freq='t')
    
    len_train = len(ETT_hour1_x)
    index_train = len_train//5

    meta_train = {
        "ETTh1_train": {
            "x_spt": ETT_hour1_x[0:index_train],
            "x_spt_mark": ETT_hour1_x_mark[0:index_train],
            "y_spt": ETT_hour1_y[0:index_train], 
            "y_spt_mark": ETT_hour1_y_mark[0:index_train],
            "x_qry": ETT_hour1_x[index_train:5*index_train],
            "x_qry_mark": ETT_hour1_x_mark[index_train:5*index_train],
            "y_qry": ETT_hour1_y[index_train:2500],
            "y_qry_mark": ETT_hour1_y_mark[index_train:5*index_train]
            }, 
        "ETTh2_train": {
            "x_spt": ETT_hour2_x[0:index_train],
            "x_spt_mark": ETT_hour2_x_mark[0:index_train],
            "y_spt": ETT_hour2_y[0:index_train], 
            "y_spt_mark": ETT_hour2_y_mark[0:index_train],
            "x_qry": ETT_hour2_x[index_train:5*index_train],
            "x_qry_mark": ETT_hour2_x_mark[index_train:5*index_train],
            "y_qry": ETT_hour2_y[index_train:5*index_train],
            "y_qry_mark": ETT_hour2_y_mark[index_train:5*index_train]
            },
        "ETTm1_train": {
            "x_spt": ETT_minute1_x[0:index_train], 
            "x_spt_mark":ETT_minute1_x_mark[0:index_train],
            "y_spt": ETT_minute1_y[0:index_train], 
            "y_spt_mark":ETT_minute1_y_mark[0:index_train],
            "x_qry": ETT_minute1_x[index_train:5*index_train], 
            "x_qry_mark":ETT_minute1_x_mark[index_train:5*index_train],
            "y_qry": ETT_minute1_y[index_train:5*index_train],
            "y_qry_mark":ETT_minute1_y_mark[index_train:5*index_train]
            },
        "ETTm2_train": {
            "x_spt": ETT_minute2_x[0:index_train],
            "x_spt_mark":ETT_minute2_x_mark[0:index_train],
            "y_spt": ETT_minute2_y[0:index_train], 
            "y_spt_mark":ETT_minute2_y_mark[0:index_train],
            "x_qry": ETT_minute2_x[index_train:5*index_train], 
            "x_qry_mark":ETT_minute2_x_mark[index_train:5*index_train],
            "y_qry": ETT_minute2_y[index_train:5*index_train],
            "y_qry_mark":ETT_minute2_y_mark[index_train:5*index_train]
            }
    }
    
    # informer的数据中val是测试集，test是验证集
    # meta—learing中test是测试集，val是验证集
    # 因此这里将数据集的test部分用于验证(命名为val)，val部分用于测试(命名为test)
    len_val = len(ETT_hour1_test_x)
    index_val = len_val//5
    
    meta_val = {
        "ETTh1_val": {
            "x_spt": ETT_hour1_test_x[0:index_val],
            "x_spt_mark": ETT_hour1_test_x_mark[0:index_val],
            "y_spt": ETT_hour1_test_y[0:index_val], 
            "y_spt_mark": ETT_hour1_test_y_mark[0:index_val],
            "x_qry": ETT_hour1_test_x[index_val:5*index_val],
            "x_qry_mark": ETT_hour1_test_x_mark[index_val:5*index_val],
            "y_qry": ETT_hour1_test_y[index_val:5*index_val],
            "y_qry_mark": ETT_hour1_test_y_mark[index_val:5*index_val]
            }, 
        "ETTh2_val": {
            "x_spt": ETT_hour2_test_x[0:index_val],
            "x_spt_mark": ETT_hour2_test_x_mark[0:index_val],
            "y_spt": ETT_hour2_test_y[0:index_val], 
            "y_spt_mark": ETT_hour2_test_y_mark[0:index_val],
            "x_qry": ETT_hour2_test_x[index_val:5*index_val],
            "x_qry_mark": ETT_hour2_test_x_mark[index_val:5*index_val],
            "y_qry": ETT_hour2_test_y[index_val:5*index_val],
            "y_qry_mark": ETT_hour1_test_y_mark[index_val:5*index_val]
            }, 
        "ETTm1_val": {
            "x_spt": ETT_minute1_test_x[0:index_val], 
            "x_spt_mark": ETT_minute1_test_x_mark[0:index_val],
            "y_spt": ETT_minute1_test_y[0:index_val], 
            "y_spt_mark": ETT_minute1_test_y_mark[0:index_val],
            "x_qry": ETT_minute1_test_x[index_val:5*index_val], 
            "x_qry_mark": ETT_minute1_test_x_mark[index_val:5*index_val],
            "y_qry": ETT_minute1_test_y[index_val:5*index_val],
            "y_qry_mark": ETT_minute1_test_y_mark[index_val:5*index_val]
            },
        "ETTm2_val": {
            "x_spt": ETT_minute2_test_x[0:index_val], 
            "x_spt_mark": ETT_minute2_test_x_mark[0:index_val],
            "y_spt": ETT_minute2_test_y[0:index_val], 
            "y_spt_mark": ETT_minute2_test_y_mark[0:index_val],
            "x_qry": ETT_minute2_test_x[index_val:5*index_val], 
            "x_qry_mark": ETT_minute2_test_x_mark[index_val:5*index_val],
            "y_qry": ETT_minute2_test_y[index_val:5*index_val],
            "y_qry_mark": ETT_minute2_test_y_mark[index_val:5*index_val]
        }
    }
    
    len_test = len(ETT_hour1_val_x)
    index_test = len_test//5
    
    meta_test = {
        "ETTh1_test": {
            "x_spt": ETT_hour1_val_x[0:index_test],
            "x_spt_mark": ETT_hour1_val_x_mark[0:index_test],
            "y_spt": ETT_hour1_val_y[0:index_test], 
            "y_spt_mark": ETT_hour1_val_y_mark[0:index_test],
            "x_qry": ETT_hour1_val_x[index_test:5*index_test],
            "x_qry_mark": ETT_hour1_val_x_mark[index_test:5*index_test],
            "y_qry": ETT_hour1_val_y[index_test:5*index_test],
            "y_qry_mark": ETT_hour1_val_y_mark[index_test:5*index_test]
            }, 
        "ETTh2_test": {
            "x_spt": ETT_hour2_val_x[0:index_test],
            "x_spt_mark": ETT_hour2_val_x_mark[0:index_test],
            "y_spt": ETT_hour2_val_y[0:index_test], 
            "y_spt_mark": ETT_hour2_val_y_mark[0:index_test],
            "x_qry": ETT_hour2_val_x[index_test:5*index_test],
            "x_qry_mark": ETT_hour2_val_x_mark[index_test:5*index_test],
            "y_qry": ETT_hour2_val_y[index_test:5*index_test],
            "y_qry_mark": ETT_hour2_val_y_mark[index_test:5*index_test]
            }, 
        "ETTm1_test": {
            "x_spt": ETT_minute1_val_x[0:index_test], 
            "x_spt_mark": ETT_minute1_val_x_mark[0:index_test],
            "y_spt": ETT_minute1_val_y[0:index_test], 
            "y_spt_mark": ETT_minute1_val_y_mark[0:index_test],
            "x_qry": ETT_minute1_val_x[index_test:5*index_test], 
            "x_qry_mark": ETT_minute1_val_x_mark[index_test:5*index_test],
            "y_qry": ETT_minute1_val_y[index_test:5*index_test],
            "y_qry_mark": ETT_minute1_val_y_mark[index_test:5*index_test],
            },
        "ETTm2_test": {
            "x_spt": ETT_minute2_test_x[0:index_test], 
            "x_spt_mark": ETT_minute2_val_x_mark[0:index_test],
            "y_spt": ETT_minute2_test_y[0:index_test], 
            "y_spt_mark": ETT_minute2_val_y_mark[0:index_test],
            "x_qry": ETT_minute2_test_x[index_test:5*index_test], 
            "x_qry_mark": ETT_minute2_val_x_mark[index_test:5*index_test],
            "y_qry": ETT_minute2_test_y[index_test:5*index_test],
            "y_qry_mark": ETT_minute2_val_y_mark[index_test:5*index_test]
            }
    }
    print("------meta_train------")
    for key in meta_train.keys():
        meta_train[key]['x_spt'] = np.array(meta_train[key]['x_spt'])
        meta_train[key]['x_spt_mark'] = np.array(meta_train[key]['x_spt_mark'])
        meta_train[key]['y_spt'] = np.array(meta_train[key]['y_spt'])
        meta_train[key]['y_spt_mark'] = np.array(meta_train[key]['y_spt_mark'])
        meta_train[key]['x_qry'] = np.array(meta_train[key]['x_qry'])
        meta_train[key]['x_qry_mark'] = np.array(meta_train[key]['x_qry_mark'])
        meta_train[key]['y_qry'] = np.array(meta_train[key]['y_qry'])
        meta_train[key]['y_qry_mark'] = np.array(meta_train[key]['y_qry_mark'])
        print("{}: x_spt_shape {}, x_spt_mark_shape {}, y_spt_shape {}, y_spt_mark_shape {}, x_qry_shape {}, x_qry_mark_shape {}, y_qry_shape {}, y_qry_mark_shape {},".format(
            key, np.array(meta_train[key]['x_spt']).shape, np.array(meta_train[key]['x_spt_mark']).shape, 
            np.array(meta_train[key]['y_spt']).shape, np.array(meta_train[key]['y_spt_mark']).shape,
            np.array(meta_train[key]['x_qry']).shape, np.array(meta_train[key]['x_qry_mark']).shape,
            np.array(meta_train[key]['y_qry']).shape, np.array(meta_train[key]['y_qry_mark']).shape
        ))
    
    print("------meta_val------")
    for key in meta_val.keys():
        meta_val[key]['x_spt'] = np.array(meta_val[key]['x_spt'])
        meta_val[key]['x_spt_mark'] = np.array(meta_val[key]['x_spt_mark'])
        meta_val[key]['y_spt'] = np.array(meta_val[key]['y_spt'])
        meta_val[key]['y_spt_mark'] = np.array(meta_val[key]['y_spt_mark'])
        meta_val[key]['x_qry'] = np.array(meta_val[key]['x_qry'])
        meta_val[key]['x_qry_mark'] = np.array(meta_val[key]['x_qry_mark'])
        meta_val[key]['y_qry'] = np.array(meta_val[key]['y_qry'])
        meta_val[key]['y_qry_mark'] = np.array(meta_val[key]['y_qry_mark'])
        print("{}: x_spt_shape {}, x_spt_mark_shape {}, y_spt_shape {}, y_spt_mark_shape {}, x_qry_shape {}, x_qry_mark_shape {}, y_qry_shape {}, y_qry_mark_shape {},".format(
            key, np.array(meta_val[key]['x_spt']).shape, np.array(meta_val[key]['x_spt_mark']).shape,
            np.array(meta_val[key]['y_spt']).shape, np.array(meta_val[key]['y_spt_mark']).shape,
            np.array(meta_val[key]['x_qry']).shape, np.array(meta_val[key]['x_qry_mark']).shape,
            np.array(meta_val[key]['y_qry']).shape, np.array(meta_val[key]['y_qry_mark']).shape
        ))
    
    print("------meta_test------")
    for key in meta_test.keys():
        meta_test[key]['x_spt'] = np.array(meta_test[key]['x_spt'])
        meta_test[key]['x_spt_mark'] = np.array(meta_test[key]['x_spt_mark'])
        meta_test[key]['y_spt'] = np.array(meta_test[key]['y_spt'])
        meta_test[key]['y_spt_mark'] = np.array(meta_test[key]['y_spt_mark'])
        meta_test[key]['x_qry'] = np.array(meta_test[key]['x_qry'])
        meta_test[key]['x_qry_mark'] = np.array(meta_test[key]['x_qry_mark'])
        meta_test[key]['y_qry'] = np.array(meta_test[key]['y_qry'])
        meta_test[key]['y_qry_mark'] = np.array(meta_test[key]['y_qry_mark'])
        print("{}: x_spt_shape {}, x_spt_mark_shape {}, y_spt_shape {}, y_spt_mark_shape {}, x_qry_shape {}, x_qry_mark_shape {}, y_qry_shape {}, y_qry_mark_shape {},".format(
            key, np.array(meta_test[key]['x_spt']).shape, np.array(meta_test[key]['x_spt_mark']).shape,
            np.array(meta_test[key]['y_spt']).shape, np.array(meta_test[key]['y_spt_mark']).shape,
            np.array(meta_test[key]['x_qry']).shape, np.array(meta_test[key]['x_qry_mark']).shape,
            np.array(meta_test[key]['y_qry']).shape, np.array(meta_test[key]['y_qry_mark']).shape
        ))

    return meta_train, meta_val, meta_test
    
    
    

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
    # get_highest_digit是为了取个整(doge)
    for i in range(0,get_highest_digit(len(data_x))):
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