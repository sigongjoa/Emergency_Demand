# load module
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import warnings
import torch
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pytorch_lightning as pl
import seaborn as sns
from scipy.signal import find_peaks

from my_funs import *
from sklearn import metrics
from matplotlib import gridspec
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet , EncoderNormalizer , GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#import tensorflow as tf 
import tensorboard as tb 
#tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
plt.rcParams['font.family'] = 'NanumGothic'

dongs = ['강남동', '교  동', '근화동', '남  면', '남산면', '동  면', '동내면', '동산면', '북산면','사북면', '서  면', '석사동', '소양동', '신동면', '신북읍', '신사우동', '약사명동', '조운동','퇴계동', '효자1동', '후평1동']

def get_val_dataloader(training , data , time_idx):
    batch_size= 32    
    # test_data를 계속 append 하는 식으로 새생성
    test_data = data[data['time_idx'] < time_idx]
    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1)
    return val_dataloader

    df = pd.DataFrame([org, pred]).T
    df.columns = ['org' , 'pred']
    df['pred_category'] = 0
    df = df.astype(int)
    pred_df = df[df['pred']==1]
    cat_num = 0
    pre_idx = 0
    for idx in pred_df.index:
        if (pre_idx + 1) == idx:
            pass
        else:
            cat_num +=1

        pre_idx = idx
        #print(idx , cat_num)
        df.loc[idx,'pred_category'] = cat_num
    df.fillna(0, inplace=True)
    df = df.astype(int)
    return df

def pred_category(org, pred):
    df = pd.DataFrame([org, pred]).T
    df.columns = ['org' , 'pred']
    df['pred_category'] = 0
    df = df.astype(int)
    pred_df = df[df['pred']==1]
    cat_num = 0
    pre_idx = 0
    for idx in pred_df.index:
        if (pre_idx + 1) == idx:
            pass
        else:
            cat_num +=1

        pre_idx = idx
        #print(idx , cat_num)
        df.loc[idx,'pred_category'] = cat_num
    df.fillna(0, inplace=True)
    df = df.astype(int)
    return df

def post_processing(org,pred , cut_off):
    df = pred_category(org, pred)
    for idx in range(1, df['pred_category'].max()+1):
        df_idx = df[df['pred_category'] == idx][cut_off:].index
        #print(df_idx)
        df['pred_category'].loc[df_idx] = 0 
    cut_off_idx = df[df['pred_category'] > 0].index
    df['pred'] = 0
    df.loc[cut_off_idx,'pred'] = 1
    return np.array(df['org']) , np.array(df['pred'])

def confusion_matrix_plot_peak(training, tft ,data ,dong , title):
    # 원본 함수 
    fig = plt.figure(figsize=(25, 7))
    gs = gridspec.GridSpec(nrows=2, # row 몇 개 
                           ncols=2, # col 몇 개 
                           height_ratios=[1,1], 
                           width_ratios=[5,1]
                          )

    gs.update(wspace=0.025, hspace=0.2)
    ax0 = plt.subplot(gs[: , 0])
    ax1 = plt.subplot(gs[1,1])
    ax2 = plt.subplot(gs[0,1])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')

    ax0.set_title(f'{dong}_plot {title}' , fontsize = 20)
    ax2.set_title('confusion_matrix', fontsize = 20)
    ax1.set_title('recall & precision , MAE', fontsize = 20)

    df = data[data['h_dong']==dong]
    df_index = df[df['time_idx'] > 24* 7 -1]['REG_DTIME']
    org_count = df[df['time_idx'] > 24* 7 -1 ]['count']

    predition5= np.array([])
    for i in range(1,52):
        val_data = get_val_dataloader(training ,data, 24*7 + 24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        predition5 = np.concatenate([predition5,pred['prediction'][idx, : , 5]])
    
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)
    
    peak_idx , peaks  = find_peaks(predition5, 1)
    ax0.plot(df_index.iloc[peak_idx] , predition5[peak_idx] , "x" , label = 'find peak')
    predition5[peak_idx] = 1
    predition5 = np.logical_not(predition5 < 1)
    
    ax0.plot(df_index,org_count , label = 'orginal target' , alpha=1)
    ax0.plot(df_index,predition5 ,label = '90% prediction round', alpha= 0.6, linestyle='--')
    ax0.legend()

    tn, fp, fn, tp = metrics.confusion_matrix(org_count,predition5).ravel()

    c_mat = np.array([[tp , fp],[fn , tn]])
    ax1 = sns.heatmap(c_mat, annot=True, cbar = False, fmt='g',cmap='Blues')

    f1_score = metrics.f1_score(org_count,predition5)
    f2_score = metrics.fbeta_score(org_count,predition5 , 2)
    recall = metrics.recall_score(org_count,predition5)
    precision = metrics.precision_score(org_count,predition5)
    MAE = abs(recall - precision)
    

    ax2.text(0.1, 2.7 , f'f1 score  {f1_score : 4f}' , fontsize=25)
    ax2.text(0.1, 3.0 , f'f2 score  {f2_score : 4f}' , fontsize=25)
    ax2.text(0.1, 3.3 , f'recall    {recall : 4f}' , fontsize=25)
    ax2.text(0.1, 3.6 , f'precision {precision : 4f}' , fontsize=25)
    ax2.text(0.1, 3.9 , f'MAE       {MAE : 4f}' , fontsize=25)
    
    
def dong_f2(training, tft ,data,cut_off, dong):
    df = data[data['h_dong']==dong]
    df_index = df[df['time_idx'] > 24* 7 -1]['REG_DTIME']
    org_count = df[df['time_idx'] > 24* 7 -1 ]['count']

    predition5= np.array([])
    for i in range(1,52):
        # 1년(8760) + 하루(24) * idx를 이용해서 하루씩 append하는 식으로 prediction 진행
        val_data = get_val_dataloader(training ,data, 24*7 + 24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        # dong의 문자열 표현을 tft.prediction을 하기 위해서 숫자로 변경
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        # 90퍼 값만 사용하기 위해서 [idx, : , 5]과 같이 슬라이싱
        predition5 = np.concatenate([predition5,pred['prediction'][idx, : , 5]])
    
    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)
    
    peak_idx , peaks  = find_peaks(predition5, 1)
    predition5[peak_idx] = 1
    predition5 = np.logical_not(predition5 < 1)
    return metrics.fbeta_score(org_count,predition5 , 2)


def dong_f2_test(training, tft ,data, dong, cut_off):
    df = data[data['h_dong']==dong]
    df_index = df[df['time_idx'] > 24* 7 -1]['REG_DTIME']
    org_count = df[df['time_idx'] > 24* 7 -1 ]['count']

    predition5= np.array([])
    for i in range(1,52):
        # 1년(8760) + 하루(24) * idx를 이용해서 하루씩 append하는 식으로 prediction 진행
        val_data = get_val_dataloader(training ,data, 24*7 + 24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        # dong의 문자열 표현을 tft.prediction을 하기 위해서 숫자로 변경
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        # 90퍼 값만 사용하기 위해서 [idx, : , 5]과 같이 슬라이싱
        predition5 = np.concatenate([predition5,pred['prediction'][idx, : , 5]])
    
    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)

    peak_idx , peaks  = find_peaks(predition5, 1)
    predition5[peak_idx] = 1
    predition5 = np.logical_not(predition5 < 1)
    return org_count , predition5

def dong_f2_all(training , tft ,data):
    val_data = get_val_dataloader(training ,data, 24*7 + 24)
    pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
    predictions = pred['prediction'][:, : , 5]

    for i in range(2,52):
        val_data = get_val_dataloader(training ,data, 24*7 + 24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        predictions = torch.cat([predictions ,pred['prediction'][:, : , 5]], dim=1) 
    predictions = predictions.reshape(-1,1)

    org_df = data[data['time_idx'] > 24*7 -1]
    org_count = []
    for dong in dongs:
        org_count.extend(org_df[org_df['h_dong'] == dong]['count'].to_list())

    org_count = np.array(org_count)
    org_count = np.logical_not(org_count < 1)
    
    peak_idx , peaks  = find_peaks(predictions.reshape(-1), 1)
    predictions[peak_idx] = 1
    predictions = np.logical_not(predictions < 1)

    return metrics.fbeta_score(org_count,predictions , 2) 

def get_f2(training, tft ,data):
    f2_df = pd.DataFrame(columns = ['h_dong' , 'f2_score'])
    for idx, dong in enumerate(dongs):
        #print(f'{dong} is processing f2_score')
        f2_df.loc[idx,'h_dong'] = dong
        f2_df.loc[idx,'f2_score'] = dong_f2(training, tft ,data, dong ,)
    return f2_df.sort_values(by='f2_score' , ascending=False)