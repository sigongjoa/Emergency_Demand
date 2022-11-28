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
from sklearn import metrics
from matplotlib import gridspec

import pytorch_lightning as pl
import seaborn as sns
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
#plt.rcParams['font.sans-serif'] = ['NanumGothic.ttf', 'sans-serif']


# functions
def data_processing(path = '../../sample_table/ewma_6h_scaling.csv'):
    data = pd.read_csv(path)    
    data = data[['REG_DTIME', 'h_dong', 'count', 'pops', 'windspd','humid', 'temp', 'precip_form', 'precip', 'isHoliday']]
    data['REG_DTIME'] = pd.to_datetime(data['REG_DTIME'])
    data['DOW'] = data['REG_DTIME'].dt.dayofweek
    data['HOD'] = data['REG_DTIME'].dt.hour
    data['MOY'] = data['REG_DTIME'].dt.month
    data["time_idx"] =  \
    data["REG_DTIME"].dt.year * 365*24 + \
    data["REG_DTIME"].dt.day_of_year * 24  + \
    data["REG_DTIME"].dt.hour 
    data["time_idx"] -= data["time_idx"].min()
    data['h_dong'] = data['h_dong'].astype(str)
    data['DOW'] = data['DOW'].astype(str)
    data['HOD'] = data['HOD'].astype(str)
    data['MOY'] = data['MOY'].astype(str)
    data['precip_form'] = data['precip_form'].astype(str)
    data['isHoliday'] = data['isHoliday'].astype(str)
    #data['count'] = data['count'].round(5)

    return data

def get_training(data , max_prediction_length, max_encoder_length):
    # traing data 생성
    training_cutoff = data["time_idx"].max() - max_prediction_length
    training = TimeSeriesDataSet(
        data[lambda x : x.time_idx <= training_cutoff],
        time_idx = "time_idx",
        target = "count",
        group_ids = ['h_dong'],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=24,
        static_categoricals=["h_dong"],
        time_varying_known_categoricals=["HOD", "DOW" , 'isHoliday', 'MOY'],
        time_varying_known_reals=['pops'],
        time_varying_unknown_categoricals=['precip_form'],
        time_varying_unknown_reals=['count','windspd' , 'temp' ,'precip'], 
        target_normalizer=GroupNormalizer(
            groups=["h_dong"], 
            transformation="relu",
            center = False
        ),
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        #allow_missing=True,
        allow_missing_timesteps = True,
        #predict_mode = False
        )
    return training
def get_val_dataloader(training , data , time_idx):
    batch_size= 32    
    test_data = data[data['time_idx'] < time_idx]
    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1)
    return val_dataloader

def countious_prediction(training, tft ,ewma_data, dong, title):
    plt.figure(figsize=(16,4))
    predition3 , predition4, predition5, predition6 = np.array([]) , np.array([]) , np.array([]) , np.array([])
    for i in range(1,59):
        val_data = get_val_dataloader(training ,ewma_data, 8760+24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        #print(idx_df['time_idx'].loc[0])
        dong_idx = idx_df[idx_df['h_dong'] == dong].index[0]
        #predition3 = np.concatenate([predition3,pred['prediction'][dong_idx, : , 3]])
        #predition4 = np.concatenate([predition4,pred['prediction'][dong_idx, : , 4]])
        predition5 = np.concatenate([predition5,pred['prediction'][dong_idx, : , 5]])
        #predition6 = np.concatenate([predition6,pred['prediction'][dong_idx, : , 6]])
    
    ewma_dong_df = ewma_data[ewma_data['h_dong']==dong]
    ewma_df_index = ewma_dong_df[ewma_dong_df['time_idx'] > 8760-1]['REG_DTIME']
    ewma_org_count = ewma_dong_df[ewma_dong_df['time_idx'] > 8760-1]['count']
    
    org_df = data_processing('../../data/data.csv')
    dong_org_df = org_df[org_df['h_dong'] == dong]
    dong_org_count = dong_org_df[dong_org_df['time_idx'] > 8760-1]['count']
        
    
    plt.plot(ewma_df_index,dong_org_count , label = 'orginal target' , alpha=1)
    plt.plot(ewma_df_index,ewma_org_count , label = 'ewma target' , alpha=0.7)
    #plt.plot(ewma_df_index,predition3 ,label = '50% prediction', alpha= 0.6, linestyle='--')
    #plt.plot(ewma_df_index,predition4 ,label = '70% prediction', alpha= 0.6, linestyle='--')
    plt.plot(ewma_df_index,predition5 ,label = '90% prediction', alpha= 0.6, linestyle='--')
    #plt.plot(ewma_df_index,predition6 ,label = '98% prediction', alpha= 0.6, linestyle='--')
    plt.title(f'{dong}_{title}')
    plt.legend()
    plt.savefig(f'{dong}_{title}.png')

def confusion_matrix_plot(training, tft ,ewma_data, dong , title):
    fig = plt.figure(figsize=(25, 5)) 
    gs = gridspec.GridSpec(nrows=1, # row 몇 개 
                           ncols=2, # col 몇 개 
                           height_ratios=[1], 
                           width_ratios=[5,1]
                          )
    gs.update(wspace=0.025, hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    predition5= np.array([]) 
    
    for i in range(1,59):
        val_data = get_val_dataloader(training ,ewma_data, 8760+24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        dong_idx = idx_df[idx_df['h_dong'] == dong].index[0]
        predition5 = np.concatenate([predition5,pred['prediction'][dong_idx, : , 5]])
        
    ewma_dong_df = ewma_data[ewma_data['h_dong']==dong]
    ewma_df_index = ewma_dong_df[ewma_dong_df['time_idx'] > 8760-1]['REG_DTIME']
    ewma_org_count = ewma_dong_df[ewma_dong_df['time_idx'] > 8760-1]['count']
    ax0.plot(ewma_df_index,predition5 , label = '90% prediction' , alpha=0.6 , color = 'violet')
    
    org_df = data_processing('../../data/data.csv')
    dong_org_df = org_df[org_df['h_dong'] == dong]
    dong_org_count = dong_org_df[dong_org_df['time_idx'] > 8760-1]['count']
    
    dong_org_count = dong_org_count.to_numpy().round()
    dong_org_count = np.logical_not(dong_org_count < 1)
    predition5 = predition5.round()
    predition5 = np.logical_not(predition5 < 1)
    
    ax0.plot(ewma_df_index,dong_org_count , label = 'orginal target' , alpha=1)
    ax0.plot(ewma_df_index,ewma_org_count , label = 'ewma target' , alpha=0.5)
    ax0.plot(ewma_df_index,predition5 ,label = '90% prediction round', alpha= 0.6, linestyle='--')
    ax0.set_title(f'{dong}_{title}')
    ax0.legend()
    
    tn, fp, fn, tp = metrics.confusion_matrix(dong_org_count,predition5).ravel()

    c_mat = np.array([[tp , fp],[fn , tn]])
    ax1 = sns.heatmap(c_mat, annot=True, cbar = False, fmt='g',cmap='Blues')
    ax1.set_title('confusion_matrix')
    ax1.set_xticks([])
    ax1.set_yticks([])
    print(f'{dong} f1_score' , metrics.f1_score(dong_org_count,predition5))
    return metrics.f1_score(dong_org_count,predition5)