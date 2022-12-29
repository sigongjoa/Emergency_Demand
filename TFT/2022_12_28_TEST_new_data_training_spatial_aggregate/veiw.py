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
import networkx as nx
from sklearn import metrics
from matplotlib import gridspec


import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet , EncoderNormalizer , GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#import tensorflow as tf 
import tensorboard as tb 
#tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import matplotlib
from matplotlib import dates
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
plt.rcParams['font.family'] = 'NanumGothic'
#plt.rcParams['font.sans-serif'] = ['NanumGothic.ttf', 'sans-serif']

from my_funs_nei import *
q = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

def prediction_plot_n(training, tft ,data, dong , title, ewma_funs, factor):
    df = data[data['h_dong']==dong]
    df_index = df[df['time_idx'] > 24* 7 -1]['REG_DTIME']
    org_data = data_processing('nei_test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_ewma_data = data_processing('nei_test.csv' , factor , ewma_funs)
    org_ewma_data = org_ewma_data[org_ewma_data['h_dong'] == dong]
    s_size = 4
    
    prediction_df = pd.DataFrame()
    for file in sorted(os.listdir('./test_data'))[1:]:
        val_data = data_processing('./test_data/' + file , factor , ewma_funs)
        val_data = val_data.fillna(0)
        validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        pred , x, idx_df = tft.predict(val_dataloader, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        
        xyz = pd.DataFrame()
        delta_h = val_data['REG_DTIME'].unique()[1] - val_data['REG_DTIME'].unique()[0] 
        s = val_data['REG_DTIME'].unique().max() + delta_h
        e = val_data['REG_DTIME'].unique().max() + 24*delta_h

        #print(pd.date_range(s,e ,freq = 'h'))
        xyz.index = pd.date_range(s,e ,freq = 'h')
        xyz['prediction50'] = pred['prediction'][idx, : , 3]
        xyz['prediction75'] = pred['prediction'][idx, : , 4]
        xyz['prediction90'] = pred['prediction'][idx, : , 5]
        prediction_df = pd.concat([prediction_df, xyz])

    fig = plt.figure(figsize=(15, 150))
    gs = gridspec.GridSpec(nrows = 25, # row 몇 개 
                       ncols=1, # col 몇 개 
                       height_ratios=np.full(25,1), 
                       width_ratios=[1]
                      )

    gs.update(wspace=0.1, hspace=0.5)
 
    for i in range(25):
        ax = plt.subplot(gs[i,0])
        ax.set_title(f'sex{i}')


        #idx = prediction_df.index[:-24]
        idx = prediction_df.index[i*48:(i+1)*48]
        org_count = org_data[org_data['REG_DTIME'].isin(idx)]['count']    
        org_ewma_count = org_ewma_data[org_ewma_data['REG_DTIME'].isin(idx)]['count']    

        ax.plot(idx ,prediction_df.loc[idx, 'prediction50'] , label = 'preidction 50%' , alpha=0.8)
        ax.plot(idx ,prediction_df.loc[idx, 'prediction75'] , label = 'preidction 75%' , alpha=0.5)
        ax.plot(idx ,prediction_df.loc[idx, 'prediction90'] , label = 'preidction 90%' , alpha=0.3)
        ax.plot(idx ,org_count , label = 'orginal target' , alpha=0.4)
        ax.plot(idx ,org_ewma_count , label = 'ewma apply target' , alpha=0.4)
        ax.xaxis.set_major_locator(dates.HourLocator())
        ax.set_xticklabels(ax.get_xticks(), rotation = 45, fontsize = 8)
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H'))


        ax.set_title(f'{dong}  {idx[0]} ~ {idx[-1]}',fontsize=15)
        ax.legend()

        
def confusion_matrix_plot50(training, tft ,data ,dong , title, ewma_funs, factor):
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

    prediction_df = pd.DataFrame()
    for file in sorted(os.listdir('./test_data'))[1:]:
        val_data = data_processing('./test_data/' + file , factor , ewma_funs)
        val_data = val_data.fillna(0)
        validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        pred , x, idx_df = tft.predict(val_dataloader, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        
        xyz = pd.DataFrame()
        delta_h = val_data['REG_DTIME'].unique()[1] - val_data['REG_DTIME'].unique()[0] 
        s = val_data['REG_DTIME'].unique().max() + delta_h
        e = val_data['REG_DTIME'].unique().max() + 24*delta_h

        #print(pd.date_range(s,e ,freq = 'h'))
        xyz.index = pd.date_range(s,e ,freq = 'h')
        xyz['prediction50'] = pred['prediction'][idx, : , 3]
        xyz['prediction75'] = pred['prediction'][idx, : , 4]
        xyz['prediction90'] = pred['prediction'][idx, : , 5]
        prediction_df = pd.concat([prediction_df, xyz])

    predition5 = prediction_df['prediction50'][:-24]
    df_index = prediction_df.index[:-24]
    
    org_data = data_processing('nei_test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['REG_DTIME'].isin(df_index)]['count']
    
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)
    predition5 = np.logical_not(predition5 < 0.5)
    
    ax0.plot(df_index,org_count , label = 'orginal target' , alpha=1)
    ax0.plot(df_index,predition5 ,label = '50% prediction round', alpha= 0.6, linestyle='--')
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
    
    
def confusion_matrix_plot75(training, tft ,data ,dong , title, ewma_funs, factor):
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

    prediction_df = pd.DataFrame()
    for file in sorted(os.listdir('./test_data'))[1:]:
        val_data = data_processing('./test_data/' + file , factor , ewma_funs)
        val_data = val_data.fillna(0)
        validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        pred , x, idx_df = tft.predict(val_dataloader, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        
        xyz = pd.DataFrame()
        delta_h = val_data['REG_DTIME'].unique()[1] - val_data['REG_DTIME'].unique()[0] 
        s = val_data['REG_DTIME'].unique().max() + delta_h
        e = val_data['REG_DTIME'].unique().max() + 24*delta_h

        #print(pd.date_range(s,e ,freq = 'h'))
        xyz.index = pd.date_range(s,e ,freq = 'h')
        xyz['prediction50'] = pred['prediction'][idx, : , 3]
        xyz['prediction75'] = pred['prediction'][idx, : , 4]
        xyz['prediction90'] = pred['prediction'][idx, : , 5]
        prediction_df = pd.concat([prediction_df, xyz])

    predition5 = prediction_df['prediction75'][:-24]
    df_index = prediction_df.index[:-24]
    
    org_data = data_processing('nei_test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['REG_DTIME'].isin(df_index)]['count']
    
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)
    predition5 = np.logical_not(predition5 < 0.5)
    
    ax0.plot(df_index,org_count , label = 'orginal target' , alpha=1)
    ax0.plot(df_index,predition5 ,label = '75% prediction round', alpha= 0.6, linestyle='--')
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
    
def confusion_matrix_plot90(training, tft ,data ,dong , title, ewma_funs, factor):
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

    prediction_df = pd.DataFrame()
    for file in sorted(os.listdir('./test_data'))[1:]:
        val_data = data_processing('./test_data/' + file , factor , ewma_funs)
        val_data = val_data.fillna(0)
        validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True, stop_randomization=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        pred , x, idx_df = tft.predict(val_dataloader, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        
        xyz = pd.DataFrame()
        delta_h = val_data['REG_DTIME'].unique()[1] - val_data['REG_DTIME'].unique()[0] 
        s = val_data['REG_DTIME'].unique().max() + delta_h
        e = val_data['REG_DTIME'].unique().max() + 24*delta_h

        #print(pd.date_range(s,e ,freq = 'h'))
        xyz.index = pd.date_range(s,e ,freq = 'h')
        xyz['prediction50'] = pred['prediction'][idx, : , 3]
        xyz['prediction75'] = pred['prediction'][idx, : , 4]
        xyz['prediction90'] = pred['prediction'][idx, : , 5]
        prediction_df = pd.concat([prediction_df, xyz])

    predition5 = prediction_df['prediction90'][:-24]
    df_index = prediction_df.index[:-24]
    
    org_data = data_processing('nei_test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['REG_DTIME'].isin(df_index)]['count']
    
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    org_count = org_count.to_numpy()
    org_count = np.logical_not(org_count < 1)
    predition5 = np.logical_not(predition5 < 0.5)
    
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
