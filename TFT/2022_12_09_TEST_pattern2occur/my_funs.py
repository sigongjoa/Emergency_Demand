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
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
plt.rcParams['font.family'] = 'NanumGothic'
#plt.rcParams['font.sans-serif'] = ['NanumGothic.ttf', 'sans-serif']

import numpy as np
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

def k_nbrs_in(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs  | set([start])

def k_nbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs 


nei_dong = {
    '동산면' : ['동내면' , '신동면', '남산면'],
    '후평1동' :['동  면' , '신사우동', '근화동', '소양동', '교  동', ],
    '사북면' : ['서  면' , '신북읍'],
    '신북읍' : ['북산면' , '사북면' , '서  면', '신사우동' , '동  면'], 
    '석사동' : ['동내면', '동  면', '퇴계동'],
    '남산면' : ['동산면', '신동면', '서  면' , '남  면'],
    '교  동' : ['후평1동' , '소양동' , '조운동' , ],
    '신동면' : ['동내면' , '퇴계동', '강남동', '서  면', '남산면', '동산면'],
    '효자1동': ['조운동', '약사명동','근화동','강남동', '퇴계동', ],
    '북산면' : ['신북읍', '동  면'],
    '서  면' : ['신사우동', '신북읍', '사북면', '남산면', '신동면', '강남동', '근화동'],
    '조운동' : ['교  동', '소양동' , '약사명동', '효자1동'],
    '동내면' : ['동  면', '석사동', '퇴계동', '신동면' , '동산면'],
    '강남동' : ['퇴계동', '효자1동', '근화동','서  면' , '신동면'],
    '퇴계동' : ['석사동', '강남동', '신동면'],
    '근화동' : ['신사우동' , '서  면' , '강남동', '약사명동', '소양동'] , 
    '동  면' : ['북산면' , '신북읍', '신사우동', '후평1동' , '석사동' , '동내면'],
    '신사우동':['동  면' , '신북읍','서  면' , '근화동' , '후평1동'],
    '약사명동':['소양동' , '근화동','효자1동', '조운동'],
    '남  면' : ['남산면']
}


def graph_count(graph_data : pd.DataFrame, w1 : float , w2 : float , w3 : float):
    graph_data = graph_data[['REG_DTIME', 'h_dong', 'count', 'pops', 'windspd' , 'humid' , 'temp', 'precip_form', 'precip', 'isHoliday']]
    dongs = graph_data['h_dong'].unique()
    
    nei_edge_df = pd.DataFrame(columns = ['source' , 'target' , 'weight'])

    idx = 0
    for key in nei_dong.keys():
        for nei in nei_dong[key]:
            nei_edge_df.loc[idx] = [key, nei , 1 ,]
            idx += 1
    nei_node_df = pd.DataFrame(
        {
        'adm' : dongs,
        'color' : np.full(len(dongs) , 'yellow' )
    })
    
    adm_G = nx.from_pandas_edgelist(nei_edge_df , source='source' , target='target' , edge_attr = ['weight'])
    nodes_attr = nei_node_df.set_index('adm').to_dict(orient='index')
    nx.set_node_attributes(adm_G, nodes_attr)
    p1 , p2 ,p3 = 1 , 0.5 , 0.3
    dts = graph_data['REG_DTIME'].unique()#.astype(str)
    dongs = graph_data['h_dong'].unique()
    graph_count = []
    for idx in graph_data.index:
        dt = graph_data['REG_DTIME'].loc[idx]
        dong = graph_data['h_dong'].loc[idx]
        #print(dt,dong)
        time_data = graph_data[graph_data['REG_DTIME'] == dt]
        nei_dong_data = time_data[time_data['h_dong'].isin(k_nbrs_in(adm_G, dong, 1))]
        #print(sum(nei_dong_data['count']))

        #graph_data.loc[idx,'count'] = sum(nei_dong_data['count'])
        graph_count.append(sum(nei_dong_data['count']))

    graph_data['count'] = graph_count
    return graph_data


def get_training(data , max_prediction_length = 24 , max_encoder_length = 24*7):
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
    batch_size= 16
    test_data = data[data['time_idx'] < time_idx]
    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1)
    return val_dataloader

def confusion_matrix_plot(training, tft ,data ,dong , title  ):
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
        predition5 = np.concatenate([predition5,pred['prediction'][idx, : , 3]])
    
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    org_data = data_processing('../../test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['time_idx'] > 24*7-1]['count']
    org_count = np.logical_not(org_count < 1)
    
    org_data = data_processing('../../test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['time_idx'] > 24*7-1]['count']
    org_count = np.logical_not(org_count < 1)
    
    ax0.plot(df_index,predition5 , label = 'prediction round' , alpha=0.8)
    ax0.plot(df_index,org_count , label = 'orginal target' , alpha=0.8)
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
    
    
def prediction_plot_n(training, tft ,data, dong , title, round_cut,n):
    fig = plt.figure(figsize=(25, 7))
    df = data[data['h_dong']==dong]
    df_index = df[df['time_idx'] > 24* 7 -1]['REG_DTIME']
    org_count_ewma = df[df['time_idx'] > 24* 7 -1 ]['count']

    predition5= np.array([])
    for i in range(1,52):
        val_data = get_val_dataloader(training ,data, 24*7 + 24*i)
        pred , x, idx_df = tft.predict(val_data, mode='raw', return_x = True , return_index = True)
        idx = idx_df[idx_df['h_dong'] == dong].index[0]
        predition5 = np.concatenate([predition5,pred['prediction'][idx, : , n]])
    
    org_data = data_processing('../../test.csv' , 0 , None)
    org_data = org_data[org_data['h_dong'] == dong]
    org_count = org_data[org_data['time_idx'] > 24*7-1]['count']
    org_count = np.logical_not(org_count < 1)
    
    plt.plot(df_index,org_count , label = 'orginal target' , alpha=0.4)
    plt.plot(df_index,predition5 , label = 'preidction' , alpha=0.8)
    plt.plot(df_index,org_count_ewma , label = 'orginal target' , alpha=0.8)
    plt.axhline(round_cut, 0 , len(df_index), alpha=0.3 , label = 'round_cut')
    plt.title(title,fontsize=25)
    plt.legend()
