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
from tqdm import tqdm

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

def moving_average_alpha(df: pd.DataFrame, unit: float):
    ret_df = pd.DataFrame()
    max_value = df['count'].max()
    # forward df
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(alpha = unit).mean()
        back_ewma = back_ewma / back_ewma.max() * (max_value)
        #print(f'{dong} max : {max_value}   ewma_max :{back_ewma.max()}')
        df['count'][dong_df.index] = back_ewma 
    
    #print(max_value , df['count'].max())
    return df


def moving_average_com(df: pd.DataFrame, unit: float):
    ret_df = pd.DataFrame()
    max_value = df['count'].max()
    # forward df
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(com = unit).mean()
        back_ewma = back_ewma / back_ewma.max() * (max_value)
        #print(f'{dong} max : {max_value}   ewma_max :{back_ewma.max()}')
        df['count'][dong_df.index] = back_ewma 
    
    #print(max_value , df['count'].max())
    return df

def moving_average_span(df: pd.DataFrame, unit: float):
    ret_df = pd.DataFrame()
    max_value = df['count'].max()
    # forward df
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(span = unit).mean()
        back_ewma = back_ewma / back_ewma.max() * (max_value)
        #print(f'{dong} max : {max_value}   ewma_max :{back_ewma.max()}')
        df['count'][dong_df.index] = back_ewma 
    
    #print(max_value , df['count'].max())
    return df
    
def moving_average_halflife(df: pd.DataFrame, unit: float):
    ret_df = pd.DataFrame()
    max_value = df['count'].max()
    # forward df
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(halflife = unit).mean()
        back_ewma = back_ewma / back_ewma.max() * (max_value)
        #print(f'{dong} max : {max_value}   ewma_max :{back_ewma.max()}')
        df['count'][dong_df.index] = back_ewma 

    #print(max_value , df['count'].max())
    return df

def moving_average_alpha_both(df: pd.DataFrame, unit: float):
    ret_df = pd.DataFrame()
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(alpha = unit).mean()

        inv_dong_df = dong_df[::-1]
        for_ewma = inv_dong_df['count'].ewm(alpha = unit).mean()
        
        ewma = (for_ewma + back_ewma) / 2
        ewma = ewma / ewma.max() * max_value
        df['count'][dong_df.index] = ewma 
    
    #print(max_value , df['count'].max())
    return df

def graph_count(graph_data : pd.DataFrame, w1 : float , w2 : float , w3 : float):
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
    '남  면' : ['남산면'],
    '북산면' : ['신북읍' , '동  면'],
    '소양동' : ['근화동', '후평1동' , '교  동' , '조운동' , '약사명동']
    }

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
    dts = graph_data['REG_DTIME'].unique()#.astype(str)
    dongs = graph_data['h_dong'].unique()
    for dong in tqdm(graph_data['h_dong'].unique()):
        dong_df = graph_data[graph_data['h_dong'] == dong]
        nei_dongs = k_nbrs(adm_G, dong, 1)
        nei_sum = np.full(len(dong_df), 0)
        for nei_dong in nei_dongs:
            nei_count = graph_data[graph_data['h_dong'] == nei_dong]['count'].to_list()
            nei_sum += nei_count
        graph_data.loc[dong_df.index, 'nei1'] = (w1*dong_df['count'].to_list()) + (w2*nei_sum)
    graph_data['count'] = graph_data['nei1'] 
    return graph_data

def data_processing(path : str  , unit : float , ewma_fun : moving_average_alpha ):
    # path  : 데이터가 저장된 경로
    # unit  : ewma의 unit
    
    data = pd.read_csv(path)    
    # 사용하는 column만 선택
    data = data[['REG_DTIME', 'h_dong', 'count', 'pops', 'windspd','humid', 'temp', 'precip_form', 'precip', 'isHoliday' , 'nei1']]
    data['REG_DTIME'] = pd.to_datetime(data['REG_DTIME'])
    data['DOW'] = data['REG_DTIME'].dt.dayofweek
    data['HOD'] = data['REG_DTIME'].dt.hour
    data['MOY'] = data['REG_DTIME'].dt.month
    # time_idx 시간을 고유하게 표현(윤년)
    #for x , y in enumerate(data['REG_DTIME'].unique()):
    #    t_data_index = data[data['REG_DTIME'] == y].index
    #    data.loc[t_data_index , 'time_idx'] = x
    data["time_idx"] =  \
    data["REG_DTIME"].dt.year * 365*24 + \
    data["REG_DTIME"].dt.day_of_year * 24  + \
    data["REG_DTIME"].dt.hour 
    data["time_idx"] -= data["time_idx"].min()
    # trainer에 넣기 위해서 category로 만들기
    data['h_dong'] = data['h_dong'].astype(str)
    data['DOW'] = data['DOW'].astype(str)
    data['HOD'] = data['HOD'].astype(str)
    data['MOY'] = data['MOY'].astype(str)
    data['precip_form'] = data['precip_form'].astype(str)
    data['isHoliday'] = data['isHoliday'].astype(str)
    # ewma 적용
    if unit == 0:
        return data
    else:
        data = ewma_fun(data,unit)
        return data    
    

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