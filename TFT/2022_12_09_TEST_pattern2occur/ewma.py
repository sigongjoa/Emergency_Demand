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

dongs = ['강남동', '교  동', '근화동', '남  면', '남산면', '동  면', '동내면', '동산면', 
         '북산면','사북면', '서  면', '석사동', '소양동', '신동면', '신북읍', '신사우동', '약사명동', '조운동','퇴계동', 
         '효자1동', '후평1동']
    

def moving_average_alpha_both(df: pd.DataFrame, unit: float):
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(alpha = unit).mean()

        inv_dong_df = dong_df[::-1]
        for_ewma = inv_dong_df['count'].ewm(alpha = unit).mean()
        
        ewma = (for_ewma + back_ewma) / 2
        ewma = ewma / ewma.max() * max_value
        df['count'][dong_df.index] = ewma 
    return df

def moving_average_alpha_noise(df: pd.DataFrame, unit: float):
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        max_value = dong_df['count'].max()
        back_ewma = dong_df['count'].ewm(alpha = unit).mean()

        inv_dong_df = dong_df[::-1]
        for_ewma = inv_dong_df['count'].ewm(alpha = unit).mean()
        
        ewma = (for_ewma + back_ewma) / 2
        ewma = ewma / ewma.max() * max_value
        df['count'][dong_df.index] = ewma 
    df['count'] += np.random.normal(0.02,0.04, len(df))
    return df

def moving_average_hour(df: pd.DataFrame, unit = 0):
    for dong in df['h_dong'].unique():
        #print(dong)
        dong_df = df[df['h_dong'] == dong]
        over0 = dong_df[dong_df['count'] > 0].index
        for idx in over0:
            #print(idx , df['time_idx'].loc[idx])
            value = df['count'].loc[idx] 
            try:
                df['count'].loc[idx-21] += value / 2
                df['count'].loc[idx+21] += value / 2
            except:
                pass    
    return df

def data_processing(path : str  , unit : float , ewma_fun ):
    # path  : 데이터가 저장된 경로
    # unit  : ewma의 unit
    
    data = pd.read_csv(path)    
    # 사용하는 column만 선택
    data = data[['REG_DTIME', 'h_dong', 'count', 'pops', 'windspd','humid', 'temp', 'precip_form', 'precip', 'isHoliday']]
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
    
