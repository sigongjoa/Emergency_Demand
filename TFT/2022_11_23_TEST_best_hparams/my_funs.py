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


def moving_average_alpha(df: pd.DataFrame, unit: float):
    '''
    training datset에 ewma를 적용하는 함수 , unit = 0.3 or 0.4
    '''
    ret_df = pd.DataFrame()
    max_value = df['count'].max()
    # forward df
    for dong in df['h_dong'].unique():
        dong_df = df[df['h_dong'] == dong]
        inv_dong_df = df[df['h_dong'] == dong][::-1]
        dong_df['count'] = dong_df['count'].ewm(alpha = unit).mean()
        for_ewma = dong_df['count'].ewm(alpha = unit).mean()
        back_ewma = inv_dong_df['count'].ewm(alpha = unit).mean()

        ewma = (for_ewma + back_ewma)/2 
        df['count'][dong_df.index] = ewma 
    df['count'] = df['count']  /  df['count'].max() * (max_value)
    
    #print(max_value , df['count'].max())
    return df

def data_processing(path : str  , unit : float):
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
        data = moving_average_alpha(data,unit)
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

def get_val_dataloader(training , data , time_idx):
    batch_size= 32    
    test_data = data[data['time_idx'] < time_idx]
    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=1)
    return val_dataloader

def confusion_matrix_plot(training, tft ,data, dong , title):
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

    ax0.set_title(f'{dong}_plot' , fontsize = 20)
    ax2.set_title('confusion_matrix', fontsize = 20)
    ax1.set_title('recall & precision , MAE', fontsize = 20)

    # 원하는 동 선택이후  예측할려는 기간의 count와 index(REG_DTIME)를 가져옴
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
    
    # tft가 예측한 값 plot(rounding 적용 X)
    ax0.plot(df_index,predition5 , label = 'preidction' , alpha=0.4)

    # 원래의 발생 빈도 값을 round 적용 및 T/F 화
    org_count = org_count.to_numpy().round()
    org_count = np.logical_not(org_count < 1)
    # prediction 값을 round 적용 및 T/F 화
    predition5 = predition5.round()
    predition5 = np.logical_not(predition5 < 1)
    
    # round된 두 값을 plot로 시각화
    ax0.plot(df_index,org_count , label = 'orginal target' , alpha=1)
    ax0.plot(df_index,predition5 ,label = '90% prediction round', alpha= 0.6, linestyle='--')
    ax0.legend()

    # confusion martix 계산 및 heatmap 시각화
    tn, fp, fn, tp = metrics.confusion_matrix(org_count,predition5).ravel()

    c_mat = np.array([[tp , fp],[fn , tn]])
    ax1 = sns.heatmap(c_mat, annot=True, cbar = False, fmt='g',cmap='Blues')

    f1_score = metrics.f1_score(org_count,predition5)
    recall = metrics.recall_score(org_count,predition5)
    precision = metrics.precision_score(org_count,predition5)
    MAE = abs(recall - precision)

    ax2.text(0.1, 3.1 , f'f1 score  {f1_score : 4f}' , fontsize=25)
    ax2.text(0.1, 3.4 , f'recall      {recall : 4f}' , fontsize=25)
    ax2.text(0.1, 3.7 , f'precision {precision : 4f}' , fontsize=25)
    ax2.text(0.1, 4.1 , f'MAE        {MAE : 4f}' , fontsize=25)
