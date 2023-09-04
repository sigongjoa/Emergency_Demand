# Test

* Goal

이 디렉토리에서는 sparsity를 해결하는 방법들을 테스트 및 관리  
각 method를 적용 후에 학습 코드에 적용할 수 있도록 만듬  

* file management

기존의 언제 작업했는지 모르는 notebook들에 대해서는 `old_*`로 해놓음  
이후 파일들은 `date_method` 형식으로 저장  

* Todo
`old`파일에 대하여 내용 정리해서 readme 파일에 저장


### olds

추후 추가

### sparsity loss

현재 TFT 모델에 사용하는 loss는 qunantile loss를 사용중(default)  

$\max(q \times (y_{real}-y_{\text{pred}}), (1-q) \times (y_{\text{pred}}-y_{real}))$

현재 사용하는 데이터가 sparsity 함  
이는 $y_{pred} = y_{real} = 0$ 이고 이는 계산을 했을 때 $0$이 나오고 이에 대해서 loss에 대한 계산이 이루어지지 않음  

다른 말로 표현을 하면 TN이 너무 많이 발생하고 TN이 많은 쪽으로 학습이 됨   
실제로 예측의 결과가 0을 많이 예측을 하면 원래의 데이터의 분포와 비슷해서 예측 값이 대부분 0이됨(모델이 보수적)  

우리는 여기에서 1로 예측하는 것을 늘려야 하는 상황임  
이를 위해서는 loss를 계산을 할 떄 $y_{real} > 0 , y_{pred} = 0$ 경우인 FP의 정보가 많이 반영되도록 변경  

* round method

confuison matrix를 계산을 위해서는 실수형인 real값과 pred 값에 대해서 반올림을 해야함  
round를 했을 때는 값이 0또는 1로 해서 기울기를 계산을 할 수 없는 상태 `grad error`가 뜸  
round를 결정을 해줘야 하는데 이는 적잘한 threshold를 설정하기가 함듬  

counfusion matrix에 넣기 위해서 dtype 변경하고 device 바꾸고 할게 많아서 에러가 많이 뜸  
float형 데이터에서의 confusion matrix를 정의해서 사용할 필요가 있음  

> html : http://202.31.200.194:8888/view/NPLAB-NAS/Members/SEO/Emergency_Demand/Traffic_Accient/TFT/Long%20Beach_5/Test/2023_09_03_sparsity%20loss_round.html

* wieght loss method

기존의 loss 대신에 가중치를 변경을 하면 confusion matrix의 정보를 어느정도 사용 할 수 있음  


| $real$ | $pred$ | $confusion matrix$ | $real - 1.5*pred$ |
|------|------|------------------|-------------------|
| 0    | 0    | TN               | 0                 |
| 0    | 0.5  |                  | 0.75              |
| 0    | 1    | FP               | 1.5               |
| 1    | 0    | FN               | 1                 |
| 1    | 0.5  |                  | 0.25              |
| 1    | 1    | TP               | 0.5               |

다음과 같은 상황에서 pred값이 0.5인 경우에는 loss가 적은 쪽인 real 값이 1인 경우로 가고 이는 앞에서 말한 상황을 재현 할 수 있음  
confusion matrix보다 코드의 변경이 적어서 에러가 별로 없음  

> test_noteboook : http://202.31.200.194:8888/notebooks/NPLAB-NAS/Members/SEO/Emergency_Demand/Traffic_Accient/TFT/Long%20Beach_5/Test/2023_09_03_sparsity%20loss.ipynb#