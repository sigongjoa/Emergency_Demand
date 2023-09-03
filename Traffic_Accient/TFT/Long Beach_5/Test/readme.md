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

$ \max(q \times (y_{real}-y_{\text{pred}}), (1-q) \times (y_{\text{pred}}-y_{real}))$

현재 사용하는 데이터가 sparsity 함  
이는 $y_{pred} = y_{real} = 0$ 이고 이는 계산을 했을 때 $0$이 나오고 이에 대해서 loss에 대한 계산이 이루어지지 않음  

다른 말로 표현을 하면 TN이 너무 많이 발생하고 TN이 많은 쪽으로 학습이 됨   
실제로 예측의 결과가 0을 많이 예측을 하면 원래의 데이터의 분포와 비슷해서 예측 값이 대부분 0이됨(모델이 보수적)  

우리는 여기에서 1로 예측하는 것을 늘려야 하는 상황임  
이를 위해서는 loss를 계산을 할 떄 $y_{real} > 0 , y_{pred} = 0$ 경우인 FP의 정보가 많이 반영되도록 변경  

