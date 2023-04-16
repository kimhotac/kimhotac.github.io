---
layout: post
title: "LogisticRegression"
---

# **로지스틱 회귀 정리(개인)**

* 분류 문제에 기본적인 알고리즘
* 지도 학습
* 불연속적인 집합으로 분류

기본적인 틀 자체는 선형 회귀와 같다. 단지 선형 회귀는 직선을 그어 가중치와 편차를 가깝게 학습해 간다면 로지스틱 회귀는 아래 보이는 로지스틱 곡선을 사용해 점점 가깝게 학습해 간다. 물론 계산과 내부 파라미터 등은 다르다. 한 선을 라벨링 된 데이터에 가까이 붙여 간다는 점이 비슷한 것이다.

단일선형회귀에서 사용하는 함수가 아래와 같다.

$\hat y = \theta_0 + x_1 \cdot \theta_1$

단일로지스틱 회귀에서 사용하는 함수는 아래와 같다.

$\hat y = \dfrac{1}{1+\exp{(\theta_0 + x_1 \cdot \theta_1)}}$


로지스틱 함수(Logistic function)를 해석하려면 odd와 로짓 함수(logit function)의 개념이 필요하다.

어떤 한 사건이 0과 1로 일어난다고 가정 했을 때(=베르누이 분포)
* 1이 일어난 사건의 확률을 $P$
* 0이 일어난 사건의 확률은 $1-P$

$Odd=\dfrac{P}{1-P}$

$logit function =\log (\dfrac{P}{1-P}) \$


**로짓 함수의 특징**
* $0<P≤1$
* $-∞<Y<∞$

**로짓 변환(logit transform)**

$log(Odd) =log(\dfrac{\dfrac{1}{1+\exp{(\theta_0 + x_1 \cdot \theta_1)}}}{1-\dfrac{1}{1+\exp{(\theta_0 + x_1 \cdot \theta_1)}}})=log(\exp{(\theta_0 + x_1 \cdot \theta_1)})=\theta_0 + x_1 \cdot \theta_1$

아래는 로짓 함수를 0.01과 0.99사이의 값을 그래프로 구현한 것이다.


```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.01, 0.99, 90)
plt.plot(x, np.log(x/(1-x)))
plt.xlabel("P")
plt.ylabel("Y")
plt.show()
```


    
![png](output_6_0.png)
    


아래는 로지스틱 함수를 구현해 본 것이다.


```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 101)
plt.plot(x, 1/(1+np.exp(-x)))
plt.xlabel("x")
plt.show()
```


    
![png](output_8_0.png)
    


위의 그래프 두개를 비교해 보면 로지스틱 함수와 로짓 함수가 역함수 관계라는 것을 직관적으로 알 수 있다.

로지스틱 함수를 모델에 어떻게 적용 했는가?

1. 머신러닝에서 사용하는 엄청난 스케일의 학습할 데이터중 특성을 일정한 구간으로 나눈다.
2. 구간에 해당하는 데이터를 '1'에 해당할 확률로 만든다.
3. 그러면 일정한 특성의 확률에서 급격하게 증가하는 부분이 생긴다. 그 부분에 편향과 가중치를 조정해 맞추어 가는 것이다.

위의 단일 로지스틱 함수와 비슷하게 다중 로지스틱 함수도 표현 가능하다.

$ y = \dfrac{1}{1+\exp{(\theta_0 + x_1 \cdot \theta_1+x_2⋅\theta_2+⋅⋅⋅+x_n⋅\theta_n)}} $

그러면 Odd와 로짓 함수도 간단하게 표현 할 수 있다.

$odd={\exp{(\theta_0 + x_1 \cdot \theta_1+x_2⋅\theta_2+⋅⋅⋅+x_n⋅\theta_n)}}$

$\log(odd)={{(\theta_0 + x_1 \cdot \theta_1+x_2⋅\theta_2+⋅⋅⋅+x_n⋅\theta_n)}}$

**파라메타 추정법**

* 로그-우도 함수(loglikelihood function)
* 로그 손실 함수(cross enrtopy)

로그-우도 함수는 최대화 할 수록 오차가 줄고

로그 손실은 최소화 할 수록 오차가 준다.


**기준값**

일반적으로 0.5이지만 보수적 결정이 필요 할 때 낮추어 사용한다.

아래는 단일로지스틱 함수를 여러개의 값을 넣어 구현해 본 것이다.


```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 101)
plt.plot(x, 1/(1+np.exp(-0-1*x)),label='Q0=0, Q1=1')
plt.plot(x, 1/(1+np.exp(-0-2*x)),label='Q0=0, Q1=2')
plt.plot(x, 1/(1+np.exp(-1-1*x)),label='Q0=1, Q1=1')
plt.plot(x, 1/(1+np.exp(-2-1*x)),label='Q0=2, Q1=1')
plt.plot(x, 0.5+x*0)
plt.legend(loc=2)
plt.xlabel("x")
plt.show()
```


    
![png](output_13_0.png)
    


<a class="anchor" id="0"></a>
# **파이썬을 활용한 로지스틱 회귀 튜토리얼(번역, 일부 수정)**



<a class="anchor" id="0.1"></a>
# **목차**

1.	[로지스틱 회귀란?](#1)
2.	[직관으로 로지스틱 회귀를 살펴보기](#2)
3.	[로지스틱 회귀의 가정](#3)
4.	[로지스틱 회귀의 유형](#4)
5.	[라이브러리 가져오기](#5)
6.	[데이터 세트 가져오기](#6)
7.	[탐색적 데이터 분석](#7)
8.	[특징 벡터 및 대상 변수 선언](#8)
9.	[데이터를 훈련 및 테스트 셋으로 분할](#9)
10.	[특성공항](#10)
11.	[특성 스케일링](#11)
12.	[모델 훈련](#12)
13.	[결과 예측](#13)
14.	[정확도 점수 확인](#14)
15.	[오차 행렬](#15)
16.	[분류 행렬](#16)
17.	[임계값 레벨 조정](#17)
18.	[ROC-AUC](#18)
19.	[K-폴드 교차 검증](#19)
20.	[GridSearch CV를 이용한 하이퍼파라미터 최적화](#20)
21.	[결과 및 결론](#21)
22. [참고자료](#22)

# **1. 로지스틱 회귀란?** <a class="anchor" id="1"></a>


[목차](#0.1)

분류문제에서 가장 기본적인 알고리즘 이라고 할 수 있다. 지도 학습 분류 알고리즙으로 여러 특성값을 불연속적인 집합들로 분류하는데에 사용된다.

# **2. 직관으로 로지스틱 회귀를 살펴보기** <a class="anchor" id="2"></a>


[목차](#0.1)


로지스틱 회귀 모델은 분류 목적으로 사용되는 통계 모델이다. 샘플틀의 주어지면 로지스틱 회귀 알고리즘을 통해 두개 이상의 불연속적인 집합으로 분류한다. 따라서 기본적으로 이산형이다.


로지스틱 회귀 알고리즘은 다음과 같이 작동한다.

## **Sigmoid Function**

z로 표시되는 이 예측값은 0과 1 사이의 확률 값으로 변환된다. 예측된 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용한다.. 그런 다음 이 시그모이드 함수는 모든 실제 값을 0과 1 사이의 확률 값으로 매핑한다.

머신 러닝에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용된다. 시그모이드 함수는 S자 모양의 곡선을 가지고 있다. 시그모이드 곡선이라고도 한다.

시그모이드 함수는 로지스틱 함수의 특수한 경우이다. 다음과 같은 수학 공식으로 표현된다.

그래픽으로는 다음 그래프로 시그모이드 함수를 나타낼 수 있다.

### Sigmoid Function

![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

## **Decision boundary**

시그모이드 함수는 0과 1 사이의 확률 값을 반환한다. 이 확률 값은 "0" 또는 "1"인 불연속형 클래스에 매핑된다. 이 확률 값을 불연속형 클래스(합격/불합격, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택한다. 이 임계값을 결정 경계라고 한다. 이 임계값을 초과하면 확률 값을 클래스 1에 매핑하고, 그 이하이면 클래스 0에 매핑한다.

수학적으로 다음과 같이 표현할 수 있다.

p ≥ 0.5 => 클래스 = 1

p < 0.5 => 클래스 = 0 

일반적으로 의사 결정 경계는 0.5로 설정된다. 따라서 확률 값이 0.8(> 0.5)인 경우, 이 관측값을 클래스 1에 매핑한다. 마찬가지로 확률 값이 0.2(<0.5)인 경우, 이 관측값을 클래스 0에 매핑한다. 이는 아래 그래프에 표시된다.

![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)

## **예측하기**

이제 로지스틱 회귀에서 시그모이드 함수와 결정 경계에 대해 알게 되었다 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있다. 로지스틱 회귀의 예측 함수는 관측값이 양수일 확률, 즉 예 또는 참을 반환한다. 이를 클래스 1이라고 부르며 P(class = 1)로 표시된다. 확률이 1에 가까워지면 관찰이 클래스 1에 속하고, 그렇지 않으면 클래스 0에 속한다고 모델에 대해 더 확신할 수 있다.


# **3. 로지스틱 회귀의 가정** <a class="anchor" id="3"></a>


[목차](#0.1)


로지스틱 회귀 모델에는 몇 가지 주요 가정이 필요한다. 이러한 가정은 다음과 같다.

1. 로지스틱 회귀 모델에서는 종속 변수가 이진, 다항식 또는 서수여야 한다.

2. 관찰이 서로 독립적이어야 한다. 따라서 관찰이 반복 측정에서 비롯되어서는 안 된다.

3. 로지스틱 회귀 알고리즘은 독립 변수 간의 다중공선성이 거의 또는 전혀 필요하지 않습니다. 이는 독립 변수가 서로 너무 높은 상관관계가 없어야 함을 의미한다.

4. 로지스틱 회귀 모델은 독립 변수의 선형성과 로그 확률을 가정한다.

5. 로지스틱 회귀 모형의 성공 여부는 표본 크기에 따라 달라집니다. 일반적으로 높은 정확도를 달성하기 위해서는 큰 표본 크기가 필요한다.

# **4. 로지스틱 회귀의 유형** <a class="anchor" id="4"></a>


[목차](#0.1)


로지스틱 회귀 모델은 대상 변수 범주에 따라 세 가지 그룹으로 분류할 수 있다. 이 세 그룹은 다음과 같이 설명된다.

### 1. 이원 로지스틱 회귀

이원 로지스틱 회귀에서 대상 변수에는 두 가지 범주가 있다. 범주의 일반적인 예로는 예 또는 아니오, 좋음 또는 나쁨, 참 또는 거짓, 스팸 또는 스팸 없음, 통과 또는 실패가 있다.


### 2. 다항 로지스틱 회귀

다항 로지스틱 회귀에서 대상 변수는 특정 순서가 아닌 세 개 이상의 범주를 갖는다. 따라서 세 개 이상의 명목 범주가 있다. 예시에는 사과, 망고, 오렌지, 바나나 등 과일 카테고리의 유형이 포함된다.


### 3. 서수 로지스틱 회귀

서수 로지스틱 회귀에서는 대상 변수에 세 개 이상의 서수 범주가 있다. 따라서 범주에는 내재적 순서가 있다. 예를 들어 학생의 성적을 나쁨, 보통, 좋음, 우수로 분류할 수 있다.


# **5. 라이브러리 가져오기** <a class="anchor" id="5"></a>


[목차](#0.1)


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

```

    /kaggle/input/weather-dataset-rattle-package/weatherAUS.csv
    


```python
import warnings

warnings.filterwarnings('ignore')
```

# **6. 데이터세트 가져오기** <a class="anchor" id="6"></a>


[목차](#0.1)


```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

# **7. 탐색적 데이터 분석** <a class="anchor" id="7"></a>


[목차](#0.1)


이제 데이터를 탐색하여 데이터에 대한 인사이트를 얻는다. 


```python
# view dimensions of dataset

df.shape
```




    (145460, 23)



데이터 집합에 145460개의 인스턴스와 23개의 변수가 있음을 알 수 있다.


```python
# preview the dataset

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
col_names = df.columns

col_names
```




    Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
           'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
           'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
           'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
           'Temp3pm', 'RainToday', 'RainTomorrow'],
          dtype='object')



### Drop  RISK_MM variable

데이터 세트 설명에 `RISK_MM` 기능 변수를 데이터 세트 설명에서 삭제해야 한다는 내용이 나와 있다. 따라서 
다음과 같이 삭제해야 한다.


```python
#df.drop(['RISK_MM'], axis=1, inplace=True)
```


```python
# view summary of dataset

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 23 columns):
    Date             145460 non-null object
    Location         145460 non-null object
    MinTemp          143975 non-null float64
    MaxTemp          144199 non-null float64
    Rainfall         142199 non-null float64
    Evaporation      82670 non-null float64
    Sunshine         75625 non-null float64
    WindGustDir      135134 non-null object
    WindGustSpeed    135197 non-null float64
    WindDir9am       134894 non-null object
    WindDir3pm       141232 non-null object
    WindSpeed9am     143693 non-null float64
    WindSpeed3pm     142398 non-null float64
    Humidity9am      142806 non-null float64
    Humidity3pm      140953 non-null float64
    Pressure9am      130395 non-null float64
    Pressure3pm      130432 non-null float64
    Cloud9am         89572 non-null float64
    Cloud3pm         86102 non-null float64
    Temp9am          143693 non-null float64
    Temp3pm          141851 non-null float64
    RainToday        142199 non-null object
    RainTomorrow     142193 non-null object
    dtypes: float64(16), object(7)
    memory usage: 25.5+ MB
    

### Types of variables


이 섹션에서는 데이터 집합을 범주형 변수와 숫자형 변수로 분리한다. 데이터 집합에는 범주형 변수와 숫자형 변수가 혼합되어 있다. 범주형 변수는 데이터 유형이 객체이다. 수치형 특성는 데이터 유형이 float64이다.


우선 범주형 변수를 찾아보겠다.


```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

    There are 7 categorical variables
    
    The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    


```python
# view the categorical variables

df[categorical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### Summary of categorical variables


- 날짜 변수가 있다. 날짜` 열로 표시된다.


- 6개의 범주형 변수가 있다. `RainTomorrow`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`, `Location`이 있다.


- 이진 범주형 변수는 `RainToday`와 `RainTomorrow` 두 가지가 있다.


- `RainTomorrow`가 대상 변수이다.

## 범주형 변수 내에서 문제 탐색


먼저 범주형 변수를 살펴 보겠다.


### 범주형 변수의 누락된 값


```python
# check missing values in categorical variables

df[categorical].isnull().sum()
```




    Date                0
    Location            0
    WindGustDir     10326
    WindDir9am      10566
    WindDir3pm       4228
    RainToday        3261
    RainTomorrow     3267
    dtype: int64




```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

    WindGustDir     10326
    WindDir9am      10566
    WindDir3pm       4228
    RainToday        3261
    RainTomorrow     3267
    dtype: int64
    

데이터 집합에 결측값이 포함된 범주형 변수가 4개만 있음을 알 수 있다. 이들은 `WindGustDir`, `WindDir9am`, `WindDir3pm` 및 `RainToday`이다.

### 범주형 변수의 빈도 수


이제 범주형 변수의 빈도 수를 확인해 보겠다.


```python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```

    2013-08-25    49
    2016-08-01    49
    2015-04-14    49
    2014-09-07    49
    2013-06-28    49
                  ..
    2008-01-19     1
    2008-01-13     1
    2007-12-30     1
    2008-01-15     1
    2007-11-21     1
    Name: Date, Length: 3436, dtype: int64
    Canberra            3436
    Sydney              3344
    Melbourne           3193
    Hobart              3193
    Brisbane            3193
    Adelaide            3193
    Perth               3193
    Darwin              3193
    Wollongong          3040
    Townsville          3040
    Albury              3040
    Cairns              3040
    Albany              3040
    GoldCoast           3040
    Ballarat            3040
    Bendigo             3040
    Launceston          3040
    AliceSprings        3040
    MountGinini         3040
    MountGambier        3040
    Tuggeranong         3039
    Newcastle           3039
    Penrith             3039
    SydneyAirport       3009
    Cobar               3009
    Woomera             3009
    Portland            3009
    WaggaWagga          3009
    Richmond            3009
    PerthAirport        3009
    Mildura             3009
    NorfolkIsland       3009
    MelbourneAirport    3009
    Moree               3009
    Dartmoor            3009
    Watsonia            3009
    Sale                3009
    BadgerysCreek       3009
    PearceRAAF          3009
    Nuriootpa           3009
    Williamtown         3009
    Witchcliffe         3009
    CoffsHarbour        3009
    Walpole             3006
    NorahHead           3004
    SalmonGums          3001
    Katherine           1578
    Uluru               1578
    Nhil                1578
    Name: Location, dtype: int64
    W      9915
    SE     9418
    N      9313
    SSE    9216
    E      9181
    S      9168
    WSW    9069
    SW     8967
    SSW    8736
    WNW    8252
    NW     8122
    ENE    8104
    ESE    7372
    NE     7133
    NNW    6620
    NNE    6548
    Name: WindGustDir, dtype: int64
    N      11758
    SE      9287
    E       9176
    SSE     9112
    NW      8749
    S       8659
    W       8459
    SW      8423
    NNE     8129
    NNW     7980
    ENE     7836
    NE      7671
    ESE     7630
    SSW     7587
    WNW     7414
    WSW     7024
    Name: WindDir9am, dtype: int64
    SE     10838
    W      10110
    S       9926
    WSW     9518
    SSE     9399
    SW      9354
    N       8890
    WNW     8874
    NW      8610
    ESE     8505
    E       8472
    NE      8263
    SSW     8156
    NNW     7870
    ENE     7857
    NNE     6590
    Name: WindDir3pm, dtype: int64
    No     110319
    Yes     31880
    Name: RainToday, dtype: int64
    No     110316
    Yes     31877
    Name: RainTomorrow, dtype: int64
    


```python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

    2013-08-25    0.000337
    2016-08-01    0.000337
    2015-04-14    0.000337
    2014-09-07    0.000337
    2013-06-28    0.000337
                    ...   
    2008-01-19    0.000007
    2008-01-13    0.000007
    2007-12-30    0.000007
    2008-01-15    0.000007
    2007-11-21    0.000007
    Name: Date, Length: 3436, dtype: float64
    Canberra            0.023622
    Sydney              0.022989
    Melbourne           0.021951
    Hobart              0.021951
    Brisbane            0.021951
    Adelaide            0.021951
    Perth               0.021951
    Darwin              0.021951
    Wollongong          0.020899
    Townsville          0.020899
    Albury              0.020899
    Cairns              0.020899
    Albany              0.020899
    GoldCoast           0.020899
    Ballarat            0.020899
    Bendigo             0.020899
    Launceston          0.020899
    AliceSprings        0.020899
    MountGinini         0.020899
    MountGambier        0.020899
    Tuggeranong         0.020892
    Newcastle           0.020892
    Penrith             0.020892
    SydneyAirport       0.020686
    Cobar               0.020686
    Woomera             0.020686
    Portland            0.020686
    WaggaWagga          0.020686
    Richmond            0.020686
    PerthAirport        0.020686
    Mildura             0.020686
    NorfolkIsland       0.020686
    MelbourneAirport    0.020686
    Moree               0.020686
    Dartmoor            0.020686
    Watsonia            0.020686
    Sale                0.020686
    BadgerysCreek       0.020686
    PearceRAAF          0.020686
    Nuriootpa           0.020686
    Williamtown         0.020686
    Witchcliffe         0.020686
    CoffsHarbour        0.020686
    Walpole             0.020665
    NorahHead           0.020652
    SalmonGums          0.020631
    Katherine           0.010848
    Uluru               0.010848
    Nhil                0.010848
    Name: Location, dtype: float64
    W      0.068163
    SE     0.064746
    N      0.064024
    SSE    0.063358
    E      0.063117
    S      0.063028
    WSW    0.062347
    SW     0.061646
    SSW    0.060058
    WNW    0.056730
    NW     0.055837
    ENE    0.055713
    ESE    0.050681
    NE     0.049038
    NNW    0.045511
    NNE    0.045016
    Name: WindGustDir, dtype: float64
    N      0.080833
    SE     0.063846
    E      0.063083
    SSE    0.062643
    NW     0.060147
    S      0.059528
    W      0.058153
    SW     0.057906
    NNE    0.055885
    NNW    0.054860
    ENE    0.053870
    NE     0.052736
    ESE    0.052454
    SSW    0.052159
    WNW    0.050969
    WSW    0.048288
    Name: WindDir9am, dtype: float64
    SE     0.074508
    W      0.069504
    S      0.068239
    WSW    0.065434
    SSE    0.064616
    SW     0.064306
    N      0.061116
    WNW    0.061006
    NW     0.059192
    ESE    0.058470
    E      0.058243
    NE     0.056806
    SSW    0.056070
    NNW    0.054104
    ENE    0.054015
    NNE    0.045305
    Name: WindDir3pm, dtype: float64
    No     0.758415
    Yes    0.219167
    Name: RainToday, dtype: float64
    No     0.758394
    Yes    0.219146
    Name: RainTomorrow, dtype: float64
    

### 레이블 개수: cardinality


범주형 변수 내의 레이블 수를 **카디널리티**라고 한다. 변수 내의 레이블 수가 많은 것을 **높은 카디널리티**라고 한다. 높은 카디널리티를 확인해 보겠다.


```python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

    Date  contains  3436  labels
    Location  contains  49  labels
    WindGustDir  contains  17  labels
    WindDir9am  contains  17  labels
    WindDir3pm  contains  17  labels
    RainToday  contains  3  labels
    RainTomorrow  contains  3  labels
    

전처리가 필요한 `Date` 변수가 있음을 알 수 있다. 다음 섹션에서 전처리를 해보겠다.


다른 모든 변수는 상대적으로 적은 수의 변수를 포함하고 있다.

### 날짜 변수의 기능 엔지니어링


```python
df['Date'].dtypes
```




    dtype('O')



날짜` 변수의 데이터 타입이 객체임을 알 수 있다. 현재 객체로 코딩된 날짜를 날짜/시간 형식으로 파싱하겠다.


```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```


```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```




    0    2008
    1    2008
    2    2008
    3    2008
    4    2008
    Name: Year, dtype: int64




```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```




    0    12
    1    12
    2    12
    3    12
    4    12
    Name: Month, dtype: int64




```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```




    0    1
    1    2
    2    3
    3    4
    4    5
    Name: Day, dtype: int64




```python
# again view the summary of dataset

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 26 columns):
    Date             145460 non-null datetime64[ns]
    Location         145460 non-null object
    MinTemp          143975 non-null float64
    MaxTemp          144199 non-null float64
    Rainfall         142199 non-null float64
    Evaporation      82670 non-null float64
    Sunshine         75625 non-null float64
    WindGustDir      135134 non-null object
    WindGustSpeed    135197 non-null float64
    WindDir9am       134894 non-null object
    WindDir3pm       141232 non-null object
    WindSpeed9am     143693 non-null float64
    WindSpeed3pm     142398 non-null float64
    Humidity9am      142806 non-null float64
    Humidity3pm      140953 non-null float64
    Pressure9am      130395 non-null float64
    Pressure3pm      130432 non-null float64
    Cloud9am         89572 non-null float64
    Cloud3pm         86102 non-null float64
    Temp9am          143693 non-null float64
    Temp3pm          141851 non-null float64
    RainToday        142199 non-null object
    RainTomorrow     142193 non-null object
    Year             145460 non-null int64
    Month            145460 non-null int64
    Day              145460 non-null int64
    dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
    memory usage: 28.9+ MB
    

`Date` 변수에서 세 개의 열이 추가로 생성된 것을 볼 수 있다. 이제 데이터 집합에서 원래의 `Date` 변수를 삭제하겠다.


```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```


```python
# preview the dataset again

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



이제 데이터 집합에서 `Date` 변수가 제거된 것을 볼 수 있다.


### 범주형 변수 살펴보기


이제 범주형 변수를 하나씩 살펴보겠다. 


```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

    There are 6 categorical variables
    
    The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    

데이터 집합에 6개의 범주형 변수가 있음을 알 수 있다. `Date` 변수는 제거되었다. 먼저 범주형 변수의 결측값을 확인하겠다.


```python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```




    Location            0
    WindGustDir     10326
    WindDir9am      10566
    WindDir3pm       4228
    RainToday        3261
    RainTomorrow     3267
    dtype: int64



`WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` 변수에 누락된 값이 있는 것을 알 수 있다. 이러한 변수를 하나씩 살펴보겠다.

### Explore `Location` variable


```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```

    Location contains 49 labels
    


```python
# check labels in location variable

df.Location.unique()
```




    array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
           'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
           'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
           'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
           'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
           'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
           'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
           'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
           'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
           'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)




```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```




    Canberra            3436
    Sydney              3344
    Melbourne           3193
    Hobart              3193
    Brisbane            3193
    Adelaide            3193
    Perth               3193
    Darwin              3193
    Wollongong          3040
    Townsville          3040
    Albury              3040
    Cairns              3040
    Albany              3040
    GoldCoast           3040
    Ballarat            3040
    Bendigo             3040
    Launceston          3040
    AliceSprings        3040
    MountGinini         3040
    MountGambier        3040
    Tuggeranong         3039
    Newcastle           3039
    Penrith             3039
    SydneyAirport       3009
    Cobar               3009
    Woomera             3009
    Portland            3009
    WaggaWagga          3009
    Richmond            3009
    PerthAirport        3009
    Mildura             3009
    NorfolkIsland       3009
    MelbourneAirport    3009
    Moree               3009
    Dartmoor            3009
    Watsonia            3009
    Sale                3009
    BadgerysCreek       3009
    PearceRAAF          3009
    Nuriootpa           3009
    Williamtown         3009
    Witchcliffe         3009
    CoffsHarbour        3009
    Walpole             3006
    NorahHead           3004
    SalmonGums          3001
    Katherine           1578
    Uluru               1578
    Nhil                1578
    Name: Location, dtype: int64




```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



### Explore `WindGustDir` variable


```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

    WindGustDir contains 17 labels
    


```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```




    array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', nan, 'ENE',
           'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'], dtype=object)




```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```




    W      9915
    SE     9418
    N      9313
    SSE    9216
    E      9181
    S      9168
    WSW    9069
    SW     8967
    SSW    8736
    WNW    8252
    NW     8122
    ENE    8104
    ESE    7372
    NE     7133
    NNW    6620
    NNE    6548
    Name: WindGustDir, dtype: int64




```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```




    ENE     8104
    ESE     7372
    N       9313
    NE      7133
    NNE     6548
    NNW     6620
    NW      8122
    S       9168
    SE      9418
    SSE     9216
    SSW     8736
    SW      8967
    W       9915
    WNW     8252
    WSW     9069
    NaN    10326
    dtype: int64



WindGustDir 변수에 9330개의 누락된 값이 있음을 알 수 있다.

### Explore `WindDir9am` variable


```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

    WindDir9am contains 17 labels
    


```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```




    array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
           'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)




```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```




    N      11758
    SE      9287
    E       9176
    SSE     9112
    NW      8749
    S       8659
    W       8459
    SW      8423
    NNE     8129
    NNW     7980
    ENE     7836
    NE      7671
    ESE     7630
    SSW     7587
    WNW     7414
    WSW     7024
    Name: WindDir9am, dtype: int64




```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```




    ENE     7836
    ESE     7630
    N      11758
    NE      7671
    NNE     8129
    NNW     7980
    NW      8749
    S       8659
    SE      9287
    SSE     9112
    SSW     7587
    SW      8423
    W       8459
    WNW     7414
    WSW     7024
    NaN    10566
    dtype: int64



WindDir9am` 변수에 10013개의 누락된 값이 있음을 알 수 있다.

### Explore `WindDir3pm` variable


```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

    WindDir3pm contains 17 labels
    


```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```




    array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
           'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)




```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```




    SE     10838
    W      10110
    S       9926
    WSW     9518
    SSE     9399
    SW      9354
    N       8890
    WNW     8874
    NW      8610
    ESE     8505
    E       8472
    NE      8263
    SSW     8156
    NNW     7870
    ENE     7857
    NNE     6590
    Name: WindDir3pm, dtype: int64




```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```




    ENE     7857
    ESE     8505
    N       8890
    NE      8263
    NNE     6590
    NNW     7870
    NW      8610
    S       9926
    SE     10838
    SSE     9399
    SSW     8156
    SW      9354
    W      10110
    WNW     8874
    WSW     9518
    NaN     4228
    dtype: int64



WindDir3pm` 변수에 3778개의 누락된 값이 있다.

### Explore `RainToday` variable


```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

    RainToday contains 3 labels
    


```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```




    array(['No', 'Yes', nan], dtype=object)




```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```




    No     110319
    Yes     31880
    Name: RainToday, dtype: int64




```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```




    Yes    31880
    NaN     3261
    dtype: int64



RainToday` 변수에 1406개의 누락 값이 있다.

### 수치형 변수 탐색


```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

    There are 19 numerical variables
    
    The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
    


```python
# view the numerical variables

df[numerical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



수치형 변수 요약 ###


- 16개의 수치형 변수가 있다. 


- `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` 및 `Temp3pm`으로 제공된다.


- 모든 수치형 변수는 선형이다.

## 수치형 특성 내에서 문제 탐색하기


이제 수치형 특성에 대해 살펴보겠다.


### 수치형 특성의 누락된 값


```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```




    MinTemp           1485
    MaxTemp           1261
    Rainfall          3261
    Evaporation      62790
    Sunshine         69835
    WindGustSpeed    10263
    WindSpeed9am      1767
    WindSpeed3pm      3062
    Humidity9am       2654
    Humidity3pm       4507
    Pressure9am      15065
    Pressure3pm      15028
    Cloud9am         55888
    Cloud3pm         59358
    Temp9am           1767
    Temp3pm           3609
    Year                 0
    Month                0
    Day                  0
    dtype: int64



16개의 수치형 특성에 모두 결측값이 포함되어 있음을 알 수 있다.

### 수치형 변수의 이상치


```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

            MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
    count  143975.0  144199.0  142199.0      82670.0   75625.0       135197.0   
    mean       12.0      23.0       2.0          5.0       8.0           40.0   
    std         6.0       7.0       8.0          4.0       4.0           14.0   
    min        -8.0      -5.0       0.0          0.0       0.0            6.0   
    25%         8.0      18.0       0.0          3.0       5.0           31.0   
    50%        12.0      23.0       0.0          5.0       8.0           39.0   
    75%        17.0      28.0       1.0          7.0      11.0           48.0   
    max        34.0      48.0     371.0        145.0      14.0          135.0   
    
           WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
    count      143693.0      142398.0     142806.0     140953.0     130395.0   
    mean           14.0          19.0         69.0         52.0       1018.0   
    std             9.0           9.0         19.0         21.0          7.0   
    min             0.0           0.0          0.0          0.0        980.0   
    25%             7.0          13.0         57.0         37.0       1013.0   
    50%            13.0          19.0         70.0         52.0       1018.0   
    75%            19.0          24.0         83.0         66.0       1022.0   
    max           130.0          87.0        100.0        100.0       1041.0   
    
           Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
    count     130432.0   89572.0   86102.0  143693.0  141851.0  145460.0   
    mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
    std            7.0       3.0       3.0       6.0       7.0       3.0   
    min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
    25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
    50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
    75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
    max         1040.0       9.0       9.0      40.0      47.0    2017.0   
    
              Month       Day  
    count  145460.0  145460.0  
    mean        6.0      16.0  
    std         3.0       9.0  
    min         1.0       1.0  
    25%         3.0       8.0  
    50%         6.0      16.0  
    75%         9.0      23.0  
    max        12.0      31.0   2
    

자세히 살펴보면 `Rainfall`, `Evaporation`, `WindSpeed9am`, `WindSpeed3pm` 열에 이상값이 포함될 수 있음을 알 수 있다.


위 변수의 이상값을 시각화하기 위해 박스 플롯을 그려보겠다. 


```python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```




    Text(0, 0.5, 'WindSpeed3pm')




    
![png](output_112_1.png)
    


위의 박스 플롯을 보면 이러한 변수에 많은 이상값이 있음을 확인할 수 있다.

### 변수 분포 확인


이제 히스토그램을 그려서 분포를 확인하여 정규 분포인지 왜곡된 분포인지 확인하겠다. 변수가 정규 분포를 따른다면 '극값 분석'을 하고, 왜곡되어 있다면 IQR(사분위수 범위)을 찾는다.


```python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```




    Text(0, 0.5, 'RainTomorrow')




    
![png](output_115_1.png)
    


네 가지 변수가 모두 왜곡되어 있음을 알 수 있다. 따라서 사분위수 간 범위를 사용하여 이상값을 찾는다.


```python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    Rainfall outliers are values < -2.4000000000000004 or > 3.2
    

`Rainfall`의 경우 최소값과 최대값은 0.0과 371.0이다. 따라서 이상값은 3.2를 초과하는 값이다.


```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004
    

`증발`의 경우 최소값과 최대값은 0.0과 145.0이다. 따라서 이상값은 21.8을 초과하는 값이다.


```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    WindSpeed9am outliers are values < -29.0 or > 55.0
    

`WindSpeed9am`의 경우 최소값과 최대값은 0.0과 130.0이다. 따라서 이상값은 55.0을 초과하는 값이다.


```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    WindSpeed3pm outliers are values < -20.0 or > 57.0
    

`WindSpeed3pm`의 경우 최소값과 최대값은 0.0과 87.0이다. 따라서 이상값은 57.0을 초과하는 값이다.

# **8. 특징 벡터 및 대상 변수 선언** <a class="anchor" id="8"></a>


[목차](#0.1)


```python
df.dropna(axis=0, subset=['RainTomorrow'], inplace = True)
df.isnull().sum()
```




    Location             0
    MinTemp            637
    MaxTemp            322
    Rainfall          1406
    Evaporation      60843
    Sunshine         67816
    WindGustDir       9330
    WindGustSpeed     9270
    WindDir9am       10013
    WindDir3pm        3778
    WindSpeed9am      1348
    WindSpeed3pm      2630
    Humidity9am       1774
    Humidity3pm       3610
    Pressure9am      14014
    Pressure3pm      13981
    Cloud9am         53657
    Cloud3pm         57094
    Temp9am            904
    Temp3pm           2726
    RainToday         1406
    RainTomorrow         0
    Year                 0
    Month                0
    Day                  0
    dtype: int64




```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

# **9. 데이터를 별도의 학습과 테스트 셋으로 분할** <a class="anchor" id="9"></a>


[목차](#0.1)


```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```


```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```




    ((113754, 24), (28439, 24))



# **10. 특성공학** <a class="anchor" id="10"></a>


[목차](#0.1)


**특성공학**은 원시 데이터를 유용한 특성으로 변환하여 모델을 더 잘 이해하고 예측력을 높이는 데 도움이 되는 프로세스를 말한다. 여기서는 다양한 유형의 변수에 대해 특성공학을 적용하겠다.


먼저 범주형 변수와 수치형 변수를 다시 구분하여 표시해 보겠다.


```python
# check data types in X_train

X_train.dtypes
```




    Location          object
    MinTemp          float64
    MaxTemp          float64
    Rainfall         float64
    Evaporation      float64
    Sunshine         float64
    WindGustDir       object
    WindGustSpeed    float64
    WindDir9am        object
    WindDir3pm        object
    WindSpeed9am     float64
    WindSpeed3pm     float64
    Humidity9am      float64
    Humidity3pm      float64
    Pressure9am      float64
    Pressure3pm      float64
    Cloud9am         float64
    Cloud3pm         float64
    Temp9am          float64
    Temp3pm          float64
    RainToday         object
    Year               int64
    Month              int64
    Day                int64
    dtype: object




```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```




    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']




```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```




    ['MinTemp',
     'MaxTemp',
     'Rainfall',
     'Evaporation',
     'Sunshine',
     'WindGustSpeed',
     'WindSpeed9am',
     'WindSpeed3pm',
     'Humidity9am',
     'Humidity3pm',
     'Pressure9am',
     'Pressure3pm',
     'Cloud9am',
     'Cloud3pm',
     'Temp9am',
     'Temp3pm',
     'Year',
     'Month',
     'Day']



### 수치형 변수의 엔지니어링 결측치 엔지니어링


```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```




    MinTemp            495
    MaxTemp            264
    Rainfall          1139
    Evaporation      48718
    Sunshine         54314
    WindGustSpeed     7367
    WindSpeed9am      1086
    WindSpeed3pm      2094
    Humidity9am       1449
    Humidity3pm       2890
    Pressure9am      11212
    Pressure3pm      11186
    Cloud9am         43137
    Cloud3pm         45768
    Temp9am            740
    Temp3pm           2171
    Year                 0
    Month                0
    Day                  0
    dtype: int64




```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```




    MinTemp            142
    MaxTemp             58
    Rainfall           267
    Evaporation      12125
    Sunshine         13502
    WindGustSpeed     1903
    WindSpeed9am       262
    WindSpeed3pm       536
    Humidity9am        325
    Humidity3pm        720
    Pressure9am       2802
    Pressure3pm       2795
    Cloud9am         10520
    Cloud3pm         11326
    Temp9am            164
    Temp3pm            555
    Year                 0
    Month                0
    Day                  0
    dtype: int64




```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

    MinTemp 0.0044
    MaxTemp 0.0023
    Rainfall 0.01
    Evaporation 0.4283
    Sunshine 0.4775
    WindGustSpeed 0.0648
    WindSpeed9am 0.0095
    WindSpeed3pm 0.0184
    Humidity9am 0.0127
    Humidity3pm 0.0254
    Pressure9am 0.0986
    Pressure3pm 0.0983
    Cloud9am 0.3792
    Cloud3pm 0.4023
    Temp9am 0.0065
    Temp3pm 0.0191
    

### 가정


데이터가 완전히 무작위로 누락되었다고 가정한다(MCAR). 결측값을 추정하는 데 사용할 수 있는 두 가지 방법이 있다. 하나는 평균 또는 중앙값 추정이고 다른 하나는 무작위 샘플 추정이다. 데이터 집합에 이상값이 있는 경우 중앙값 보정을 사용해야 한다. 중앙값 대입은 이상치에 강하기 때문에 중앙값 대입을 사용하겠다.


데이터의 적절한 통계적 측정값(이 경우 중앙값)을 사용하여 결측치를 추정한다. 추정은 훈련 집합에 대해 수행한 다음 테스트 집합으로 전파해야 한다. 즉, 훈련 세트와 테스트 세트 모두에서 결측치를 채우는 데 사용할 통계적 측정값은 훈련 세트에서만 추출해야 한다. 이는 과적합을 피하기 위한 것이다.


```python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
      
```


```python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```




    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64




```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```




    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64



이제 학습 및 테스트 집합의 숫자 열에 누락된 값이 없음을 알 수 있다.


```python
y_train.isnull().sum()
```




    0




```python
y_test.isnull().sum()
```




    0



### 범주형 변수의 엔지니어링 결측치 설계


```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```




    Location       0.000000
    WindGustDir    0.065114
    WindDir9am     0.070134
    WindDir3pm     0.026443
    RainToday      0.010013
    dtype: float64




```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

    WindGustDir 0.06511419378659213
    WindDir9am 0.07013379749283542
    WindDir3pm 0.026443026179299188
    RainToday 0.01001283471350458
    


```python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```




    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64




```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```




    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64



최종 점검으로 X_train과 X_test에서 누락된 값이 있는지 확인한다.


```python
# check missing values in X_train

X_train.isnull().sum()
```




    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64




```python
# check missing values in X_test

X_test.isnull().sum()
```




    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64



X_train과 X_test에 누락된 값이 없음을 알 수 있다.

### 수치형 특성의 엔지니어링 이상값


`Rainfall`, `Evaporation`, `WindSpeed9am`, `WindSpeed3pm`열에 이상값이 포함되어 있음을 확인했다. 위 변수의 최대값을 제한하고 이상값을 제거하기 위해 탑 코딩 방식을 사용하겠다.


```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```




    (3.2, 3.2)




```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```




    (21.8, 21.8)




```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```




    (55.0, 55.0)




```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```




    (57.0, 57.0)




```python
X_train[numerical].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>1017.640649</td>
      <td>1015.241101</td>
      <td>4.651801</td>
      <td>4.703588</td>
      <td>16.995062</td>
      <td>21.688643</td>
      <td>2012.759727</td>
      <td>6.404021</td>
      <td>15.710419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>6.738680</td>
      <td>6.675168</td>
      <td>2.292726</td>
      <td>2.117847</td>
      <td>6.463772</td>
      <td>6.855649</td>
      <td>2.540419</td>
      <td>3.427798</td>
      <td>8.796821</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.300000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>




이제 `Rainfall`, `Evaporation`, `WindSpeed9am`, `WindSpeed3pm` 열의 이상값에 상한이 설정된 것을 볼 수 있다.

### 범주형 특성 인코딩


```python
categorical
```




    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']




```python
X_train[categorical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113462</th>
      <td>Witchcliffe</td>
      <td>S</td>
      <td>SSE</td>
      <td>S</td>
      <td>No</td>
    </tr>
    <tr>
      <th>89638</th>
      <td>Cairns</td>
      <td>ENE</td>
      <td>SSE</td>
      <td>SE</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>138130</th>
      <td>AliceSprings</td>
      <td>E</td>
      <td>NE</td>
      <td>N</td>
      <td>No</td>
    </tr>
    <tr>
      <th>87898</th>
      <td>Cairns</td>
      <td>ESE</td>
      <td>SSE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16484</th>
      <td>Newcastle</td>
      <td>W</td>
      <td>N</td>
      <td>SE</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113462</th>
      <td>Witchcliffe</td>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>S</td>
      <td>41.0</td>
      <td>SSE</td>
      <td>S</td>
      <td>...</td>
      <td>1013.4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>18.8</td>
      <td>20.4</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>89638</th>
      <td>Cairns</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>ENE</td>
      <td>33.0</td>
      <td>SSE</td>
      <td>SE</td>
      <td>...</td>
      <td>1013.1</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>26.4</td>
      <td>27.5</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>138130</th>
      <td>AliceSprings</td>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>E</td>
      <td>31.0</td>
      <td>NE</td>
      <td>N</td>
      <td>...</td>
      <td>1013.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.5</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>87898</th>
      <td>Cairns</td>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>ESE</td>
      <td>37.0</td>
      <td>SSE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.8</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>27.3</td>
      <td>29.4</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>16484</th>
      <td>Newcastle</td>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>W</td>
      <td>39.0</td>
      <td>N</td>
      <td>SE</td>
      <td>...</td>
      <td>1015.2</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>22.2</td>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



`RainToday` 변수에서 `RainToday_0`과 `RainToday_1`이라는 두 개의 변수가 추가로 생성된 것을 확인할 수 있다.

이제 `X_train` 훈련 집합을 생성하겠다.


```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113462</th>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>28.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89638</th>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>138130</th>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87898</th>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>59.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16484</th>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>



마찬가지로 `X_test` 테스트 집합을 생성한다.


```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>88578</th>
      <td>17.4</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>11.1</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>63.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59016</th>
      <td>6.8</td>
      <td>14.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>8.5</td>
      <td>46.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>80.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127049</th>
      <td>10.1</td>
      <td>15.4</td>
      <td>3.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>120886</th>
      <td>14.4</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.6</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136649</th>
      <td>6.8</td>
      <td>14.3</td>
      <td>3.2</td>
      <td>0.2</td>
      <td>7.3</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>92.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>



이제 모델 구축을 위한 훈련 및 테스트가 준비되었다. 그 전에 모든 특징 변수를 동일한 스케일로 매핑해야 한다. 이를 '피처 스케일링'이라고 한다. 다음과 같이 하겠다.

# **11. 특성 스케일링** <a class="anchor" id="11"></a>


[목차](#0.1)


```python
X_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>




```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.484406</td>
      <td>0.530004</td>
      <td>0.210962</td>
      <td>0.236312</td>
      <td>0.554562</td>
      <td>0.262667</td>
      <td>0.254148</td>
      <td>0.326575</td>
      <td>0.688675</td>
      <td>0.515095</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151741</td>
      <td>0.134105</td>
      <td>0.369949</td>
      <td>0.129528</td>
      <td>0.190999</td>
      <td>0.101682</td>
      <td>0.160119</td>
      <td>0.152384</td>
      <td>0.189356</td>
      <td>0.205307</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.375297</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.479810</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.220183</td>
      <td>0.586207</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593824</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.247706</td>
      <td>0.600000</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>



이제 로지스틱 회귀 분류기에 입력할 `X_train` 데이터 집합이 준비되었다. 다음과 같이 하겠다.

# **12. 모델 학습** <a class="anchor" id="12"></a>


[목차](#0.1)


```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)

```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)



# **13. 예측** <a class="anchor" id="13"></a>


[목차](#0.1)


```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```




    array(['No', 'No', 'No', ..., 'No', 'No', 'Yes'], dtype=object)




### predict_proba 메서드


**predict_proba** 메서드는 이 경우 대상 변수(0, 1)에 대한 확률을 배열 형태로 반환한다.

'0'은 비가 오지 않을 확률, '1'은 비가 올 확률이다.


```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```




    array([0.91382428, 0.83565645, 0.82033915, ..., 0.97674285, 0.79855098,
           0.30734161])




```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```




    array([0.08617572, 0.16434355, 0.17966085, ..., 0.02325715, 0.20144902,
           0.69265839])



# **14. 정확도 점수 확인** <a class="anchor" id="14"></a>


[목차](#0.1)


```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

    Model accuracy score: 0.8502
    

여기서 **y_test**는 실제 클래스 레이블이고 **y_pred_test**는 테스트 세트에서 예측된 클래스 레이블이다.

### 훈련 세트와 테스트 세트의 정확도 비교하기


이제 훈련 세트와 테스트 세트의 정확도를 비교하여 과적합 여부를 확인하겠다.


```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```




    array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)




```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

    Training-set accuracy score: 0.8476
    

### 오버피팅 및 언더피팅 확인


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

    Training set score: 0.8476
    Test set score: 0.8502
    

훈련 세트 정확도 점수는 0.8476이고 테스트 세트 정확도는 0.8501이다. 이 두 값은 상당히 비슷하다. 따라서 과적합에 대한 의문의 여지가 없다. 



로지스틱 회귀에서는 기본값인 C = 1을 사용한다 이 값은 훈련과 테스트 집합 모두에서 약 85%의 정확도로 우수한 성능을 제공한다 그러나 훈련과 테스트 집합 모두에서 모델 성능은 매우 비슷하다 과소 적합의 경우일 가능성이 높다. 

C를 늘려서 더 유연한 모델을 맞춰 보겠다.


```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```




    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

    Training set score: 0.8478
    Test set score: 0.8505
    

C=100이 테스트 세트 정확도를 높이고 훈련 세트 정확도도 약간 증가한다는 것을 알 수 있다. 따라서 더 복잡한 모델일수록 성능이 더 좋다는 결론을 내릴 수 있다.

이제 C=0.01을 설정하여 기본값인 C=1보다 더 정규화된 모델을 사용하면 어떻게 되는지 살펴보겠다.


```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```




    LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

    Training set score: 0.8409
    Test set score: 0.8448
    

따라서 C=0.01을 설정하여 보다 정규화된 모델을 사용하면 학습 및 테스트 세트 정확도가 모두 기본 매개변수에 비해 감소한다.

### Compare model accuracy with null accuracy


따라서 모델 정확도는 0.8501이다. 하지만 위의 정확도만으로는 모델이 매우 우수하다고 말할 수는 없다. '널 정확도'와 비교해야 한다. 널 정확도는 항상 가장 빈번한 클래스를 예측함으로써 얻을 수 있는 정확도이다.

따라서 먼저 테스트 세트의 클래스 분포를 확인해야 한다. 


```python
# check class distribution in test set

y_test.value_counts()
```




    No     22067
    Yes     6372
    Name: RainTomorrow, dtype: int64



가장 빈번한 클래스의 발생 횟수가 22067회임을 알 수 있다. 따라서 22067을 총 발생 횟수로 나누면 널 정확도를 계산할 수 있다.


```python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

    Null accuracy score: 0.7759
    

모델 정확도 점수는 0.8501이지만 널 정확도 점수는 0.7759임을 알 수 있다. 따라서 로지스틱 회귀 모델이 클래스 레이블을 예측하는 데 매우 효과적이라는 결론을 내릴 수 있다.

이제 위의 분석을 바탕으로 분류 모델의 정확도가 매우 우수하다는 결론을 내릴 수 있다. 우리 모델은 클래스 레이블을 예측하는 측면에서 매우 잘 작동하고 있다.


하지만 값의 기본 분포는 제공하지 않는다. 또한 분류기가 어떤 유형의 오류를 범하고 있는지에 대해서도 알려주지 않는다. 


' 행렬'이라는 또 다른 도구가 우리를 구해준다.

# **15. 오차 행렬** <a class="anchor" id="15"></a>


[목차](#0.1)


오차행렬은 분류 알고리즘의 성능을 요약하는 도구이다. 오차행렬은 분류 모델의 성능과 모델에서 발생하는 오류 유형을 명확하게 파악할 수 있게 해준다. 오차행렬은 각 범주별로 분류된 올바른 예측과 잘못된 예측에 대한 요약을 제공한다. 요약은 표 형식으로 표시된다.


분류 모델 성능을 평가하는 동안 네 가지 유형의 결과가 나올 수 있다. 이 네 가지 결과는 다음과 같이 설명된다.


**True Positives (TP)** – True Positives는 관찰이 특정 클래스에 속할 것으로 예측하고 관찰이 실제로 해당 클래스에 속할 때 발생한다.


**True Negatives (TN)** – True Negatives는 관찰이 특정 클래스에 속하지 않는다고 예측했지만 실제로 관찰이 해당 클래스에 속하지 않을 때 발생한다.


**False Positives (FP)** – False Positives 관찰이 특정 클래스에 속한다고 예측했지만 실제로는 해당 클래스에 속하지 않을 때 발생한다.  이러한 유형의 오류를 **Type I error.**
라고 한다.


**False Negatives (FN)** – False Negatives는 관측값이 특정 클래스에 속하지 않는다고 예측했지만 실제로는 해당 클래스에 속할 때 발생한다. 이는 매우 심각한 오류이며 **Type II error.** 라고 한다.



이 네 가지 결과는 아래 혼동 매트릭스에 요약되어 있다.



```python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

    Confusion matrix
    
     [[20892  1175]
     [ 3086  3286]]
    
    True Positives(TP) =  20892
    
    True Negatives(TN) =  3286
    
    False Positives(FP) =  1175
    
    False Negatives(FN) =  3086
    

오차행렬은 `20892 + 3285 = 24177개의 올바른 예측`과 `3087 + 1175 = 4262개의 잘못된 예측`을 보여준다.


이 경우, 우리는


- `True Positives` (실제 양성:1  예측 양성:1) - 20892


- `True Negatives` (실제 음성:0  예측 음성:0) - 3285


- `False Positives` (실제 음성:0  예측 양성:1) - 1175 `(Type I error)`


- `False Negatives` (실제 양성:1  예측 음성:0) - 3087 `(Type II error)`


```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x75e507e01208>




    
![png](output_217_1.png)
    


# **16. 분류 행렬** <a class="anchor" id="16"></a>


[목차](#0.1)

## 분류 보고서


**분류 보고서**는 분류 모델 성능을 평가하는 또 다른 방법이다. 여기에는 모델의 **정확도**, **회상률**, **F1** 및 **지원** 점수가 표시된다. 이러한 용어는 나중에 설명하겠다.

다음과 같이 분류 보고서를 보일 수 있다.


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

                  precision    recall  f1-score   support
    
              No       0.87      0.95      0.91     22067
             Yes       0.74      0.52      0.61      6372
    
        accuracy                           0.85     28439
       macro avg       0.80      0.73      0.76     28439
    weighted avg       0.84      0.85      0.84     28439
    
    

## 분류 정확도


```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

```

    Classification accuracy : 0.8502
    

## 분류 오류


```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

```

    Classification error : 0.1498
    

## 정확도


**정확도**는 예측된 모든 양성 결과 중 올바르게 예측된 양성 결과의 비율로 정의할 수 있다. 이는 오탐과 미탐의 합계(TP + FP)에 대한 정탐(TP)의 비율로 나타낼 수 있다. 


따라서 **정확도**는 정확하게 예측된 긍정적인 결과의 비율을 식별한다. 이는 부정 클래스보다 긍정 클래스에 더 관심이 있다.



수학적으로 정확도는 `TP 대 (TP + FP)`의 비율로 정의할 수 있다.





```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

```

    Precision : 0.9468
    

## 회수율


리콜은 실제 양성 결과 중 올바르게 예측된 양성 결과의 비율로 정의할 수 있다.
이는 오탐과 미탐의 합계(TP + FN)에 대한 정탐(TP)의 비율로 나타낼 수 있다. **리콜**은 **감도(Sensitivity)**라고도 한다.


**리콜**은 정확하게 예측된 실제 양성 반응의 비율을 식별한다.


수학적으로 리콜은 `TP 대 (TP + FN)`의 비율로 나타낼 수 있다.






```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

    Recall or Sensitivity : 0.8713
    

## True Positive


**True Positive**는 **리콜**과 같은 의미이다.



```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

    True Positive Rate : 0.8713
    

## False Positive 비율


```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

    False Positive Rate : 0.2634
    

## 특이성


```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

    Specificity : 0.7366
    

## f1-score


**f1-score**는 정확도와 리콜의 가중치 조화 평균이다. 가능한 최고 **f1-score**는 1.0이고 최악은 0.0이다. 
는 0.0이다.  **f1-score**는 정확도와 회수율의 조화 평균이다. 따라서 **f1-score**는 정확도와 재인식을 계산에 포함하기 때문에 정확도 측정값보다 항상 낮다. 'f1-score'의 가중 평균은 다음과 같은 경우에 사용해야 한다. 
분류기 모델을 비교하는 데 사용해야 한다.



## 지원


**Support**는 데이터 세트에서 클래스의 실제 발생 횟수이다.

# **17. 임계값 수준 조정하기** <a class="anchor" id="17"></a>


[목차](#0.1)


```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```




    array([[0.91382428, 0.08617572],
           [0.83565645, 0.16434355],
           [0.82033915, 0.17966085],
           [0.99025322, 0.00974678],
           [0.95726711, 0.04273289],
           [0.97993908, 0.02006092],
           [0.17833011, 0.82166989],
           [0.23480918, 0.76519082],
           [0.90048436, 0.09951564],
           [0.85485267, 0.14514733]])



###  관찰 사항


- 각 행에서 숫자는 1로 합산된다.


- 0과 1의 두 클래스에 해당하는 2개의 열이 있다.

    - 클래스 0 - 내일 비가 내리지 않을 것으로 예상되는 확률이다.    
    
    - 클래스 1 - 내일 비가 올 것으로 예상되는 확률이다.
        
    
- 예측 확률의 중요성

    - 비가 올 확률 또는 비가 오지 않을 확률에 따라 관측값의 순위를 매길 수 있다.


- 예측_프로바(predict_proba) 프로세스

    - 확률을 예측한다.    
    
    - 가장 높은 확률을 가진 클래스를 선택한다.    
    
    
- 분류 임계값 수준

    - 분류 임계값 수준은 0.5이다.    
    
    - 클래스 1 - 확률이 0.5를 초과하면 비가 올 확률이 예측된다.    
    
    - 클래스 0 - 확률이 0.5 미만인 경우 비가 내리지 않을 확률이 예측된다.    



```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913824</td>
      <td>0.086176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835656</td>
      <td>0.164344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.820339</td>
      <td>0.179661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.990253</td>
      <td>0.009747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.957267</td>
      <td>0.042733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.979939</td>
      <td>0.020061</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.178330</td>
      <td>0.821670</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.234809</td>
      <td>0.765191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.900484</td>
      <td>0.099516</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.854853</td>
      <td>0.145147</td>
    </tr>
  </tbody>
</table>
</div>




```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```




    array([0.08617572, 0.16434355, 0.17966085, 0.00974678, 0.04273289,
           0.02006092, 0.82166989, 0.76519082, 0.09951564, 0.14514733])




```python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_244_1.png)
    


 ### 관찰 사항


- 위의 히스토그램이 매우 양으로 치우쳐 있음을 알 수 있다.


- 첫 번째 열은 확률이 0.0에서 0.1 사이인 관측값이 약 15,000개임을 알려줍니다.


- 확률이 0.5를 초과하는 소수의 관측값이 있다.


- 따라서 이 소수의 관측은 내일 비가 올 것이라고 예측한다.


- 대부분의 관측은 내일 비가 내리지 않을 것이라고 예측한다.

### 임계값 낮추기


```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

    With 0.1 threshold the Confusion Matrix is  
    
     [[12726  9341]
     [  547  5825]] 
    
     with 18551 correct predictions,  
    
     9341 Type I errors( False Positives),  
    
     547 Type II errors( False Negatives),  
    
     Accuracy score:  0.6523084496641935 
    
     Sensitivity:  0.9141556811048337 
    
     Specificity:  0.5766982371867494 
    
     ==================================================== 
    
    
    With 0.2 threshold the Confusion Matrix is  
    
     [[17066  5001]
     [ 1234  5138]] 
    
     with 22204 correct predictions,  
    
     5001 Type I errors( False Positives),  
    
     1234 Type II errors( False Negatives),  
    
     Accuracy score:  0.7807588171173389 
    
     Sensitivity:  0.8063402385436284 
    
     Specificity:  0.7733720034440568 
    
     ==================================================== 
    
    
    With 0.3 threshold the Confusion Matrix is  
    
     [[19080  2987]
     [ 1872  4500]] 
    
     with 23580 correct predictions,  
    
     2987 Type I errors( False Positives),  
    
     1872 Type II errors( False Negatives),  
    
     Accuracy score:  0.8291430781673055 
    
     Sensitivity:  0.7062146892655368 
    
     Specificity:  0.8646395069560883 
    
     ==================================================== 
    
    
    With 0.4 threshold the Confusion Matrix is  
    
     [[20191  1876]
     [ 2517  3855]] 
    
     with 24046 correct predictions,  
    
     1876 Type I errors( False Positives),  
    
     2517 Type II errors( False Negatives),  
    
     Accuracy score:  0.845529027040332 
    
     Sensitivity:  0.6049905838041432 
    
     Specificity:  0.9149861784565188 
    
     ==================================================== 
    
    
    

### 의견


- 이진 문제에서는 예측 확률을 클래스 예측으로 변환하기 위해 기본적으로 0.5의 임계값이 사용된다.


- 임계값을 조정하여 민감도 또는 특이도를 높일 수 있다. 


- 민감도와 특이도는 반비례 관계에 있다. 하나를 높이면 항상 다른 하나는 낮아지고 그 반대의 경우도 마찬가지이다.


- 임계값 수준을 높이면 정확도가 높아진다는 것을 알 수 있다.


- 임계값 수준 조정은 모델 구축 프로세스의 마지막 단계 중 하나로 수행해야 한다.

# **18. ROC - AUC** <a class="anchor" id="18"></a>


[목차](#0.1)



## ROC 곡선


분류 모델 성능을 시각적으로 측정하는 또 다른 도구는 **ROC 곡선**이다. ROC 곡선은 **수신기 작동 특성 곡선**의 약자이다. ROC 곡선은 다양한 분류 임계값 수준에서 분류 모델의 성능을 보여주는 도표이다. 
분류 임계값 수준에서 분류 모델의 성능을 보여주는 도표이다. 



ROC 곡선**은 다양한 임계값 수준에서 **False Positive Rate(FPR)**에 대한 **True Positive Rate(TPR)**을 플롯한다.



**True Positive Rate(TPR)**은 **리콜(Recall)**이라고도 한다. 이는 `TP 대 (TP + FN)`의 비율로 정의된다.



**False Positive Rate(FPR)**은 `FP 대 (FP + TN)`의 비율로 정의된다.




ROC 곡선에서는 단일 포인트의 TPR(True Positive Rate)과 FPR(False Positive Rate)에 초점을 맞출 것이다. 이렇게 하면 다양한 임계값 수준에서 TPR과 FPR로 구성된 ROC 곡선의 일반적인 성능을 알 수 있다. 따라서 ROC 곡선은 다양한 분류 임계값 수준에서 TPR과 FPR을 그래프로 표시한다. 임계값 수준을 낮추면 더 많은 항목이 양성으로 분류될 수 있다. 그러면 TP와 FP가 모두 증가한다.





```python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

```


    
![png](output_250_0.png)
    


ROC 곡선은 특정 컨텍스트에 대한 민감도와 특이도의 균형을 맞추는 임계값 수준을 선택하는 데 도움이 된다.

## ROC-AUC


**ROC AUC**는 **Receiver Operating Characteristic - Area Under Curve**의 약자이다. 분류기 성능을 비교하기 위한 기법이다. 이 기법에서는 '곡선 아래 면적(AUC)`을 측정한다. 완벽한 분류기는 ROC AUC가 1이 되는 반면, 순수 무작위 분류기는 ROC AUC가 0.5가 된다. 


따라서 **ROC AUC**는 곡선 아래에 있는 ROC 플롯의 백분율이다.


```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

    ROC AUC : 0.8729
    

### 의견


- ROC AUC는 분류기 성능을 단일 수치로 요약한 것이다. 값이 높을수록 분류기가 더 우수하다는 것을 의미한다.

- 우리 모델의 ROC AUC는 1에 가까워지고 있다. 따라서 내일 비가 올지 안 올지를 예측하는 데 있어 우리 분류기가 잘 작동한다고 결론 내릴 수 있다.


```python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

    Cross validated ROC AUC : 0.8695
    

# **19. k-Fold Cross Validation** <a class="anchor" id="19"></a>


[목차](#0.1)


```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

    Cross-validation scores:[0.84686387 0.84624852 0.84633642 0.84963298 0.84773626]
    

교차 검증 정확도는 평균을 계산하여 요약할 수 있다.


```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

    Average cross-validation score: 0.8474
    

당사의 오리지널 모델 점수는 0.8476으로 확인되었다. 교차 검증 평균 점수는 0.8474이다. 따라서 교차 검증이 성능 향상으로 이어지지 않는다는 결론을 내릴 수 있다.

# **20. GridSearch CV를 이용한 하이퍼파라미터 최적화** <a class="anchor" id="20"></a>


[목차](#0.1)


```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)

```




    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='warn',
                                              n_jobs=None, penalty='l2',
                                              random_state=0, solver='liblinear',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

    GridSearch CV best score : 0.8474
    
    
    Parameters that give the best results : 
    
     {'penalty': 'l1'}
    
    
    Estimator that was chosen by the search : 
    
     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l1',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)
    


```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

    GridSearch CV score on test set: 0.8507
    

### 의견


- 원래 모델 테스트 정확도는 0.8501이고 GridSearch CV 정확도는 0.8507이다.


- 이 특정 모델의 경우 GridSearch CV가 성능을 향상시킨다는 것을 알 수 있다.

# **21. 결과 및 결론** <a class="anchor" id="21"></a>


[목차](#0.1)

1.	로지스틱 회귀 모델 정확도 점수는 0.8501이다. 따라서 이 모델은 내일 호주에 비가 올지 여부를 예측하는 데 매우 잘 작동한다.

2.	소수의 관측치가 내일 비가 올 것이라고 예측한다. 대다수의 관측은 내일 비가 내리지 않을 것으로 예측한다.

3.	모델이 과적합의 징후를 보이지 않는다.

4.	C 값을 늘리면 테스트 세트 정확도가 높아지고 훈련 세트 정확도도 약간 증가한다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘할 것이라는 결론을 내릴 수 있다.

5.	임계값 수준을 높이면 정확도가 높아집니다.

6.	모델의 ROC AUC가 1에 가까워집니다. 따라서 분류기가 내일 비가 올지 안 올지 예측하는 데 잘 작동한다는 결론을 내릴 수 있다.

7.	원래 모델의 정확도 점수는 0.8501이고 RFECV 후 정확도 점수는 0.8500이다. 따라서 거의 비슷한 정확도를 얻을 수 있지만 특징 집합이 줄어든다.

8.	원래 모델에서 FP = 1175인 반면 FP1 = 1174이다. 따라서 거의 동일한 수의 오탐을 얻는다. 또한 FN = 3087이고 FN1 = 3091이다. 따라서 오탐이 약간 더 높다.

9.	우리의 원래 모델 점수는 0.8476으로 확인되었다. 교차 검증 평균 점수는 0.8474이다. 따라서 교차 검증이 성능 향상으로 이어지지 않는다는 결론을 내릴 수 있다.

10.	원래 모델 테스트 정확도는 0.8501이고 GridSearch CV 정확도는 0.8507이다. 이 특정 모델에 대해 GridSearch CV가 성능을 향상시킨다는 것을 알 수 있다.

# **22. 참고자료** <a class="anchor" id="22"></a>


[목차](#0.1)



The work done in this project is inspired from following books and websites:-


1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido

3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves

4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli

5. Udemy course – Feature Selection for Machine Learning by Soledad Galli

6. https://en.wikipedia.org/wiki/Logistic_regression

7. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

8. https://en.wikipedia.org/wiki/Sigmoid_function

9. https://www.statisticssolutions.com/assumptions-of-logistic-regression/

10. https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

11. https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression

12. https://www.ritchieng.com/machine-learning-evaluate-classification-model/


참조의 참조
https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial
https://www.deepl.com/translator


```python
[Go to Top](#0)
```


      File "<ipython-input-144-7aee13e58179>", line 1
        [Go to Top](#0)
             ^
    SyntaxError: invalid syntax
    
