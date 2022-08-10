# simulated annealing 구현을 위해 기본적으로 필요한 함수들입니다.
# 다른 변수선택 구현에도 동일하거나 유사한 함수가 사용됩니다.
# import하여 simulated annealing 구현 함수 내에서 작동시킵니다.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def get_aic(x, y):
    '''
        aic를 구하는 함수
        input: data x, data y
        output: aic 값
    '''
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(x)
    residual = y - prediction

    N = len(x)
    s = len(x.columns)
    AIC = N*np.log(sum(residual**2)/N) + 2*(s + 2)

    return AIC

def get_bic(x, y):
    '''
        bic를 구하는 함수
        input: data x, data y
        output: bic 값
    '''
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(x)
    residual = y - prediction

    N = len(x)
    s = len(x.columns)
    BIC = N*np.log(sum(residual**2)/N) + np.log(N)*(s + 2)

    return BIC

def get_mspe(x, y):
    '''
        mspe를 구하는 함수
        input: data x, data y
        output: mspe 값
    '''
    N = len(x)
    train_x = x[:round(N * 0.7)]
    train_y = y[:round(N * 0.7)]
    test_x = x[round(N * 0.7):]
    test_y = y[round(N * 0.7):]
    reg = LinearRegression().fit(train_x, train_y)
    prediction = reg.predict(test_x)
    MSPE = sum((test_y - prediction)**2) / len(test_y)

    return MSPE

def get_bin(x, p):
    '''
        선택된 변수의 정수 index를 [01100110..] 방식으로 변환해주는 함수
        input: 정수 index array, 총 변수 개수 p
        output: binary 방식 변수 선택 결과
    '''
    zeros = np.zeros(p, dtype=int)
    zeros[x] = 1

    return zeros

def flip(k, x, p):
    '''
        기존 선택된 변수들에서 k개만큼 flip해주는 함수
        input: flip할 횟수 k, 정수 index array, 총 변수 개수 p
        output: 새롭게 선택된 변수 결과
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(p, size = k, replace = False)
    zeros[idx] = 1

    old = get_bin(x, p)
    new = abs(old - zeros)

    return new
