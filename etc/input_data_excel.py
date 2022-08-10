import pandas as pd
import numpy as np

from dataclasses import dataclass, field, asdict, astuple
from typing import List, ClassVar
import ast                              # 입력 문자 beta coef, integer 로 변환에 사용하는 부분

from kqc_custom import generate_dependent_sample, generate_independent_sample

def read_excel2tuple():
    df = pd.read_excel('input_parameter.xlsx')
    return list(df.itertuples(name=None))
'''
    Data generator Class from KQC_custom function
    - 각각의 parameter 를 EXCEL 로 부터 parameter 를 입력받는 class 입니다.
    입력받는 parmeter c
    1. n_samples
    2. n_features
    3. beta_coef
    4. epsilon
    5. correlation_parameter
    6. kqc_kustom // dependent or independent 선택 가능 
    7. (optional) data name : 별도로 지정하지 않는 경우 자동으로 data + # 으로 생성
'''
from typing import ClassVar
@dataclass
class data_generator:
    count_data : ClassVar[int] = 0  # 전치 처리위한 class variable

    # KQC custom file // data generator function's parameter
    n_samples : int                 # 데이터 샘플 수
    n_features : int                # 데이터 피쳐 개수
    beta_coef : float               # beta 계수
    epsilon : float                 # 오차의 표준편차 정도
    correlation_parameter : float   # 상관계수 ?

    # kqc data generator dependent / inpependent 선택 부분 
    is_independent : str            # is_independent
    data_name : str                 # 지정 데이터명

    def __post_init__(self):
        data_generator.count_data += 1  # class 생성 data counter
      
    # parameter 입력 받기 
    @classmethod
    def create(cls): 
        data = cls() 
        return data
    
    @classmethod
    def data_counter(cls):
        return [ 'data' + str(i) for i in range(cls.count_data)]

    #KQC_custom 의 data generator 로 부터 data 생성 class
    def data_gen(self):
        data_gened = []
        if self.is_independent == 'dependent':
            x, y = generate_dependent_sample(   self.n_samples,
                                                self.n_features,
                                                self.beta_coef,
                                                self.epsilon,
                                                self.correlation_parameter
            )
        else:
            x, y = generate_independent_sample( self.n_samples,
                                                self.n_features,
                                                self.beta_coef,
                                                self.epsilon,
                                                self.correlation_parameter
            )
        data_gened.append(x)
        data_gened.append(y)
        return data_gened

# class data generator 로 부터, 입력 데이터 리스트의 수 만큼 리스트 생성 
def list_gen_from_class(df):
    generated_data_list = []
    Tupled_df_KQC_data_parameter = read_excel2tuple()
    for i in range(len(Tupled_df_KQC_data_parameter)):
        data_list_Row = list(list(df.itertuples(name=None))[i])
        generated_data_list.append(data_generator(  n_samples= data_list_Row[2],
                                                    n_features= data_list_Row[3],
                                                    beta_coef= ast.literal_eval(data_list_Row[4]),
                                                    epsilon= data_list_Row[5],
                                                    correlation_parameter= data_list_Row[6],
                                                    is_independent= data_list_Row[7],
                                                    data_name=data_list_Row[1]
            )
        )
    return generated_data_list

# 생성된 데이터 검색을 위한 dict 생성 function
def dict_gen_from_list(list):
    generated_data_dict = {}
    for i in range(len(list)):
        '''
            클래스에서 선언 인스턴스 개수로 자동 카운트 되는 이름명은 윗 코드
            dataframe 에서 선언한 '지정 데이터명'을 사용한다면 아래 코드를 사용하면 됩니다.
        '''
        generated_data_dict[list[0].data_counter()[i]] = list[i]
        # list(KQC_data_parameter.loc[:,'지정 데이터명'])[i] = generated_data_list[i]
    return generated_data_dict