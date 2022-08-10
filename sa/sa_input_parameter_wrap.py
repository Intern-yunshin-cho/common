from os import lstat
import pandas as pd
import numpy as np

from dataclasses import dataclass, field, asdict, astuple
from typing import List, ClassVar
import ast                              # 입력 문자 beta coef, integer 로 변환에 사용하는 부분

from sa.SimulatedAnnealing import SimulatedAnnealing as SA
from sa.basefunctions import get_aic, get_bic, get_mspe, get_bin, flip

def sa_read_excel2tuple():
    df = pd.read_excel('sa_input_parameter.xlsx')
    return list(df.itertuples(name=None))

from typing import ClassVar
@dataclass
class sa_parameter_wrap:
    count_parameter_data : ClassVar[int] = 0  # 전치 처리위한 class variable
    # return SA to value, result, log as SA's parameter
    lst : float                     # ?
    k : float                       # ?
    alpha : float                   # ?
    tau : float                     # ?

    # sa_basefunctions 의 objective function 선택하는 부분
    objective : str 
    sa_input_parameter_name :str

    def __post_init__(self):
        sa_parameter_wrap.count_parameter_data += 1  # class 생성 data counter

    @classmethod
    def data_counter(cls):
        return [ 'sa_parameter' + str(i) for i in range(cls.count_parameter_data)]

# input data 를 묶어서 생성 
# 여기서 df 는 SA_input_parameter = pd.read_excel('sa_input_parameter.xlsx')
def sa_list_gen_from_class(df):
    generated_sa_parameter_wrap_list = []
    Tupled_sa_input_parameter = sa_read_excel2tuple()

    for i in range(len(Tupled_sa_input_parameter)):
        data_list_Row = list(list(df.itertuples(name=None))[i])
        generated_sa_parameter_wrap_list.append(sa_parameter_wrap(  lst= ast.literal_eval(data_list_Row[2]),
                                                                    k= data_list_Row[3],
                                                                    alpha= data_list_Row[4],
                                                                    tau= data_list_Row[5],
                                                                    objective=data_list_Row[6],
                                                                    sa_input_parameter_name = data_list_Row[1]
            )
        )
    return generated_sa_parameter_wrap_list
