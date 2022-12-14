import pandas as pd
import numpy as np

# input data 부분 필요 라이브러리
import etc.kqc_custom
from dataclasses import dataclass, field, asdict, astuple
from typing import List, ClassVar
import ast  # 입력 문자 beta coef, integer 로 변환에 사용하는 부분

# 출력을 저장할 데이터 클래스 생성 
@dataclass
class sa_Results:
    #사용환경에 따른 수정요
    # default_information =  'N = 500, p = 20, beta_coef=[4, 3, 2, 2], epsilon =, covariance_parameter = 5, cooling_schedule = lundymee'#.format(eps)
    value : List[float] = field(default_factory=list)
    result : List[float] = field(default_factory=list)
    log : List[int] = field(default_factory=list)
    
    # pre-variable
    # def __post_init__(self):
        # self.information = self.default_information 
    # dict allocation
    def __getitem__(self,key):
        return getattr(self, key)
    def __setitem__(self,key,value):
        return setattr(self, key, value)
# class instance generator
def sa_Results_gen(value, result, log):
    return sa_Results(value, result, log)