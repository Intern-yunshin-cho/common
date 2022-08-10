import pandas as pd
import numpy as np

# input data 부분 필요 라이브러리
import etc.kqc_custom
from dataclasses import dataclass, field, asdict, astuple
from typing import List, ClassVar
import ast  # 입력 문자 beta coef, integer 로 변환에 사용하는 부분

# 출력을 저장할 데이터 클래스 생성 
class qaoa_Results:
    def __init__(self, best_solution, result_log):
        self.best_solution, self.result_log = best_solution, result_log
    def __getitem__(self,key):
        return getattr(self, key)
    def __setitem__(self,key,value):
        return setattr(self, key, value)

# class instance generator
def qaoa_Results_gen(best_score, best_solution, result_log):
    return qaoa_Results(best_score, best_solution, result_log)