import etc.kqc_custom
import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit import Aer,IBMQ
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tools import add_constant

from dataclasses import dataclass, field, asdict, astuple
from typing import List, ClassVar
import ast  # 입력 문자 beta coef, integer 로 변환에 사용하는 부분

@dataclass
class result_container:
    SSE : float
    AIC : float
    BIC : float
    lst : List[float] = field(default_factory=list)
    def print_result(self):
        print(self.lst)
        print(f"SSE : {self.SSE}")
        print(f"AIC : {self.AIC}")
        print(f"BIC : {self.BIC}")

def QAOA (x,y, backend) :
    data_x = pd.DataFrame(x)
    data_y = pd.DataFrame(y)
    p = data_x.shape[1]
    Q = data_x.corr()
    
    #베타 값 구하기
    beta = -Q.apply(sum)
    Q2 = Q
    for i in range(p) : 
        Q2.iloc[i,i] = Q2.iloc[i,i]-1 
    r_squared_list = []
    names = []
    for i in range( len(data_x.columns )) :
        names.append("x"+str(i))
    data_x.columns = names    
    
    # 개개인 변수에 대한 R_2 (결정계수) 구하기
    for i in range(p):
        model = LinearRegression()
        X, y = data_x[["x"+str(i)]], data_y
        model.fit(X, y)
        r_squared = model.score(X, y)
        r_squared_list.append(model.score(X, y))
    beta2 = -Q2.apply(sum)
    beta2_with_r2 = beta2 - r_squared_list
    
    # KQC_custom...함수로 QAOA결과값 
    result_qaoa=etc.kqc_custom.qubo_qaoa(Q2,beta2_with_r2,backend)

    # QAOA 결과값에 따라서 데이터 나누기
    x_vector = []
    for i in range(p):
        x_vector.append(result_qaoa[0][i])
    x_vector = pd.DataFrame(x_vector).T
    x_vector.columns = data_x.columns
    x_vector.index = ["turn"]
    concat_x = pd.concat([x_vector, data_x], axis = 0)
    concat_x_one=concat_x.T[concat_x.T["turn"]==1].T
    concat_x_zero = concat_x.T[concat_x.T["turn"]==0].T
    data_x_one =concat_x_one.loc[0:,:]
    data_x_zero =concat_x_zero.loc[0:,:]
    
    # 모든 변수를 포함한 모델의 aic,mse, bic    
    model = LinearRegression()
    model.fit(data_x,data_y)
    mse = (np.sum( (model.predict(data_x)-data_y) **2 ) / len(data_x))
    aic = (len(data_x)* np.log(mse) - 10 *np.log(len(data_x)) + 2* (len(data_x.columns)+1) )
    bic = (len(data_x)* np.log(mse) - 10 *np.log(len(data_x)) + np.log(len(data_x))* (len(data_x.columns)+1) )
    
    # QAOA에서 모델에 포함되지 않음 변수들에 대한 aic, mse, bic (없어도 괜찮을듯) 
    model0 = LinearRegression()
    model0.fit(data_x_zero,y)
    mse0 = (np.sum( (model0.predict(data_x_zero)-y) **2 ) / len(data_x_zero))
    aic0 = (len(data_x_zero)* np.log(mse0) - 10 *np.log(len(data_x_zero)) + 2* (len(data_x_zero.columns)+1) )  
    bic0 = (len(data_x_zero)* np.log(mse0) - 10 *np.log(len(data_x_zero)) + np.log(len(data_x_zero))* (len(data_x_zero.columns)+1) )   
    
    # QAOA에서 모델에 포함된 변수들에 대한 aic, mse, bic
    model1 = LinearRegression()
    model1.fit(data_x_one,y)
    mse1 = (np.sum( (model1.predict(data_x_one)-y) **2 ) / len(data_x_one))
    aic1 = (len(data_x_one)* np.log(mse1) - 10 *np.log(len(data_x_one)) + 2* (len(data_x_one.columns)+1) )  
    bic1 = (len(data_x_one)* np.log(mse1) - 10 *np.log(len(data_x_one)) + np.log(len((data_x_one)+1))* (len(data_x_one.columns)+1) )    

    result_data = []
    result_data.append(result_container(r_squared_list, mse, aic, bic))
    result_data.append(result_container(data_x_zero.columns, mse0, aic0, bic0))
    result_data.append(result_container(data_x_one.columns, mse1, aic1, bic1))

    # print(result_qaoa)
    # print(r_squared_list)
    # print(f"SSE : {mse}")
    # print(f"AIC : {aic}")
    # print(f"BIC : {bic}")
    # print(data_x_zero.columns)
    # print(f"SSE 0 : {mse0}")
    # print(f"AIC 0 : {aic0}")
    # print(f"BIC 0 : {bic0}")
    # print(data_x_one.columns)
    # print(f"SSE 1 : {mse1}")
    # print(f"AIC 1 : {aic1}")
    # print(f"BIC 1 : {bic1}")

    best_solution = list(result_qaoa[0])

    return best_solution, result_data