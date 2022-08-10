from input_data_excel import read_excel2tuple, list_gen_from_class, dict_gen_from_list
from sa_SimulatedAnnealing import SimulatedAnnealing as SA
from sa_input_parameter_wrap import sa_read_excel2tuple, sa_parameter_wrap, sa_list_gen_from_class
import pandas as pd

'''
    전체를 동시에 실행시키는 방향으로 정리 
'''
if __name__ == '__main__':
    # read Excel & input data
    KQC_data_parameter = pd.read_excel('./input_parameter.xlsx')

    # data_tuple = read_excel2tuple()
    # generated_data_list = list_gen_from_class(data_tuple)
    # generated_data_dict = dict_gen_from_list(generated_data_list)

    # # read Excel & input simulated annealing parameter data
    # sa_parameter = sa_read_excel2tuple()
    print(KQC_data_parameter)
