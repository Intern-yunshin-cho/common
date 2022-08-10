import pickle 
with open('generated_data_dict', 'rb') as f:
    generated_data_dict = pickle.load(f)
'''
    REPL 환경에서 load data pickle 을 호출하시면 
    generated_data_dict로 변수를 불러올 수 있습니다. 
    
    파일명은 main 의 저장함수의 이름을 따라 자동으로 선언됩니다.
'''