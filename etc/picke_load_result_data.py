import pickle 
with open('Result_dict', 'rb') as f:
    Result_dict = pickle.load(f)
'''
    REPL 환경에서 load data pickle 을 호출하시면 
    Result_dict로 변수를 불러올 수 있습니다. 
    
    파일명은 main 의 저장함수의 이름을 따라 자동으로 선언됩니다.
'''