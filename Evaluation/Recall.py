from Evaluation import Evaluation
import numpy as np
from typing_extensions import override

class Recall(Evaluation) :

    def __init__(self, data, top_n, *, path_evaluation = None, toyData, N = 30,n : int|None = 20):
        super().__init__(data, top_n, path_evaluation=path_evaluation, toyData=toyData, N=N, n=n)

    def numerator(self,u : None|int = None, indexTrain : None|int = None) -> float :
        if not self.toyData :
            return np.array([len(set(self.top_n[indexTrain][u][:i]).intersection(set(self.test[indexTrain][u]))) for i in range(1,self.N+1)])
        print([len(set(self.top_n[:i]).intersection(set(self.data_test))) for i in range(1,self.N+1)])
        return np.array([len(set(self.top_n[:i]).intersection(set(self.data_test))) for i in range(1,self.N+1)])

    def denominator(self,u : None|int = None, indexTrain : None|int = None) -> float :
        if not self.toyData :
            return np.array([len(self.test[indexTrain][u]) for _ in range(1,self.N+1)])
        return np.array([len(self.data_test) for _ in range(1,self.N+1)])

    @override
    def result(self, u = None, indexTrain = None):
        return self.numerator(u,indexTrain=indexTrain)/self.denominator(u,indexTrain=indexTrain)