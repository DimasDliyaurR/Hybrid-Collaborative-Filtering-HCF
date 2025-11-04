from Evaluation import Evaluation,Recall, Precision
from typing_extensions import override

class F1_score(Evaluation) :

    def __init__(self, data, top_n, *, path_evaluation = None, toyData, N = 30, n = 20):
        
        self.recall = Recall(data, top_n, path_evaluation=path_evaluation, toyData=toyData, N=N,n=n)
        self.precision = Precision(data, top_n, path_evaluation=path_evaluation, toyData=toyData, N=N,n=n)
        super().__init__(data, top_n, path_evaluation=path_evaluation, toyData=toyData, N=N, n=n)

    def numerator(self,u : None|int = None, indexFold : None|int = None) -> float :
        return 2*(self.precision.result_evaluation * self.recall.result_evaluation)

    def denominator(self,u : None|int = None, indexFold : None|int = None) -> float :
        return self.precision.result_evaluation + self.recall.result_evaluation
        

    @override
    def result(self, u = None, indexFold = None):
        return self.numerator(u,indexFold=indexFold)/self.denominator(u,indexFold=indexFold)