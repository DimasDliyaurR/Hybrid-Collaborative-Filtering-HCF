
from Evaluation import Evaluation
from typing_extensions import override

class Precision(Evaluation) :

    def __init__(self,data, top_n : list[list[list[int]]],*, path_evaluation : str = None,toyData : bool, N : int = 30):
        super().__init__(data, top_n, path_evaluation=path_evaluation,toyData = toyData, N=N)

    def numerator(self,u : None|int = None, indexTrain : None|int = None) -> float :
        if not self.toyData :
            return sum([len(set(self.top_n[indexTrain][u][:i]).intersection(set(self.test[indexTrain][u]))) for i in range(1,self.N+1)])
        return sum([len(set(self.top_n[:i]).intersection(set(self.data_test))) for i in range(1,self.N+1)])

    def denominator(self,u : None|int = None, indexTrain : None|int = None) -> float :
        if not self.toyData :
            return sum([len(self.top_n[indexTrain][u][:i]) for i in range(1,self.N+1)])
        return sum([len(self.top_n[:i]) for i in range(1,self.N+1)])

    @override
    def result(self, u : None|int = None, indexTrain : None|int = None) -> float :
        return self.numerator(u,indexTrain)/self.denominator(u,indexTrain)
