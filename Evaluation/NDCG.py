from prediction import Prediction
import numpy as np

class NDCG(Prediction) :

    def __init__(self,data, opsional : str, similarity : list[list[float]],*, N : int, path_file : str|None = None, k : int, toyData : bool = False):
        Prediction.__init__(self,data, opsional, similarity,path_file=path_file, k=k, toyData=toyData)

    def groundTruth(self, u, trainIndex,N) -> np.array :
         test_index_of_rating = [i for i in range(len(self.test[u])) if self.test[u][i] > 0]
         return np.array([1 if self.prediction.get_top_n_specific_user(u,trainIndex)[:i][-1] in test_index_of_rating else 0 for i in range(N)])

    def IDCG(self,N) -> float :
        return sum(1/np.log2(np.arange(2,N+1))).real

    def DCG(self, N, u) -> float :
        return sum(self.groundTruth(u)/np.log2(np.arange(2,N+1))).real

    def NDCG(self,N,u) :
        return self.DCG(N,u) / self.IDCG(N)
    
    def main_calculation(self) : ...