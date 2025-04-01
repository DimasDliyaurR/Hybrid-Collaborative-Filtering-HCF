import DistanceBased as SDB
import prediction as P
import helper.helper as hp
from typing_extensions import override
import pandas as pd

class TverskyIndex(SDB.Similarity,SDB.Mean,P.Prediction) :
    
    def __init__(self, data, *, opsional="user-based",k=2,alpha=0.7,beta=0.2):
        SDB.Mean.__init__(self,data,opsional=opsional)
        self.beta = beta
        self.alpha = alpha
        self.result_similarity = self.main_calculation()
        P.Prediction.__init__(self,self.result_mean_centered,self.result_similarity,data,meanList=self.result_mean,opsional=opsional,k=k)

    @override
    def numerator(self, vector1:list, vector2:list) -> int:
        return len( set(hp.indexOfNonZero(vector1)) & set(hp.indexOfNonZero(vector2)) )

    @override
    def denominator(self, vector1:list, vector2:list) -> int:
        return len( set(hp.indexOfNonZero(vector1)) & set(hp.indexOfNonZero(vector2)) ) + (   self.alpha * len( set(hp.indexOfNonZero(vector1)) - set(hp.indexOfNonZero(vector2)) ) + ( self.beta * len( set(hp.indexOfNonZero(vector2)) - set(hp.indexOfNonZero(vector1)) ) ) )

    @override
    def similarity_calculation(self,index1: list, index2: list, matrix : list[float]) -> list[float]:
        return self.numerator(matrix[index1],matrix[index2]) / self.denominator(matrix[index1],matrix[index2])

    @override
    def main_calculation(self):
        result = [[] for _ in range(len(self.matrix))]
        
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.similarity_calculation(i, j, self.matrix)
                result[i].append(similarity_result)
        return result

    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    @override
    def show(self) :
        return pd.DataFrame(self.result_similarity)