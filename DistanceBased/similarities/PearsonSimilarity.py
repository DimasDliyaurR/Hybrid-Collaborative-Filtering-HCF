import cmath

import DistanceBased as SDB
import prediction as P
from typing_extensions import override
import pandas as pd
from helper.helper import reverseMatrix
from MatrixRating import MatrixRating
from operator import itemgetter,mul
class PearsonSimilarity (SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :

    def __init__(self, data, *, opsional="user-based",k=2):
        MatrixRating.__init__(self,data)
        SDB.Mean.__init__(self,self.matrixRating, reverseMatrix(self.matrixRating),opsional=opsional)
        self.opsional = opsional
        self.__data = self.matrixRating if opsional == "user-based" else self.reverseMatrixRating
        self.result_similarity = self.main_calculation()
        P.Prediction.__init__(self,data,opsional,self.result_similarity,k=k)

    @override
    def numerator(self,u : int, v : int, setOfRated : list) -> float:
        if len(setOfRated) == 0 :
            return 0
        return sum(map(mul, (itemgetter(*setOfRated)(self.result_mean_centered[u])) , (itemgetter(*setOfRated)(self.result_mean_centered[v])) )) if len(setOfRated) > 1 else (self.result_mean_centered[u][setOfRated[0]] * self.result_mean_centered[u][setOfRated[0]])
        
    @override
    def denominator(self, u : int, v : int, setOfRated : list) -> float:
        if len(setOfRated) == 0 :
            return 0
        return cmath.sqrt(sum( list(map(lambda x : x**2,itemgetter(*setOfRated)(self.result_mean_centered[u] ))) ) if len(setOfRated) > 1 else self.result_mean_centered[u][setOfRated[0]]**2) * (cmath.sqrt( sum(list(map(lambda x:x**2,itemgetter(*setOfRated)(self.result_mean_centered[v])))) if len(setOfRated) > 1 else self.result_mean_centered[v][setOfRated[0]]**2))

    @override
    def similarity_calculation(self, u : int, v : int) -> float:
        
        set1 = self.getItem(u) if self.opsional == "user-based" else self.getUser(u)
        set2 = self.getItem(v) if self.opsional == "user-based" else self.getUser(v)
        commonlyRated = list(set(set1) & set(set2))

        denominator = self.denominator(u,v,commonlyRated).real

        numerator = self.numerator(u,v,commonlyRated).real

        return (numerator / denominator) if denominator != 0 and numerator != 0 else 0

    @override
    def main_calculation(self) -> list[list[float]]:
        result = [[] for _ in range(len(self.__data))]
        for i in range(len(self.__data)): 
            if i % 10 == 0 :
                print(f"Sim({i})")
            for j in range(i, len(self.__data)):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.similarity_calculation(i, j)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        print("Sim : Selesai")
        return result
    
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity
        
    @override
    def show(self) -> object:
        return pd.DataFrame(self.result_similarity)
    
