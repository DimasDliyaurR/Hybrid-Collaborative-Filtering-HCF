import cmath

import DistanceBased as SDB
import prediction as P
from typing_extensions import override
import pandas as pd
from helper.helper import reverseMatrix
from MatrixRating import MatrixRating
from operator import mul,itemgetter

class CosineSimilarity (SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :

    def __init__(self, data,*, opsional="user-based",k=2):
        MatrixRating.__init__(self,data)
        SDB.Mean.__init__(self,self.matrixRating, reverseMatrix(self.matrixRating),opsional=opsional)
        self.opsional = opsional
        self.__data = self.matrixRating if opsional == "user-based" else self.reverseMatrixRating
        self.result_similarity = self.main_calculation()
        P.Prediction.__init__(self,data,opsional,self.result_similarity,k=k)

    @override
    def numerator(self,u:int,v:int, commonlyRated : list[int]) -> float:
        if len(commonlyRated) == 0 :
            return 0
        return sum(map(mul, list(itemgetter(*commonlyRated)(self.__data[u])) , list(itemgetter(*commonlyRated)(self.__data[v])) )) if len(commonlyRated) > 1 else self.__data[u][commonlyRated[0]] * self.__data[v][commonlyRated[0]]

    @override
    def denominator(self,u : int,v : int, set1 : list[int] ,set2 : list[int]) -> float:
        if len(set1) == 0 and len(set2) == 0 :
            return 0
        return cmath.sqrt( sum(list(map(lambda x : x**2, itemgetter(*set1)(self.__data[u]) ))) if len(set1) > 1 else self.__data[u][set1[0]]**2 ) * cmath.sqrt(sum(list(map(lambda x : x**2,itemgetter(*set2)(self.__data[v])) )) if len(set2) > 1 else self.__data[v][set2[0]]**2)

    @override
    def similarity_calculation(self, u : int, v : int):

        set1 = self.getItem(u) if self.opsional == "user-based" else self.getUser(u)
        set2 = self.getItem(v) if self.opsional == "user-based" else self.getUser(v)
        commonlyRated = list( set(set1) & set(set2) )

        numerator = self.numerator(u,v, commonlyRated).real
        denominator = self.denominator(u,v, set1, set2).real

        return (numerator / denominator) if denominator != 0 and numerator != 0 else 0

    @override
    def main_calculation(self):
        result = [[] for _ in range(len(self.__data))]
        
        for i in range(len(self.__data)): 
            if i % 10 == 0 :
                print(f"Sim({i})")
            for j in range(i, len(self.__data)):
                # print(f"Sim({type(j)})")
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.similarity_calculation(int(i), int(j))
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        print("Sim : Selesai")
        return result
    
    def similarity_result(self) :
        return self.result_similarity
        
    @override
    def show(self) :
        return pd.DataFrame(self.result_similarity)
    
