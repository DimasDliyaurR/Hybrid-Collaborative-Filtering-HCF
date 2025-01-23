import cmath

import numpy as np
import helper.helper as hp
import DistanceBased as SDB
import prediction as p
from typing_extensions import override
import pandas as pd

class CosineSimilarity (SDB.DistanceBased,SDB.Mean,p.Prediction) :

    def __init__(self, data, *, opsional="user-based",k=2):
        SDB.Mean.__init__(self,data,opsional=opsional)
        self.result_similarity = self.matrix_calculation()
        p.Prediction.__init__(self,self.result_mean_centered,self.result_similarity,data,meanList=self.result_mean,opsional=opsional,k=k)

    @override
    def numerator(self,vector1 : list, vector2 : list) -> int:
        return sum((vector1[i] * vector2[i]) for i in range(len(vector1)))

    @override
    def denominator(self,vector1 : list, vector2: list) -> int:
        return cmath.sqrt(sum((vector1[i]**2) for i in range(len(vector1)))) * cmath.sqrt(sum((vector2[i]**2) for i in range(len(vector2))))

    @override
    def similarity_calculation(self, u, v, matrix):
        matrix_filtered = [
                [matrix[u][mc1]for mc1 in range(len(matrix[u]))],
                [matrix[v][mc2]for mc2 in range(len(matrix[v]))],
            ]

        vector1 = np.delete(matrix_filtered[0],hp.indexOfZero(matrix[u],matrix[v])).tolist()
        vector2 = np.delete(matrix_filtered[1],hp.indexOfZero(matrix[u],matrix[v])).tolist()

        denominator = self.denominator(matrix[v],matrix[u]).real
        numerator = self.numerator(vector1,vector2).real

        return (numerator / denominator) if denominator != 0 and numerator != 0 else 0

    @override
    def matrix_calculation(self):
        result = [[] for _ in range(len(self.matrix))]
        
        for i in range(len(self.matrix)): 
            for j in range(i, len(self.matrix)):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.similarity_calculation(i, j, self.matrix)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        return result
    
    def similarity_result(self) :
        return self.result_similarity
        
    @override
    def show(self) :
        return pd.DataFrame(self.result_similarity)
    
