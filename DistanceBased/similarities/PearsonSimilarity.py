import cmath

import numpy as np
import helper.helper as hp
import DistanceBased as SDB
import prediction as P
from typing_extensions import override
import pandas as pd
import time
from joblib import Parallel,delayed,Memory

class PearsonSimilarity (SDB.Similarity,SDB.Mean,P.Prediction) :
    """
    Measure Similarity of Pearson

    Args:
    -----
        data : pass the user-item matrix rating
        opsional : Determinate based of threating matrix such a , user-based or item-based

    Attributes:
    -----------
        matrix : user-item matrix rating
        result_similarity : Provide the matrix of similarity
    """

    def __init__(self, data, *, opsional="user-based",k=2):
        SDB.Mean.__init__(self,data,opsional=opsional)
        self.result_similarity = self.matrix_calculation()
        P.Prediction.__init__(self,self.result_mean_centered,self.result_similarity,data,meanList=self.result_mean,opsional=opsional,k=k)

    @override
    def numerator(self,vector1 : list, vector2 : list) -> int:
        return sum((vector1[i] * vector2[i]) for i in range(len(vector1)))

    @override
    def denominator(self,vector1 : list, vector2: list) -> int:
        return cmath.sqrt(sum((vector1[i]**2) for i in range(len(vector1)))) * cmath.sqrt(sum((vector2[i]**2) for i in range(len(vector2))))

    @override
    def similarity_calculation(self, index1 : int, index2 : int, matrix : list[float], mean_centered : list[float]) -> float:
        matrix_filtered = [
                [mean_centered[index1][mc1]for mc1 in range(len(matrix[index1]))],
                [mean_centered[index2][mc2]for mc2 in range(len(matrix[index2]))],
            ]

        vector1 = np.delete(matrix_filtered[0],hp.indexOfZero(matrix[index1],matrix[index2])).tolist()
        vector2 = np.delete(matrix_filtered[1],hp.indexOfZero(matrix[index1],matrix[index2])).tolist()

        denominator = self.denominator(matrix[index2],matrix[index1]).real
        numerator = self.numerator(vector1,vector2).real

        return (numerator / denominator) if denominator != 0 and numerator != 0 else 0

    @override
    def matrix_calculation(self) -> list[list[float]]:
        result = [[] for _ in range(len(self.matrix))]
        
        for i in range(len(self.matrix)): 
            for j in range(i, len(self.matrix)):
                if i == j:
                    result[i].append(1)
                    continue
                similarity_result = self.similarity_calculation(i, j, self.matrix,self.result_mean_centered)
                result[i].append(similarity_result)
                result[j].append(similarity_result)
        print("Sim : Selesai")
        return result
    
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity
        
    @override
    def show(self) -> object:
        return pd.DataFrame(self.result_similarity)
    
