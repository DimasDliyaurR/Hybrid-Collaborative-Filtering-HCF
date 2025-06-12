import DistanceBased as SDB
import prediction as P

from typing_extensions import override

import pandas as pd
from numpy import (
    transpose, 
    zeros, 
    array
)

from MatrixRating import MatrixRating
# from Evaluation import NDCG
import os
import joblib
from tqdm import tqdm

class DiceCoefficientSimilarity(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, * ,toyData : bool|None = False, opsional="user-based",k=2):
        SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)

        if not toyData :
            
            path_file = "cache/dice_coefficient/" + ("user_based" if opsional == "user-based" else "item_based") + f"/dice_coefficient_similarity.joblib"
            path_file_prediction = "cache/dice_coefficient/" + ("user_based" if opsional == "user-based" else "item_based") + f"/prediction/{k}/dice_coefficient_prediction.joblib"
            

            if os.path.exists(path_file) :
                self.result_similarity = joblib.load(path_file)
            else :
                self.result_similarity = self.main_calculation()
                joblib.dump(self.result_similarity,path_file)
                
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData,path_file=path_file_prediction)

        else :
            self.result_similarity = self.main_calculation()
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData)
    
    @override
    def numerator(self, A:list, B:list) -> list[float]:
        return 2 * len( set(A) & set(B) )

    @override
    def denominator(self, A:list, B:list) -> list[float]:
        return len( A ) + len( B )
    
    @override
    def similarity_calculation(self,A, B, indexFold : int|None = None) -> float :
        setA = set(self.getItem(A,indexFold) if self.opsional == "user-based" else self.getUser(A,indexFold))
        setB = set(self.getItem(B,indexFold) if self.opsional == "user-based" else self.getUser(B,indexFold))

        denom = self.denominator(setA,setB).real
        
        return (self.numerator(setA,setB).real / denom) if denom != 0 else 0

    @override
    def main_calculation(self):
        if self.toyData :
            matrix = self.matrixRating if self.opsional == "user-based" else self.reverseMatrixRating
            result = [[] for _ in range(len(matrix))]
            
            for i in tqdm(range(len(matrix)),desc="Dice Coefficient"):
                for j in range(i,len(matrix)):
                    if i == j:
                        result[i].append(1)
                        continue
                    similarity_result = self.similarity_calculation(int(i), int(j))
                    result[i].append(similarity_result)
                    result[j].append(similarity_result)
            return result
            
        else :
            result = []
            for indexFold in tqdm(range(len(self.train)),desc="Dice Coefficient") :
                matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])
                
                temp = array(zeros((len(matrix),len(matrix)))).tolist()
                for i in range(len(matrix)):
                    
                    result_inner = temp.copy()
                    for j in range(i,len(matrix)):
                        if i == j:
                            result_inner[i][j] = 1
                            continue
                        similarity_result = self.similarity_calculation(i, j, indexFold)
                        result_inner[j][i] = similarity_result
                        result_inner[i][j] = similarity_result
                    result.append(result_inner)
            return result
        
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    def show_similarity(self,indexFold : int|None = None) :
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        
        if indexFold is None :
            return pd.DataFrame(self.result_similarity)
        return pd.DataFrame(self.result_similarity[indexFold])