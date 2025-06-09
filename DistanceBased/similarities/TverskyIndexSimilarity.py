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

class TverskyIndex(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, *,path_evaluation : None|str = None, toyData : bool|None = False, opsional="user-based",k=2,alpha_1=0.7,alpha_2=0.2):
        SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        if not toyData :
            
            path_file = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/tversky_index_similarity.joblib"
            path_file_prediction = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/prediction/{k}/tversky_index_prediction.joblib"
            # print(path_file)
            # print(os.path.exists(path_file))
            if os.path.exists(path_file) : # and (not os.path.exists(path_file_prediction)) :
                self.result_similarity = joblib.load(path_file)
            else :
                self.result_similarity = self.main_calculation()
                joblib.dump(self.result_similarity,path_file)
                
            # if not os.path.exists(path_file_prediction) :
            #     SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData,path_file=path_file_prediction)
            
            # if not toyData and path_evaluation is None :
            #     self.evaluation = NDCG(self.prediction,path_evaluation=path_evaluation)
        else :
            self.result_similarity = self.main_calculation()
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData)
    
    def __checkSymmetric(self) -> bool :
        return self.alpha_1 == self.alpha_2
    
    @override
    def numerator(self, A:list, B:list) -> list[float]:
        return len( set(A) & set(B) )

    @override
    def denominator(self, A:list, B:list) -> list[float]:
        return len( set(A) & set(B) ) + (self.alpha_1 * len( set(A) - set(B) )) + self.alpha_2 * len( set(B) - set(A) )
    
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
            
            if self.__checkSymmetric() :
                for i in range(len(matrix)):
                    for j in range(i,len(matrix)):
                        if i == j:
                            result[i].append(1)
                            continue
                        similarity_result = self.similarity_calculation(int(i), int(j))
                        result[i].append(similarity_result)
                        result[j].append(similarity_result)
                return result
            
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i == j:
                        result[i].append(1)
                        continue
                    similarity_result = self.similarity_calculation(int(i), int(j))
                    result[i].append(similarity_result)
            return result
            
        else :
            if self.__checkSymmetric() :
                result = []
                for indexFold in range(len(self.train)) :
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
            
            result = []
            for indexFold in range(len(self.train)) :
                matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])
                # print("Train Fold ",indexFold)
                result_train = []
                
                for i in range(len(matrix)):
                    result_inner = []
                    for j in range(len(matrix)):
                        if i == j:
                            result_inner.append(1)
                            continue
                        similarity_result = self.similarity_calculation(i, j, indexFold)
                        result_inner.append(similarity_result)
                    result_train.append(result_inner)
                result.append(result_train)
            return result
        
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    def get_similarity_dataframe(self,indexFold : int|None = None) :
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        
        if indexFold is None :
            return pd.DataFrame(self.result_similarity)
        return pd.DataFrame(self.result_similarity[indexFold])