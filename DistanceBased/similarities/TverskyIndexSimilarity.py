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
from operator import mul,itemgetter
import os
import joblib

class TverskyIndex(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, *, toyData : bool|None = False, opsional="user-based",k=2,alpha_1=0.7,alpha_2=0.2):
        # SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        if not toyData :
            
            # path_file = "cache/tversky_index/" + ("user_based" if self.opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/tversky_index_similarity.joblib"
            path_file = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/tversky_index_similarity.joblib"
            # path_file_prediction = "cache/tversky_index/" + ("user_based" if self.opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/prediction/{k}/tversky_index_prediction.joblib"
            path_file_prediction = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/prediction/{k}/tversky_index_prediction.joblib"
            print(os.path.exists(path_file))
            print(path_file)
            print(os.path.exists(path_file_prediction))
            print(path_file_prediction)
            if os.path.exists(path_file) and (not os.path.exists(path_file_prediction)) :
                self.result_similarity = joblib.load(path_file)
            # else :
            #     print("Sim Selesai")
            #     self.result_similarity = self.main_calculation()
            #     print("Sim Mulai")
            #     joblib.dump(self.result_similarity,path_file)
            
            if not os.path.exists(path_file_prediction) :
                SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
                P.Prediction.__init__(self,data,opsional,self.result_similarity, path_file=path_file_prediction,k=k,toyData=toyData)
        else :
            print("Sim Mulai")
            self.result_similarity = self.main_calculation()
            print("Sim Selesai")
            P.Prediction.__init__(self,data,opsional,self.result_similarity,k=k,toyData=toyData)

    def __checkSymmetric(self) -> bool :
        return self.alpha_1 == self.alpha_2
    
    @override
    def numerator(self, A:list, B:list) -> list[float]:
        return len( set(A) & set(B) )

    @override
    def denominator(self, A:list, B:list) -> list[float]:
        return len( set(A) & set(B) ) + (self.alpha_1 * len( set(A) - set(B) )) + self.alpha_2 * len( set(B) - set(A) )
        
    @override
    def similarity_calculation(self,A, B, indexTrain : int|None = None) -> float :
        setA = set(self.getItem(A,indexTrain) if self.opsional == "user-based" else self.getUser(A,indexTrain))
        setB = set(self.getItem(B,indexTrain) if self.opsional == "user-based" else self.getUser(B,indexTrain))

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
                for indexTrain in range(len(self.train)) :
                    print(f"Train Index = {indexTrain}")
                    matrix = self.train[indexTrain] if self.opsional == "user-based" else transpose(self.train[indexTrain])
                    
                    temp = array(zeros((len(matrix),len(matrix)))).tolist()
                    for i in range(len(matrix)):
                        if i%100 == 0 :
                            print(f"sim({i})")
                        result_inner = temp.copy()
                        for j in range(i,len(matrix)):
                            if i == j:
                                result_inner[i][j] = 1
                                continue
                            similarity_result = self.similarity_calculation(i, j, indexTrain)
                            result_inner[j][i] = similarity_result
                            result_inner[i][j] = similarity_result
                        result.append(result_inner)
                return result
            
            result = []
            for indexTrain in range(len(self.train)) :
                matrix = self.train[indexTrain] if self.opsional == "user-based" else transpose(self.train[indexTrain])
                
                result_train = []
                print(f"Train Index = {indexTrain}")
                
                for i in range(len(matrix)):
                    result_inner = []
                    for j in range(len(matrix)):
                        if i == j:
                            result_inner.append(1)
                            continue
                        similarity_result = self.similarity_calculation(i, j, indexTrain)
                        result_inner.append(similarity_result)
                    result_train.append(result_inner)
                result.append(result_train)
            return result
        
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    @override
    def show(self,indexTrain : int|None = None) :
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        
        if indexTrain is None :
            return pd.DataFrame(self.result_similarity)
        return pd.DataFrame(self.result_similarity[indexTrain])