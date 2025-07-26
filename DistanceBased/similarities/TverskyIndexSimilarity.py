import DistanceBased as SDB
import prediction as P
from tqdm import tqdm

from typing_extensions import override

import pandas as pd
from numpy import (
    transpose, 
    zeros, 
    array
)

from MatrixRating import MatrixRating

import os
import joblib
import time

class TverskyIndexSimilarity(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, *, toyData : bool|None = False, opsional="user-based",k=2,alpha_1=0.7,alpha_2=0.2 ,time  : bool = False):
        SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.time = time
        
        if not toyData and time :
            self.time_computation_similarity_per_fold = []
            self.higher_time_computation_similarity = 0


        if not toyData :
            
            path_file = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/tversky_index_similarity.joblib"
            path_file_prediction = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/prediction/{k}/tversky_index_prediction.joblib"
            
            if not time and os.path.exists(path_file) :
                self.result_similarity = joblib.load(path_file)
            else :
                self.result_similarity = self.main_calculation()
                joblib.dump(self.result_similarity,path_file) if not time else ""
        
                
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData,path_file=path_file_prediction)
            
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
                for i in tqdm(range(len(matrix))):
                    for j in range(i,len(matrix)):
                        if i == j:
                            result[i].append(1)
                            continue
                        similarity_result = self.similarity_calculation(int(i), int(j))
                        result[i].append(similarity_result)
                        result[j].append(similarity_result)
                return result
            
            for i in tqdm(range(len(matrix))):
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
                
                for indexFold in tqdm(range(len(self.train)),desc="Tversky Index") :
                    matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])
                    
                    temp = array(zeros((len(matrix),len(matrix)))).tolist()

                    time_computation = []
                    for i in range(len(matrix)):
                        
                        result_inner = temp.copy()
                        time_computation_similarity_per_user = []
                        
                        for j in range(i,len(matrix)):
                            if i == j:
                                result_inner[i][j] = 1
                                continue
                            
                            t1 = time.time()
                            similarity_result = self.similarity_calculation(i, j, indexFold)
                            t2 = time.time()
                            time_computation_similarity_per_user += [t2-t1]
                            
                            result_inner[j][i] = similarity_result
                            result_inner[i][j] = similarity_result
                        time_computation += [time_computation_similarity_per_user]
                        result.append(result_inner)
                    self.time_computation_similarity_per_fold.append(time_computation)
                return result
            
            result = []
            for indexFold in tqdm(range(len(self.train)),desc="Tversky Index") :
                matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])

                result_train = []
                time_computation = []
                
                for i in range(len(matrix)):
                    result_inner = []
                    time_computation_similarity_per_user = []
                    for j in range(len(matrix)):
                        if i == j:
                            result_inner.append(1)
                            continue
                        t1 = time.time()
                        similarity_result = self.similarity_calculation(i, j, indexFold)
                        t2 = time.time()
                        time_computation_similarity_per_user += [t2-t1]
                        result_inner.append(similarity_result)
                    
                    time_computation += [time_computation_similarity_per_user]
                    result_train.append(result_inner)
                
                result.append(result_train)
                self.time_computation_similarity_per_fold.append(time_computation)
            
            return result
        
    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    def show_similarity(self,indexFold : int|None = None) :
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        
        if indexFold is None :
            return pd.DataFrame(self.result_similarity)
        return pd.DataFrame(self.result_similarity[indexFold])
    
    def show_time_computation_similarity_array(self, indexFold : int|None = None) : 
        if self.toyData :
            ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
        
        if self.time :
            if indexFold is None :
                return self.time_computation_similarity_per_fold
            return self.time_computation_similarity_per_fold[indexFold]

        ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")

    def show_time_computation_similarity(self, indexFold : int|None = None) : 
        if self.toyData :
            ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
        
        if self.time :
            if indexFold is None :
                return pd.DataFrame(self.time_computation_similarity_per_fold)
            return pd.DataFrame(self.time_computation_similarity_per_fold[indexFold])

        ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")