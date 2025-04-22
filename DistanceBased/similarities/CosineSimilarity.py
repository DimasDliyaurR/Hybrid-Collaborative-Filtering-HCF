import cmath

import DistanceBased as SDB
import prediction as P
from typing_extensions import override
import pandas as pd
from numpy import transpose
from MatrixRating import MatrixRating
from operator import mul,itemgetter
import os
import joblib
from Evaluation import NDCG


class CosineSimilarity (SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :

    def __init__(self, data,*, opsional="user-based", N : int|None = None, k=2, toyData: bool = False):
        SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
        
        if not toyData :
            if N is None :
                raise ValueError("NDCG parameter is missing")
            
            path_file = "cache/cosine/" + ("user_based" if self.opsional == "user-based" else "item_based") + "/cosine_similarity.joblib"
            path_file_prediction = "cache/cosine/" + ("user_based" if self.opsional == "user-based" else "item_based") + f"/prediction/{str(k)}/cosine_prediction.joblib"
            print(os.path.exists(path_file))
            if os.path.exists(path_file) :
                self.result_similarity = joblib.load(path_file)
            else :
                self.result_similarity = self.main_calculation()
                joblib.dump(self.result_similarity,path_file)
            NDCG.__init__(self,data,opsional,self.result_similarity,N=N, path_file=path_file_prediction, k=k,toyData=toyData)
        else :
            self.__data = self.matrixRating if opsional == "user-based" else self.reverseMatrixRating
            self.result_similarity = self.main_calculation()
            P.Prediction.__init__(self,data,opsional,self.result_similarity,k=k,toyData=toyData)

    @override
    def numerator(self,u:int,v:int, commonlyRated : list[int], matrix : list|None = None) -> float:
        if len(commonlyRated) == 0 :
            return 0
        if not self.toyData :
            return sum(map(mul, list(itemgetter(*commonlyRated)(matrix[u])) , list(itemgetter(*commonlyRated)(matrix[v])) )) if len(commonlyRated) > 1 else matrix[u][commonlyRated[0]] * matrix[v][commonlyRated[0]]
        return sum(map(mul, list(itemgetter(*commonlyRated)(self.__data[u])) , list(itemgetter(*commonlyRated)(self.__data[v])) )) if len(commonlyRated) > 1 else self.__data[u][commonlyRated[0]] * self.__data[v][commonlyRated[0]]

    @override
    def denominator(self,u : int,v : int, set1 : list[int] ,set2 : list[int], matrix : list|None = None) -> float:
        if len(set1) == 0 or len(set2) == 0 :
            return 0
        if not self.toyData :
            return cmath.sqrt( sum(list(map(lambda x : x**2, itemgetter(*set1)(matrix[u]) ))) if len(set1) > 1 else matrix[u][set1[0]]**2 ) * cmath.sqrt(sum(list(map(lambda x : x**2,itemgetter(*set2)(matrix[v])) )) if len(set2) > 1 else matrix[v][set2[0]]**2)
        return cmath.sqrt( sum(list(map(lambda x : x**2, itemgetter(*set1)(self.__data[u]) ))) if len(set1) > 1 else self.__data[u][set1[0]]**2 ) * cmath.sqrt(sum(list(map(lambda x : x**2,itemgetter(*set2)(self.__data[v])) )) if len(set2) > 1 else self.__data[v][set2[0]]**2)

    @override
    def similarity_calculation(self, u : int, v : int, indexTrain : int|None = None, matrix : list|None = None):

        if self.toyData :
            set1, set2 = self.getItem(u) if self.opsional == "user-based" else self.getUser(u) , self.getItem(v) if self.opsional == "user-based" else self.getUser(v)
            commonlyRated = list( set(set1) & set(set2) )
            numerator , denominator = self.numerator(u,v, commonlyRated).real, self.denominator(u,v, set1, set2).real
            return (numerator / denominator) if denominator != 0 and numerator != 0 else 0
        else :
            set1, set2 = self.getItem(u, indexTrain) if self.opsional == "user-based" else self.getUser(u,indexTrain) , self.getItem(v,indexTrain) if self.opsional == "user-based" else self.getUser(v,indexTrain)
            commonlyRated = list( set(set1) & set(set2) )
            numerator , denominator = self.numerator(u,v, commonlyRated,matrix).real, self.denominator(u,v, set1, set2,matrix).real
            return (numerator / denominator) if denominator != 0 and numerator != 0 else 0

    @override
    def main_calculation(self):

        if self.toyData :
            matrix = self.matrixRating if self.opsional == "user-based" else self.reverseMatrixRating
            
            result = [[] for _ in range(len(matrix))]
            
            for i in range(len(matrix)):
                for j in range(i, len(matrix)):
                    if i == j:
                        result[i].append(1)
                        continue
                    similarity_result = self.similarity_calculation(int(i), int(j))
                    result[i].append(similarity_result)
                    result[j].append(similarity_result)
            print("Sim : Selesai")
            return result

        else :
            result = []
            for indexTrain in range(len(self.train)) :
                matrix = self.train[indexTrain] if self.opsional == "user-based" else transpose(self.train[indexTrain])
                
                result_inner = [[] for _ in range(len(matrix))]
                for i in range(len(matrix)):
                    if i%10 == 0 :
                        print(f"Sim({i}) = Train {indexTrain}")
                    for j in range(i, len(matrix)):
                        if i == j:
                            result_inner[i].append(1)
                            continue
                        similarity_result = self.similarity_calculation(i, j, indexTrain, matrix)
                        result_inner[i].append(similarity_result)
                        result_inner[j].append(similarity_result)
                print("Sim : Selesai")
                result.append(result_inner)
            return result
    
    def similarity_result(self) :
        return self.result_similarity
        
    @override
    def show(self,u : int|None = None) :
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        if u is None :
            raise ValueError("Argument is missing!")
        return pd.DataFrame(self.result_similarity[u])
    
