import DistanceBased as SDB
import prediction as P
from typing_extensions import override
import pandas as pd
from helper.helper import reverseMatrix
from MatrixRating import MatrixRating


class DiceCoefficient(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, *, opsional="user-based",k=2):
        MatrixRating.__init__(self,data)
        SDB.Mean.__init__(self,self.matrixRating, reverseMatrix(self.matrixRating),opsional=opsional)
        self.__opsional = opsional
        self.__data = self.matrixRating if opsional == "user-based" else self.reverseMatrixRating
        self.result_similarity = self.main_calculation()
        P.Prediction.__init__(self,data,opsional,self.result_similarity,k=k)

    @override
    def numerator(self, A:list, B:list) -> float:
        return 2*len(list( A & B ))

    @override
    def denominator(self, A:list, B:list) -> float:
        return len(A) + len(B)

    @override
    def similarity_calculation(self,A: int, B: int) -> float :

        setA = set(self.getItem(A) if self.__opsional == "user-based" else self.getUser(A))
        setB = set(self.getItem(B) if self.__opsional == "user-based" else self.getUser(B))
        
        return self.numerator(setA,setB) / self.denominator(setA,setB)

    @override
    def main_calculation(self):
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
        print("Sim selesai")
        return result

    def similarity_result(self) -> list[list[float]]:
        return self.result_similarity

    @override
    def show(self) :
        return pd.DataFrame(self.result_similarity)