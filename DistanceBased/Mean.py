import pandas as pd
from numpy import transpose
from MatrixRating import MatrixRating
class Mean(MatrixRating) :
    def __init__(self, data,*,opsional="user-based", toyData : bool = False):
        super().__init__(data,toyData=toyData)
        self.opsional = opsional
        if toyData :
            self.result_mean = self.__mean_calculation()
            print(self.result_mean)
            self.result_mean_centered = self.__mean_centered_calculation()
        else :
            self.result_mean_training = self.__mean_calculation()
            self.result_mean_centered_training = self.__mean_centered_calculation()
    
    @staticmethod
    def __numerator(vector) -> int:
        return sum(vector)

    @staticmethod
    def __denominator(vector) -> int:
        return len([i for i in vector if i != 0])

    def __mean_calculation(self) -> list[float]:
        if not self.toyData :
            return [ [ (self.__numerator(vector)/self.__denominator(vector)) if self.__denominator(vector) != 0 else 0 for vector in (self.train[trainIndex] if self.opsional == "user-based" else transpose(self.train[trainIndex])) ] for trainIndex in range(len(self.train)) ]
        return [ (self.__numerator(vector)/self.__denominator(vector)) if self.__denominator(vector) != 0 else 0 for vector in (self.matrixRating if self.opsional == "user-based" else self.reverseMatrixRating)]
    def __mean_centered_calculation(self) -> list[float]:
        if not self.toyData :
            
            result_training = []
            for trainIndex in range(len(self.train)) :
                result = []
                for index,vector in enumerate(self.train[trainIndex] if self.opsional == "user-based" else transpose(self.train[trainIndex])) :
                    result_inner = []
                    for item in vector :
                        result_inner.append((item - self.result_mean_training[trainIndex][index]) if item != 0 else 0)
                    result.append(result_inner)
                result_training.append(result if self.opsional == "user-based" else transpose(result))
            return result_training

            # return [
            #         [
            #             [
            #                 (item - self.result_mean_training[trainIndex][index]) if item != 0 else 0 for item in vector
            #             ] 
            #             for index,vector in enumerate(self.train[trainIndex] if self.opsional == "user-based" else transpose(self.train[trainIndex]))
            #         ]
            #             for trainIndex in range(len(self.train))
            #         ]
        
        return [[ ((self.matrixRating[u][i] - self.result_mean[u]) if self.opsional == "user-based" else (self.matrixRating[u][i] - self.result_mean[i])) if self.matrixRating[u][i] != 0 else 0 for i in range(len(self.matrixRating[u]))] for u in range(len(self.matrixRating))]

    def show_mean_centered(self, indexTrain : bool |None = None) :
        if not self.toyData :
            return pd.DataFrame(self.result_mean_centered_training[indexTrain])
        return pd.DataFrame(self.result_mean_centered)
    
    def show_mean(self, indexTrain : bool | None = None) :
        if not self.toyData :
            if indexTrain is None :
                return pd.DataFrame(self.result_mean_training)
            return pd.DataFrame(self.result_mean_training[indexTrain])
        return pd.DataFrame(self.result_mean)