import pandas as pd
from numpy import transpose
from MatrixRating import MatrixRating
class Mean(MatrixRating) :

    def __init__(self, data,*,opsional="user-based", toyData : bool = False) :
        super().__init__(data,toyData=toyData)
        self.opsional = opsional
        if toyData :
            self.result_mean = self.__mean_calculation()
            self.result_mean_centered = self.__mean_centered_calculation()
        else :
            self.result_mean_training = self.__mean_calculation()
            self.result_mean_centered_training = self.__mean_centered_calculation()
    
    def __numerator(self, u : int, indexFold : None|int = None) -> int:
        return sum(self.getItemWithValue(u,indexFold=indexFold) if self.opsional == "user-based" else self.getUserWithValue(u,indexFold=indexFold))

    def __denominator(self, u : int, indexFold : None|int = None) -> int:
        return len(self.getItem(u,indexFold=indexFold) if self.opsional == "user-based" else self.getUser(u,indexFold=indexFold))

    def __mean_calculation(self) -> list[float]:
        if not self.toyData :
            return [ [ (self.__numerator(u,indexFold=indexFold)/self.__denominator(u,indexFold=indexFold)) if self.__denominator(u, indexFold=indexFold) != 0 else 0 for u in range(len(self.train[indexFold] if self.opsional == "user-based" else self.train[indexFold][0])) ] for indexFold in range(len(self.train)) ]
        return [ (self.__numerator(u)/self.__denominator(u)) if self.__denominator(u) != 0 else 0 for u in range(len(self.matrixRating if self.opsional == "user-based" else self.reverseMatrixRating))]
    
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
        
        return [[ ((self.matrixRating[u][i] - self.result_mean[u]) if self.opsional == "user-based" else (self.matrixRating[u][i] - self.result_mean[i])) if self.matrixRating[u][i] != 0 else 0 for i in range(len(self.matrixRating[u]))] for u in range(len(self.matrixRating))]

    def show_mean_centered(self, indexFold : bool |None = None) :
        if not self.toyData :
            return pd.DataFrame(self.result_mean_centered_training[indexFold])
        return pd.DataFrame(self.result_mean_centered)
    
    def show_mean(self, indexFold : bool | None = None) :
        if not self.toyData :
            if indexFold is None :
                return pd.DataFrame(self.result_mean_training)
            return pd.DataFrame(self.result_mean_training[indexFold])
        return pd.DataFrame(self.result_mean)