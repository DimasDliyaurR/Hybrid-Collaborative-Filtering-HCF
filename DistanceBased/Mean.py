import pandas as pd
import helper.helper as hp

class Mean() :
    '''
    Calculation of mean of user-item matrix rating
    
    Args:
        data : matrix rating
        opsional : 

    Attributes:
    ----------
        matrix : user-item matrix rating
        result_mean : Result mean of matrix rating

    Methods:
    --------
        numerator(self,vector)
            Sum the vector as a numerator of calculation mean
        
        denominator(self,vector)
            Count the vector except number of 0 as a denominator of calculation mean

        mean_calculation(self)
            Calculation mean to user-item matrix rating
    '''
    def __init__(self,data,*,opsional="user-based"):
        self.matrix = data if opsional == "user-based" else hp.reverseMatrix(data)
        self.result_mean = self.mean_calculation()
        self.result_mean_centered = self.mean_centered_calculation()

    def __numerator(self,vector) -> int:
        return sum([i for i in vector])

    def __denominator(self,vector) -> int:
        return len([i for i in vector if i != 0])

    def mean_calculation(self) -> list[float]:
        return [(self.__numerator(vector)/self.__denominator(vector)) for vector in self.matrix]

    def mean_centered_calculation(self) -> list[float]:
        return [
                [
                    (item - self.result_mean[index]) if item != 0 else 0 for item in vector
                ] 
                for index,vector in enumerate(self.matrix)
            ]

    def show(self) :
        return {
            "mean" : pd.DataFrame(self.result_mean),
            "mean_centered" : pd.DataFrame(self.result_mean_centered)
        }