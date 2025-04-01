import pandas as pd
import helper.helper as hp
import numpy as np
from MatrixRating import MatrixRating
class Mean(MatrixRating) :
    '''
    Measurement of mean of the user-item matrix rating

    Args:
    -----
        data : matrix rating
        opsional : Determinate based of threating matrix such a , user-based or item-based

    Attributes:
    -----------
        matrix : user-item matrix rating
        result_mean : Result mean of matrix rating

    Methods:
    --------
        numerator(self,vector)
            Sum the vector as a numerator of measurement mean

        denominator(self,vector)
            Determinate the number of member vector except of zero value

        mean_calculation(self)
            Measurement mean to user-item matrix rating
    '''
    def __init__(self,data,*,opsional="user-based"):
        self.opsional = opsional
        self.matrix = np.array(data) if opsional == "user-based" else hp.reverseMatrix(np.array(data))
        self.result_mean = self.mean_calculation()
        self.result_mean_centered = self.mean_centered_calculation()

    def __numerator(self,vector) -> int:
        return sum(vector)

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

    def show_mean(self) :
        return {
            "mean" : pd.DataFrame(self.result_mean),
            "mean_centered" : pd.DataFrame(self.result_mean_centered if self.opsional == "user-based" else hp.reverseMatrix(self.result_mean_centered))
        }