from Evaluation import Evaluation, Precision
import numpy as np
from typing_extensions import override
from MatrixRating import MatrixRating

class AP(Evaluation) :

    def __init__(self, data, top_n, *, matrix_object : MatrixRating, path_evaluation = None, toyData, N = 30, n = 20):
        self.precision = Precision(data, top_n ,matrix_object=matrix_object, path_evaluation=path_evaluation, toyData=toyData, N=N)
        super().__init__(data, top_n, matrix_object=matrix_object, path_evaluation=path_evaluation, toyData=toyData, N=N, n=n)

    def groundTruth(self, u : None | int = None, indexFold : int|None = None) -> np.array :
        if not self.toyData :
            data_test = self.matrix_object.getItemTest(u,indexFold=indexFold)
            return np.array([1 if (self.get_top_n_specific_user(u,indexFold=indexFold)[:i+1][-1] in data_test) and len(data_test) > i else 0 for i in range(self.N)])
        return np.array([1 if (self.top_n[:i+1][-1] in self.data_test) and len(self.top_n) > i else 0 for i in range(self.N)])    

    @override
    def result(self, u = None, indexFold = None):
        return self.precision.result(u,indexFold=indexFold) * self.groundTruth(u,indexFold=indexFold)