from Evaluation import Evaluation,Recall, Precision
from typing_extensions import override
from MatrixRating import MatrixRating
import numpy as np

class F1_score(Evaluation) :

    def __init__(self, data, top_n, *, matrix_object : MatrixRating, path_evaluation = None, toyData, N = 30, n = 20):
        self.recall = Recall(data, top_n,matrix_object=matrix_object, path_evaluation=path_evaluation, toyData=toyData, N=N,n=n)
        self.precision = Precision(data, top_n,matrix_object=matrix_object, path_evaluation=path_evaluation, toyData=toyData, N=N,n=n)
        
        self.__precision_result = self.precision.result_evaluation
        self.__recall_result = self.recall.result_evaluation
        
        self.__numerator_process_result = self.__numerator_process()
        self.__denominator_process_result = self.__denominator_process()

        super().__init__(data, top_n, matrix_object=matrix_object, path_evaluation=path_evaluation, toyData=True, N=N, n=n)
    
    @staticmethod
    def __result_process(num,denom) :

        num = np.array(num)
        denom = np.array(denom)

        non_zero_mask = denom != 0

        # Initialize result array with a default value (e.g., 0)
        result = np.zeros_like(num, dtype=float)

        # Perform division only where denominator is not zero
        result[non_zero_mask] = num[non_zero_mask] / denom[non_zero_mask]
        return result

    def __numerator_process(self) -> list :
        return [2*(np.array(self.__recall_result[fold]) * np.array(self.__precision_result[fold])) for fold in range(len(self.__recall_result))]
    
    def numerator(self,u : None|int = None, indexFold : None|int = None) -> float :
        return self.__numerator_process_result[indexFold]

    def __denominator_process(self) -> list :
        return [np.array(self.__recall_result[fold]) + np.array(self.__precision_result[fold]) for fold in range(len(self.__recall_result))]
    
    def denominator(self,u : None|int = None, indexFold : None|int = None) -> float :
        return self.__denominator_process_result[indexFold]
        
    @override
    def result(self, u = None, indexFold = None):
        return [self.__result_process(self.numerator(u,indexFold=fold),self.denominator(u,indexFold=fold)) for fold in range(len(self.__recall_result))]