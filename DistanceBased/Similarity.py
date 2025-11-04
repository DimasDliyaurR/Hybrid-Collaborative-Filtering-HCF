from abc import abstractmethod
from prediction import Prediction

class Similarity(Prediction) :
    '''
    Property of Distance based when calculation similarity

    Methods:
    --------
        numerator(vector1,vector2)
            Measurement the value of numerator by threating matrix as a vector
        
        denominator(vector1,vector2)
            Measurement the value of denominator by threating matrix as a vector

        similarity_calculation(u, v, matrix)
            Provide the result similarity of the user-item matrix rating as a vector

        main_calculation(u, v, matrix)
            Provide measurement similarity of the user-item matrix rating as a matrix
    '''

    def __init__(self, data, mean_object, similarity = None, *, opsional = None, hybrid = False, k = None, path_file = None, toyData = False, time = False):
        super().__init__(data, mean_object, similarity, opsional=opsional, hybrid=hybrid, k=k, path_file=path_file, toyData=toyData, time=time)

    @property
    @abstractmethod
    def numerator(self, vector1:list, vector2:list) -> int: ...

    @property
    @abstractmethod
    def denominator(self, vector1:list, vector2:list) -> int: ...

    @property
    @abstractmethod
    def similarity_calculation(self) -> list[float]: ...

    @property
    @abstractmethod
    def similarity_result(self) -> list[list[float]]: ...
    
    @property
    @abstractmethod
    def main_calculation(self) -> list[list[float]]: ...
