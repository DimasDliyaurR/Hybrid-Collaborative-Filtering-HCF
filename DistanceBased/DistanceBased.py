from abc import abstractmethod

class DistanceBased :
    '''
    Property of Distance based when calculation similarity

    Methods:
    --------
    numerator(vector1,vector2)
        Measurement the value of numerator
    
    denominator(vector1,vector2)
        Measurement the value of denominator

    mean_calculation(u, v, matrix)
        Calculation similarity of the user-item matrix rating
    '''

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
    def main_calculation(self) -> list[list[float]]: ...
    
    @property
    @abstractmethod
    def similarity_result(self) -> list[list[float]]: ...
