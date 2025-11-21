from abc import ABC, abstractmethod
class Similarity(ABC) :
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

    @abstractmethod
    def numerator(self, vector1:list, vector2:list) -> int: ...

    @abstractmethod
    def denominator(self, vector1:list, vector2:list) -> int: ...

    @abstractmethod
    def similarity_calculation(self) -> list[float]: ...
    
    @abstractmethod
    def main_calculation(self) -> list[list[float]]: ...
