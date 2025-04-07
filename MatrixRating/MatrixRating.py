import numpy as np 
import pandas as pd
from helper.helper import reverseMatrix
from operator import itemgetter

class MatrixRating():
    
    def __init__(self, matrix : str | list[list[float]]):
        """
        Parameters :
            matrix : str | list[list[float]]
                The path of dataset or Data matrix

            data : str | list[list[float]]
                The path of data or Data matrix

            reverseMatrixRating : list[list[float]]
                Reverse of matrix rating
        """
        self.matrixRating = self.__processData(matrix) if type(matrix) is str else matrix
        self.reverseMatrixRating = reverseMatrix(self.matrixRating)
        
    
    def getItem(self,user : int, *, interacted : bool = True) -> list[float]:
        """
        Get set of item have rated by specific user
            Representation of I_u or \widehat{I}_u (depend on parameter interacted) Notation
        ------------------------------------------------------------------------------------
        Parameters :
            user : int 
                specific user 

            interacted : bool 
                The item have rated or not the item

        Returns :
            list[float] : Set of item
        """
        return [i for i in range(len(self.matrixRating[user])) if self.matrixRating[user][i] != 0] if interacted else [i for i in range(len(self.matrixRating[user])) if self.matrixRating[user][i] == 0]

    def getUser(self,item : int,*,interacted: bool=True) -> list[float] :
        """
        Get set of user have rated by specific item
            Representation of U_i or \widehat{U}_i (depend on parameter interacted) Notation
        ------------------------------------------------------------------------------------
        Parameters
        ----------
            item : int 
                specific item
            interacted : bool 
                The item have rated or not the user

        Returns
        -------
            list[float] : Set of item
        """
        return [i for i in range(len(self.reverseMatrixRating[item])) if self.reverseMatrixRating[item][i] != 0] if interacted else [i for i in range(len(self.reverseMatrixRating[item])) if self.reverseMatrixRating[item][i] == 0]

    def getItemWithValue(self,index : int) -> list[int] :
        return list(itemgetter(*self.getItem(index))(self.matrixRating[index])) if len(self.getItem(index)) > 1 else [self.matrixRating[index][self.getItem(index)[0]]] 
    
    def getUserWithValue(self,index : int) -> list[int] :
        return list(itemgetter(*self.getUser(index))(self.reverseMatrixRating[index])) if len(self.getUser(index)) > 1 else [self.reverseMatrixRating[index][self.getUser(index)[0]]] 

    @staticmethod
    def __processData(matrix) -> list[list[float]]:
        """
        Convert Dataset into Matriks
        -----------------------------------

        Returns
        -------
            list[list[float]] : Set of user
        """

        data = pd.read_csv(matrix,sep="\t", names=["user_id","item_id","rating","timestamp"])

        matrix_rating = pd.DataFrame(np.zeros((943,1682)),index=list(range(1,944)),columns=list(range(1,1683))).rename_axis(index="user_id",columns="item_id")
        data_old = data.pivot_table(index="user_id",columns="item_id",values="rating")
        data_old = data_old.fillna(0)
        matrix_rating.update(data_old)

        return np.array(matrix_rating).tolist()
    
    def showMatrix(self) -> pd :
        """
        Show Matrix Rating
        -----------------------------------

        Returns
        ------
            Pandas
        """
        return pd.DataFrame(self.matrixRating)
