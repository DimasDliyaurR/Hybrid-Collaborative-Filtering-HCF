import numpy as np 
import pandas as pd
from operator import itemgetter

class MatrixRating():

    def __init__(self, data : list[list[float]] | None ,*, toyData : bool = False):
        """
        Parameters :
            matrix : str | list[list[float]]
                The path of folder movielens or matrix rating

            data : str | list[list[float]]
                The path of data or Data matrix

            reverseMatrixRating : list[list[float]]
                Reverse of matrix rating
        """
        self.data = data
        self.toyData = toyData
        if toyData :
            if type(data) is str :
                raise ValueError("matrix should be type list")

            self.matrixRating = data
            self.reverseMatrixRating = np.transpose(self.matrixRating).tolist()
        else :
            if type(data) is list :
                raise ValueError("data should be type str")

            self.train = [self.__processDataTrain(i) for i in range(1,6)]
            self.test = [self.__processDataTest(i) for i in range(1,6)]
        
        self.matrix_component_of_item = self.__dozenOfItem()
        self.matrix_component_of_user = self.__dozenOfUser()

    def __dozenOfItem(self) -> dict :
        if self.toyData :
            return {
                "unrated" : [[j for j in range(len(self.matrixRating[0])) if self.matrixRating[i][j] == 0 ] for i in range(len(self.matrixRating))],
                "rated" : [[j for j in range(len(self.matrixRating[0])) if self.matrixRating[i][j] != 0 ] for i in range(len(self.matrixRating))]
            }

        return {
            "unrated" : [[[ j for j in range(len(self.train[indexTrain][i])) if self.train[indexTrain][i][j] == 0] for i in range(len(self.train[indexTrain]))] for indexTrain in range(len(self.train))],
            "rated" : [[[ j for j in range(len(self.train[indexTrain][i])) if self.train[indexTrain][i][j] != 0] for i in range(len(self.train[indexTrain]))] for indexTrain in range(len(self.train))],
        }

    def __dozenOfUser(self) -> dict :
        if self.toyData :
            return {
                "unrated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] == 0 ] for i in range(len(self.reverseMatrixRating))],
                "rated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] != 0 ] for i in range(len(self.reverseMatrixRating))]
            }
        
        transpose_matrix = [np.transpose(matrix) for matrix in self.train]
        return {
            "unrated" : [[[ j for j in range(len(transpose_matrix[indexTrain][i])) if transpose_matrix[indexTrain][i][j] == 0] for i in range(len(transpose_matrix[indexTrain]))] for indexTrain in range(len(self.train))],
            "rated" : [[[ j for j in range(len(transpose_matrix[indexTrain][i])) if transpose_matrix[indexTrain][i][j] != 0] for i in range(len(transpose_matrix[indexTrain]))] for indexTrain in range(len(self.train))],
        }

    def getItem(self, user : int, indexTrain : int|None = None,*, interacted : bool = True) -> list[int]:
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

        label = "rated" if interacted else "unrated"

        if self.toyData :
            return self.matrix_component_of_item[label][user]
        else :
            if type(indexTrain) is None :
                raise ValueError("Index of train is missing")
            return self.matrix_component_of_item[label][indexTrain][user]

    def getUser(self ,item : int,indexTrain : int|None = None,*,interacted: bool=True) -> list[float] :
        """
        Get set of user have rated by specific item
            Representation of U_i or \widehat{U}_i (depend on parameter interacted) Notation
        ------------------------------------------------------------------------------------
        Parameters
        ----------
            item : int
                specific item
            interacted : bool
                The item already rated or not by user

        Returns
        -------
            list[float] : Set of item
        """
        label = "rated" if interacted else "unrated"
        
        if self.toyData :
            return self.matrix_component_of_user[label][item]
        else :
            if type(indexTrain) is None :
                raise ValueError("Index of train is missing")
            return self.matrix_component_of_user[label][indexTrain][item]

    def getItemWithValue(self,data : list ,index : int|None = None) -> list[int] :
        return list(itemgetter(*self.getItem(data,index))(data[index])) if len(self.getItem(data,index)) > 1 else [data[index][self.getItem(data,index)[0]]] 

    def getUserWithValue(self, data : list, index : int|None = None) -> list[int] :
        data = np.transpose(data)
        return list(itemgetter(*self.getUser(index))(data[index])) if len(self.getUser(index)) > 1 else [data[index][self.getUser(index)[0]]] 

    @staticmethod
    def __transformationData(data : list) -> pd :
        matrix_rating = pd.DataFrame(np.zeros((943,1682)),index=list(range(1,944)),columns=list(range(1,1683))).rename_axis(index="user_id",columns="item_id")
        data_old = data.pivot_table(index="user_id",columns="item_id",values="rating")
        data_old = data_old.fillna(0)
        matrix_rating.update(data_old)
        return matrix_rating

    @staticmethod
    def __checkNumberOfRating(data) :
        NumberOfRating = 0

        for i in range(len(data)) :
            numberInnerOfRating = len(list(filter(lambda x : x!=0, data[i])))
            NumberOfRating += numberInnerOfRating
        return NumberOfRating

    def __processDataTrain(self, fold) -> list[list[float]] :
        """
        Convert Dataset into Matriks
        -----------------------------------

        Returns
        -------
            list[list[float]] : Set of user
        """

        train = pd.read_csv(f"{self.data}/u{str(fold)}.base",sep="\t", names=["user_id","item_id","rating","timestamp"])

        result = np.array(self.__transformationData(train)).tolist()
        return result
    
    def __processDataTest(self, fold) -> list[list[float]]:
        """
        Convert Dataset into Matriks
        -----------------------------------

        Returns
        -------
            list[list[float]] : Set of user
        """

        test = pd.read_csv(f"{self.data}/u{fold}.test",sep="\t", names=["user_id","item_id","rating","timestamp"])
        result = np.array(self.__transformationData(test)).tolist()
        return result
    
    def showMatrix(self,u) -> pd :
        """
        Show Matrix Rating
        -----------------------------------

        Returns
        ------
            Pandas
        """
        return pd.DataFrame(self.train[u])
