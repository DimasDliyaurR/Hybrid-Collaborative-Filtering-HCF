import numpy as np 
import pandas as pd
from operator import itemgetter

Vector = list[float|int]
Matrix = list[list[float|int]]

class MatrixRating():

    def __init__(self, data : Matrix | str ,*, toyData : bool = False):
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
                raise ValueError(f"{self} matrix should be type list")

            self.matrixRating = data
            self.reverseMatrixRating = np.transpose(self.matrixRating).tolist()
        else :
            if type(data) is list :
                raise ValueError(f"{self} data should be type str")
            
            # Keseuluruhan Dataset
            self.dataset = pd.read_csv(f"{self.data}/u.data",sep="\t", names=["user_id","item_id","rating","timestamp"])
            
            numberOfUserAndItem = self.__getNumberOfUserAndItemData()            
            self.numberOfUser = numberOfUserAndItem["users"]
            self.numberOfItem = numberOfUserAndItem["items"]
            
            self.maxIdOfUserAndItem = self.__getMaxOfIdUserAndItemData()            
            self.maxIdOfUser = numberOfUserAndItem["users"]
            self.maxIdOfItem = numberOfUserAndItem["items"]
            
            self.uniqueIdOfUserAndItemTrain = self.__getUniqueOfUserAndItemDataTrain()

            self.uniqueIdOfUserAndItemTest = self.__getUniqueOfUserAndItemDataTest() 

            self.train = [self.__processDataTrain(i) for i in range(1,6)]
            self.test = [self.__processDataTest(i) for i in range(1,6)]

            self.train_transpose = [np.transpose(matrix) for matrix in self.train]
            self.test_transpose = [np.transpose(matrix) for matrix in self.test]
        
        self.__matrix_component_of_item = self.__dozenOfItem()
        self.__matrix_component_of_user = self.__dozenOfUser()
        self.__matrix_component_of_item_test = self.__dozenOfItemTest()
        self.__matrix_component_of_user_test = self.__dozenOfUserTest()

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

    def __dozenOfUser(self) -> dict[str : Matrix ] :
        if self.toyData :
            return {
                "unrated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] == 0 ] for i in range(len(self.reverseMatrixRating))],
                "rated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] != 0 ] for i in range(len(self.reverseMatrixRating))]
            }
        
        return {
            "unrated" : [[[ j for j in range(len(self.train_transpose[indexTrain][i])) if self.train_transpose[indexTrain][i][j] == 0] for i in range(len(self.train_transpose[indexTrain]))] for indexTrain in range(len(self.train))],
            "rated" : [[[ j for j in range(len(self.train_transpose[indexTrain][i])) if self.train_transpose[indexTrain][i][j] != 0] for i in range(len(self.train_transpose[indexTrain]))] for indexTrain in range(len(self.train))],
        }
    
    def __dozenOfItemTest(self) -> dict :

        return {
            "unrated" : [[[ j for j in range(len(self.test[indexTrain][i])) if self.test[indexTrain][i][j] == 0] for i in range(len(self.test[indexTrain]))] for indexTrain in range(len(self.test))],
            "rated" : [[[ j for j in range(len(self.test[indexTrain][i])) if self.test[indexTrain][i][j] != 0] for i in range(len(self.test[indexTrain]))] for indexTrain in range(len(self.test))],
        }

    def __dozenOfUserTest(self) -> dict[str : Matrix ] :
        if self.toyData :
            return {
                "unrated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] == 0 ] for i in range(len(self.reverseMatrixRating))],
                "rated" : [[j for j in range(len(self.reverseMatrixRating[0])) if self.reverseMatrixRating[i][j] != 0 ] for i in range(len(self.reverseMatrixRating))]
            }
        
        return {
            "unrated" : [[[ j for j in range(len(self.test_transpose[indexTrain][i])) if self.test_transpose[indexTrain][i][j] == 0] for i in range(len(self.test_transpose[indexTrain]))] for indexTrain in range(len(self.test))],
            "rated" : [[[ j for j in range(len(self.test_transpose[indexTrain][i])) if self.test_transpose[indexTrain][i][j] != 0] for i in range(len(self.test_transpose[indexTrain]))] for indexTrain in range(len(self.test))],
        }

    def __getUniqueOfUserAndItemDataTrain(self) -> dict[int] :
        train_per_train_user = []
        train_per_train_item = []
        for index in range(1,6) :
            df = pd.read_csv(f"data/ml-100k/u{index}.base",sep="\t", names=["user_id","item_id","rating","timestamp"])
            train_per_train_user.append(pd.unique(df["user_id"]))
            train_per_train_item.append(pd.unique(df["item_id"]))
        
        return {
                "users" : train_per_train_user,
                "items" : train_per_train_item
            }
    
    def __getUniqueOfUserAndItemDataTest(self) -> dict[int] :
        train_per_test_user = []
        train_per_test_item = []
        for index in range(1,6) :
            df = pd.read_csv(f"data/ml-100k/u{index}.test",sep="\t", names=["user_id","item_id","rating","timestamp"])
            train_per_test_user.append(pd.unique(df["user_id"]))
            train_per_test_item.append(pd.unique(df["item_id"]))
        
        return {
                "users" : train_per_test_user,
                "items" : train_per_test_item
            }
    
    def __getNumberOfUserAndItemData(self) -> dict[int] :        
        return {
            "users" : len(pd.unique(self.dataset["user_id"])),
            "items" : len(pd.unique(self.dataset["item_id"]))
            }

    def __getMaxOfIdUserAndItemData(self) -> dict[int] :  
        return {
            "users" : max(pd.unique(self.dataset["user_id"])),
            "items" : max(pd.unique(self.dataset["item_id"]))
            }
    
    def getUniqueIdOfUserTrain(self,indexTrain : None|int = None) :
        return self.uniqueIdOfUserAndItemTrain["users"][indexTrain]
    
    def getUniqueIdOfItemTrain(self,indexTrain : None|int = None) :
        return self.uniqueIdOfUserAndItemTrain["items"][indexTrain]
    
    def getUniqueIdOfUserTest(self,indexTrain : None|int = None) :
        return self.uniqueIdOfUserAndItemTest["users"][indexTrain]
    
    def getUniqueIdOfItemTest(self,indexTrain : None|int = None) :
        return self.uniqueIdOfUserAndItemTest["items"][indexTrain]
        


    def getItem(self, user : int, indexTrain : int|None = None,*, interacted : bool = True) -> Vector :
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
            return self.__matrix_component_of_item[label][user]
        else :
            if indexTrain is None :
                raise ValueError(f"Index of train is missing {self}")
            return self.__matrix_component_of_item[label][indexTrain][user]

    def getUser(self ,item : int,indexTrain : int|None = None,*,interacted: bool=True) -> Vector :
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
            return self.__matrix_component_of_user[label][item]
        else :
            if indexTrain is None :
                raise ValueError(f"Index of train is missing {self}")
            return self.__matrix_component_of_user[label][indexTrain][item]
    
    def getItemTest(self, user : int, indexTrain : int|None = None,*, interacted : bool = True) -> Vector :
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
            return self.__matrix_component_of_item_test[label][user]
        else :
            if indexTrain is None :
                raise ValueError(f"Index of train is missing {self}")
            return self.__matrix_component_of_item_test[label][indexTrain][user]

    def getUserTest(self ,item : int,indexTrain : int|None = None,*,interacted: bool=True) -> Vector :
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
            return self.__matrix_component_of_user_test[label][item]
        else :
            if indexTrain is None :
                raise ValueError(f"Index of train is missing {self}")
            return self.__matrix_component_of_user_test[label][indexTrain][item]

    def getItemWithValue(self,user : int, *,indexTrain : int|None = None, interacted : bool = True) -> Vector :
        if self.toyData :
            return list(itemgetter(*self.getItem(user,indexTrain,interacted=interacted))(self.matrixRating)) if len(self.getItem(user,indexTrain,interacted=interacted)) < 1 else [self.matrixRating[self.getItem(user,indexTrain,interacted=interacted)[0]]]
        else :
            return list(itemgetter(*self.getItem(user,indexTrain,interacted=interacted))(self.train[indexTrain])) if len(self.getItem(user,indexTrain,interacted=interacted)) > 1 else [self.train[indexTrain][self.getItem(user,indexTrain,interacted=interacted)[0]]]

    def getUserWithValue(self, item : int, indexTrain : int|None = None, interacted : bool = True) -> Vector :
        if self.toyData :
            return list(itemgetter(*self.getUser(item,indexTrain,interacted=interacted))(self.reverseMatrixRating)) if len(self.getUser(item,indexTrain,interacted=interacted)) > 1 else [self.reverseMatrixRating[self.getUser(item,indexTrain,interacted=interacted)[0]]]
        else :
            return list(itemgetter(*self.getUser(item,indexTrain,interacted=interacted))(self.train_transpose[indexTrain])) if len(self.getUser(item,indexTrain,interacted=interacted)) > 1 else [self.train_transpose[indexTrain][self.getUser(item,indexTrain,interacted=interacted)[0]]] 

    def transformationData(self,data : pd) -> pd :
        matrix_rating = pd.DataFrame(np.zeros((self.numberOfUser,self.numberOfItem)),index=list(range(1,self.numberOfUser+1)),columns=list(range(1,self.numberOfItem+1))).rename_axis(index="user_id",columns="item_id")
        data_old = data.pivot_table(index="user_id",columns="item_id",values="rating")
        data_old = data_old.fillna(0)
        matrix_rating.update(data_old)
        return matrix_rating

    @staticmethod
    def checkNumberOfRating(data) :
        NumberOfRating = 0

        for i in range(len(data)) :
            numberInnerOfRating = len(list(filter(lambda x : x!=0, data[i])))
            NumberOfRating += numberInnerOfRating
        return NumberOfRating

    def __processDataTrain(self, fold) -> Matrix :
        """
        Convert Dataset into Matriks
        -----------------------------------

        Returns
        -------
            list[list[float]] : Set of user
        """

        train = pd.read_csv(f"{self.data}/u{str(fold)}.base",sep="\t", names=["user_id","item_id","rating","timestamp"])

        result = np.array(self.transformationData(train)).tolist()
        return result
    
    def __processDataTest(self, fold) -> Matrix :
        """
        Convert Dataset into Matriks
        -----------------------------------

        Returns
        -------
            list[list[float]] : Set of user
        """

        test = pd.read_csv(f"{self.data}/u{fold}.test",sep="\t", names=["user_id","item_id","rating","timestamp"])
        result = np.array(self.transformationData(test)).tolist()
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
    
    def show_dataset(self):
        return self.transformationData(self.dataset)
