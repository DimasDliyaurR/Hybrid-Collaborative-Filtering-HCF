import pandas as pd
from prediction import Prediction
from DistanceBased import Mean, Similarity
from MatrixRating import MatrixRating
from operator import itemgetter

class HybridCollaborativeFiltering(MatrixRating) :

    def __init__(self, similarity_user_based : Prediction, similarity_item_based : Prediction,*,data : str|None = None, gamma : float, N : int = 2) -> None :
        self.gamma = gamma
        self.N = N
        self.user_based = similarity_user_based
        self.item_based = similarity_item_based

        self.__validationObject()
        self.toyData = similarity_user_based.toyData
        self.k = similarity_user_based.k
        MatrixRating.__init__(self,data,toyData=self.user_based.toyData)

        self.prediction_user_based = self.user_based.get_prediction_array()
        self.prediction_item_based = self.item_based.get_prediction_array()

        self.result_hybrid = self.main_calculation()
        self.topN = self.get_top_n()

    def __validationObject(self) :
        """ 
            Rule :
            ------
            1. Similaritas harus sama
            2. Parameter toyData harus sama
            3. Parameter K harus sama
            4. Method harus beda (UCF dan ICF)

            Returns :
            ---------
            None
        """

        if self.__checkType() and self.__checkToyData() and self.__checkKParam() and self.__checkOpsionalParam() :
            return

        if not self.__checkType() :
            raise ValueError("The Type between object should be same !")
        elif not self.__checkToyData() :
            raise ValueError("The toyData parameter between object should be same !")
        elif not self.__checkKParam() :
            raise ValueError("The K parameter between object should be same !")
        elif not self.__checkOpsionalParam() :
            raise ValueError("The opsional parameter between object should be different !")
    
    def __checkType(self) :
        return type(self.user_based) == type(self.user_based)

    def __checkToyData(self) :
        return self.user_based.toyData == self.item_based.toyData

    def __checkKParam(self) :
        return self.user_based.k == self.item_based.k

    def __checkOpsionalParam(self) :
        return self.user_based.opsional != self.item_based.opsional

    def fusion(self,user : int,item : int,*,indexTrain : int|None = None) -> float:
        if self.toyData :
            return (self.gamma * self.prediction_user_based[user][item] + (1-self.gamma) * self.prediction_item_based[user][item])
        return (self.gamma * self.prediction_user_based[indexTrain][user][item] + (1-self.gamma) * self.prediction_item_based[indexTrain][user][item])

    def main_calculation(self) -> list[list[float]]:
        if self.toyData :
            return [[(self.fusion(user,item)) for item in self.getItem(user)]for user in range(len(self.matrixRating))]

        result = []
        for indexTrain in range(len(self.train)) :
            result_train = []
            for user in range(len(self.train[indexTrain])) :
                result_inner = []
                unrated_item = self.getItem(user,indexTrain=indexTrain,interacted=False)
                for item in range(len(self.train[indexTrain][user])) :
                    result_inner.append(self.fusion(user,item,indexTrain=indexTrain) if item in unrated_item else self.train[indexTrain][user][item])
                result_train.append(result_inner)
            result.append(result_train)
        return result

    def get_data_frame(self,indexTrain : None|int = None) -> pd :
        """
        Mengembalikan hasil prediksi dalam bentuk dataframe

        Returns:
        --------
        object
             Data prediksi
        """
        if self.user_based.toyData :
            return pd.DataFrame(self.result_hybrid)
        
        if indexTrain is None :
            return pd.DataFrame(self.result_hybrid)
        return pd.DataFrame(self.result_hybrid[indexTrain])
    
    def get_top_n(self) :
        if self.toyData :
            result = []
            for i in range(len(self.matrixRating)) :
                print(f"Train index = {indexTrain}")
                unratedItem = self.getItem(i,interacted=False)
                
                if len(unratedItem) > 1 :
                    # valueOfPrediction = itemgetter(*unratedItem)(self.prediction[i]) if len(unratedItem) > 1 else self.prediction[i][unratedItem[0]]

                    if len(unratedItem) < 1 :
                        result_inner.append([])

                    result.append(unratedItem,key=lambda x: self.result_hybrid[i][x],reverse=True) if len(unratedItem) > 1 else self.result_hybrid[i][unratedItem[0]]
                else :
                    result.append(unratedItem)
            return result
        else :
            result = []
            for indexTrain in range(len(self.train)) :
                print(f"Train index = {indexTrain}")
                result_inner = []
                for u in range(len(self.train[indexTrain])) :
                    unratedItem = self.getItem(u,indexTrain=indexTrain,interacted=False)
                    if len(unratedItem) > 1 :
                        # Prediction training : 5 x 943 x 1682 
                        # Prediction training : indexTrain x u x {iteration}
                        # If number of unrated Item have less then 1 : Acces Prediction training used index
                        # However, if The number of unrated item have more than 1 , Acces prediction should be with itemgetter function
                        # valueOfPrediction = itemgetter(*unratedItem)(self.prediction[indexTrain][u]) if len(unratedItem) > 1 else self.prediction[indexTrain][u][unratedItem[0]]

                        if len(unratedItem) == 0 :
                            result_inner.append([])
                            continue
                        
                        # Algoritma Seharusnya disamakan dengan sebelumnya (Skripsi Tahun Kemarin)
                        sorted_array = sorted(unratedItem,key=lambda x: self.result_hybrid[indexTrain][u][x],reverse=True) if len(unratedItem) > 1 else self.result_hybrid[indexTrain][u][unratedItem[0]]

                        result_inner.append(sorted_array)
                    else :
                        result_inner.append(unratedItem)
                result.append(result_inner)
            return result
    
    def get_top_n_fusion(self) :
        """
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi hasil prediksi.
        """
        return pd.DataFrame(self.topN)
