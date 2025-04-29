import pandas as pd
from prediction import Prediction
from DistanceBased import Mean, Similarity
from MatrixRating import MatrixRating
from operator import itemgetter

class HybridCollaborativeFiltering(Similarity,Prediction) :

    def __init__(self, similarity_user_based : Prediction, similarity_item_based : Prediction,*,data : str|None = None, gamma : float, N : int = 2) -> None :
        self.gamma = gamma
        self.N = N
        self.user_based = similarity_user_based
        self.item_based = similarity_item_based

        self.__validationObject()
        self.toyData = similarity_user_based.toyData
        self.k = similarity_user_based.k
        MatrixRating.__init__(data,toyData=self.toyData)

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
            result.append([[(self.fusion(user,item))for item in self.getItem(user,indexTrain=indexTrain,interacted=False)] 
                            for user in range(len(self.train[indexTrain]))])
        return result

    def get_data_frame(self) -> pd :
        """
        Mengembalikan hasil prediksi dalam bentuk dataframe

        Returns:
        --------
        object
             Data prediksi
        """
        return pd.DataFrame(self.result_hybrid)
    
    def get_top_n(self) :
        """
        Mengembalikan hasil dari Top-N dari prediksi

        Returns:
        --------
        array
            Array yang berisi tentang Top-N
        """
        result = []
        for i in range(len(self.matrixRating)) :
            valuePrediction = itemgetter(*self.getItem(i,interacted=False))(self.result_hybrid[i]) if len(self.getItem(i,interacted=False)) > 1 else self.result_hybrid[i][self.getItem(i,interacted=False)[0]]
            result.append(sorted(len(range(valuePrediction)),key=lambda x : self.result_hybrid[i][x],reverse=True)[:self.k])
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
