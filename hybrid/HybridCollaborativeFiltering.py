import pandas as pd
from prediction import Prediction
from DistanceBased import Similarity
import DistanceBased.similarities as S
from Evaluation import NDCG, Precision, Recall
from MatrixRating import MatrixRating

class EntryClass :
    def __init__(self, **kwargs):
        
        if kwargs.get("NDCG") :
            self.kwarg = kwargs
        
        if kwargs.get("Precision") :
            self.kwarg = kwargs
        
        if kwargs.get("Recall") :
            self.kwarg = kwargs

class GetClassEvaluation :
    def __init__(self, **kwargs) :
        if kwargs.get("NDCG") :
            self.name = NDCG
        
        if kwargs.get("Precision") :
            self.name = Precision
        
        if kwargs.get("Recall") :
            self.name = Recall
        

class HybridCollaborativeFiltering(NDCG,Prediction,MatrixRating) :

    def __init__(self,
                data : str,
                object : Similarity, 
                *, 
                k_user : int,
                k_item : int,
                gamma : float,
                toyData : None|bool = False, 
                N : int = 100,
                n : int|None = None,
                **kwargs
            ) -> None :

        self.gamma = gamma
        self.N = N

        if object == S.TI :
            if kwargs.keys() not in ["alpha_1","alpha_2"] :
                raise ValueError("Parameter Alpha 1 dan alpha 2 seharusnya ada")
            self.user_based = object(data,opsional="user-based",k=k_user,alpha_1=kwargs["alpha_1"],alpha_2=kwargs["alpha_2"],toyData=toyData)
            self.item_based = object(data,opsional="item-based",k=k_item,alpha_1=kwargs["alpha_1"],alpha_2=kwargs["alpha_2"],toyData=toyData)
        else :
            self.user_based = object(data,opsional="user-based",k=k_user,toyData=toyData)
            self.item_based = object(data,opsional="item-based",k=k_item,toyData=toyData)

        self.toyData = toyData

        MatrixRating.__init__(self,data,toyData=toyData)

        self.prediction_user_based = self.user_based.get_prediction_array()
        self.prediction_item_based = self.item_based.get_prediction_array()

        self.result_hybrid = self.main_calculation()
        Prediction.__init__(self,data,prediction=self.result_hybrid,hybrid=True)
        NDCG.__init__(self,data,self.topN,toyData=toyData,N=N,n=n)

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
    
    def get_top_n_fusion(self) :
        """
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi hasil prediksi.
        """
        return pd.DataFrame(self.topN)
