import pandas as pd
from prediction import Prediction
from DistanceBased import Similarity
import DistanceBased.similarities as S
from Evaluation import NDCG, Evaluation
from MatrixRating import MatrixRating
import os
import joblib
import time
from tqdm import tqdm

def HybridCollaborativeFilteringMain(
                data : str,
                object : Similarity, 
                *,
                object_evaluation : Evaluation = NDCG,
                k_user : int,
                k_item : int,
                gamma : float,
                toyData : None|bool = False, 
                N : int = 100,
                n : int|None = None,
                path_file : str|None,
                **kwargs
                ) :
    
    
    class HybridCollaborativeFiltering(object_evaluation,Prediction,MatrixRating) :

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
                    path_file : str|None,
                    time : bool = False,
                    **kwargs
                ) -> None :

            self.gamma = gamma
            self.N = N
            self.time = time

            if not toyData and path_file != None and os.path.exists(path_file) and not time :
                self.result_hybrid = joblib.load(path_file)
            else :

                if object == S.TI :
                    if "alpha_1" not in kwargs and "alpha_2" not in kwargs :
                        raise ValueError("Parameter Alpha 1 dan alpha 2 seharusnya ada")
                    
                    self.user_based = object(data,opsional="user-based",k=k_user,alpha_1=kwargs["alpha_1"],alpha_2=kwargs["alpha_2"],toyData=toyData,time=time)
                    
                    self.item_based = object(data,opsional="item-based",k=k_item,alpha_1=kwargs["alpha_1"],alpha_2=kwargs["alpha_2"],toyData=toyData,time=time)
                    
                else :
                    self.user_based = object(data,opsional="user-based",k=k_user,toyData=toyData,time=time)
                    self.item_based = object(data,opsional="item-based",k=k_item,toyData=toyData,time=time)

                self.toyData = toyData

                if not toyData and time :
                    self.time_computation_prediction_per_fold = []

                MatrixRating.__init__(self,data,toyData=toyData)

                
                self.prediction_user_based = self.user_based.get_prediction_array()
                
                self.prediction_item_based = self.item_based.get_prediction_array()

                self.result_hybrid = self.main_calculation()
            
            Prediction.__init__(self,data,prediction=self.result_hybrid,hybrid=True,toyData=toyData,time=time)
            object_evaluation.__init__(self,data,self.topN,toyData=toyData,N=N,n=n)

        def fusion(self,user : int,item : int,*,indexFold : int|None = None) -> float:
            if self.toyData :
                return (self.gamma * self.prediction_user_based[user][item] + (1-self.gamma) * self.prediction_item_based[user][item])
            return (self.gamma * self.prediction_user_based[indexFold][user][item] + (1-self.gamma) * self.prediction_item_based[indexFold][user][item])

        def main_calculation(self) -> list[list[float]]:
            if self.toyData :
                result = []
                for user in tqdm(range(len(self.matrixRating)),desc="Prediction Hybrid Collaborative Filtering") :
                    result_inner = []
                    unrated_item = self.getItem(user,interacted=False)
                    for item in range(len(self.matrixRating[user])) :
                        result_inner.append(self.fusion(user,item) if item in unrated_item else self.matrixRating[user][item])
                    result.append(result_inner)
                return result
            
            result = []
            for indexFold in tqdm(range(len(self.train)),desc="Prediction Hybrid Collaborative Filtering") :
                t1 = time.time() if self.time else ""
                result_train = []
                for user in range(len(self.train[indexFold])) :
                    result_inner = []
                    unrated_item = self.getItem(user,indexFold=indexFold,interacted=False)
                    for item in range(len(self.train[indexFold][user])) :
                        result_inner.append(self.fusion(user,item,indexFold=indexFold) if item in unrated_item else self.train[indexFold][user][item])
                    result_train.append(result_inner)
                result.append(result_train)
                t2 = time.time() if self.time else ""
                self.time_computation_prediction_per_fold.append(t2-t1) if self.time else ""
            return result

        def get_data_frame(self,indexFold : None|int = None) -> pd :
            """
            Mengembalikan hasil prediksi dalam bentuk dataframe

            Returns:
            --------
            object
                Data prediksi
            """
            if self.user_based.toyData :
                return pd.DataFrame(self.result_hybrid)
            
            if indexFold is None :
                return pd.DataFrame(self.result_hybrid)
            return pd.DataFrame(self.result_hybrid[indexFold])
        
        def get_top_n_fusion(self) :
            """
            Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

            Returns:
            --------
            pandas.DataFrame
                DataFrame yang berisi hasil prediksi.
            """
            return pd.DataFrame(self.topN)
        
        def show_time_computation_prediction_hybrid(self) :
            
            if self.toyData :
                ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
            
            if self.time :
                return pd.DataFrame(self.time_computation_prediction_per_fold)

            ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")
        
        def show_time_computation_similarity_hybrid(self,opsional = "user-based",indexFold : int|None = None) :
            if self.toyData :
                ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
            
            if self.time :
                return self.user_based.show_time_computation_similarity(indexFold) if opsional == "user-based" else self.item_based.show_time_computation_similarity(indexFold)

            ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")

        
        def show_time_computation_array(self,opsional,indexFold : int|None = None) :
            if self.toyData :
                ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
            
            if self.time :
                return self.user_based.show_time_computation_similarity_array(indexFold) if opsional == "user-based" else self.item_based.show_time_computation_similarity_array(indexFold)

            ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")

    
    return HybridCollaborativeFiltering(
        data,
        object,
        k_user=k_user,
        k_item=k_item,
        gamma=gamma,
        toyData=toyData,
        N=N,
        n=n,
        path_file=path_file,
        **kwargs
    )