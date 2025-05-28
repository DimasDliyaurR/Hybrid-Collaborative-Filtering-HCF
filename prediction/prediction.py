from numpy import transpose, array
import pandas as pd
from DistanceBased import Mean
from operator import itemgetter,mul
import os
import joblib

class Prediction(Mean):
    
    def __init__(
                self,
                data,
                similarity : None| list[list[float]] = None,
                *, 
                opsional : str|None = None, 
                prediction : None| list[list[float]] = None, 
                hybrid : None|bool = False, 
                k : int|None = None, 
                path_file : str|None = None, 
                toyData : bool = False
            ) -> None :
        
        super().__init__(data,opsional=opsional,toyData=toyData)
        if not self.toyData :
            self.k = k
            
            if not hybrid :
                self.similarity = similarity
            
                self.result_mean_centered_training_prediction = [transpose(self.result_mean_centered_training[i]).tolist() for i in range(5)] if opsional == "user-based" else self.result_mean_centered_training
                self.prediction = self.main_prediction_calculation()
            else :
                self.prediction = prediction
        else :
            self.similarity = similarity if opsional == "user-based" else transpose(similarity)
            self.result_mean_centered_for_prediction = transpose(self.result_mean_centered).tolist() if opsional == "user-based" else self.result_mean_centered
            self.k = k
            self.prediction = self.main_prediction_calculation()

        self.topN = self.get_top_n()

    def __numerator(self,u,i,nearestNeighborhood, indexFold : int|None = None) -> float :
        if self.toyData :
            return sum( list( 
                map(mul,itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ]),
                itemgetter(*nearestNeighborhood)(self.result_mean_centered_for_prediction[ u  if self.opsional == "item-based" else i ])) 
                )) if len(nearestNeighborhood) > 1 else (self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered_for_prediction[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]])
        else :
            return sum( list( map(mul,itemgetter(*nearestNeighborhood)(self.similarity[indexFold][ u  if self.opsional == "user-based" else i ]),itemgetter(*nearestNeighborhood)(self.result_mean_centered_training_prediction[indexFold][ i  if self.opsional == "user-based" else u ])) ) ) if len(nearestNeighborhood) > 1 else (self.similarity[indexFold][ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered_training_prediction[indexFold][ i  if self.opsional == "user-based" else u ][nearestNeighborhood[0]])
    
    def __denominator(self,u,i,nearestNeighborhood, indexFold : int|None = None) -> float :
        if self.toyData :
            return sum( list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[u if self.opsional == "user-based" else i])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[u if self.opsional == "user-based" else i][nearestNeighborhood[0]])
        else :
            return sum(list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[indexFold][u if self.opsional == "user-based" else i])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[indexFold][u if self.opsional == "user-based" else i][nearestNeighborhood[0]])

    def selectedNeighborhood(self,u,i, indexFold : None|int = None) -> list[float]:

        if self.toyData :
            indices = list(set(self.getUser(i))) if self.opsional == "user-based" else list(set(self.getItem(u)))
            similarity_selected = self.similarity[u if self.opsional == "user-based" else i]
        else :
            similarity_selected = self.similarity[indexFold][u if self.opsional == "user-based" else i]
            indices = list(set(self.getUser(i, indexFold)) - set([u])) if self.opsional == "user-based" else list(set(self.getItem(u,indexFold)) - set([i]))

        if len(indices) > 1 :
            indices = sorted(indices,key=lambda x : similarity_selected[x],reverse=True)
        else :
            return indices if len(indices) >= 1 else []

        return indices[:self.k]
        

    def prediction_calculation(self, u, i, indexFold : int|None = None) -> float :
        
        if self.toyData :
            nearestNeighborhood = self.selectedNeighborhood(u,i)
            average = self.result_mean[u if self.opsional == "user-based" else i]
        else :
            nearestNeighborhood = self.selectedNeighborhood(u,i, indexFold)
            average = self.result_mean_training[indexFold][u if self.opsional == "user-based" else i]

        if len(nearestNeighborhood) != 0 :
            numerator = self.__numerator(u,i,nearestNeighborhood,indexFold)
            denom = self.__denominator(u,i,nearestNeighborhood,indexFold)
        else :
            return 0
        return (average + (numerator / denom)) if denom != 0 else 0

    def main_prediction_calculation(self) -> None :

        if self.toyData :
            result = []
            for u in range(len(self.matrixRating)) :
                result_inner = []
                for i in range(len(self.matrixRating[0])) :
                    value = self.prediction_calculation(u,i) if self.matrixRating[u][i] == 0 else  self.matrixRating[u][i]
                    result_inner.append(value)
                result.append(result_inner)
            return result
        
        else :
            result = []
            for indexFold in range(len(self.train)) :
                print(f"Prediction Train {indexFold}")
                
                result_train = []
                for u in range(len(self.train[indexFold])) :
                    result_inner = []
                    for i in range(len(self.train[indexFold][u])) :
                        result_inner.append(self.prediction_calculation(u,i,indexFold) if self.train[indexFold][u][i] == 0 else  self.train[indexFold][u][i])
                    result_train.append(result_inner)
                result.append(result_train)
            return result
    
    def get_top_n(self) :
        if self.toyData :
            result = []
            for i in range(len(self.matrixRating)) :
                unratedItem = self.getItem(i,interacted=False)
                
                if len(unratedItem) > 1 :
                    # valueOfPrediction = itemgetter(*unratedItem)(self.prediction[i]) if len(unratedItem) > 1 else self.prediction[i][unratedItem[0]]

                    if len(unratedItem) < 1 :
                        result_inner.append([])

                    sorted_array = sorted(unratedItem,key=lambda x: self.prediction[i][x],reverse=True)

                    result.append(sorted_array) if len(unratedItem) > 1 else self.prediction[i][unratedItem[0]]
                else :
                    result.append(unratedItem)
            return result
        else :
            result = []
            for indexFold in range(len(self.train)) :
                print(f"Train index = {indexFold}")
                result_inner = []
                for u in range(len(self.train[indexFold])) :
                    unratedItem = self.getItem(u,indexFold=indexFold,interacted=False)
                    if len(unratedItem) > 1 :
                        # Prediction training : 5 x 943 x 1682 
                        # Prediction training : indexFold x u x {iteration}
                        # If number of unrated Item have less then 1 : Acces Prediction training used index
                        # However, if The number of unrated item have more than 1 , Acces prediction should be with itemgetter function
                        # valueOfPrediction = itemgetter(*unratedItem)(self.prediction[indexFold][u]) if len(unratedItem) > 1 else self.prediction[indexFold][u][unratedItem[0]]

                        if len(unratedItem) == 0 :
                            result_inner.append([])
                            continue
                        
                        # Algoritma Seharusnya disamakan dengan sebelumnya (Skripsi Tahun Kemarin)
                        sorted_array = sorted(unratedItem,key=lambda x: self.prediction[indexFold][u][x],reverse=True) if len(unratedItem) > 1 else self.prediction[indexFold][u][unratedItem[0]]

                        result_inner.append(sorted_array)
                    else :
                        result_inner.append(unratedItem)
                result.append(result_inner)
            return result

    def get_top_n_specific_user(self, u, indexFold : None|int = None) :
        if not self.toyData and indexFold is None :
            raise ValueError(f"Index of train should be passed ! {self}")
        return self.topN[indexFold][u] if not self.toyData else self.topN[u]

    def get_prediction_array(self):
        return array(self.prediction).tolist()

    def get_prediction_data_frame(self, indexFold : None|int = None):
        if indexFold is None :
            return pd.DataFrame(self.prediction)
        return pd.DataFrame(self.prediction[indexFold])

    def get_top_n_array(self) :
        return self.topN
