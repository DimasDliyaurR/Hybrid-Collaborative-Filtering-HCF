from numpy import transpose, array
import pandas as pd
from DistanceBased import Mean
from operator import itemgetter,mul
import os
import copy
import joblib
from tqdm import tqdm

class Prediction():
    
    def __init__(
                self,
                data,
                mean_object : Mean,
                similarity : None| list[list[float]] = None,
                *, 
                opsional : str|None = None, 
                prediction : None| list[list[float]] = None, 
                hybrid : None|bool = False, 
                k : int|None = None,
                path_file : str|None = None, 
                toyData : bool = False,
                time : bool = False
            ) -> None :
        
        self.mean_object = mean_object
        self.opsional = opsional
        self.toyData = toyData

        if not self.toyData :
            self.k = k
            
            if not hybrid :
                self.similarity = similarity

                if not time and os.path.exists(path_file) :
                    self.prediction = joblib.load(path_file)
                else :
                    self.result_mean_centered_training_prediction = [transpose(self.mean_object.result_mean_centered_training[i]) for i in range(5)] if opsional == "user-based" else self.mean_object.result_mean_centered_training
                    self.prediction = self.main_prediction_calculation()
                    # joblib.dump(self.prediction,path_file) if not time else ""
            else :
                self.prediction = prediction      
        else :
            if not hybrid :
                self.similarity = similarity if opsional == "user-based" else transpose(similarity)
                self.data_for_prediction = copy.deepcopy(data)
                self.matrixRating = data
                self.reverseMatrixRating = transpose(self.mean_object.matrixRating).tolist()
                self.result_mean_centered_for_prediction = transpose(self.mean_object.result_mean_centered).tolist() if opsional == "user-based" else self.result_mean_centered
                self.k = k
                self.prediction = self.main_prediction_calculation()
            else :
                self.data_for_prediction = copy.deepcopy(data)
                self.prediction = prediction      

        self.topN = self.get_top_n()

    def __numerator(self,u,i,nearestNeighborhood, indexFold : int|None = None) -> float :
        if self.toyData :
            return sum( list( 
                map(mul,itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ]),
                itemgetter(*nearestNeighborhood)(self.result_mean_centered_for_prediction[ u  if self.opsional == "item-based" else i ])) 
                )) if len(nearestNeighborhood) > 1 else (self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered_for_prediction[ u  if self.opsional == "item-based" else i ][nearestNeighborhood[0]])
        else :
            return sum( list( map(mul,itemgetter(*nearestNeighborhood)(self.similarity[indexFold][ u  if self.opsional == "user-based" else i ]),itemgetter(*nearestNeighborhood)(self.result_mean_centered_training_prediction[indexFold][ i  if self.opsional == "user-based" else u ])) ) ) if len(nearestNeighborhood) > 1 else (self.similarity[indexFold][ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered_training_prediction[indexFold][ i  if self.opsional == "user-based" else u ][nearestNeighborhood[0]])
    
    def __denominator(self,u,i,nearestNeighborhood, indexFold : int|None = None) -> float :
        if self.toyData :
            return sum( list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[u if self.opsional == "user-based" else i])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[u if self.opsional == "user-based" else i][nearestNeighborhood[0]])
        else :
            return sum(list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[indexFold][u if self.opsional == "user-based" else i])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[indexFold][u if self.opsional == "user-based" else i][nearestNeighborhood[0]])

    def selected_neighborhood(self,u,i, indexFold : None|int = None) -> list[float]:

        if self.toyData :
            indices = list(set(self.mean_object.getUser(i))) if self.opsional == "user-based" else list(set(self.mean_object.getItem(u)))
            similarity_selected = self.similarity[u if self.opsional == "user-based" else i]
        else :
            similarity_selected = self.similarity[indexFold][u if self.opsional == "user-based" else i]
            indices = list(set(self.mean_object.getUser(i, indexFold)) - set([u])) if self.opsional == "user-based" else list(set(self.mean_object.getItem(u,indexFold)) - set([i]))
        
        if len(indices) > 1 :
            indices = sorted(indices,key=lambda x : similarity_selected[x],reverse=True)
        else :
            return indices if len(indices) >= 1 else []

        return indices[:self.k]
        

    def prediction_calculation(self, u, i, indexFold : int|None = None) -> float :
        
        if self.toyData :
            nearestNeighborhood = self.selected_neighborhood(u,i)
            average = self.mean_object.result_mean[u if self.opsional == "user-based" else i]
        else :
            nearestNeighborhood = self.selected_neighborhood(u,i, indexFold)
            average = self.mean_object.result_mean_training[indexFold][u if self.opsional == "user-based" else i]

        if len(nearestNeighborhood) != 0 :
            numerator = self.__numerator(u,i,nearestNeighborhood,indexFold)
            denom = self.__denominator(u,i,nearestNeighborhood,indexFold)
        else :
            return 0

        return (average + (numerator / denom)) if denom != 0 else 0

    def main_prediction_calculation(self) -> None :

        if self.toyData :
            result = copy.deepcopy(self.data_for_prediction)
            for u in tqdm(range(len(self.matrixRating)),desc="Prediction") :
                for i in range(len(self.matrixRating[0])) :
                    result[u][i] = self.prediction_calculation(u,i) if self.matrixRating[u][i] == 0 else self.matrixRating[u][i]
            return result
        else :
            result = self.mean_object.train.copy()
            for indexFold in tqdm(range(len(self.mean_object.train)),desc="Prediction") :
                for u in range(len(self.mean_object.train[indexFold])) :
                    for i in self.mean_object.getItem(u,indexFold=indexFold, interacted=False) :
                        result[indexFold][u][i] = self.prediction_calculation(u,i,indexFold)
            return result
    
    def get_top_n(self) :
        if self.toyData :
            result = []
            for i in range(len(self.data_for_prediction) ) :
                unratedItem = self.mean_object.getItem(i,interacted=False)
                if len(unratedItem) > 1 :

                    if len(unratedItem) == 0 :
                        result_inner.append([])
                        continue
                    
                    sorted_array = sorted(unratedItem,key=lambda x: self.prediction[i][x],reverse=True) if len(unratedItem) > 1 else self.prediction[u][unratedItem[0]]

                    result.append(sorted_array)
                else :
                    result.append(unratedItem)
            return result
        else :
            result = []
            for indexFold in range(len(self.mean_object.train)) :
                result_inner = []
                for u in range(len(self.mean_object.train[indexFold])) :
                    unratedItem = self.mean_object.getItem(u,indexFold=indexFold,interacted=False)
                    if len(unratedItem) > 1 :
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

    def show_prediction(self, indexFold : None|int = None):
        if indexFold is None :
            return pd.DataFrame(self.prediction)
        return pd.DataFrame(self.prediction[indexFold])

    def get_top_n_array(self) :
        return self.topN
