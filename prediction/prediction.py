from numpy import transpose, array
import pandas as pd
from MatrixRating import MatrixRating
from DistanceBased import Mean
from operator import itemgetter,mul

class Prediction(Mean,MatrixRating):
    
    def __init__(self,data, opsional, similarity,*, k):
        MatrixRating.__init__(self,data)
        Mean.__init__(self,self.matrixRating, transpose(self.matrixRating),opsional=opsional)
        self.similarity = similarity if opsional == "user-based" else transpose(similarity)
        self.k = k
        self.opsional = opsional
        self.prediction = self.main_prediction_calculation()
        self.topN = self.get_top_n()

    def __numerator(self,u,i,nearestNeighborhood):
        return sum( list( map(mul,itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ]),itemgetter(*nearestNeighborhood)(self.result_mean_centered[ u  if self.opsional == "user-based" else i ])) ) ) if len(nearestNeighborhood) > 1 else (self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]])

    def __denominator(self,u,i,nearestNeighborhood):
        return sum(list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]])

    def selectedNeighborhood(self,u,i) -> list[float]:
        similarity_selected = self.similarity[u if self.opsional == "user-based" else i]
        indices = list(set(self.getUser(i)) - set([u])) if self.opsional == "user-based" else list(set(self.getItem(u)) - set([i]))
        
        if len(indices) > 1 :
            similarity_selected = list(itemgetter(*indices)(similarity_selected))
            indices = sorted(range(len(similarity_selected)),key=lambda x : similarity_selected[x],reverse=True)  # Sort indices by value
        else :
            return indices if len(indices) >= 1 else []
        
        return indices[:self.k]
        

    def prediction_calculation(self, u, i) -> float:

        nearestNeighborhood = self.selectedNeighborhood(u,i)
        average = self.result_mean[u if self.opsional == "user-based" else i]
        
        if len(nearestNeighborhood) != 0 :
            numerator = self.__numerator(u,i,nearestNeighborhood)
            denom = self.__denominator(u,i,nearestNeighborhood)
        else :
            return 0

        return (average + (numerator / denom)) if denom != 0 else 0

    def main_prediction_calculation(self):
        return [
            [
                (self.prediction_calculation(u, i) if self.matrixRating[u][i] == 0 else self.matrixRating[u][i]) 
                for i in range(len(self.matrixRating[0]))
            ]
            for u in range(len(self.matrixRating))
        ]
    
    def get_top_n(self) :
        result = []
        for i in range(len(self.matrixRating)) :
            unratedItem = self.getItem(i,interacted=False)
            if len(unratedItem) > 1 :
                valueOfPrediction = itemgetter(*unratedItem)(self.prediction[i])
                result.append(sorted(range(len(valueOfPrediction)),key=lambda x: valueOfPrediction[x],reverse=True)[:self.k])
            else :
                result.append(unratedItem)
        return result

    def get_prediction_array(self):
        return array(self.prediction).tolist()
    
    def get_prediction_data_frame(self):
        return pd.DataFrame(self.prediction)
    
    def get_top_n_array(self) :
        return self.topN
