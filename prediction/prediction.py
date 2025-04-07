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
       
        # return sum(self.similarity[ u  if self.opsional == "user-based" else i ][neighbor] * self.result_mean_centered[ u  if self.opsional == "user-based" else i ][neighbor] for neighbor in nearestNeighborhood)
        
        return sum( list( map(mul,itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ]),itemgetter(*nearestNeighborhood)(self.result_mean_centered[ u  if self.opsional == "user-based" else i ])) ) ) if len(nearestNeighborhood) > 1 else (self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]]*self.result_mean_centered[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]])

    def __denominator(self,u,i,nearestNeighborhood):
       
        return sum(list(map(lambda x : abs(x),itemgetter(*nearestNeighborhood)(self.similarity[ u  if self.opsional == "user-based" else i ])))) if len(nearestNeighborhood) > 1 else abs(self.similarity[ u  if self.opsional == "user-based" else i ][nearestNeighborhood[0]])

    def selectedNeighborhood(self,u,i) -> list[float]:

        if self.opsional == "user-based" :
            # similarity_selected = np.array(self.similarity[u])
            # indices = np.array(list(set(self.getUser(i)) - set([u])))
            similarity_selected = self.similarity[u]
            indices = list(set(self.getUser(i)) - set([u]))
        else :
            # similarity_selected = transpose(self.similarity)[i]
            similarity_selected = self.similarity[i]
            indices = list(set(self.getItem(u)) - set([i]))
            # similarity_selected = np.array(hp.reverseMatrix(self.similarity)[i])
            # indices = np.array(list(set(self.getItem(u)) - set([i])))

        # print(f"indices : {indices}")
        
        if len(indices) > 1 :
            # indices = np.argsort(np.array(list(itemgetter(*indices)(similarity_selected))))[::-1]  # Sort indices by value
            similarity_selected = list(itemgetter(*indices)(similarity_selected))
            indices = sorted(range(len(similarity_selected)),key=lambda x : similarity_selected[x],reverse=True)  # Sort indices by value
        else :
            return indices if len(indices) >= 1 else []
        
        return indices[:self.k]
        # return [indices.tolist()[:self.k], neighborhood.tolist()[:self.k]]
        

    def prediction_calculation(self, u, i) -> float:
        
        if u % 10 == 0 :
            print(f"predict({u},{i})")
        # print(f"user : {u} | item : {i}")
        # t1 = time.time()
        nearestNeighborhood = self.selectedNeighborhood(u,i)
        # print(f"Tetangga : {nearestNeighborhood}")
        average = self.result_mean[u if self.opsional == "user-based" else i]
        
        if len(nearestNeighborhood) != 0 :
            numerator = self.__numerator(u,i,nearestNeighborhood)
            denom = self.__denominator(u,i,nearestNeighborhood)
        else :
            # t2 = time.time()
            # print(f"Time Taken : {t2-t1}")
            return average

        # t2 = time.time()
        # print(f"Time Taken : {t2-t1}")
        return (average + (numerator / denom)) if denom != 0 else 0
        # print(f"result : {result}")
        # return result

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
            result.append(
                sorted([self.prediction[i][inner] for inner in range(len(self.matrixRating[i])) if self.matrixRating[i][inner] == 0])[::-1][0:self.k]
                # Sorting a similarities
                # list(sim[np.array(sim).argsort()[::-1]][0:self.k])
                )
        return result

    def get_prediction_array(self):
      
        return array(self.prediction).tolist()
    
    def get_prediction_data_frame(self):
       
        return pd.DataFrame(self.prediction)
    
    def get_top_n_array(self) :
      
        return self.topN
