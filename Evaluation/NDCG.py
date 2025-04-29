import numpy as np
from MatrixRating import MatrixRating
from prediction import Prediction
from operator import itemgetter

class NDCG() :

    def __init__(self,data : MatrixRating, prediction : Prediction,*, N : int) -> None:
        self.data = data
        self.N = N
        self.prediction = prediction

    def groundTruth(self, u, indexTrain) -> np.array :
         ground_truth = itemgetter(*self.data.getItem(u,indexTrain=indexTrain))(self.test[u])

         return np.array([1 if self.prediction.get_top_n_specific_user(u,indexTrain)[:i+1][-1] in ground_truth else 0 for i in range(self.N) if len(ground_truth) > i])

    def IDCG(self) -> float :
        return sum(1/np.log2(np.arange(2,self.N+1))).real

    def DCG(self, u) -> float :
        return sum(self.groundTruth(u)/np.log2(np.arange(2,self.N+1))).real

    def NDCG(self,u) -> float :
        return self.DCG(self.N,u) / self.IDCG(self.N)
    
    def main_calculation(self) -> float :
        """
        Komputasi evaluasi setiap fold data training
        ---------------------------------------

        1. Tahap 1
        Per fold: hitung evaluasi untuk setiap user (target) yang ada di dalam data test
        
        2. Tahap 2
        Per fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 1) dari seluruh user (target) dan kemudian dibagi dengan jumlah user (target)
        
        3. Tahap 3
        Seluruh fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 2) dari seluruh fold kemudian dibagi dengan jumlah fold
        
        """
        result_per_fold = []
        for indexTrain in range(len(self.data.train)) :
            
            number_of_evaluation_per_fold = 0
            for u in range(len(self.data.train[indexTrain])) :
                # Proses Tahap 1
                number_of_evaluation_per_fold += self.NDCG(u)
            
            # Proses Tahap 2
            mean_evaluation_per_fold = number_of_evaluation_per_fold/self.data.numberOfUser
            result_per_fold.append(mean_evaluation_per_fold)
        
        # Proses Tahap 3
        total_mean_evaluation = sum(result_per_fold)/len(self.data.train)
        return total_mean_evaluation
        
