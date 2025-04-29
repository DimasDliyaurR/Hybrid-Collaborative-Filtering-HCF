import numpy as np
from MatrixRating import MatrixRating
from operator import itemgetter

class NDCG() :

    def __init__(self,data : MatrixRating, top_n : list[list[list[int]]],*, N : int):
        self.data = data
        self.N = N
        self.top_n = top_n

    def groundTruth(self, u, indexTrain) -> np.array :
         test_index_of_rating = itemgetter(*self.data.getItem(u,indexTrain=indexTrain))(self.test[u])
         return np.array([1 if self.top_n[:i][-1] in test_index_of_rating else 0 for i in range(self.N)])

    def IDCG(self) -> float :
        return sum(1/np.log2(np.arange(2,self.N+1))).real

    def DCG(self, u) -> float :
        return sum(self.groundTruth(u)/np.log2(np.arange(2,self.N+1))).real

    def NDCG(self,u) :
        return self.DCG(self.N,u) / self.IDCG(self.N)
    
    # Tahap 1
    # Per fold: hitung evaluasi untuk setiap user (target) yang ada di dalam data test
    # Tahap 2
    # Per fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 1) dari seluruh user (target) dan kemudian dibagi dengan jumlah user (target)
    # Tahap 3
    # Seluruh fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 2) dari seluruh fold kemudian dibagi dengan jumlah fold
    def main_calculation(self) : ...