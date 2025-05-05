import numpy as np
from MatrixRating import MatrixRating
from operator import itemgetter

class NDCG(MatrixRating) :

    def __init__(self,data, top_n : list[list[list[int]]],*,path_evaluation : str = None,toyData : bool, N : int = 30) -> None:
        MatrixRating.__init__(self,data,toyData=toyData)

        self.N = N
        self.top_n = top_n
        self.result_evaluation = self.main_calculation()


    def get_top_n_specific_user(self,u,*,indexTrain : int|None = None) -> list[float] :
        return self.top_n[indexTrain][u]

    def __groundTruth(self, u, indexTrain : int|None = None) -> np.array :
         data_test = self.getItemTest(u,indexTrain=indexTrain)
         return np.array([1 if (self.get_top_n_specific_user(u,indexTrain=indexTrain)[:i+1][-1] in data_test) and len(data_test) > i else 0 for i in range(self.N)])

    def IDCG(self) -> float :
        return sum(1/np.log2(np.arange(2,self.N+2))).real

    def DCG(self, u, indexTrain : int|None = None) -> float :
        return sum(self.__groundTruth(u,indexTrain=indexTrain)/np.log2(np.arange(2,self.N+2))).real

    def NDCG(self,u,indexTrain) -> float :
        print("DCG = ",self.DCG(u,indexTrain=indexTrain),"IDCG = ",self.IDCG())
        return self.DCG(u,indexTrain=indexTrain) / self.IDCG()
    
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
        for indexTrain in range(len(self.train)) :
            
            number_of_evaluation_per_fold = 0
            for u in range(len(self.train[indexTrain])) :
                # Proses Tahap 1
                number_of_evaluation_per_fold += self.NDCG(u,indexTrain)
            
            # Proses Tahap 2
            print("Tahap 1 =", number_of_evaluation_per_fold)
            mean_evaluation_per_fold = number_of_evaluation_per_fold/self.numberOfUser
            result_per_fold.append(mean_evaluation_per_fold)
        
        # Proses Tahap 3
        print("Tahap 2 =", result_per_fold)
        total_mean_evaluation = sum(result_per_fold)/len(self.train)
        print("Tahap 3 =", total_mean_evaluation)
        return total_mean_evaluation
    
    def show_evaluation(self) :
        return self.result_evaluation