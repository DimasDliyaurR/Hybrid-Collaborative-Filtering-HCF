from Evaluation import Evaluation
import numpy as np
from typing_extensions import override, overload

class NDCG(Evaluation) :

    def __init__(self,data, top_n : list[list[list[int]]],*, path_evaluation : str = None,toyData : bool, N : int = 30):
        super().__init__(data, top_n, path_evaluation=path_evaluation,toyData = toyData, N=N)

    def groundTruth(self, u : None | int = None, indexFold : int|None = None) -> np.array :
        if not self.toyData :
            data_test = self.getItemTest(u,indexFold=indexFold)
            return np.array([1 if (self.get_top_n_specific_user(u,indexFold=indexFold)[:i+1][-1] in data_test) and len(data_test) > i else 0 for i in range(self.N)])

        return np.array([1 if (self.top_n[:i+1][-1] in self.data_test) and len(self.top_n) > i else 0 for i in range(self.N)])

    def ideal(self,n) :
        return np.array([sum(1/np.log2(np.arange(2,n+2))) for n in range(1,n+1)])

    def ideal_iteration(self,n) -> np.array :
        return np.array([sum(self.ideal(i)) for i in range(1,n+1)])

    def IDCG(self) -> float :
        return self.ideal_iteration(self.N)

    def DCG(self, u: None | int = None, indexFold : int|None = None) -> float :
        ground_truth = self.groundTruth(u,indexFold=indexFold)
        dcg_iteration = [ideal_iteration*ground_truth_index for ideal_iteration,ground_truth_index in zip(self.ideal(self.N),ground_truth)]

        result = [[sum(dcg_iteration[:n]) for n in range(1,i+2)] for i in range(self.N)]

        return result[-1]

    def NDCG(self,u : None|int = None,indexFold : None|int = None) -> np.array :
        
        if not self.toyData :
            return self.DCG(u,indexFold=indexFold)/self.IDCG()
        
        return self.DCG()/self.IDCG()
    
    @override
    def main_calculation_evaluation(self) -> float :
        """
        The main calculation of evaluation 
        ---------------------------------------

        1. Tahap 1
        Per fold: hitung evaluasi untuk setiap user (target) yang ada di dalam data test
        
        2. Tahap 2
        Per fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 1) dari seluruh user (target) dan kemudian dibagi dengan jumlah user (target)
        
        3. Tahap 3
        Seluruh fold: hitung rata-rata hasil evaluasi dengan menjumlahkan hasil evaluasi (hasil Tahap 2) dari seluruh fold kemudian dibagi dengan jumlah fold
        
        """
        if not self.toyData :
            # list[5]
            result_per_fold = []
            for indexFold in range(len(self.train)) :

                # Ukuran : Jumlah pengguna x N
                number_ndcg_of_evaluation_per_fold = []
                unique_user_of_test_fold = self.getUniqueIdOfUserTest(indexFold=indexFold)

                for u in unique_user_of_test_fold :

                    # DCG dan IDCG
                    number_ndcg_of_evaluation_per_fold += [self.NDCG(u,indexFold=indexFold)]

                # Mencari rata-rata NDCG dengan menjumlahkan DCG dibagi dengan jumlah pengguna pada data test fold
                
                #  Ukuran : 1 x N
                ndcg_per_fold = [ sum([row[col] for row in number_ndcg_of_evaluation_per_fold ])/len(unique_user_of_test_fold) for col in range(len(number_ndcg_of_evaluation_per_fold[0]))]

                result_per_fold.append(ndcg_per_fold)

            result = [sum([row[col] for row in result_per_fold]) for col in range(len(result_per_fold[0]))]

            return (result)
        

        return self.NDCG()
