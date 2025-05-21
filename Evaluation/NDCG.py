from Evaluation import Evaluation
import numpy as np
from typing_extensions import override, overload

class NDCG(Evaluation) :

    def __init__(self,data, top_n : list[list[list[int]]],*, path_evaluation : str = None,toyData : bool, N : int = 30):
        super().__init__(data, top_n, path_evaluation=path_evaluation,toyData = toyData, N=N)

    def groundTruth(self, u : None | int = None, indexTrain : int|None = None) -> np.array :
        if not self.toyData :
            data_test = self.getItemTest(u,indexTrain=indexTrain)
            return np.array([1 if (self.get_top_n_specific_user(u,indexTrain=indexTrain)[:i+1][-1] in data_test) and len(data_test) > i else 0 for i in range(self.N)])

        return np.array([1 if (self.top_n[:i+1][-1] in self.data_test) and len(self.top_n) > i else 0 for i in range(self.N)])

    def ideal(self,n) :
        return np.array([sum(1/np.log2(np.arange(2,n+2))) for n in range(1,n+1)])

    def ideal_iteration(self,n) -> np.array :
        return np.array([sum(self.ideal(i)) for i in range(1,n+1)])

    def IDCG(self) -> float :
        return sum(self.ideal_iteration(self.N))

    def DCG(self, u: None | int = None, indexTrain : int|None = None) -> float :
        ground_truth = self.groundTruth(u,indexTrain=indexTrain)
        dcg_iteration = [ideal_iteration*ground_truth_index for ideal_iteration,ground_truth_index in zip(self.ideal(self.N),ground_truth)]

        result = [[sum(dcg_iteration[:n]) for n in range(1,i+2)] for i in range(self.N)]

        return sum(result[-1])

    def NDCG(self,dcg : list[float]|None = None ,idcg : list[float] | None = None) -> np.array :
        if not self.toyData :
            return dcg/idcg
        return self.DCG()/self.IDCG()
    
    @override
    def main_calculation(self) -> float :
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
            for indexTrain in range(len(self.train)) :

                number_dcg_of_evaluation_per_fold = []
                number_idcg_of_evaluation_per_fold = []

                for u in self.getUniqueIdOfUserTest(indexTrain=indexTrain) :

                    # DCG dan IDCG
                    number_dcg_of_evaluation_per_fold += [self.DCG(u,indexTrain=indexTrain)]
                    number_idcg_of_evaluation_per_fold += [self.IDCG()]

                # NDCG -> float
                mean_dcg_per_fold = self.NDCG((sum(number_dcg_of_evaluation_per_fold)/len(self.getUniqueIdOfUserTest(indexTrain=indexTrain))),(sum(number_idcg_of_evaluation_per_fold)/len(self.getUniqueIdOfUserTest(indexTrain=indexTrain))))
                print(mean_dcg_per_fold)
                # print("Tahap 1 =", mean_evaluation_per_fold)
                result_per_fold.append(mean_dcg_per_fold)

            # fold / 5 -> list[5]
            result = [ fold/len(self.train) for fold in (result_per_fold)]

            return (result)
        

        return self.NDCG()
