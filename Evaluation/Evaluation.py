from MatrixRating import MatrixRating
from tqdm.notebook import tqdm

class Evaluation(MatrixRating) :

    def __init__(self,data, top_n : list[list[list[int]]], *, matrix_object : MatrixRating, path_evaluation : str = None,toyData : bool, N : int = 30, n : int|None = 20) -> None:

        self.N = N
        self.n = n
        self.top_n = top_n
        self.matrix_object = matrix_object
        self.toyData = toyData
        
        if toyData :
            self.data_test = data

        self.result_evaluation = self.main_calculation_evaluation()

    def get_top_n_specific_user(self,u,*,indexFold : int|None = None) -> list[float] :
        return self.top_n[indexFold][u] if not self.toyData else self.top_n

    def result(self,u: None | int = None,indexFold : None | int = None) -> float : ...

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
            result_per_fold = []
            for indexFold in tqdm(range(len(self.matrix_object.train)),desc="Evaluation") :

                number_of_evaluation_per_fold = []

                for u in self.matrix_object.getUniqueIdOfUserTest(indexFold=indexFold) :
                    # Proses Tahap 1
                    number_of_evaluation_per_fold += [self.result(u,indexFold)]

                result_per_fold.append(number_of_evaluation_per_fold)

            return result_per_fold
        
        return self.result()
    
    def show_evaluation(self) :
        return self.result_evaluation