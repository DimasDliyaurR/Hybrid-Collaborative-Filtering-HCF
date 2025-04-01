import pandas as pd
import prediction as p

class HybridCollaborativeFiltering() :

    def __init__(self,data : list[list[float]], similarity : "p.prediction",*, gamma_user : float,k : list, N : int = 2):
        self.matrix = data
        self.gamma_user = gamma_user
        self.N = N
        self.user_based = similarity(self.matrix,opsional="user-based",k=k[0])
        self.item_based = similarity(self.matrix,opsional="item-based",k=k[1])
        self.prediction_user_based = self.user_based.get_prediction_array()
        self.prediction_item_based = self.item_based.get_prediction_array()
        self.result_hybrid = self.main_calculation()
        self.topN = self.get_top_n()

    def fusion(self,outer : int,inner : int) -> float:
        return (self.gamma_user * self.prediction_user_based[outer][inner] + (1-self.gamma_user) * self.prediction_item_based[outer][inner])
    
    def main_calculation(self) -> list[list[float]]:
         return [
            [
                (self.fusion(outer,inner) if self.matrix[outer][inner] == 0 else self.matrix[outer][inner])
                for inner in range(len(self.matrix[0]))
            ]
            for outer in range(len(self.matrix))
        ]
    
    def get_data_frame(self) -> object :
        """
        Mengembalikan hasil prediksi dalam bentuk dataframe

        Returns:
        --------
        object
             Data prediksi
        """
        return pd.DataFrame(self.result_hybrid)
    
    def get_top_n(self) :
        """
        Mengembalikan hasil dari Top-N dari prediksi

        Returns:
        --------
        array
            Array yang berisi tentang Top-N
        """
        result = []
        for i in range(len(self.matrix)) :
            result.append(
                sorted([self.result_hybrid[i][inner] for inner in range(len(self.matrix[i])) if self.matrix[i][inner] == 0])[::-1][0:self.N]
                # Sorting a similarities
                # list(sim[np.array(sim).argsort()[::-1]][0:self.k])
                )
        return result
    
    def get_top_n_fusion(self) :
        """
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi hasil prediksi.
        """
        return pd.DataFrame(self.topN)
