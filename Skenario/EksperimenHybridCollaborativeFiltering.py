import DistanceBased.similarities as S
from DistanceBased import Similarity
from hybrid import HCF
import os
from itertools import product
import joblib

class EksperimenHybridCollaborativeFiltering() :

    def __init__(self,
                 data : str ,
                 *, 
                 object : Similarity,
                 path_file : str, 
                 k_user : float|list[float], 
                 k_item : float|list[float], 
                 gamma : float|list[float], 
                 n : int, 
                 params : dict|None = None
        ) :

        self.object = object
        self.params = params
        self.data = data
        self.k_user = k_user
        self.k_item = k_item
        self.gamma = gamma
        self.n = n

        if os.path.exists(path_file) :
            self.result_skenario = joblib.load(path_file)
        else :
            self.result_skenario = self.skenario_executed()

    def skenario_executed(self) :

        if self.object == S.TI :
            alpha_1 = self.params["alpha_1"]
            alpha_2 = self.params["alpha_2"]

            result = []
            for gamma_index, k_user_index, k_item_index, alpha_1_index, alpha_2_index in product(self.gamma, self.k_user, self.k_item, alpha_1, alpha_2) :

                hybrid = HCF(self.data,self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index,additional={
                    "alpha_1" : alpha_1_index,
                    "alpha_2" : alpha_2_index,
                })

                result.append(hybrid.result_evaluation[:self.n-1])
            return result
        else :
            result = []
            for gamma_index, k_user_index, k_item_index in product(self.gamma, self.k_user, self.k_item) :
                hybrid = HCF(self.data,self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index)

                result.append(hybrid.result_evaluation[:self.n-1])

            return result