import DistanceBased.similarities as S
from DistanceBased import Similarity
from hybrid import HCF
from itertools import product

class EksperimenHybridCollaborativeFiltering() :

    def __init__(self, object : Similarity, params : dict) :
        self.object = object
        self.params = params

        self.result_skenario = self.skenario_executed()

    def skenario_executed(self) :
        gamma = self.params["gamma"]
        k_item = self.params["k_item"]
        k_user = self.params["k_user"]

        if self.object == S.TI :
            alpha_1 = self.params["alpha_1"]
            alpha_2 = self.params["alpha_2"]
            
            result = []
            for gamma_index, k_user_index, k_item_index, alpha_1_index, alpha_2_index in product(gamma, k_user,k_item,alpha_1,alpha_2) :
                hybrid = HCF("data/ml-100k",self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index,additional={
                    "alpha_1" : alpha_1_index,
                    "alpha_2" : alpha_2_index,
                })

                result.append(hybrid.result_evaluation[:19])
            
            return result
        else :
            result = []
            for gamma_index, k_user_index, k_item_index in product(gamma, k_user,k_item) :
                hybrid = HCF("data/ml-100k",self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index)

                result.append(hybrid.result_evaluation[:19])
            
            return result

