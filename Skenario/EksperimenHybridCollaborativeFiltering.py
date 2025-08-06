import DistanceBased.similarities as S
from DistanceBased import Similarity
from hybrid import HCF
import os
from tqdm.contrib.itertools import product
import joblib
from Evaluation import Evaluation

class EksperimenHybridCollaborativeFiltering() :

    def __init__(self,
            data : str ,
            *,
            object : Similarity,
            path_file : str, 
            k_user : float|list[float], 
            k_item : float|list[float], 
            gamma : float|list[float], 
            N : int,
            n : int,
            Evaluation : None|Evaluation = None,
            **params,
        ) :

        self.object = object
        self.params = params
        self.data = data
        self.k_user = k_user
        self.k_item = k_item
        self.gamma = gamma
        self.N = N
        self.n = n
        self.Evaluation = Evaluation

        if os.path.exists(path_file) :
            self.result_skenario = joblib.load(path_file)
        else :
            self.result_skenario = self.skenario_executed()
            joblib.dump(self.result_skenario,path_file)

    def skenario_executed(self) :

        if self.object == S.TI :
            alpha_1 = self.params["alpha_1"]
            alpha_2 = self.params["alpha_2"]

            result = {}
            for gamma_index, k_user_index, k_item_index, alpha_1_index, alpha_2_index in product(self.gamma, self.k_user, self.k_item, alpha_1, alpha_2) :

                hybrid = HCF(self.data,self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index,alpha_1=alpha_1_index,alpha_2=alpha_2_index,N=self.N,n=self.n,Evaluation=self.Evaluation)

                result.setdefault(gamma_index, {}).setdefault(k_user_index, {}).setdefault(k_item_index, {}).setdefault(alpha_1_index, {})[alpha_2_index] = hybrid.result_evaluation
            return result
        else :
            result = {}
            for gamma_index, k_user_index, k_item_index in product(self.gamma, self.k_user, self.k_item) :
                
                hybrid = HCF(self.data,self.object,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index, N=self.N,n=self.n)

                result.setdefault(gamma_index, {}).setdefault(k_user_index, {})[k_item_index] = hybrid.result_evaluation

            return result