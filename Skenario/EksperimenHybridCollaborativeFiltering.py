import DistanceBased.similarities as S
from DistanceBased import Similarity, Mean
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
            path_folder : str, 
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

        self.__skenario = {
            "k_user" : k_user,
            "k_item" : k_item,
            "gamma" : gamma
        } | params

        print("Start")
        self.path_folder = path_folder
        self.ls_directory = os.listdir(path_folder)
        
        if not self.__skenario_check() :
            self.mean_user = Mean("data/ml-100k",opsional="user-based")
            self.mean_item = Mean("data/ml-100k",opsional="item-based")
        
        print("End")
    
    def update_lisdir(self) :
        self.ls_directory = os.listdir(self.path_folder)

    def __skenario_check(self) :
        print("Check skenario")

        number_skenario = 1
        for i in self.__skenario :
            number_skenario *= len(self.__skenario[i])
        

        print(f"Jumlah skenario : {number_skenario}, Jumlah skenario telah selesai : {len(self.ls_directory)}")
        if number_skenario == len(self.ls_directory) :
            print("Semua skenario telah selesai")
        else :
            print("Skenario belum selesai!")

        return number_skenario == len(self.ls_directory)

    
    def check_skenario(self,*,gamma,k_user,k_item,alpha_1 = None, alpha_2 = None) -> bool :
        self.update_lisdir()

        if self.object == S.TI :
            
            if alpha_1 == None and alpha_2 == None :
                raise ValueError("Alpha 1 and Alpha 2 should be given as parameter when using Tversky Index Function")
            
            path_skenario =  f"{gamma}_{k_user}_{k_item}_{alpha_1}_{alpha_2}.joblib"

        elif self.object == S.DC :
            path_skenario =  f"{gamma}_{k_user}_{k_item}.joblib"

        print(path_skenario) if path_skenario not in self.ls_directory else ""
        
        return path_skenario in self.ls_directory

    def skenario_executed(self) :

        if self.object == S.TI :
            alpha_1 = self.params["alpha_1"]
            alpha_2 = self.params["alpha_2"]

            for gamma_index, k_user_index, k_item_index, alpha_1_index, alpha_2_index in product(self.gamma, self.k_user, self.k_item, alpha_1, alpha_2) :

                if not self.check_skenario(gamma=gamma_index,k_user=k_user_index,k_item=k_item_index, alpha_1=alpha_1_index,alpha_2=alpha_2_index) :
                    hybrid = HCF(self.data,self.object,mean_object_user=self.mean_user,mean_object_item=self.mean_item,object_evaluation=self.Evaluation,k_user=k_user_index,k_item=k_item_index,gamma=gamma_index,alpha_1=alpha_1_index,alpha_2=alpha_2_index,N=self.N,n=self.n)
                    joblib.dump(hybrid.result_evaluation, self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}_{alpha_1_index}_{alpha_2_index}.joblib")
                    print("success create :",self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}_{alpha_1_index}_{alpha_2_index}.joblib")
                
        elif self.object == S.DC :
            for gamma_index, k_user_index, k_item_index in product(self.gamma, self.k_user, self.k_item) :
                if not self.check_skenario(gamma=gamma_index,k_user=k_user_index,k_item=k_item_index) :
                    hybrid = HCF(self.data,self.object,mean_object_user=self.mean_user,mean_object_item=self.mean_item,k_user=k_user_index,object_evaluation=self.Evaluation,k_item=k_item_index,gamma=gamma_index, N=self.N,n=self.n)
                    joblib.dump(hybrid.result_evaluation, self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}.joblib")
                    print("success create :",self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}.joblib")

        else :
            raise ValueError("Unavailable similarity function !")
        
    def get_all_skenario(self) -> dict :
        result = {}

        print("Mengambil semua skenario...")
        if self.object == S.TI :
            alpha_1 = self.params["alpha_1"]
            alpha_2 = self.params["alpha_2"]

            for gamma_index, k_user_index, k_item_index, alpha_1_index, alpha_2_index in product(self.gamma, self.k_user, self.k_item, alpha_1, alpha_2) :
                evaluation = joblib.load(self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}_{alpha_1_index}_{alpha_2_index}.joblib")
                result.setdefault(gamma_index, {}).setdefault(k_user_index, {}).setdefault(k_item_index, {}).setdefault(alpha_1_index, {})[alpha_2_index] = evaluation

        elif self.object == S.DC :
            for gamma_index, k_user_index, k_item_index in product(self.gamma, self.k_user, self.k_item) :
                evaluation = joblib.load(self.path_folder + "/" + f"{gamma_index}_{k_user_index}_{k_item_index}.joblib")
                result.setdefault(gamma_index, {}).setdefault(k_user_index, {})[k_item_index] = evaluation
        
        print("Berhasil mendapatkan semua skenario")
        return result