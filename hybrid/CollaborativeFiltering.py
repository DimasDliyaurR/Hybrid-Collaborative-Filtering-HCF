import pandas as pd
import prediction as p

class CollaborativeFiltering() :

    def __init__(self,data, similarity : p.prediction,*, gamma_user : float,gamma_item : float,k):
        self.matrix = data
        self.gamma_item = gamma_item
        self.gamma_user = gamma_user
        self.user_based = similarity(self.matrix,opsional="user-based",k=k)
        self.item_based = similarity(self.matrix,opsional="item-based",k=k)
        self.prediction_user_based = self.user_based.get_prediction_array()
        self.prediction_item_based = self.item_based.get_prediction_array()
        self.result_hybrid = self.main_calculation()

    def fusion(self,outer,inner) :
        print(f"{self.gamma_user} * {self.prediction_user_based[outer][inner]} + (1 - {self.gamma_item}) * {self.prediction_item_based[outer][inner]} = {(self.gamma_user * self.prediction_user_based[outer][inner] + (1-self.gamma_item) * self.prediction_item_based[outer][inner])}")
        return (self.gamma_user * self.prediction_user_based[outer][inner] + (1-self.gamma_item) * self.prediction_item_based[outer][inner])
    
    def main_calculation(self) :
         return [
            [
                (self.fusion(outer,inner) if self.matrix[outer][inner] == 0 else self.matrix[outer][inner]) 
                for inner in range(len(self.matrix[0]))
            ]
            for outer in range(len(self.matrix))
        ]
    
    def get_data_frame(self) :
        return pd.DataFrame(self.result_hybrid)