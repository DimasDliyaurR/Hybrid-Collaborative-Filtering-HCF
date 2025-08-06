import DistanceBased as SDB
import prediction as P
from tqdm import tqdm

from typing_extensions import override

import pandas as pd
from numpy import (
    transpose, 
    zeros, 
    array
)

from MatrixRating import MatrixRating

import os
import joblib
import time

class TverskyIndexSimilarity(SDB.Similarity, P.Prediction, SDB.Mean, MatrixRating) :
    
    def __init__(self, data, *, toyData : bool|None = False, opsional="user-based",k=2,alpha_1=0.7,alpha_2=0.2 ,time  : bool = False) -> None :
        """
        Tversky Index similarity function calculation

        Parameters:
        -----------
            data : Any
                The data used for calculation (can be a DataFrame or array)
            
            toyData : bool
                Determines whether to use toy data or real data
            
            opsional : str
                Calculation mode, "user-based" or "item-based"
            
            k : int
                Number of nearest neighbors used in prediction
            
            alpha_1 : float
                Alpha_1 parameter in the Tversky Index formula
            
            alpha_2 : float
                Alpha_2 parameter in the Tversky Index formula
            
            time : bool
                Determines whether to compute computation time
        
        Attributes:
        -----------
            alpha_1 : float
                Stores alpha_1 weighting
            
            alpha_2 : float
                Stores alpha_2 weighting
            
            time_computation_similarity_per_fold : list[list[float]]
                Stores computation time for each fold
            
        Methods:
        --------
            numerator(A,B)
                Calculates the numerator of the Tversky Index similarity function for sets A and B
            
            denominator(A,B)
                Calculates the denominator of the Tversky Index similarity function for sets A and B

            similarity_calculation(A,B,indexFold)
                Calculates the overall Tversky Index similarity function for sets A and B at fold indexFold
            
            main_calculation()
                Calculates the Tversky Index similarity for all users (user-based) or items (item-based)
            
            show_similarity(indexFold)
                Displays the similarity calculation results
            
            show_time_computation_similarity_array(indexFold)
                Displays the computation time results for similarity calculation
        
        Guide for usage
        ---------------

        Toy Data
        >> TI(data,k=5,opsional="user-based",alpha_1=0.7,alpha_2=0.3,toyData=True)
        
        Real Data
        >> TI(path_data,k=5,opsional="user-based",alpha_1=0.7,alpha_2=0.3)
        
        Time Computation
        >> TI(path_data,k=5,opsional="user-based",alpha_1=0.7,alpha_2=0.3,time=True)
        """
        
        SDB.Mean.__init__(self,data,opsional=opsional,toyData=toyData)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.time = time
        
        if not toyData and time :
            self.time_computation_similarity_per_fold = []


        if not toyData :
            
            path_file = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/tversky_index_similarity.joblib"
            path_file_prediction = "cache/tversky_index/" + ("user_based" if opsional == "user-based" else "item_based") + f"/{str(alpha_1)}/{str(alpha_2)}/prediction/{k}/tversky_index_prediction.joblib"
            
            if not time and os.path.exists(path_file) :
                self.result_similarity = joblib.load(path_file)
            else :
                self.result_similarity = self.main_calculation()
                joblib.dump(self.result_similarity,path_file) if not time else ""
        
                
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData,path_file=path_file_prediction,time=time)
            
        else :
            self.result_similarity = self.main_calculation()
            P.Prediction.__init__(self,data,self.result_similarity,opsional=opsional,k=k,toyData=toyData,time=time)
        

    def __checkSymmetric(self) -> bool :
        """
        Check the value of alpha_1 and alpha_2 are same
        
        Returns
        -------
            Boolean
        """
        return self.alpha_1 == self.alpha_2
    
    @override
    def numerator(self, A:list, B:list) -> float:
        """
        Calculates the numerator of the Tversky Index similarity function for sets A and B 

        Return:
        -------
        float
        """
        return len( set(A) & set(B) )

    @override
    def denominator(self, A:list, B:list) -> float:
        """
        Calculates the denominator of the Tversky Index similarity function for sets A and B 

        Parameters:
        -----------
        A : list
            First sets
        B : list
            Second sets

        Return:
        -------
            float
        """
        return len( set(A) & set(B) ) + (self.alpha_1 * len( set(A) - set(B) )) + self.alpha_2 * len( set(B) - set(A) )
    
    @override
    def similarity_calculation(self,A, B, indexFold : int|None = None) -> float :
        """
        Calculates the overall Tversky Index similarity function for sets A and B at fold indexFold

        Parameters:
        -----------
        A : list
            Himpunan pertama (misal: daftar item yang telah diberi rating oleh user A)
        B : list
            Himpunan kedua (misal: daftar item yang telah diberi rating oleh user B)
        
        Return:
        -------
            float
        """
        setA = set(self.getItem(A,indexFold) if self.opsional == "user-based" else self.getUser(A,indexFold))
        setB = set(self.getItem(B,indexFold) if self.opsional == "user-based" else self.getUser(B,indexFold))

        denom = self.denominator(setA,setB).real
        
        return (self.numerator(setA,setB).real / denom) if denom != 0 else 0

    @override
    def main_calculation(self) -> list[list[float]]:
        """
        Calculates the Tversky Index similarity matrix for all users (user-based) or items (item-based).
        Handles both toy data and real datasets, supporting symmetric and non-symmetric similarity computations.
        For real datasets, also tracks computation time per similarity calculation.
       
        Return:
        -------
            list[list[float]]
        """
        if self.toyData :
            matrix = self.matrixRating if self.opsional == "user-based" else self.reverseMatrixRating
            result = [[] for _ in range(len(matrix))]
            
            if self.__checkSymmetric() :
                for i in tqdm(range(len(matrix))):
                    for j in range(i,len(matrix)):
                        if i == j:
                            result[i].append(1)
                            continue
                        similarity_result = self.similarity_calculation(int(i), int(j))
                        result[i].append(similarity_result)
                        result[j].append(similarity_result)
                return result
            
            for i in tqdm(range(len(matrix))):
                for j in range(len(matrix)):
                    if i == j:
                        result[i].append(1)
                        continue
                    similarity_result = self.similarity_calculation(int(i), int(j))
                    result[i].append(similarity_result)
            return result
            
        else :
            if self.__checkSymmetric() :
                result = []
                
                for indexFold in tqdm(range(len(self.train)),desc="Tversky Index") :
                    matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])
                    
                    temp = array(zeros((len(matrix),len(matrix)))).tolist()

                    time_computation = [] if self.time else ""
                    for i in range(len(matrix)):
                        
                        result_inner = temp.copy()
                        time_computation_similarity_per_user = temp.copy() if self.time else ""
                        
                        for j in range(i,len(matrix)) :

                            if i == j:
                                result_inner[i][j] = 1
                                time_computation_similarity_per_user[i][j] = 1e-10 if self.time else ""
                                continue

                            t1 = time.time() if self.time else ""
                            similarity_result = self.similarity_calculation(i, j, indexFold)
                            t2 = time.time() if self.time else ""
                            time_computation_similarity_per_user += [t2-t1] if self.time else ""

                            result_inner[j][i] = similarity_result
                            result_inner[i][j] = similarity_result

                            if self.time :
                                time_computation_similarity_per_user[j][i] = t2-t1
                                time_computation_similarity_per_user[i][j] = t2-t1

                        result.append(result_inner)
                    self.time_computation_similarity_per_fold.append(time_computation) if self.time else ""
                return result

            result = []
            
            for indexFold in tqdm(range(len(self.train)),desc="Tversky Index") :
                matrix = self.train[indexFold] if self.opsional == "user-based" else transpose(self.train[indexFold])

                result_train = []
                time_computation = [] if self.time else ""

                for i in range(len(matrix)):
                    result_inner = []
                    time_computation_similarity_per_user = [] if self.time else ""
                    for j in range(len(matrix)):
                        if i == j:
                            result_inner.append(1)
                            continue

                        t1 = time.time() if self.time else ""
                        similarity_result = self.similarity_calculation(i, j, indexFold)
                        t2 = time.time() if self.time else ""

                        time_computation_similarity_per_user += [t2-t1] if self.time else ""
                        result_inner.append(similarity_result)
                    
                    time_computation += [time_computation_similarity_per_user] if self.time else ""
                    result_train.append(result_inner)
                
                result.append(result_train)
                self.time_computation_similarity_per_fold.append(time_computation) if self.time else ""
            
            return result

    def show_similarity(self,indexFold : int|None = None) :
        """
        Displays the similarity calculation results

        Parameters:
        -----------
            indexFold : int | None
                Indeks fold yang ingin ditampilkan. Jika None, tampilkan seluruh hasil.

        Returns:
        --------
            pd.DataFrame
                DataFrame hasil perhitungan similaritas.
        """
        if self.toyData :
            return pd.DataFrame(self.result_similarity)
        
        if indexFold is None :
            return pd.DataFrame(self.result_similarity)
        return pd.DataFrame(self.result_similarity[indexFold])
    
    def show_time_computation_similarity_array(self, indexFold : int|None = None) : 
        """
        Displays the computation time results for similarity calculation

        Parameters:
        -----------
            indexFold : int | None
                Indeks fold yang ingin ditampilkan. Jika None, tampilkan seluruh hasil.

        Returns:
        --------
            list[list[float]]
                Array time computation per fold.
        """
        if self.toyData :
            ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
        
        if self.time :
            if indexFold is None :
                return self.time_computation_similarity_per_fold
            return self.time_computation_similarity_per_fold[indexFold]

        ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")

    def show_time_computation_similarity(self, indexFold : int|None = None) : 
        """
        Displays the computation time results for similarity calculation, it will come with form DataFrame.

        Parameters:
        -----------
            indexFold : int | None
                Index fold that will be display. If the value contain None, overall folds of data will be display.

        Returns:
        --------
            pd.DataFrame
                The result of time computation that proceed at the each fold will be display with form DataFrame.
        """
        if self.toyData :
            ValueError("Mode waktu komputasi, tidak bisa dalam mode toy data")
        
        if self.time :
            if indexFold is None :
                return pd.DataFrame(self.time_computation_similarity_per_fold)
            return pd.DataFrame(self.time_computation_similarity_per_fold[indexFold])

        ValueError("Tidak dalam mode waktu komputasi, perlu nilai True pada parameter time")