import numpy as np
import pandas as pd
import helper.helper as hp

class Prediction:
    """
    Kelas Prediction digunakan untuk menghasilkan prediksi pada sistem rekomendasi berbasis Collaborative Filtering
    dengan menggunakan data mean-centered dan similarity matrix. Kelas ini mendukung pendekatan user-based atau item-based,
    serta opsi untuk menggunakan mean-centered dari "brother dataset" jika tersedia.

    Attributes:
    -----------
    similarity : list of list
        Matriks similarity yang digunakan untuk perhitungan prediksi.
    mean_centered : list of list
        Hasil mean-centered dari data asli yang diberikan.
    data : list of list
        Matriks data input yang akan digunakan untuk menghitung prediksi.
    k : int
        Jumlah tetangga terdekat (neighbors) yang akan digunakan dalam perhitungan prediksi.
    meanList : list of float
        Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
    opsional : str
        item-based,user-based.
    prediction : list of list
        Matriks hasil prediksi yang dihasilkan oleh algoritma.
    twins : bool, optional
        Menentukan apakah menggunakan dataset kembar (brother dataset) untuk perhitungan mean centered.

    Methods:
    --------
    numerator(similarity, meanCentered)
        Menghitung nilai pembilang (numerator) dari formula prediksi.

    denominator(similarity)
        Menghitung nilai penyebut (denominator) dari formula prediksi.

    selectedNeighborhood(neighborhood, index, indexUser, k, data, meanCentered, *, opsional, twins)
        Memilih `k` tetangga terdekat dari similarity matrix untuk perhitungan prediksi.

    prediction_calculation(userTarget, item)
        Menghasilkan nilai prediksi untuk user dan item tertentu.

    main_prediction_calculation(data)
        Menghasilkan matriks prediksi untuk seluruh data.

    getPredictionArray()
        Mengembalikan hasil prediksi dalam bentuk array numpy.

    getPredictionDataFrame()
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.
    """

    def __init__(self, meanC, similarity, data, *, meanList, meanListBrother=[], mean_centered_result_brother=[], opsional, k, twins=False):
        """
        Inisialisasi objek Prediction.

        Parameters:
        -----------
        meanC : list of list
            Mean-centered matrix dari data yang diberikan.
        similarity : list of list
            Matriks similarity yang dihitung sebelumnya (dari kelas BC atau sejenis).
        data : list of list
            Matriks data input yang akan diprediksi.
        meanList : list of float
            Daftar mean dari masing-masing vektor (baris atau kolom) dalam data.
        meanListBrother : list of float, optional
            Daftar mean dari brother dataset, jika menggunakan dataset kembar.
        mean_centered_result_brother : list of list, optional
            Mean-centered dari brother dataset, jika menggunakan dataset kembar.
        opsional : str
            item-based,user-based.
        k : int
            Jumlah tetangga terdekat yang akan digunakan dalam prediksi.
        twins : bool, optional
            Menentukan apakah menggunakan dataset kembar (brother dataset) untuk perhitungan mean-centered.
        """
        self.similarity = similarity
        self.twins = twins
        if twins:
            self.meanListBrother = meanListBrother
            self.mean_centered_result_brother = mean_centered_result_brother
        self.mean_centered = meanC
        self.data = data
        self.k = k
        self.meanList = meanList
        self.opsional = opsional
        self.prediction = self.main_prediction_calculation(self.data)
        self.topN = self.get_top_n()

    @staticmethod
    def __numerator(similarity, meanCentered):
        """
        Menghitung nilai pembilang (numerator) dari formula prediksi.

        Parameters:
        -----------
        similarity : list of float
            Daftar nilai similarity untuk tetangga terdekat.
        meanCentered : list of float
            Mean-centered nilai untuk tetangga terdekat.

        Returns:
        --------
        float
            Hasil pembilang dari formula prediksi.
        """
        return sum(sim * meanC for sim, meanC in zip(similarity, meanCentered))

    @staticmethod
    def __denominator(similarity):
        """
        Menghitung nilai penyebut (denominator) dari formula prediksi.

        Parameters:
        -----------
        similarity : list of float
            Daftar nilai similarity untuk tetangga terdekat.

        Returns:
        --------
        float
            Hasil penyebut dari formula prediksi.
        """
        return sum(abs(sim) for sim in similarity)

    @staticmethod
    def selectedNeighborhood(neighborhood, index, indexUser, k, data, meanCentered, *, opsional, twins) -> list:
        """
        Memilih `k` tetangga terdekat dari similarity matrix untuk perhitungan prediksi.

        Parameters:
        -----------
        neighborhood : list of list
            Matriks similarity yang akan dipilih tetangga terdekatnya.
        index : int
            Indeks item atau user yang menjadi referensi.
        indexUser : int
            Indeks user yang sedang diproses.
        k : int
            Jumlah tetangga terdekat yang akan dipilih.
        data : list of list
            Matriks data yang sedang diproses.
        meanCentered : list of list
            Mean-centered data dari matriks asli.
        opsional : str
            item-based, user-based.
        twins : bool
            Apakah menggunakan brother dataset untuk mean-centered.

        Returns:
        --------
        list of list
            Daftar `k` tetangga terdekat beserta mean-centered data dari tetangga tersebut.
        """
        meanCentered = hp.reverseMatrix(meanCentered) if not twins or opsional == "item-based" else meanCentered

        indexZero = hp.checkIndexZeroOfData(data=hp.reverseMatrix(data) if opsional == "user-based" else data, fixIndex=indexUser if opsional == "user-based" else index, indexUser=indexUser if opsional == "item-based" else index,maxIndex=len(neighborhood))

        indexOfNeighborhood = list(np.delete(hp.createList(0, len(neighborhood[0]) - 1), indexZero).tolist())

        neighborhood = list(np.delete(neighborhood[indexUser if opsional == "user-based" else index], indexZero).tolist())

        lengthLoop = len(neighborhood)
        for i in range(lengthLoop - 2, -1, -1):
            indexFlag = i
            prevNeighborhood = np.real(neighborhood[i])
            prevIndexList = indexOfNeighborhood[i]
            innerCondition = True
            j = i + 1
            while innerCondition and j < lengthLoop and prevNeighborhood < np.real(neighborhood[j]):
                if prevNeighborhood < np.real(neighborhood[j]):
                    indexFlag = j
                    neighborhood[j - 1] = np.real(neighborhood[j])
                    indexOfNeighborhood[j - 1] = indexOfNeighborhood[j]
                    j += 1
                else:
                    j += 1
                    innerCondition = False
            neighborhood[indexFlag] = prevNeighborhood
            indexOfNeighborhood[indexFlag] = prevIndexList

        meanCenteredBasedIndexNeighborhood = [
            (meanCentered[indexUser if opsional ==  "item-based" else index][i]) 
            for i in indexOfNeighborhood[0:k]
        ]
        return [neighborhood[0:k], meanCenteredBasedIndexNeighborhood ]

    def prediction_calculation(self, userTarget, item) -> float:
        """
        Menghasilkan nilai prediksi untuk user dan item tertentu.

        Parameters:
        -----------
        userTarget : int
            Indeks user yang sedang diprediksi.
        item : int
            Indeks item yang sedang diprediksi.

        Returns:
        --------
        float
            Nilai prediksi berdasarkan formula Collaborative Filtering.
        """
        target = self.selectedNeighborhood(self.similarity, item, userTarget, self.k, self.data, self.mean_centered if not self.twins else hp.reverseMatrix(self.mean_centered_result_brother), opsional=self.opsional, twins=self.twins)
        average = self.meanList[userTarget if self.opsional == "user-based" else item] if not self.twins else (self.meanListBrother[userTarget if self.opsional == "user-based" else item])
        numerator = self.__numerator(target[0], target[1])
        denom = self.__denominator(target[0])
        result = (average + (numerator / denom)) if denom != 0 else 0
        return result

    def main_prediction_calculation(self, data):
        """
        Menghasilkan matriks prediksi untuk seluruh data.

        Parameters:
        -----------
        data : list of list
            Matriks data yang akan dihitung prediksinya.

        Returns:
        --------
        list of list
            Matriks prediksi untuk seluruh data.
        """
        return [
            [
                (self.prediction_calculation(i, j) if data[i][j] == 0 else data[i][j]) 
                for j in range(len(data[0]))
            ]
            for i in range(len(data))
        ]
    
    def get_top_n(self) :
        """
        Mengembalikan hasil dari Top-N dari prediksi

        Returns:
        --------
        array
            Array yang berisi tentang Top-N
        """
        result = []
        for i in range(len(self.data)) :
            result.append(
                sorted([self.prediction[i][inner] for inner in range(len(self.data[i])) if self.data[i][inner] == 0])[::-1][0:self.k]
                # Sorting a similarities
                # list(sim[np.array(sim).argsort()[::-1]][0:self.k])
                )
        return result

    def get_prediction_array(self):
        """
        Mengembalikan hasil prediksi dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi hasil prediksi.
        """
        return np.array(self.prediction).tolist()
    
    def get_prediction_data_frame(self):
        """
        Mengembalikan hasil prediksi dalam bentuk array numpy.

        Returns:
        --------
        numpy.ndarray
            Array numpy yang berisi hasil prediksi.
        """
        return pd.DataFrame(self.prediction)
    
    def get_top_n_array(self) :
        """
        Mengembalikan hasil prediksi dalam bentuk DataFrame pandas.

        Returns:
        --------
        pandas.DataFrame
            DataFrame yang berisi hasil prediksi.
        """
        return self.topN