# first line: 41
    @override
    def similarity_calculation(self, index1 : int, index2 : int, matrix : list[float], mean_centered : list[float]) -> float:
        print(f"Sim({index1},{index2})")
        matrix_filtered = [
                [mean_centered[index1][mc1]for mc1 in range(len(matrix[index1]))],
                [mean_centered[index2][mc2]for mc2 in range(len(matrix[index2]))],
            ]

        vector1 = np.delete(matrix_filtered[0],hp.indexOfZero(matrix[index1],matrix[index2])).tolist()
        vector2 = np.delete(matrix_filtered[1],hp.indexOfZero(matrix[index1],matrix[index2])).tolist()

        denominator = self.denominator(matrix[index2],matrix[index1]).real
        numerator = self.numerator(vector1,vector2).real

        return (numerator / denominator) if denominator != 0 and numerator != 0 else 0
