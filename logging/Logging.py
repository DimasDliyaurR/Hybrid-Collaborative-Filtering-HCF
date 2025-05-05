import pandas as pd

class Logging() :

    def __init__(self, meta_data : dict ,*, sheet_name : str, path_file : str) :
        self.meta_data = meta_data
        self.sheet_name = sheet_name
        self.path_file = path_file

    def __modification(self) : ...

    def __writing(self) : ...