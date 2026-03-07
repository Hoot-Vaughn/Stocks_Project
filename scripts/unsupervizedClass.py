import pandas as pd
from sklearn import preprocessing


class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path).dropna()
        
        
        print(data.head())

        data = data.to_numpy()

        return data