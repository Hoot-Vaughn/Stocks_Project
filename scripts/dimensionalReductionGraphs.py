
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pacmap
import pandas as pd
import numpy as np

import sys
import os

#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dimReduction.dReduce import *

df = pd.read_csv('Stocks_Project/data/World-Stock-Prices-Dataset_ContData.csv')

dfint = df.drop(columns=['Date'])
dfint = dfint.astype(float)
y = dfint.columns.to_numpy()


print(dfint.columns)
print(dfint.dtypes)
print(y)


#graphPCA(dfint, 2, 2)
graphtSNE(dfint, 2)