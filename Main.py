import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplleaflet
import SOM as som
import SC as sc
from sklearn.metrics import silhouette_score

df = pd.read_csv('fire_archive_M6_81128.csv')

df = df[df['confidence'] > 30]
df = df.iloc[:10000,:]
# df['acq_date'] = pd.to_datetime(df['acq_date'])
# df_year = df[df['acq_date'].dt.year == 2016]

fitur = ['latitude', 'longitude', 'brightness', 'confidence']
data = df[fitur]

dataset = data.copy()
# normalisasi
dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
# Inisialisasi bobot

weight = np.array([[0.61, 0.88, 0.79, 0.60],
          [0.59, 0.96, 0.41, 0.97],
          [0.76, 0.15, 0.05, 0.38]])
# weight = np.around(np.random.uniform(low=0.01, high=0.99, size=(5, 4)), decimals=2)

R = 1
Alpha = 0.1
c = 0.1
Et = 18
E0 = 1
model = som.train_SOM(weight, dataset.values, Alpha, c, R, Et, E0)
klaster = som.test_SOM(model, dataset.values)
klaster = np.ravel(klaster)
sil = silhouette_score(dataset.values, klaster, metric='euclidean')
print(sil)