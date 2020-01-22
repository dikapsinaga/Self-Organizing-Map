import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def calculate_distance(neuron, data):
    distance = []
    for idx, J in enumerate(neuron):
        distance.append(np.sum((J - data) ** 2))
    print(distance)
    return distance
 
def find_winner_neuron(neuron, data):
    distance = calculate_distance(neuron, data)
    idx_winner = distance.index(min(distance))
    return idx_winner

def update_weight(neuron, data, idx_winner, alpha, R):
    neuron[idx_winner] += ( alpha * (data - neuron[idx_winner]))
    for idx in range(1, R+1) :
        if idx_winner + idx < len(neuron) :
            neuron[idx_winner + idx] += ( alpha * (data - neuron[idx_winner + idx]) )
        if idx_winner - idx >= 0 : 
            neuron[idx_winner - idx] += ( alpha * (data - neuron[idx_winner - idx]) )
    return neuron

def train_SOM(neuron, data, alpha, c, R, Et, E0):
    epoch = 0
    
    while epoch < Et:
        epoch += 1
        print("Epoch", epoch)

        for row_data in data:
            idx = find_winner_neuron(neuron, row_data)
            print("winner", idx)
            neuron = update_weight(neuron, row_data, idx, alpha, R)
#             print(neuron[idx])
        if epoch % E0 == 0 and R > 1:
            R -= 1

        alpha *= c
        
    print("Total Epoch:", epoch)
    return neuron


def test_SOM(neuron, data):
    cluster = []
    for row in data:
        cluster.append(find_winner_neuron(neuron, row) + 1)
    return pd.DataFrame({"cluster": cluster})

def fit(neuron, data_learn, data_test, alpha, c, R, Et, E0):
    model = train_SOM(neuron, data_learn, alpha, c, R, Et, E0)
    klaster =  test_SOM(model,data_test)
    klaster = np.ravel(klaster)
    sil = silhouette_score(data_test, klaster, metric='euclidean')
    return sil