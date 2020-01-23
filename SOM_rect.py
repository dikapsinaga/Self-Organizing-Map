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

def update_weight(neuron, data, idx_winner, alpha, R, num_of_col):
    print(idx_winner)
    idx_win_row = int(idx_winner / num_of_col)
    idx_win_col = idx_winner % num_of_col
    print("Center", idx_win_row, idx_win_col)
#         if len(neuron)/num_of_col == num_of_col and :
#             neuron[idx_winner] += ( alpha * (data - neuron[idx_winner]) )
#             print("T")
#             print(idx_win_row, idx_win_col)
    for i in range(-R, R+1):
        if idx_win_row + i >= 0 and (idx_win_row + i) < (len(neuron) / num_of_col) : # check if row out of bounds
            if R - abs(i) == 0 : # Special case on top and bottom neuron
                index = (num_of_col * (idx_win_row + i)) + idx_win_col
                print(index)
                neuron[index] += ( alpha * (data - neuron[index]) )
                print("A")
                print( (idx_win_row + i), idx_win_col)
            else :
                if R - abs(i) != idx_win_row : # updating neuron on center for each row but not in row winner neuron 
                    index = (num_of_col * (idx_win_row + i)) + idx_win_col
                    print(index)
                    neuron[index] += ( alpha * (data - neuron[index]) )
                    print("B")
                    print( (idx_win_row + i), idx_win_col )
                    
                if R - abs(i) == idx_win_row and len(neuron)/num_of_col == num_of_col:
                    index = (num_of_col * (idx_win_row + i)) + idx_win_col
                    print(index)
                    neuron[index] += ( alpha * (data - neuron[index]) )
                    print("T")
                    print( (idx_win_row + i), idx_win_col )
                
                for j in range(1, (R - abs(i) + 1)) :
                    if idx_win_col-j > -1 :
                        index = (num_of_col * (idx_win_row + i)) + idx_win_col-j
                        print(index)
                        neuron[index] += ( alpha * (data - neuron[index]) )
                        print("C")
                        print( (idx_win_row + i), idx_win_col-j)
                    if idx_win_col+j < num_of_col :
                        index = (num_of_col * (idx_win_row + i)) + idx_win_col+j 
                        print(index)
                        neuron[index] += ( alpha * (data - neuron[index]) )
                        print("D")
                        print( (idx_win_row + i), idx_win_col+j )
        else:
            print("SKIP")
    neuron = np.around(neuron, decimals=6)
    print(neuron)
    return neuron

def train_SOM(neuron, data, alpha, c, R, Et, E0, num_of_col):
    epoch = 0
    
    while epoch < Et:
        epoch += 1
        print("Epoch", epoch)

        for row_data in data:
            idx = find_winner_neuron(neuron, row_data)
            print("winner", idx)
            neuron = update_weight(neuron, row_data, idx, alpha, R, num_of_col)
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