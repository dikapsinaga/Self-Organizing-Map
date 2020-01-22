import numpy as np
import pandas as pd

def silhouette(data):
    a = []
    b = []

    for i, row1 in data.iterrows(): 
        sum_a = 0
        count_a = 0
        sum_b = {}
        count_b = {}

        for j, row2 in data.iterrows():
            if(row1.klaster == row2.klaster):
                count_a += 1
                sum_a += euclidian_distance(row1, row2)
            else:
                klaster = row2.klaster
                if klaster in sum_b:
                    sum_b[klaster] += euclidian_distance(row1,row2)
                    count_b[klaster] += 1 
                else:
                    sum_b[klaster] = euclidian_distance(row1, row2)
                    count_b[klaster] = 1
            
        for key in sum_b:
            sum_b[key] = sum_b[key]/count_b[key]
        
        b.append((sum_b[min(sum_b, key=sum_b.get)]))
        a.append(1/(count_a-1)*sum_a)
                
    a = np.array(a)
    b = np.array(b)
    
    sil = (b-a)/np.maximum(a,b)
    
    dictSil = {}
    
    for i in range (len(sil)):
        klaster = data.klaster.iloc[i]
        if klaster in dictSil:
            dictSil[klaster] += sil[i]
        else:
            dictSil[klaster] = sil[i]

    count = data["klaster"].value_counts()
    score = []
    for key in dictSil:
        score.append(dictSil[key]/count[key])
    return np.average(score)

def euclidian_distance(data1, data2):
    data1 = data1.drop('klaster')
    data2 = data2.drop('klaster')
    return np.sqrt(np.sum((data1-data2)**2))