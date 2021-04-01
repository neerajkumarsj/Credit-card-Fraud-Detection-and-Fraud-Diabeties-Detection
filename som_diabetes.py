# -*- coding: utf-8 -*-

#Self Organizing Map

##Install MiniSom Package
"""

!pip install MiniSom

"""### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv('E:\\Neeraj\\Exam and Careers\\DataScience\\AI , Deep Learning and NLP\\Vision AI\\SOM\\diabetes.csv')
n = dataset.shape[0]
j = []
i = 0
while i <= n:
    j.append(i)
    i += 1
    if i == 768:
        break
dataset.insert(0,'Patient ID',j,True )
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values


"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 9, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""
"""#
bone() - uses a bare bone canvas
som.distance_map is the way of finfing the mean and then using the neuron distance


"""#
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding Who are Falsely Diagonised as Diabeties"""

mappings = som.win_map(X)
True_diab = np.concatenate((mappings[(1,8)], mappings[(1,7)]), axis = 0)
True_diab = sc.inverse_transform(True_diab)

"""##Printing the Fraunch Clients"""


True_diab = pd.DataFrame(True_diab, columns = ['Patient ID','Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'] )
print('Patient IDs Who are Falsely Diagonised as no Diabeties ')
print(True_diab.iloc[:,:1])
"""
From The figure We can see that these are the, 
Patient IDs Who are Falsely Diagonised as no Diabeties 
   Patient ID
0         2.0
1        22.0
2        44.0
3       131.0
4       167.0
"""

"""## Finding Who are Falsely Diagonised as no Diabeties but have diabeties"""
false_diab = np.concatenate((mappings[(4,0)], mappings[(2,9)], mappings[(5,9)]), axis = 0)
false_diab = sc.inverse_transform(false_diab)

"""##Printing the Fraunch Clients"""


false_diab = pd.DataFrame(false_diab, columns = ['Patient ID','Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'] )
print('Patient IDs Who are Falsely Diagonised as Diabeties ')
print(false_diab.iloc[:,:1])
"""
From The figure We can see that these are the, 
Patient IDs Who are Falsely Diagonised as Diabeties 
"""
"""
From The figure We can see that these are the, 
Patient IDs Who are Falsely Diagonised as Diabeties 
    Patient ID
0         24.0
1         25.0
2         86.0
3         88.0
4        159.0
5        215.0
6        254.0
7        298.0
8        306.0
9        323.0
10       358.0
11       375.0
12       455.0
13       154.0
14       185.0
15       209.0
16       238.0
17       327.0
18       408.0
19        69.0
20        73.0
21       110.0
22       132.0
23       144.0
24       189.0
25       195.0
26       199.0
27       244.0
28       335.0
29       415.0
"""