# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

col_names = []	#contiene i nomi delle colonne
#leggo i nomi delle colonne dal file adult.names
with open("adult.names",'r') as f:
	for line in f:
		line = line.strip()
		if len(line) == 0 or line[0] == '|':
			continue
		p = line.find(':')
		if p < 0:
			continue
		col_names.append(line[:p])
col_names.append("income")
#print(col_names)
#leggo il dataset di training
training = pd.read_csv("adult.data", names = col_names, skipinitialspace = True)
X_t = pd.get_dummies(training[training.columns[:-1]])	#tutte le colonne tranne l'ultima. Get dummies esplode le colonne non numeriche in N colonne di 0 e 1
y_t = training[training.columns[-1]]	#ultima colonna

#leggo il dataset di validazione
validation = pd.read_csv("adult.test", names = col_names, skipinitialspace = True, skiprows = 1)
X_v = pd.get_dummies(validation[validation.columns[:-1]]).reindex(columns = X_t.columns, fill_value = 0)
y_v = validation[validation.columns[-1]]
translation = {v:v[:-1] for v in y_v.unique()}	#alcune righe del dataset hanno un punto finale. Lo tolgo
y_v = y_v.replace(translation)

#creo e alleno l'albero di decisione, provando diverse profondità dell'albero: da 1 a 30
depths = range(1,30)
accuracies = []
for depth in depths:
	classifier = tree.DecisionTreeClassifier(max_depth = depth)
	classifier.fit(X_t,y_t)

	y_p = classifier.predict(X_v)
	accuracy = sum(y_p == y_v) / float(len(y_v))		#accuratezza del modello
	accuracies.append(accuracy)
	print depth, accuracy
#disegno un grafico con le accuratezze in base alla profondità
plt.plot(depths, accuracies)
plt.show()

#più l'albero è profondo, più cresce il problema di overfitting: l'albero impara tutte le anomalie del dataset di training!
