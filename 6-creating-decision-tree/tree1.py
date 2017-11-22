# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import pandas as pd
import pprint

#y = [1,2,1,1,1,2,2,3,3,3,2,2,2,1,3]	#dataset provvisorio

def distribution(y):
	m = len(y)
	c = Counter(y)	#conta le ricorrenze di ciascun elemento in y
	return {v:float(n)/m for v,n in c.items()}	#distribuzione (di stima) di probabilità per ogni campione di y

def entropy(distr):
	H = 0.0
	for p in distr.values():
		H -= p * np.log(p)
	return H / np.log(2)	#utilizzo il log in base 2 per il calcolo dell'entropia, in termini di bit

def gini(distr):
	G = 1.0
	for p in distr.values():
		G - p**2
	return G

def split(X,y):
	#itero su valori diversi di theta
	bestEH = 1e10
	for j in range(4):
		#per ogni colonna trovo tutti i valori possibili e provo con ciascun valore quale è il migliore
		unique_values = np.unique(X[:,j])
		for i in range(len(unique_values) - 1):
			theta = (unique_values[i] + unique_values[i+1])/2		
			selector = X[:,j] <= theta
			X1 = X[selector]	#X1 sottomatrice di X dove i valori sono minori di theta
			y1 = y[selector]	#y1 come sopra rispetto a y

			selector = X[:,j] > theta
			X2 = X[selector]
			y2 = y[selector]

			H1 = entropy(distribution(y1))
			H2 = entropy(distribution(y2))
			EH = (H1 * len(y1) + H2 * len(y2)) / len(y)	#entropia attesa: media pesata tra le due entropie
			if EH < bestEH:
				bestEH = EH
				best_split = {
					"column" : j,
					"threshold" : theta,
					"split1" : (X1,y1,H1),
					"split2" : (X2, y2, H2)		
				}
	return best_split

def create_tree(X,y):	#crea un albero sotto forma di dizionario ricorsivamente
	H = entropy(distribution(y))
	if H == 0:
		return {'y' : y[0]}
	best_split = split(X,y)
	return {
		"column" : best_split["column"],
		"threshold" : best_split["threshold"],
		"yes" : create_tree(
								best_split["split1"][0],
								best_split["split1"][1]
									),
		"no" : create_tree(
								best_split["split2"][0],
								best_split["split2"][1]
									)
	}

def predict(x, tree):
	if 'y' in tree:	#foglia
		return tree['y']
	if x[tree["column"]] <= tree["threshold"]:
		return predict(x, tree["yes"])
	else:
		return predict(x, tree["no"])

iris = pd.read_csv(
	'iris.data', 
	names=[
		'Sepal width',
		'Sepal length',
		'Petal width',
		'Petal length',
		'Class'
	]
)
X = iris.iloc[:,0:4].as_matrix()
y = iris.iloc[:,4].as_matrix()
H = entropy(distribution(y))
#print(H)

tree = create_tree(X,y)
#pprint.pprint(tree)
test = [5.6, 2.5, 3.9, 1.1]
print(predict(test,tree))
