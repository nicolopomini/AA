# -*- coding: utf-8 -*-

###############################################################
# Clustering agglomerativo con complete linkage
# 
# Requisiti: python 2.7 con pandas, numpy, scipy e matplotlib
#
# Eseguire con:
# > python clustering.py
# oppure
# > chmod +x clustering.py
# > ./clustering.py
###############################################################

### Importiamo i moduli necessari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
######

### Leggiamo il dataset (che si deve trovare nella stessa directory dello script)
iris = pd.read_csv("iris.data",header=None)
m, n = iris.shape # m esempi, n-1 attributi
n -= 1 # in modo da non considerare la classe (ultima colonna del dataset)
######

### Riduciamo il numero di esempi per velocizzare la "nostra" procedura di clustering (inefficiente)
m = 30
iris = iris.iloc[list(range(0,10)) + list(range(50,60)) + list(range(100,110))] # 10 esempi per ogni classe
######

### Rimescoliamo i dati per non essere influenzati dall'ordinamento originario (per classe)
np.random.seed(1) # fissiamo il seed del generatore di numeri casuali per riproducibilita'
idx = list(range(m))
np.random.shuffle(idx) # rimescoliamo gli indici
iris = iris.iloc[idx] # permutiamo le righe del dataset
######

### Normalizziamo gli attributi all'intervallo [0,1]
for j in range(n):
	iris[j] = (iris[j] - np.min(iris[j])) / (np.max(iris[j]) - np.min(iris[j]))
######

### Calcoliamo la matrice delle distanze
distances = np.zeros((m,m))
triangle = np.empty(m*(m-1)//2) # il modulo hierarchy di SciPy si aspetta solo il triangolo superiore della matrice come un array 1D
k = 0
for i in range(m-1):
	for j in range(i+1,m):
		triangle[k] = distances[i,j] = distances[j,i] \
                    = np.linalg.norm(iris.iloc[i,:n] - iris.iloc[j,:n]) # distanza Euclidea
		k += 1
######

### Visualizziamo la matrice delle distanze come una "mappa di calore" (blu per distanze piccole, rosso per distanze grandi)
plt.imshow(distances) # immagine apparentemente casuale (ma simmetrica)
plt.show()
#plt.savefig("distances.png")
#plt.close()
######

### Creiamo dizionario per memorizzare tutti i cluster "correnti" durante il clustering
clusters = {}
for i in range(m):
	clusters[i] = [i] # ogni cluster e' una lista di indici, con chiave numerica da 0 a m-1 per i cluster iniziali contenenti singoli esempi, chiave da m a 2m-2 per i cluster di piu' elementi creati in seguito
######

### Memorizziamo per ogni iterazione (in totale m-1 iterazioni) i 2 cluster da unire, la loro distanza e il numero di esempi nel nuovo cluster
H = np.zeros((m-1,4)) # H[i,0] e H[i,1] sono i cluster da unire all'iterazione i
                      # H[i,2] distanza tra i 2 cluster
                      # H[i,3] numero di esempi nel nuovo cluster
for it in range(m-1):
	clust_idxs = list(clusters) # lista delle chiavi del dizionario
	H[it,2] = +np.infty # vogliamo trovare i cluster a distanza minima
	for i in range(len(clust_idxs)-1):
		for j in range(i+1,len(clust_idxs)):
			clust_couple_dist = -np.infty # la distanza tra 2 cluster e' la massima distanza tra ogni possibile coppia di esempi dei 2 cluster
			for ii in clusters[clust_idxs[i]]:
				for jj in clusters[clust_idxs[j]]:
					current_dist \
					   = np.linalg.norm(iris.iloc[ii,:n] - iris.iloc[jj,:n])
					if current_dist > clust_couple_dist:
						clust_couple_dist = current_dist
			if clust_couple_dist < H[it,2]: # se la coppia di cluster attuale e' quella con la distanza minima trovata finora, aggiorniamo H
				H[it,2] = clust_couple_dist
				H[it,0] = clust_idxs[i]
				H[it,1] = clust_idxs[j]
				H[it,3] = len(clusters[clust_idxs[i]]) + len(clusters[clust_idxs[j]])
	clusters[m+it] = clusters[H[it,0]] + clusters[H[it,1]] # uniamo i 2 cluster piu' vicini
	del clusters[H[it,0]] # rimuoviamo dal dizionario i cluster precedenti
	del clusters[H[it,1]]
######

### Visualizziamo il "nostro" dendrogramma
dn = hierarchy.dendrogram(H)
plt.show()
#plt.savefig("dendrogram.png")
#plt.close()
######

### Rifacciamo il clustering usando la libreria
Z = hierarchy.complete(triangle)
######

### Visualizziamo il dendrogramma restituito dalla libreria
hierarchy.dendrogram(Z)
plt.show()
#plt.savefig("dendrogram_lib.png")
#plt.close()
######

### Usiamo l'ordine delle foglie del dendrogramma per permutare righe e colonne della matrice delle distanze. La mappa di calore adesso dovrebbe essere "meno casuale" ed evidenziare i cluster
permutation = dn["leaves"]
permuted_distances = distances[permutation]
permuted_distances = permuted_distances[:,permutation]
plt.imshow(permuted_distances)
plt.show()
#plt.savefig("permuted_distances.png")
#plt.close()
######

