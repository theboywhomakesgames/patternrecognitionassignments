from tkinter import W
import numpy as np
from scipy.spatial import distance

class KNN:
	def __init__(self, n):
		self.k = n

	def prepare(self, X, Y):
		# find all the distances of all the data nxn matrix of distances
		self.X = X
		self.Y = Y

		self.distances = np.zeros((Y.shape[0], Y.shape[0]))

		# calculate the distances for later
		for i in range(self.distances.shape[0]):
			for j in range(self.distances.shape[1]):
				if(i == j):
					self.distances[i][j] = 0
				else:
					self.distances[i][j] = self.distances[j][i] = distance.euclidean(X[i], X[j])

		return

	def find_label(self, i):
		# after fitting this can predict the label for a dp
		row = self.distances[i].tolist()
		labels = self.Y.tolist()

		knn_labels = []
		visited = []

		for i in range(self.k):
			min = float("inf")
			best_idx = -1
			for idx, distance in enumerate(row):
				if(idx not in visited and  distance < min and distance != 0):
					min = distance
					best_idx = idx

			visited.append(best_idx)
			knn_labels.append(labels[best_idx])

		counts = {}
		for l in knn_labels:
			if(str(l) in counts):
				counts[str(l)] += 1
			else:
				counts[str(l)] = 1
		
		label = ""
		count = 0
		for l in counts:
			if(counts[l] > count):
				label = l
				
		return label

	def predict(self, dp):
		return

		return counts