from __future__ import division, print_function
from typing import List, Callable
import numpy
import scipy


class KNN:
	
	def __init__(self, k: int, distance_function) -> float:
		self.k = k
		self.distance_function = distance_function

	def getKNeighbors(self, currentFeature: List[float]) -> List[int]:
		allNeighbors = []
		for x in range(len(self.trainingFeatures)):
			dist = self.distance_function(currentFeature, self.trainingFeatures[x])
			allNeighbors.append([dist, self.trainingLabels[x]])

		allNeighbors.sort(key=lambda x: x[0])
		kNeighborsLabels = []
		counter=0
		while(counter<self.k):
			kNeighborsLabels.append(allNeighbors[counter][1])
			counter = counter+1
		return kNeighborsLabels

	def getMajorityLabel(self, kNeighborsLabels) -> int:
		zeroLabel = 0
		oneLabel = 0
		for x in range(len(kNeighborsLabels)):
			if(kNeighborsLabels[x] == 0):
				zeroLabel = zeroLabel+1
			else:
				oneLabel = oneLabel+1
		
		if(zeroLabel>oneLabel):
			return 0
		else:
			return 1

	def train(self, features: List[List[float]], labels: List[int]):
		self.trainingFeatures=features
		self.trainingLabels=labels

	def predict(self, features: List[List[float]]) -> List[int]:
		predictionLabels = []
		for i in range(len(features)):
			kNeighborsLabels = self.getKNeighbors(features[i])
			predictionLabels.append(self.getMajorityLabel(kNeighborsLabels))
		return predictionLabels


if __name__ == '__main__':
	print(numpy.__version__)
	print(scipy.__version__)
