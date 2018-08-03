from __future__ import division, print_function
from typing import List
import numpy
import scipy


class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.weights = []

    def train(self, features: List[List[float]], values: List[float]):
        modifiedFeatures=[]
        for i in range(len(features)):
            modifiedFeature=[]
            currentFeature=features[i]
            modifiedFeature.append(1)
            for j in range(len(currentFeature)):
                modifiedFeature.append(currentFeature[j])
            modifiedFeatures.append(modifiedFeature)

        xMatrix=numpy.array(modifiedFeatures)
        xMatrixTranspose=xMatrix.transpose()
        product=numpy.matmul(xMatrixTranspose, xMatrix)
        inverse=numpy.linalg.inv(product)
        product2=numpy.matmul(inverse, xMatrixTranspose)

        yVector=numpy.array(values)
        product3=numpy.matmul(product2, yVector)

        self.weights=product3.tolist()

    def predict(self, features: List[List[float]]) -> List[float]:
        predictedValues=[]

        for i in range(len(features)):
            predictedValue=self.weights[0]
            currentFeature=features[i]
            for j in range(len(currentFeature)):
                predictedValue=predictedValue+(currentFeature[j]*self.weights[j+1])
            predictedValues.append(predictedValue)
        return predictedValues

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights


class LinearRegressionWithL2Loss:
    ''' L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.weights = []

    def train(self, features: List[List[float]], values: List[float]):
        modifiedFeatures=[]
        for i in range(len(features)):
            modifiedFeature=[]
            currentFeature=features[i]
            modifiedFeature.append(1)
            for j in range(len(currentFeature)):
                modifiedFeature.append(currentFeature[j])
            modifiedFeatures.append(modifiedFeature)

        xMatrix=numpy.array(modifiedFeatures)
        xMatrixTranspose=xMatrix.transpose()
        product=numpy.matmul(xMatrixTranspose, xMatrix)
        dim=product.shape[0]
        identity=numpy.identity(dim)
        product4=self.alpha*identity
        summed=numpy.add(product, product4)

        inverse=numpy.linalg.inv(summed)
        product2=numpy.matmul(inverse, xMatrixTranspose)

        yVector=numpy.array(values)
        product3=numpy.matmul(product2, yVector)

        self.weights=product3.tolist()

    def predict(self, features: List[List[float]]) -> List[float]:
        predictedValues=[]

        for i in range(len(features)):
            predictedValue=self.weights[0]
            currentFeature=features[i]
            for j in range(len(currentFeature)):
                predictedValue=predictedValue+(currentFeature[j]*self.weights[j+1])
            predictedValues.append(predictedValue)
        return predictedValues

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
