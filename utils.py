from typing import List
from scipy.spatial import distance
import numpy as np
import math


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    sum=0
    for i in range(len(y_true)):
        error=(y_true[i]-y_pred[i])*(y_true[i]-y_pred[i])
        sum=sum+error
    mse = sum/len(y_true)
    return round(mse, 6)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp=0
    fp=0
    fn=0
    for x in range(len(real_labels)):
        if(real_labels[x]==1):
            if(predicted_labels[x]==1):
                tp=tp+1
            else:
                fn=fn+1
        else:
            if(predicted_labels[x]==1):
                fp=fp+1

    if((tp+fp)==0):
        return 0

    if((tp+fn)==0):
        return 0

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)

    if((precision+recall)==0):
        return 0
    return (2*precision*recall)/(precision+recall)


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    modifiedFeatures=[]
    for i in range(len(features)):
        modifiedFeature=[]
        currentFeature=features[i]
        count=0
        while(count<k):
            s=currentFeature[0]
            num=math.pow(s,count+1)
            modifiedFeature.append(num)
            count=count+1

        modifiedFeatures.append(modifiedFeature)

    return modifiedFeatures


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return distance.euclidean(point1, point2)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    if len(point1) != len(point2):
        return 0

    return sum([x*y for x,y in zip(point1,point2)])


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    dist = distance.euclidean(point1, point2)
    return (-1)*math.exp(-0.5*dist*dist)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalizedFeatures=[]
        for x in range(len(features)):
            normalizedFeature=[]
            currentFeature=features[x]
            magnitude=math.sqrt(sum(currentFeature[i]*currentFeature[i] for i in range(len(currentFeature))))
            if(magnitude==0):
                    normalizedFeature=currentFeature
            else:
                for j in range(len(currentFeature)):
                    normalizedFeature.append(round(currentFeature[j]/magnitude, 6))

            normalizedFeatures.append(normalizedFeature)
        return normalizedFeatures


class MinMaxScaler:
    """
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    
    def __init__(self):
        self.isTrain=1
        self.maximum = []
        self.minimum = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample 
        """
        if(self.isTrain==1):

            dimCount=len(features[0])

            for i in range(dimCount):
                feature=[item[i] for item in features]
                self.maximum.append(max(feature))
                self.minimum.append(min(feature))

            minmaxFeatures=[]
            for j in range(len(features)):
                minmaxFeature=[]
                currentFeature=features[j]
                for w in range(len(currentFeature)):
                    newVal=(currentFeature[w]-self.minimum[w])/(self.maximum[w]-self.minimum[w])
                    minmaxFeature.append(newVal)
                minmaxFeatures.append(minmaxFeature)

            self.isTrain=0;
            return minmaxFeatures

        else:
            minmaxFeatures=[]
            for j in range(len(features)):
                minmaxFeature=[]
                currentFeature=features[j]
                for w in range(len(currentFeature)):
                    newVal=(currentFeature[w]-self.minimum[w])/(self.maximum[w]-self.minimum[w])
                    minmaxFeature.append(newVal)
                minmaxFeatures.append(minmaxFeature)

            return minmaxFeatures