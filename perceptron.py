from __future__ import division, print_function
from typing import List, Tuple, Callable
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. The algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''

        converged=False
        mistake=False
        iterations=0
        while(iterations<self.max_iteration):
            np.random.shuffle(np.array(features))
            weightVector=np.array(self.w)
            for i in range(len(features)):
                currentFeatureVector=features[i]

                dotProduct=np.dot(weightVector,currentFeatureVector)
                denominator=np.linalg.norm(weightVector)+sys.float_info.epsilon
                check=dotProduct/denominator

                if(check>((-1)*(self.margin/2))):
                    if(check<(self.margin/2)):
                        correction=labels[i]*(currentFeatureVector/np.linalg.norm(currentFeatureVector))
                        weightVector=np.add(weightVector, correction)
                        mistake=True

                if((labels[i]*dotProduct)<=0):
                    correction=labels[i]*(currentFeatureVector/np.linalg.norm(currentFeatureVector))
                    weightVector=np.add(weightVector, correction)
                    mistake=True
            
            if(mistake==False):
                converged=True
                break

            mistake=False
            iterations=iterations+1
            self.w=weightVector


        if(converged==True):
            self.w=weightVector
            return True
        else:
            self.w=weightVector
            return False
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        predictions=[]
        for i in range(len(features)):
            currentFeatureVector=features[i]
            dotProduct=np.dot(self.w,currentFeatureVector)
            if(dotProduct<0):
                predictions.append(-1)
            else:
                predictions.append(1)
        return predictions

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    