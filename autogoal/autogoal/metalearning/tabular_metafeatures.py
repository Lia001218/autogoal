from autogoal.metalearning.metafeatures_extractor import MetafeatureExtractor
from scipy import stats
from scipy.sparse import csr_matrix
import cv2
import numpy as np
from scipy.stats import skew, kurtosis


class TabularMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []

    def extract_features(self, X, y=None):
        self.__issupervised__(y)
       
        # print('x: ', X)
        if isinstance(X, csr_matrix):
            X = X.toarray()

        self.features.append(X.shape[0]) #cardinality
        self.features.append(X.shape[1]) #attributes
        self.features.append(X.shape[0]/X.shape[1]) # proportion cardinality/attributes
        missing_values = np.argwhere(np.isnan(X))
        numeric_values = len(X.shape) - len(missing_values)
        self.features.append(numeric_values) #number of numeric attrubutes
        self.features.append(len(missing_values))
        self.features.append(np.std(X))
        self.features.append(X.std()/X.shape[1])

        skewness = np.array([skew(row) for row in X.T])

        skewness = np.nan_to_num(skewness,nan= -1)
        self.features.append(skewness.min())
        self.features.append(skewness.max())
        self.features.append(skewness.std())
        self.features.append(skewness.mean())

        kurtosis_val = np.array([kurtosis(row) for row in X.T])
        kurtosis_val = np.nan_to_num(kurtosis_val,nan= -1)
        self.features.append(kurtosis_val.min())
        self.features.append(kurtosis_val.max())
        self.features.append(kurtosis_val.std())
        self.features.append(kurtosis_val.mean())
        standardization_factor = self.__normalized_class_entropy__(X,y)
        attr_entropy = self.__normalized_attr_entropy__(X)
        self.__joint_entropy__(X,y)
        cummon_info = self.__mutual_information__(X,y,standardization_factor)
        self.__equivalent_number_of_attr__(standardization_factor,cummon_info)
        self.__noise_signal_ratio__(cummon_info,attr_entropy)
        return self.features

    def __issupervised__(self,y):
        if y is None: # is supervised
            self.features.append(1)
        else : 
            self.features.append(0)

    def __normalized_class_entropy__(self,X, y):
        if not y is None:
            _, counts = np.unique(y, return_counts=True)
            self.features.append(stats.entropy(counts) / np.log2(len(counts)))
            return stats.entropy(counts) / np.log2(len(counts))
        else:
            self.features.append(-1)
            return -1

    def __attr_i_entropy__(self,x_i):
        _, counts = np.unique(x_i, return_counts=True)
        return stats.entropy(counts) / np.log2(len(counts))
    
    def __normalized_attr_entropy__(self,X):
        attributes = [X[:, j] for j in range(X.shape[1])]
        attr_entropy = np.array([self.__attr_i_entropy__(xi) for xi in attributes])
        attr_entropy = np.nan_to_num(attr_entropy,nan= -1)
        self.features.append(attr_entropy.mean())
        self.features.append(attr_entropy.max())
        self.features.append(attr_entropy.std())
        self.features.append(attr_entropy.min())
        return attr_entropy.mean()
        
    def __equivalent_number_of_attr__(self,standardization_factor,cummon_info):
        self.features.append(standardization_factor/cummon_info)
            
    def __mutual_information__(self,X, y,standardization_factor):
    
        result = []
        
        if standardization_factor == -1:
            self.features.append(-1)
            self.features.append(-1)
            self.features.append(-1)
            return - 1
        else:
            for j in range(X.shape[1]):
                attr_entropy = self.__attr_i_entropy__(X[:, j])
                joint_entropy = self.__attr_i_joint_entropy__(X[:, j], y)
                result.append(standardization_factor + attr_entropy - joint_entropy)
        result = np.array(result)
        result = np.nan_to_num(result, nan= -1)
        self.features.append(result.mean())
        self.features.append(result.max())
        self.features.append(result.std())
        self.features.append(result.mean())
        return sum(result) / len(result)
    

    def __joint_entropy__(self,X, y):
        if y is None:
            return -1
        attributes_labels = [(X[:, j], y) for j in range(X.shape[1])]
        joint_entropy = np.array([self.__attr_i_joint_entropy__(attr_i, y) for attr_i, y in attributes_labels])
        self.features.append(joint_entropy.mean())
        self.features.append(joint_entropy.max())
        self.features.append(joint_entropy.std())
        self.features.append(joint_entropy.min())


    def __attr_i_joint_entropy__(self,x_i, y):
        _, counts = np.unique(list(zip(x_i, y)), return_counts=True)
        return stats.entropy(counts)

    def __numeric_values__(self,X):
        count = 0
        for _ in X[0]:
            for i in X[1]:
                if isinstance(i, (float, int)):
                    count +=1
        return count
    
    def __missing_values__(self, X):
        total = 0
        for i in X:
            if i is np.isnan:
                total +=1
        self.features.append(total)
        return total
    
    def __noise_signal_ratio__(self,cummon_info,attr_entropy ):
        self.features.append(attr_entropy /cummon_info)
 