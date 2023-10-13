from scipy.stats import skew, kurtosis
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from distfit import distfit
from lexical_diversity import lex_div as ld
from nltk.corpus import stopwords


class MetafeatureExtractor:
    def __init__(self):
        self.features = []
    

    def extract_features(self, X, y=None):
        pass

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
        self.features.append(skewness.min())
        self.features.append(skewness.max())
        self.features.append(skewness.std())
        self.features.append(skewness.mean())

        kurtosis_val = np.array([kurtosis(row) for row in X.T])
        
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
        print('finish')
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
 




class ImageMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []


    def extract_features(self, X, y=None):
        self.__issupervised__(y)
        self.features.append(X.shape[0])
        self.__output_cardinality__(y)
        self.__average_size__(X)
        self.__type_image__(X)
        return self.features

    def __type_image__(self,X):
        twoD = False
        threeD = False
        for x in X : 
            if len(x) == 2:
                twoD = True
            elif len(x) == 3:
                threeD = True
            if twoD and threeD:
                self.features.append(2)
                break
        if twoD:
            self.features.append(0)
        if threeD:
            self.features.append(1)


    def __average_size__(self,X):
        total_height = 0
        total_width = 0
        height_width =[]
        for x in X:
            height, width, _ = x.shape
            height_width.append(height/width)
            total_height += height
            total_width += width
        height_width = np.array(height_width)
        self.features.append(height_width.mean())
        self.features.append(height_width.min())
        self.features.append(height_width.max())
        self.features.append(height_width.std())
        self.features.append(total_height/len(X))
        self.features.append(total_width/len(X))


    def __output_cardinality__(self,y):
        if y is None: 
            self.features.append(-1)
        else : 
            self.features.append(len(y))
    
    def __issupervised__(self,y):
        if y is None: # is supervised
            self.features.append(1)
        else : 
            self.features.append(0)

class TextMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []


    def extract_features(self, X, y=None):
        # print(type(X), ' que tu eres mi alma')
        if isinstance(X, list):
            X = np.array(X)

        self.__issupervised__(y)
        cardinality = X.shape[0]
        self.features.append(cardinality)
        output_cardinality = self.__output_cardinality__(y)

        self.features.append(cardinality/output_cardinality)
        self.__average_len__(X)
        self.__len_distribution__(X)
        self.__lexical_diversity__(X)
        # self.__stop_words_proportion__(X)
        return self.features
    def __average_len_words__(self,X):
        average = []
        for x in X.shape[0]:
            large = 0
            for words in x:
                large += len(words)
            average.append(large/len(x))
        average = np.array(average)
        self.features.append(average.mean())
        self.features.append(average.min())
        self.features.append(average.max())
        self.features.append(average.std())

    def __stop_words_proportion__(self,X):
        stopwords_proportion = []
        for x in X[0]:
            count =0
            for words in ld.tokenize(x):
                if words in stopwords.words():
                    count +=1
            stopwords_proportion.append(count/len(x))
        stopwords_proportion = np.array(stopwords_proportion)
        self.features.append(stopwords_proportion.mean())
        self.features.append(stopwords_proportion.min())
        self.features.append(stopwords_proportion.max())
        self.features.append(stopwords_proportion.std())

    def __lexical_diversity__(self,X):
        list_lexical_diversity = np.array([ld.ttr(ld.tokenize(x)) for x in X[0]])
        self.features.append(list_lexical_diversity.mean())
        self.features.append(list_lexical_diversity.min())
        self.features.append(list_lexical_diversity.max())
        self.features.append(list_lexical_diversity.std())

    def __len_distribution__(self, X):
        doc_len = np.array([len(x) for x in X[0]])
        dift = distfit()
        sumary = dift.fit_transform(doc_len)
        self.features.append(sumary['model']['name'])

    def __average_len__(self,X):
        len_doc = np.array([len(x) for x in X[0] ])
        self.features.append(len_doc.mean())

    def __output_cardinality__(self,y):
        if y is None: 
            self.features.append(-1)
            return -1
        else : 
            self.features.append(len(y))
            return len(y)
    
    def __issupervised__(self,y):
        if y is None: # is supervised
            self.features.append(1)
        else : 
            self.features.append(0)
