from autogoal.metalearning.metafeatures_extractor import MetafeatureExtractor
import numpy as np
from distfit import distfit
from lexical_diversity import lex_div as ld
from nltk.corpus import stopwords


class TextMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []


    def extract_features(self, X, y=None):
        # print(type(X), ' que tu eres mi alma')
        # if isinstance(X, list):
        #     print(X[1],'alal')
        #     # print(y[0:10],'vamos')
        #     X = np.array(X)

        self.__issupervised__(y)
        # cardinality = X.shape[0]
        cardinality = len(X)
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
        list_lexical_diversity = np.array([ld.ttr(ld.tokenize(self._concatenate_words(x))) for x in X])
        self.features.append(list_lexical_diversity.mean())
        self.features.append(list_lexical_diversity.min())
        self.features.append(list_lexical_diversity.max())
        self.features.append(list_lexical_diversity.std())

    def __len_distribution__(self, X):
        try:
            doc_len = np.array([len(x) for x in X])
            dift = distfit()
            sumary = dift.fit_transform(doc_len)
            self.features.append(sumary['model']['name'])
        except:
            self.features.append('None')

    def __average_len__(self,X):
        try:
            len_doc = np.array([len(x) for x in X])
            self.features.append(len_doc.mean())
        except :
            self.features.append(-1)

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

    def _concatenate_words(self, words: list[str]):
        return " ".join(words)