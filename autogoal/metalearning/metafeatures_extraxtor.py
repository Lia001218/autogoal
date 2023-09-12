


class MetafeatureExtractor:
    def __init__(self,datatype, X, y=None):
        self.X = X
        self.y = y
        self.datatype = datatype
    

    def extract_features(self):
        pass

class TabularMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self,datatype, X, y=None):
        super().__init__(datatype, X, y)
        self.features = []

    def extract_features(self):
        pass


class ImageMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self,datatype, X, y=None):
        super().__init__(datatype, X, y)
        self.features = []


    def extract_features(self):
        pass


class TextMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self,datatype, X, y=None):
        super().__init__(datatype, X, y)
        self.features = []


    def extract_features(self):
        pass