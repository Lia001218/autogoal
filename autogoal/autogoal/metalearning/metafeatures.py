


class MetafeatureExtractor:
    def __init__(self):
        self.features = []
    

    def extract_features(self, X, y=None):
        pass

class TabularMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []

    def extract_features(self, X, y=None):
        pass
        


class ImageMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self, X, y=None):
        self.features = []


    def extract_features(self, X, y=None):
        pass


class TextMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self, X, y=None):
        self.features = []


    def extract_features(self, X, y=None):
        pass