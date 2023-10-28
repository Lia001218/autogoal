from autogoal.metalearning.metafeatures_extractor import MetafeatureExtractor
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from scipy.stats import mode

class ImageMetafeatureExtractor(MetafeatureExtractor):
    def __init__(self):
        self.features = []


    def extract_features(self, X, y=None):
        self.__issupervised__(y)
        self.features.append(X.shape[0])
        self.__output_cardinality__(y)
        self.__average_size__(X)
        self.__type_image__(X)
        self.__rgb_value__(X)
        self.__predominate_warm_or_cool_colors__(X)


        return [float(i) for i in self.features]
    
    def __average_intensity__(self,X):
        if len(X.shape) == 3:
            intensity = []
            for x in X:
                grayscale = np.dot(x[2])
                intensity.append(grayscale.mean())
            intensity = np.array(intensity)
            self.features.append(intensity.mean())
            self.features.append(intensity.min())
            self.features.append(intensity.max())
            self.features.append(intensity.std())
        else:
            self.features.append(-1)
            self.features.append(-1)
            self.features.append(-1)
            self.features.append(-1)


    # def __analysis_object__(self,X):
    #     # print(os.listdir())
    #     # model = YOLO('/home/coder/autogoal/yolo/yolov8n-oiv7.pt')
    #     # print(len(model(X[0])), 'tyutu')
    #     model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    #     model.eval()
        
    #     print()
    #     print(torch.from_numpy(X[0]))
    #     model_result = model(torch.from_numpy(X))
    #     count_object = [len([i[0]['boxes']- i[0]['score']]) for i in model_result]
    #     print('tu que ')
    #     count_object = np.array(count_object)
    #     self.features.append(count_object.mean())
    #     self.features.append(count_object.min())
    #     self.features.append(count_object.max())
    #     self.features.append(count_object.std())  

          
    def __predominate_warm_or_cool_colors__(self,X):
        
        rgb_matrix = np.array([x[2] for x in X if x.shape[2] == 3])
        if len(rgb_matrix) > 0:
            hsv_matrix = [matplotlib.colors.rgb_to_hsv(rgb) for rgb in rgb_matrix]
            predominat_hue = np.array([mode(x)[0] for x in hsv_matrix])
            self.features.append(predominat_hue.mean())
            self.features.append(predominat_hue.min())
            self.features.append(predominat_hue.max())
            self.features.append(predominat_hue.std())
        else:
            self.features.append(-1)
            self.features.append(-1)
            self.features.append(-1)
            self.features.append(-1)

    def __rgb_value__(self,X):
        rgb = []
        for x in X:
            rgb.append(x[2])
        rgb = np.array(rgb)
        self.features.append(rgb.mean())
        self.features.append(rgb.min())
        self.features.append(rgb.max())
        self.features.append(rgb.std())



    def __type_image__(self,X):
        twoD = False
        threeD = False
        for x in X : 
            if len(x.shape) == 2:
                twoD = True
            elif len(x.shape) == 3:
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
