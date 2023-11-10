
from autogoal.utils import Hour, Min
from autogoal.ml import AutoML
from autogoal.search import (
    RichLogger,
    PESearch,
)

from autogoal.kb import *

from autogoal_contrib import find_classes
from autogoal.metalearning.image_metafeatures import ImageMetafeatureExtractor

classifier = AutoML(
    dataset_type= ImageMetafeatureExtractor(),
    search_algorithm= PESearch,
    input=(Tensor4, Supervised[VectorCategorical]),
    output=VectorCategorical,
    cross_validation_steps=1,
    # Since we only want to try neural networks, we restrict
    # the contrib registry to algorithms matching with `Keras`.
    registry=find_classes("Keras"),
    # errors="raise",
    # Since image classifiers are heavy to train, let's give them a longer timeout...
    evaluation_timeout=5 * Min,
    search_timeout=1 * Hour,
)

import openml

    # download dataset with DATASET_ID. Check Dataset detail page for DATASET_ID
dataset = openml.datasets.get_dataset("44285", download_data=True, download_all_files=True)

    # display dataset info

x, y , _, _ = dataset.get_data(target=dataset.default_target_attribute)
ids = {}
count = 0
vectorized_y = []
for i in range(y.shape[0]):
    if y[i] not in ids:
        ids[y[i]] = count
        count += 1
    vectorized_y.append(ids[y[i]])

import os 
dataset.data_file

def build_path():
    data_file_path = ''
    data_name_path = ''
    for i  in range(len(dataset.data_file) -1,0,-1):
        if dataset.data_file[i] == '/':
            data_file_path = dataset.data_file[:i+1]
            break

    count = 0
    for i in range(len(dataset.name)):
        if count == 2:
          data_name_path = dataset.name[i:]
          break
        if dataset.name[i] == '_':
            count += 1

    return data_file_path + data_name_path + '/images/'


dataset_id = 44285
dataset_name = 'BRD_Mini'
ruta = build_path()

import numpy as np
from PIL import Image
from numpy import asarray

X = []
for i in os.listdir(ruta):
    temp_path =os.path.join(ruta, i)
    X.append(asarray(Image.open(temp_path)))
X = np.array(X)

y = np.array(vectorized_y)
classifier.fit(X, y, logger=RichLogger())   

