from datasets import load_dataset

dataset = load_dataset("nguha/legalbench","abercrombie",split="train")



# X_train = []
# sentences1 = dataset.data[0].to_pylist()
# sentences2 = dataset.data[1].to_pylist()
# for i,x in enumerate(sentences1):
#     if i == 1:
#         break
#     X_train.append((sentences1[i],sentences2[i]))


# Ytrain = dataset.data[3].to_pylist()
X_train = dataset.data[2].to_pylist()
Y_train = dataset.data[0].to_pylist()


from autogoal.utils import Hour, Min
from autogoal.ml import AutoML
from autogoal.search import (
    RichLogger,
    PESearch,
)
from autogoal.datasets import dorothea
from autogoal.ml import AutoML
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import ConsoleLogger, RichLogger
from autogoal.kb import *
from autogoal.metalearning.text_metafeatures import TextMetafeatureExtractor 
from autogoal.search._base import ConsoleLogger
from autogoal.kb import SemanticType
import numpy as np
from autogoal_contrib import find_classes
from autogoal.metalearning.image_metafeatures import ImageMetafeatureExtractor
from typing import Tuple
# inputType = SemanticType.infer(np.array(X_train))
outputType = SemanticType.infer(np.array(Y_train))

classifier = AutoML(
    dataset_type= TextMetafeatureExtractor(),
    search_algorithm= PESearch,
    input= Seq[Word],
    output= Word,
    cross_validation_steps=1,
    evaluation_timeout=5 * Min,
    search_timeout=1 * Hour,
)

classifier.fit(X_train,Y_train, logger=RichLogger())  