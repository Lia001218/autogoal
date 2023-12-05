
import openml
import numpy as np
benchmark_suite = openml.study.get_suite(293) # obtain the benchmark suite
from scipy import sparse as sp
i = 85

a = []
tasks = []
stop = 100

for j in range(i, len(benchmark_suite.tasks), 1):  # iterate over all tasks
    task = openml.tasks.get_task(benchmark_suite.tasks[j])  # download the OpenML task
    tasks.append(task)
    if isinstance(task, openml.OpenMLSupervisedTask):
        features, targets = task.get_X_and_y()  # get the data
    else: 
        features, targets = task.get_X(), None
    a.append((features, targets))
   

    if j == stop:
        break



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
from autogoal.metalearning.tabular_metafeatures import TabularMetafeatureExtractor
from autogoal.search._base import ConsoleLogger
from autogoal.kb import SemanticType

from autogoal_contrib import find_classes
from autogoal.metalearning.image_metafeatures import ImageMetafeatureExtractor

failures = 0

for i in range(len(a)):

    inputType = SemanticType.infer(a[i][0])
    outputType = SemanticType.infer(a[i][1])
    classifier = AutoML(
        dataset_type= TabularMetafeatureExtractor(),
        search_algorithm= PESearch,
        input= inputType,
        output= outputType,
        cross_validation_steps=1,
        # Since we only want to try neural networks, we restrict
        # the contrib registry to algorithms matching with `Keras`.
        # errors="raise",
        # Since image classifiers are heavy to train, let's give them a longer timeout...
        evaluation_timeout=5 * Min,
        search_timeout=1 * Hour,
    )
    try:
        classifier.fit(np.nan_to_num(a[i][0]),a[i][1], logger=RichLogger())   
    except:
        failures+=1
       
print(failures , "count fail pipelines")
    