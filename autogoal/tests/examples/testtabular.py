
import openml
import numpy as np
benchmark_suite = openml.study.get_suite(293) # obtain the benchmark suite
from scipy import sparse as sp
i = 1

a = []
tasks = []

for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    tasks.append(task)
    features, targets = task.get_X_and_y()  # get the data
    a.append((features, targets))
    i+=1

    if i == 12:
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

for i in a:

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
    classifier.fit(np.nan_to_num(a[i][0]),a[i][1], logger=RichLogger())   