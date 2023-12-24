
import argparse
from autogoal.kb import Supervised
from autogoal.logging import logger
import json
from autogoal.metalearning.tabular_metafeatures import TabularMetafeatureExtractor
# From `sklearn` we will use `train_test_split` to build train and validation sets.

from sklearn.model_selection import train_test_split

# From `autogoal` we need a bunch of datasets.

from autogoal import datasets
from autogoal.datasets import (
    abalone,
    cars,
    dorothea,
    gisette,
    shuttle,
    yeast,
    german_credit,
)
from autogoal.utils import Min, Gb, Hour, Sec
# We will also import this annotation type.

from autogoal.kb import MatrixContinuousDense, VectorCategorical

# This is the real deal, the class `AutoML` does all the work.

from autogoal.ml import AutoML

# And from the `autogoal.search` module we will need a couple logging utilities
# and the `PESearch` class.

from autogoal.search import (
    RichLogger,
    MemoryLogger,
    PESearch,
)



data = abalone.load()

if len(data) == 4:
    X_train, X_test, y_train, y_test = data
else:
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


classifier = AutoML(
            dataset_type= TabularMetafeatureExtractor(),
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            cross_validation_steps=1,
            # measure_time= True,
            evaluation_timeout=5 * Min,
            search_timeout=1 * Hour,
            random_state=42,
)

# logger = MemoryLogger()
loggers = [RichLogger()]

classifier.fit(X_train, y_train, logger=loggers)