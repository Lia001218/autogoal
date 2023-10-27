from autogoal.ml import AutoML
from autogoal.datasets import movie_reviews
from autogoal.search import (
    PESearch,
    RichLogger,
)
from autogoal.kb import Seq, Sentence, VectorCategorical, Supervised
from autogoal_contrib import find_classes
from sklearn.metrics import f1_score
from autogoal.metalearning.text_metafeatures import TextMetafeatureExtractor
from sklearn.model_selection import train_test_split

test = AutoML(
    dataset_type= TextMetafeatureExtractor(),
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    output=VectorCategorical
)

X, y = movie_reviews.load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
loggers = [RichLogger()]
test.fit(X_train, y_train, logger=loggers)