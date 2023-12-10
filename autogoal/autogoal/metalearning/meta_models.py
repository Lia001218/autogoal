from autogoal.metalearning.prepare_data import create_db_from_tabular_metafeatures, create_db_from_text_metafeatures, create_db_from_image_metafeatures
from autogoal.ml import AutoML
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.kb import *
from sklearn.model_selection import train_test_split
from autogoal.metalearning.tabular_metafeatures import TabularMetafeatureExtractor
from autogoal.metalearning.text_metafeatures import TextMetafeatureExtractor
from autogoal.metalearning.image_metafeatures import ImageMetafeatureExtractor

# Load tabular dataset
tabular_data = create_db_from_tabular_metafeatures()
X_train, y_train, X_test, y_test = train_test_split(tabular_data[:-2], tabular_data[-1:])

automltabular = AutoML(
    # Declare the input and output types
    dataset_type= TabularMetafeatureExtractor(),
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    # Search space configuration
    search_timeout=5*Min,
    evaluation_timeout= 30 * Sec,
    memory_limit=4*Gb,
    validation_split=0.3,
    cross_validation_steps=2,
    # remote_sources=[("
)

automltabular.fit(X_train, y_train)

# Load text dataset
text_data = create_db_from_text_metafeatures()
X_train, y_train, X_test, y_test = train_test_split(text_data[:-2], text_data[-1:])

automltext = AutoML(
    # Declare the input and output types
    dataset_type= TextMetafeatureExtractor(),
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,
    # Search space configuration
    search_timeout=5*Min,
    evaluation_timeout= 30 * Sec,
    memory_limit=4*Gb,
    validation_split=0.3,
    cross_validation_steps=2,
    # remote_sources=[("
)

automltext.fit(X_train, y_train)

# Load image dataset
image_data = create_db_from_image_metafeatures()
X_train, y_train, X_test, y_test = train_test_split(image_data[:-2], image_data[-1:])
automlimage = AutoML(
    # Declare the input and output types
    dataset_type= ImageMetafeatureExtractor(),
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,
    # Search space configuration
    search_timeout=5*Min,
    evaluation_timeout= 30 * Sec,
    memory_limit=4*Gb,
    validation_split=0.3,
    cross_validation_steps=2,
    # remote_sources=[("
)

automlimage.fit(X_train, y_train)