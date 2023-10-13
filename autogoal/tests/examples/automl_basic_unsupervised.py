# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars, dorothea
from autogoal.kb import MatrixContinuousSparse, Supervised, VectorCategorical
from autogoal.ml import AutoML, calinski_harabasz_score, silhouette_score, accuracy
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import PESearch, JsonLogger, ConsoleLogger
from autogoal_sklearn import AffinityPropagation, Birch, KMeans
from autogoal.metalearning.metafeatures import TabularMetafeatureExtractor,TextMetafeatureExtractor,ImageMetafeatureExtractor
# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()



automl = AutoML(
    # Declare the input and output types
    name = 'dorothea_unsupervised',
    dataset_type= TabularMetafeatureExtractor(),
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,

    # Search space configuration
    search_timeout=1*Hour,
    evaluation_timeout= 30 * Sec,
    memory_limit=4*Gb,
    validation_split=0.3,
    cross_validation_steps=2,
)

    # Run the pipeline search process
automl.fit(X_train, y_train)

    # Report the best pipelines
print(automl.best_pipelines_)
# print(automl.score(X_test, y_test))
 