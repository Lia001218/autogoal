# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search._base import ConsoleLogger

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    name='l',
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
)

# Run the pipeline search process
automl.fit(X, y, logger=ConsoleLogger())

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)
