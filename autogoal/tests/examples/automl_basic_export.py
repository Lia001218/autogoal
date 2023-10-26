# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import dorothea
from autogoal.ml import AutoML
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import ConsoleLogger, RichLogger
from autogoal.kb import *
from autogoal.metalearning.tabular_metafeatures import TabularMetafeatureExtractor
from autogoal.search._base import ConsoleLogger



# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()



automl = AutoML(
    # Declare the input and output types
    name= 'dorothea',
    dataset_type= TabularMetafeatureExtractor(),
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,
    # Search space configuration
    search_timeout=5*Min,
    evaluation_timeout= 30 * Sec,
    memory_limit=4*Gb,
    validation_split=0.3,
    cross_validation_steps=2,
    # remote_sources=[("172.19.0.3", 8000, "remote-sklearn")],
)

    # Run the pipeline search process
automl.fit(X_train, y_train)

# import autogoal_remote.production.client as client

# response = client.post_eval(X_test)

# total = len(response)
# ok_count = 0
# for i in range(len(response)):
#     ok_count = ok_count + 1 if y_test[i] == response[i] else ok_count
# print(f"ok: {ok_count} out of {total}")

# automl = AutoML.folder_load()

# print(automl.best_pipelines_)
# print(automl.best_scores_)

# print(automl.score(X_test, y_test))
# print(automl.predict(X_test))
# print(automl.predict_all(X_test))
# # Instantiate AutoML and define input/output types
# automl = AutoML(
#     input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
#     output=VectorCategorical,
#     # remote_sources=["remote-sklearn"],
# )

# # Run the pipeline search process
# automl.fit(X_train, y_train, logger=ConsoleLogger())

# # Report the best pipeline
# print(automl.best_pipelines_)

# print(automl.score(X_test, y_test))

# # Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
# # automl.export_portable()

# automl.export_portable(generate_zip=True)

# Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
automl.export_portable(generate_zip=True)
