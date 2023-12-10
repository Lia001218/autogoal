from autogoal.kb import *
from autogoal.ml import AutoML, calinski_harabasz_score
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import PESearch, JsonLogger, ConsoleLogger
from autogoal_sklearn import AffinityPropagation, Birch, KMeans
from autogoal.metalearning.tabular_metafeatures import TabularMetafeatureExtractor
from autogoal.search import ConsoleLogger, RichLogger
from autogoal.search import (
    RichLogger,
    PESearch,
)
from odmantic import SyncEngine
from autogoal.database.metafeature_model import MetafeatureModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faulthandler

faulthandler.enable()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_db_from_tabular_metafeatures():
    db = SyncEngine(database= 'Metalearning')
    tabular_features_db = []
    X = []
    Y = []
    tabular_modality = db.find(MetafeatureModel, MetafeatureModel.dataset_type == 'TabularMetafeatureExtractor'
    ,MetafeatureModel.pipelines != [])
    for i in tabular_modality:
        if i.pipelines:
            for j in i.pipelines:
                if j.eval_result:
                    current= [i.metacaracteristic_model,i.metric,j.algorithm_flow,j.eval_result]
                    tabular_features_db.append(current)
                    temp = np.array(current[0])
                    if current[1][11:19] == "accuracy" :
                      
                        np.append(temp,1)
                    else:
                        np.append(temp,0)
                    np.append(temp,model.encode(j.algorithm_flow))
                    
                    X.append(temp)

                    Y.append(j.eval_result[0])           
    return  X, Y


from sklearn.model_selection import train_test_split

X, Y = create_db_from_tabular_metafeatures()


X_train, X_test,Y_train, Y_test = train_test_split(X, Y)
output = SemanticType.infer(np.array(Y_train))
input = SemanticType.infer(np.array(X_train))
automltabular = AutoML(
    # Declare the input and output types
   
    input= MatrixContinuousDense,
    output= Supervised[VectorContinuous],
    search_algorithm= PESearch,
    # Search space configuration
    cross_validation_steps=1,
    evaluation_timeout=5 * Min,
    search_timeout=1 * Hour,
    # remote_sources=[("
)
automltabular.fit(X_train, Y_train, logger=RichLogger())