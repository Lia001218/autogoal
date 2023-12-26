from typing import Optional
from odmantic import Field, Model
from .pipeline_model import PipelineModel
import numpy as np
from sentence_transformers import SentenceTransformer

class MetafeatureModel(Model):

    metacaracteristic_model: Optional[list] = None
    metric: str
    input_type: str
    output_type: str
    pipelines : Optional[list[PipelineModel]] = None 
    dataset_type: str
    
 

def transform_metafeatures(metafeature_instance: MetafeatureModel, solution, model):
    X = np.array([])
    X = np.append(X, metafeature_instance.metacaracteristic_model)
    X = np.append(X, model.encode(solution))

    X = X.reshape((1,X.shape[0]))
    return X
   