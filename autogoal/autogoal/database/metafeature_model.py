from typing import Optional
from odmantic import Field, Model
from .pipeline_model import PipelineModel

class MetafeatureModel(Model):

    metacaracteristic_model: Optional[list] = None
    metric: str
    input_type: str
    output_type: str
    pipelines : Optional[list[PipelineModel]] = None 
    dataset_type: str
    memory_limit: Optional[float] = None
    evaluation_timeout: Optional[float] = None

