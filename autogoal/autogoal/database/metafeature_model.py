from typing import Optional
from odmantic import Field, Model
from .pipeline_model import PipelineModel

class MetafeatureModel(Model):
    dataset_name : str
    metacaracteristic_model: Optional[list] = None
    metric: str
    input_type: str
    output_type: str
    pipelines : Optional[list[PipelineModel]] = None 

