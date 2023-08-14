from typing import Optional
from odmantic import Field, Model

class MetafeatureModel(Model):
    dataset_name : str
    metacaracteristic_model: Optional[list[float]] = None
    metric: str
    input_type: str
    output_type: str

