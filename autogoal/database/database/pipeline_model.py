from typing import Optional
from odmantic import Field, Model

class PipelineModel(Model):
    algorithm_flow: str
    eval_result: Optional[float] = None
    error_result : Optional[str] = None