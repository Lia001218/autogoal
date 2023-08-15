from typing import Optional
from odmantic import Field, Model, EmbeddedModel

class PipelineModel(EmbeddedModel):
    algorithm_flow: str
    eval_result: Optional[float] = None
    error_result : Optional[str] = None