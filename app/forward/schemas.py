from typing import List
from pydantic import BaseModel, Field


class ForwardRequest(BaseModel):
    x: List[float] = Field(..., description="Координата x")
    y: List[float] = Field(..., description="Координата y")
    t: List[float] = Field(..., description="Время t")

    omega_x: float
    omega_y: float
    g_param: float
