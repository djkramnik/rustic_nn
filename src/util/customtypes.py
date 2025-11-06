from numpy import ndarray
from typing import Callable, List, Dict, Tuple

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]
Batch = Tuple[ndarray, ndarray]
DictOfArrays = Dict[str, ndarray]
