from typing import List, Union

import numpy as np
ndarray = np.ndarray

type TensorLike = Union[ndarray, "Tensor"]

def to_tensor(t: TensorLike) -> "Tensor":
  if isinstance(t, Tensor):
    return t
  return Tensor(t)

class Tensor(object):
  def __init__(self,
    arr: ndarray,
    depends_on: List["Tensor"] = None,
    creation_op: str = '',
    ):
    self.arr = arr
    self.grad = None
    self.depends_on = depends_on or []
    self.creation_op = creation_op
  def __add__(self, other: TensorLike):
    ot = to_tensor(other)
    return Tensor(
      self.arr + ot.arr,
      depends_on = [self, ot],
      creation_op = 'add'
    )
  def __mul__(self, other: TensorLike):
    ot = to_tensor(other)
    return Tensor(
      self.arr * ot.arr,
      depends_on = [self, ot],
      creation_op = 'mul'
    )
  def backward(self, backward_grad: "Tensor" = None):
    if backward_grad is None:
      # first time.. or reset I presume
      self.grad = Tensor(np.ones(self.arr.shape))
    else:
      if self.grad is None:
        self.grad = backward_grad
      else:
        self.grad += backward_grad

    if self.creation_op == 'add':
      self.depends_on[0].backward(self.grad)
      self.depends_on[1].backward(self.grad)

    if self.creation_op == 'mul':
      [left, right] = self.depends_on
      left.backward(self.grad * right)
      right.backward(self.grad * left)

