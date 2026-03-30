from typing import List, Union

import numpy as np

type Numerable = Union[float, int, "NumWithGrad"]

def to_number_grad(num: Numerable) -> "NumWithGrad":
  if isinstance(num, "NumWithGrad"):
    return num
  return NumWithGrad(num)

class NumWithGrad(object):
  def __init__(self,
    num: Numerable,
    depends_on: List[Numerable] = None,
    creation_op: str = ''):
    self.num = num
    self.grad = None
    self.depends_on = depends_on or []
    self.creation_op = creation_op

  def __add__(self, other: Numerable) -> "NumWithGrad":
    return NumWithGrad(
      self.num + to_number_grad(other).num,
      depends_on = [self, to_number_grad(other)],
      creation_op = 'add'
      )

  def __mul__(self, other: Numerable) -> "NumWithGrad":
    return NumWithGrad(
      self.num * to_number_grad(other).num,
      depends_on=[self, to_number_grad(other)],
      creation_op = 'mul'
    )

  def backward(self, backward_grad: "NumWithGrad" = None) -> None:
    if backward_grad is None:
      # first time apparently
      self.grad = 1
    else:
      # these lines allow gradients to accumulate
      # if the gradient doesn't exist yet.. which is not supposed to be possible..
      # set as equal to backward_grad
      if self.grad is None:
        self.grad = backward_grad
      else:
        self.grad += backward_grad

    if self.creation_op == 'add':
      # wont' this recurse endlessly? # no. depends_on[0] is not self
      # simply send backward self.grad, since increasing either of these
      # elements will increase the output by that same amount
      self.depends_on[0].backward(self.grad)
      self.depends_on[1].backward(self.grad)

    if self.creation_op == 'mul':
      # also endless recursion afaict. # no.  depends_on[0] is not self.

      # calculate the derivative with respect to the first element
      doutdfirst = self.depends_on[1] * self.grad
      # send backward the derivative with respect to that element
      self.depends_on[0].backward(doutdfirst.num)

      # calculate the derivative with respect to the second element
      doutdsecond = self.depends_on[0] * self.grad
      # send backward the derivative with respect to that element
      self.depends_on[1].backward(doutdsecond.num)









