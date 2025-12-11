from util.customtypes import *
from util.assert_shape import assert_shape
from operation.operation import Operation

# issubclass _should_ work and is the better way?
def has_ndarray(op: Operation, attr: str) -> bool:
  return str(type(getattr(op, attr, None))) == "<class 'numpy.ndarray'>"

class Layer(object):
  def __init__(self, neurons:int):
    self.neurons = neurons
    self.first = True
    self.params: List[ndarray] = []
    self.param_grads: List[ndarray] = []
    self.operations: List[Operation] = []
  # self requires input.. which is why setup layer is called from forward
  # I assume it needs it to initialize random weights of the corresponding shape

  def _setup_layer(self, input_: ndarray):
    raise NotImplementedError()

  def forward(self, input_: ndarray) -> ndarray:
    if (self.first):
      self._setup_layer(input_)
      self.first = False

    # unsure whether this gets used anywhere but we cache
    # the last input passed to forward
    self.input_ = input_
    for operation in self.operations:
      input_ = operation.forward(input_)
    self.output = input_
    return self.output

  def backward(self, output_grad: ndarray) -> ndarray:
    assert_shape(self.output, output_grad)
    for operation in reversed(self.operations):
      output_grad = operation.backward(output_grad)

    self._param_grads()
    return output_grad

  def _param_grads(self):
    self.param_grads = []
    for op in self.operations:
      if (has_ndarray(op, 'param_grad')):
        self.param_grads.append(op.param_grad)

  def _params(self):
    self.params = []
    for op in self.operations:
      if (has_ndarray(op, 'param')):
        self.params.append(op.param)
