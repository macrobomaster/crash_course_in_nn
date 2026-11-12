import math
from collections.abc import Callable, Iterator
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import Context


def _as_list(value: object) -> object:
  return value.tolist() if isinstance(value, Tensor) else value


def _shape(value: object) -> tuple[int, ...]:
  if not isinstance(value, (list, tuple)): return ()
  if not value: return (0,)
  child_shape = _shape(value[0])
  if any(_shape(child) != child_shape for child in value):
    raise ValueError("Expected a rectangular tensor or nested list")
  return (len(value),) + child_shape


def _flatten(value: object) -> Iterator[object]:
  if isinstance(value, (list, tuple)):
    for child in value: yield from _flatten(child)
  else: yield value


def format_array(x: object, precision: int = 4) -> str:
  """Format a tensor or nested Python sequence using standard Python types."""
  def format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
      return "[" + ", ".join(format_value(child) for child in value) + "]"
    if isinstance(value, (bool, int)): return str(value)
    value = float(value)
    if abs(value) < 0.5 * 10 ** (-precision): value = 0.0
    if precision == 0: return f"{value:.0f}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")
  return format_value(_as_list(x))


def check_close(name: str, actual: object, expected: object, atol: float = 1e-5, rtol: float = 1e-5) -> None:
  actual_values, expected_values = _as_list(actual), _as_list(expected)
  actual_shape, expected_shape = _shape(actual_values), _shape(expected_values)
  if actual_shape != expected_shape:
    raise AssertionError(f"{name}: shape {actual_shape} != expected {expected_shape}")
  pairs = zip(_flatten(actual_values), _flatten(expected_values))
  if not all(math.isclose(float(a), float(e), abs_tol=atol, rel_tol=rtol) for a, e in pairs):
    raise AssertionError(f"{name}:\nactual {format_array(actual_values)}\nexpected {format_array(expected_values)}")
  print(f"PASS: {name}")


def count_parameters(model: object) -> int:
  return sum(p.numel() for p in get_parameters(model))


def binary_accuracy(model: Callable[[Tensor], Tensor], x: Tensor, y: Tensor) -> float:
  prediction = [value >= 0.5 for value in _flatten(model(x).sigmoid().tolist())]
  truth = [bool(value) for value in _flatten(y.tolist())]
  return float(sum(p == t for p, t in zip(prediction, truth)) / len(truth))


def fit_binary_classifier(model: Callable[[Tensor], Tensor], train_x: Tensor, train_y: Tensor, steps: int = 200, lr: float = 0.03, log_every: int = 0) -> list[float]:
  """A deliberately visible training loop for the small classifiers in this course."""
  optimizer = nn.optim.AdamW([p for p in get_parameters(model) if p.is_param], lr=lr, weight_decay=0.0)
  losses = []
  with Context(TRAINING=1):
    for step in range(steps):
      optimizer.zero_grad()
      logits = model(train_x)
      loss = logits.binary_crossentropy_logits(train_y)
      loss.backward()
      optimizer.step()
      value = float(loss.item())
      losses.append(value)
      if log_every and (step == 0 or (step + 1) % log_every == 0):
        print(f"  step {step + 1:4d}/{steps}: loss={value:.5f}")
  return losses
