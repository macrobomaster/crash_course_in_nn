import math
import random
from tinygrad import Tensor, nn
from common import binary_accuracy, count_parameters, fit_binary_classifier

def make_rings(count: int = 256, seed: int = 7) -> tuple[list[list[float]], list[list[float]]]:
  """Create (points, binary_labels); no single straight boundary separates the classes."""
  rng = random.Random(seed)
  x, y = [], []
  for i in range(count):
    label = i % 2
    angle = rng.uniform(0, 2 * math.pi)
    radius = (0.65 if label == 0 else 1.8) + rng.gauss(0, 0.09)
    x.append([
      math.cos(angle) * radius + rng.gauss(0, 0.04),
      math.sin(angle) * radius + rng.gauss(0, 0.04),
    ])
    y.append([float(label)])
  return x, y

class LinearModel:
  def __init__(self) -> None:
    self.linear = nn.Linear(2, 1)

  def __call__(self, x: Tensor) -> Tensor:
    return self.linear(x)

class MLP:
  def __init__(self, hidden: int = 16) -> None:
    self.hidden = nn.Linear(2, hidden)
    self.output = nn.Linear(hidden, 1)

  def activation(self, x: Tensor) -> Tensor:
    # TODO: add a nonlinearity; a stack of linear maps is only linear.
    return x

  def __call__(self, x: Tensor) -> Tensor:
    return self.output(self.activation(self.hidden(x)))

def train_and_evaluate(
  name: str, model: LinearModel | MLP, train_x: Tensor, train_y: Tensor,
  test_x: Tensor, test_y: Tensor,
) -> float:
  print(f"\n{name} ({count_parameters(model):,} trainable scalars)")
  losses = fit_binary_classifier(model, train_x, train_y, steps=200, lr=0.05, log_every=100)
  train_accuracy = binary_accuracy(model, train_x, train_y)
  test_accuracy = binary_accuracy(model, test_x, test_y)
  print(f"  first loss={losses[0]:.4f}, final loss={losses[-1]:.4f}")
  print(f"  train accuracy={train_accuracy:.1%}, test accuracy={test_accuracy:.1%}")
  return test_accuracy

def main() -> None:
  x_values, y_values = make_rings()
  train_x, train_y = Tensor(x_values).detach(), Tensor(y_values).detach()
  x_test, y_test = make_rings(256, seed=42)
  test_x, test_y = Tensor(x_test).detach(), Tensor(y_test).detach()
  linear_accuracy = train_and_evaluate("Linear", LinearModel(), train_x, train_y, test_x, test_y)
  mlp_accuracy = train_and_evaluate("MLP", MLP(), train_x, train_y, test_x, test_y)
  assert linear_accuracy < 0.85
  assert mlp_accuracy > 0.95

if __name__ == "__main__":
  main()
