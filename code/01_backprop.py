from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from common import check_close

def analytical_gradients(weight: float, bias: float, x: list[float], y: list[float]) -> tuple[float, float]:
  """Return (dL/dweight, dL/dbias) for y_hat = weight*x + bias and MSE."""
  # TODO: differentiate (prediction - y)^2 and average over samples.
  raise NotImplementedError("Use the chain rule: average each derivative with sum(...) / len(x)")

def autograd_gradients(weight: float, bias: float, x: list[float], y: list[float]) -> tuple[float, float, float]:
  """Return (loss_value, dL/dweight, dL/dbias)."""
  # Keep trainable values as explicit tensors, even when they contain just one number.
  weight_t, bias_t = Tensor([float(weight)]), Tensor([float(bias)])
  x_t, y_t = Tensor(x).detach(), Tensor(y).detach()
  prediction = weight_t * x_t + bias_t
  loss = ((prediction - y_t) ** 2).mean()
  loss.backward()
  return float(loss.item()), float(weight_t.grad.item()), float(bias_t.grad.item())

def fit_line() -> None:
  x_values = [-1.0 + 2.0 * i / 63 for i in range(64)]
  y_values = [3.0 * value - 2.0 for value in x_values]
  x, y = Tensor(x_values).detach(), Tensor(y_values).detach()
  weight, bias = Tensor([0.0]), Tensor([0.0])
  optimizer = nn.optim.SGD([weight, bias], lr=0.2)
  with Context(TRAINING=1):
    for _ in range(100):
      optimizer.zero_grad()
      prediction = weight * x + bias
      loss = ((prediction - y) ** 2).mean()
      loss.backward()
      optimizer.step()
  print(f"100 steps: y_hat = {weight.item():.4f} * x + {bias.item():.4f}, loss = {loss.item():.6f}")
  assert abs(weight.item() - 3.0) < 0.02
  assert abs(bias.item() + 2.0) < 0.02

def main() -> None:
  x = [1., 2., 3.]
  y = [4., 7., 10.]
  analytic = analytical_gradients(0.5, -1.0, x, y)
  value, dw, db = autograd_gradients(0.5, -1.0, x, y)
  print(f"loss = {value:.6f}")
  print(f"your gradients:     dW={analytic[0]: .6f}, db={analytic[1]: .6f}")
  print(f"autograd gradients: dW={dw: .6f}, db={db: .6f}")
  check_close("manual backprop", analytic, [dw, db], atol=2e-5)
  fit_line()

if __name__ == "__main__":
  main()
