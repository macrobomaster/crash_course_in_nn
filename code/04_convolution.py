# Lesson 4: convolution is sliding, shared, local weighted summation.
from tinygrad import Tensor, nn
from common import check_close, count_parameters, format_array

def correlate_at(image: list[list[float]], kernel: list[list[float]], row: int, col: int) -> float:
  """Extract one output pixel: elementwise multiplication followed by a sum."""
  height, width = len(kernel), len(kernel[0])
  assert 0 <= row <= len(image) - height
  assert 0 <= col <= len(image[0]) - width
  # TODO: loop over the overlapping image patch and calculate a dot product.
  raise NotImplementedError("Multiply each image/kernel pair and return the sum")

def convolution_demo() -> None:
  image = [
    [1., 2., 0., 1.],
    [3., 1., 2., 2.],
    [0., 1., 3., 1.],
    [1., 0., 2., 2.],
  ]
  kernel = [
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.],
  ]
  height, width = len(kernel), len(kernel[0])
  output_rows = len(image) - height + 1
  output_cols = len(image[0]) - width + 1
  manual = [[correlate_at(image, kernel, r, c) for c in range(output_cols)] for r in range(output_rows)]
  x = Tensor(image).reshape(1, 1, 4, 4).detach()  # (batch, in channels, height, width)
  weights = Tensor(kernel).reshape(1, 1, 3, 3).detach()  # (out channels, in/groups, kh, kw)
  actual = x.conv2d(weights).tolist()[0][0]
  print("Manual output:", format_array(manual, precision=2))
  print("tinygrad output shape:", (len(actual), len(actual[0])))
  check_close("manual 2-D correlation equals tinygrad conv2d", actual, manual, atol=1e-6)
  assert (len(actual), len(actual[0])) == (2, 2)

def translation_demo() -> None:
  image_a = [[0.0] * 8 for _ in range(6)]
  image_b = [[0.0] * 8 for _ in range(6)]
  for row in range(2, 4):
    image_a[row][2] = 3.0
    image_b[row][6] = 3.0  # same feature, translated four pixels
  detector = [[[[-1., 2., -1.]]]]  # 1 output, 1 input, 1 x 3 spatial
  filter_t = Tensor(detector).detach()
  response_a = Tensor(image_a).reshape(1, 1, 6, 8).detach().conv2d(filter_t)
  response_b = Tensor(image_b).reshape(1, 1, 6, 8).detach().conv2d(filter_t)
  print("Original response position, maximum, pooled response:", response_a.argmax().item(), response_a.max().item(), response_a.max().item())
  print("Shifted response position, maximum, pooled response :", response_b.argmax().item(), response_b.max().item(), response_b.max().item())
  assert response_a.argmax().item() != response_b.argmax().item()
  check_close("translation preserves the pooled maximum", response_a.max().item(), response_b.max().item(), atol=1e-6)

class TinyConvolutionalClassifier:
  def __init__(self) -> None:
    self.features = nn.Conv2d(1, 4, 3, padding=1)  # four learned 3x3 filters
    self.output = nn.Linear(4, 1)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.features(x).relu()  # (B,4,H,W)
    x = x.mean((2, 3))  # global average pooling: (B,4)
    return self.output(x)

class TinyFlattenedMLP:
  def __init__(self) -> None:
    self.hidden = nn.Linear(8 * 8, 8)
    self.output = nn.Linear(8, 1)

  def __call__(self, x: Tensor) -> Tensor:
    return self.output(self.hidden(x.reshape(x.shape[0], -1)).relu())

def capacity_demo() -> None:
  image = Tensor.zeros(2, 1, 8, 8).detach()
  cnn, mlp = TinyConvolutionalClassifier(), TinyFlattenedMLP()
  print("CNN output shape and parameter count:", cnn(image).shape, count_parameters(cnn))
  print("MLP output shape and parameter count:", mlp(image).shape, count_parameters(mlp))
  assert cnn(image).shape == (2, 1)
  assert count_parameters(cnn) < count_parameters(mlp)
  print("Smaller is not automatically better: weight sharing adds a useful spatial assumption.")

def main() -> None:
  convolution_demo()
  translation_demo()
  capacity_demo()
  print("Try it: swap ReLU for max/mean pooling, change padding/stride, or add another channel.")

if __name__ == "__main__":
  main()
