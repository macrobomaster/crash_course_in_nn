# Lesson 3: losses are design choices, not just a function called loss_fn.
import math
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from common import check_close, format_array

def cross_entropy_per_example(logits: Tensor, one_hot_targets: Tensor) -> Tensor:
  """One-of-N classification. logits: (B,C), targets: (B,C), unnormalized scores in."""
  return -logits.log_softmax(-1).mul(one_hot_targets).sum(-1)

def focal_loss_per_element(logits: Tensor, binary_targets: Tensor, gamma: float = 2.0) -> Tensor:
  """Down-weight easy binary-classification examples, commonly used with data imbalance."""
  probabilities = logits.sigmoid()
  cross_entropy = logits.binary_crossentropy_logits(binary_targets, reduction="none")
  p_t = probabilities * binary_targets + (1 - probabilities) * (1 - binary_targets)
  return cross_entropy * (1 - p_t) ** gamma

def masked_mean(losses: Tensor, valid: Tensor) -> Tensor:
  # TODO: sum selected losses and divide by the number of selected items.
  # Add a tiny epsilon to the denominator so an empty mask still produces zero.
  raise NotImplementedError("Hint: use valid.cast(...), valid.where(...), and mask.sum()")

def two_hot_targets(values: Tensor, bins: int = 16, low: float = 0.0, high: float = 1.0) -> Tensor:
  """Spread a continuous value between its two neighboring centers on a uniform grid."""
  scaled = ((values - low) / (high - low) * (bins - 1)).clamp(0, bins - 1)
  left = scaled.floor().cast(dtypes.int32).clamp(0, bins - 2)
  right_weight = (scaled - left.cast(dtypes.float32)).unsqueeze(-1)
  left_distribution = left.one_hot(bins).cast(dtypes.float32) * (1 - right_weight)
  right_distribution = (left + 1).one_hot(bins).cast(dtypes.float32) * right_weight
  return left_distribution + right_distribution

def expectation_over_bins(probabilities: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
  bins = probabilities.shape[-1]
  centers = low + Tensor.arange(bins, dtype=dtypes.float32) / (bins - 1) * (high - low)
  return (probabilities * centers).sum(-1)

def regression_demo() -> None:
  truth = Tensor([0., 0., 0., 0.]).detach()
  prediction = Tensor([0., 0., 2., 6.]).detach()  # the last entry is an outlier
  squared = ((prediction - truth) ** 2).mean()
  absolute = (prediction - truth).abs().mean()
  print(f"Regression with one outlier: MSE={squared.item():.3f}, MAE={absolute.item():.3f}")
  assert abs(squared.item() - 10.0) < 1e-5
  assert abs(absolute.item() - 2.0) < 1e-5

def classification_demo() -> None:
  logits = Tensor([[3., 1., 0.], [0., 3., 0.]]).detach()
  labels = Tensor([[1., 0., 0.], [1., 0., 0.]]).detach()  # second prediction is confidently wrong
  ours = cross_entropy_per_example(logits, labels)
  reference = logits.cross_entropy(labels, reduction="none")
  print("Per-example cross-entropy:", format_array(ours.tolist()))
  check_close("softmax cross-entropy", ours.tolist(), reference.tolist(), atol=2e-6)
  assert ours[1].item() > ours[0].item() * 10

def masking_demo() -> None:
  losses = Tensor([1., 100., 3., 1000.]).detach()
  valid = Tensor([True, False, True, False])
  answer = masked_mean(losses, valid)
  check_close("mask invalid labels", answer.item(), 2.0, atol=1e-6)
  empty = masked_mean(losses, Tensor([False, False, False, False]))
  assert abs(empty.item()) < 1e-8

def distribution_demo() -> None:
  values = Tensor([0., 0.2, 0.49, 0.97, 1.]).detach()
  distribution = two_hot_targets(values)
  reconstructed = expectation_over_bins(distribution)
  print("One soft target row (16 bins):", format_array(distribution[1].tolist(), precision=3))
  check_close("two-hot probabilities sum to one", distribution.sum(-1).tolist(), [1.0] * 5, atol=1e-6)
  check_close("expectation reconstructs coordinate", reconstructed.tolist(), values.tolist(), atol=2e-6)
  logits = [[math.log(max(probability, 1e-20)) for probability in row] for row in distribution.tolist()]
  reference_loss = cross_entropy_per_example(Tensor(logits), distribution)
  print("Near-correct soft target CE values:", format_array(reference_loss.tolist()))

def main() -> None:
  regression_demo()
  classification_demo()
  masking_demo()
  distribution_demo()
  binary_targets = Tensor([1., 0., 0.]).detach()
  logits = Tensor([1., -1., -8.]).detach()
  ce = logits.binary_crossentropy_logits(binary_targets, reduction="none")
  focal = focal_loss_per_element(logits, binary_targets)
  print("Binary examples: BCE", format_array(ce.tolist()), "focal", format_array(focal.tolist()))
  assert focal[2].item() < ce[2].item() * 0.001  # the third is an easy, well-classified example

if __name__ == "__main__":
  main()
