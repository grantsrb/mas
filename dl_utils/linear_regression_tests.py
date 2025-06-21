import unittest
import torch
try:
    from utils import ( analytical_linear_regression )
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from utils import ( analytical_linear_regression )

# Helper to generate synthetic data
def generate_linear_data(N=100, D=5, noise_std=0.0, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, D)
    w_true = torch.randn(D)
    y = X @ w_true + torch.randn(N) * noise_std
    return X, y, w_true

class TestLinearRegressionAnalytical(unittest.TestCase):

    def test_output_shape(self):
        X, y, _ = generate_linear_data(N=50, D=4)
        weights, _ = analytical_linear_regression(X, y)
        self.assertEqual(weights.shape, torch.Size([5]), f"Expected shape (5,), got {weights.shape}")

    def test_zero_noise_recovery(self):
        X, y, w_true = generate_linear_data(N=100, D=4, noise_std=0.0)
        weights, val_loss = analytical_linear_regression(X, y, val_split=0.2)
        estimated_w = weights[:-1]
        bias = weights[-1]

        assert torch.allclose(estimated_w, w_true, atol=1e-4),  "Recovered weights not close to true weights"
        self.assertAlmostEqual(bias, 0.0, delta=1e-4, msg="Bias should be near zero for zero-centered X")
        self.assertLess(val_loss, 1e-6, f"Validation loss too high: {val_loss}")

    def test_with_noise(self):
        X, y, w_true = generate_linear_data(N=200, D=3, noise_std=0.1)
        weights, val_loss = analytical_linear_regression(X, y, val_split=0.3)
        estimated_w = weights[:-1]

        assert torch.allclose(estimated_w, w_true, atol=0.1), "Recovered weights should be close despite noise"
        self.assertLess(val_loss, 0.1, "Expected reasonably low validation loss")

    def test_nonzero_bias(self):
        X, y, _ = generate_linear_data(N=100, D=2, noise_std=0.01)
        y += 5.0  # Add constant bias
        weights, _ = analytical_linear_regression(X, y)
        bias = weights[-1].item()
        self.assertAlmostEqual(bias, 5.0, delta=0.1, msg=f"Expected bias ~5.0, got {bias}")

if __name__ == "__main__":
    unittest.main()

