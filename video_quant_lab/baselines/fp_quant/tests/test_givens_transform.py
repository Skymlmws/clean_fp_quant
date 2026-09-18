import torch

from src.transforms.transforms import GivensTransform, HadamardTransform, build_transform


def test_givens_is_registered_and_uses_requested_group_size():
    transform = build_transform("givens", size=16, group_size=8)
    assert isinstance(transform, GivensTransform)
    assert transform.group_size == 8


def test_givens_calibration_is_orthogonal_and_handles_negative_outliers():
    x = torch.randn(2, 3, 16)
    x[1, 2, 5] = -100
    transform = GivensTransform(size=16, group_size=8, outlier_threshold=50)

    transformed = transform(x)
    identity = torch.eye(16)

    assert transformed.shape == x.shape
    matrix = transform.to_matrix()
    torch.testing.assert_close(matrix @ matrix.T, identity, atol=1e-5, rtol=1e-5)
    assert not torch.equal(matrix[:8, :8], identity[:8, :8])


def test_givens_preserves_linear_layer_reparametrization():
    x = torch.randn(4, 16)
    x[0, 2] = 100
    weight = torch.randn(7, 16)
    transform = GivensTransform(size=16, group_size=8)

    rotated_x = transform(x)
    rotated_weight = transform(weight, inv_t=True)

    torch.testing.assert_close(
        rotated_x @ rotated_weight.T,
        x @ weight.T,
        atol=2e-5,
        rtol=2e-5,
    )


def test_givens_supports_non_last_dimension():
    x = torch.randn(16, 3)
    x[4, 0] = 100
    transform = GivensTransform(size=16, group_size=8)

    assert transform(x, dim=0).shape == x.shape


def test_givens_accumulates_multiple_observations():
    transform = GivensTransform(size=8, group_size=4, outlier_threshold=20)
    first = torch.zeros(2, 8)
    first[0, 1] = 10
    second = torch.zeros(2, 8)
    second[1, 2] = -100

    transform.observe(first)
    transform.observe(second)
    transform.finalize_calibration()

    assert transform.mat is not None
    matrix = transform.to_matrix()
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(8), atol=1e-5, rtol=1e-5)


def test_givens_calibration_state_preserves_offline_inputs():
    transform = GivensTransform(size=8, group_size=4, outlier_threshold=20)
    source = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    transform.observe(source)

    state = transform.calibration_state()

    assert state["group_size"] == 4
    assert state["representative_vectors"].shape == (2, 4)
    torch.testing.assert_close(state["group_max_abs"], torch.tensor([11.0, 15.0]))

    restored = GivensTransform(size=8, group_size=4, outlier_threshold=12)
    restored.load_calibration_state(state)
    restored.finalize_calibration()
    assert restored.givens_blocks == 1
    assert restored.hadamard_blocks == 1


def test_givens_all_fallback_matches_randomized_hadamard_exactly():
    torch.manual_seed(11)
    x = torch.randn(3, 16)
    givens = GivensTransform(
        size=16,
        group_size=8,
        outlier_threshold=float("inf"),
        fallback_randomize=True,
        seed=7,
    )
    hadamard = HadamardTransform(group_size=8, randomize=True, seed=7)

    transformed = givens(x)

    assert givens.givens_blocks == 0
    assert givens.hadamard_blocks == 2
    torch.testing.assert_close(transformed, hadamard(x), atol=0, rtol=0)
