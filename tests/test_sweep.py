"""Tests for sweep functionality."""
from trainformer.sweep import Choice, Grid, IntUniform, Sweep, Uniform, choice, grid, log_uniform


def test_grid_sweep():
    """Test grid search."""
    def objective(x: int, y: int) -> float:
        return -(x - 2) ** 2 - (y - 3) ** 2

    sweep = Sweep(objective, {"x": Grid([1, 2, 3]), "y": Grid([2, 3, 4])})
    results = sweep.run(method="grid")

    assert len(results) == 9  # 3x3 grid
    # Best should be x=2, y=3
    best = results[0]
    assert best["config"]["x"] == 2
    assert best["config"]["y"] == 3
    assert best["metric"] == 0.0


def test_random_sweep():
    """Test random search."""
    def objective(x: float) -> float:
        return -x ** 2

    sweep = Sweep(objective, {"x": Uniform(-10, 10)})
    results = sweep.run(method="random", n_trials=20, seed=42)

    assert len(results) == 20
    # All results should be negative or zero
    for r in results:
        assert r["metric"] <= 0


def test_sweep_space_constructors():
    """Test space constructor functions."""
    assert isinstance(grid(1, 2, 3), Grid)
    assert grid(1, 2, 3).values == [1, 2, 3]

    assert isinstance(choice(1, 2, 3), Choice)
    assert choice(1, 2, 3).values == [1, 2, 3]

    assert isinstance(log_uniform(1e-5, 1e-3), Uniform)
    assert log_uniform(1e-5, 1e-3).log is True

    assert isinstance(IntUniform(1, 10), IntUniform)
    assert IntUniform(1, 10).low == 1
    assert IntUniform(1, 10).high == 10


def test_sweep_failed_trial():
    """Test handling of failed trials."""
    call_count = [0]

    def objective(x: int) -> float:
        call_count[0] += 1
        if call_count[0] == 2:
            raise ValueError("Intentional failure")
        return -x ** 2

    sweep = Sweep(objective, {"x": Grid([1, 2, 3])})
    results = sweep.run(method="grid")

    assert len(results) == 3
    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
