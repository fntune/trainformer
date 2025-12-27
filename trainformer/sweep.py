"""Hyperparameter sweep utilities."""
import itertools
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Grid:
    """Grid search over values."""
    values: list


@dataclass
class Choice:
    """Random choice from values."""
    values: list


@dataclass
class Uniform:
    """Uniform distribution."""
    low: float
    high: float
    log: bool = False


@dataclass
class IntUniform:
    """Uniform distribution over integers."""
    low: int
    high: int


# Convenience constructors
def grid(*values) -> Grid:
    return Grid(list(values))


def choice(*values) -> Choice:
    return Choice(list(values))


def uniform(low: float, high: float) -> Uniform:
    return Uniform(low, high)


def log_uniform(low: float, high: float) -> Uniform:
    return Uniform(low, high, log=True)


def int_uniform(low: int, high: int) -> IntUniform:
    return IntUniform(low, high)


class Sweep:
    """Hyperparameter sweep runner.

    Supports grid search, random search, and Bayesian optimization (via optuna).

    Args:
        fn: Function to optimize (should return a metric to maximize)
        spaces: Dict mapping parameter names to search spaces
    """

    def __init__(self, fn: Callable, spaces: dict[str, Any]):
        self.fn = fn
        self.spaces = spaces

    def run(
        self,
        method: str = "random",
        n_trials: int = 10,
        n_parallel: int = 1,
        seed: int = 42,
    ) -> list[dict]:
        """Run the sweep.

        Args:
            method: Search method ('grid', 'random', 'bayes')
            n_trials: Number of trials for random/bayes search
            n_parallel: Number of parallel workers
            seed: Random seed

        Returns:
            List of results sorted by metric (best first)
        """
        random.seed(seed)

        if method == "grid":
            configs = self._grid_configs()
        elif method == "random":
            configs = [self._sample() for _ in range(n_trials)]
        elif method == "bayes":
            return self._run_bayes(n_trials, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        if n_parallel == 1:
            results = [self._run_one(c) for c in configs]
        else:
            results = self._run_parallel(configs, n_parallel)

        return sorted(
            results,
            key=lambda r: r.get("metric") if r.get("metric") is not None else float("-inf"),
            reverse=True,
        )

    def _grid_configs(self) -> list[dict]:
        """Generate all grid configurations."""
        grid_params = {
            name: space.values
            for name, space in self.spaces.items()
            if isinstance(space, (Grid, Choice))
        }

        configs = []
        for combo in itertools.product(*grid_params.values()):
            config = dict(zip(grid_params.keys(), combo))
            # Add sampled values for non-grid params
            for name, space in self.spaces.items():
                if name not in config:
                    config[name] = self._sample_one(space)
            configs.append(config)

        return configs

    def _sample(self) -> dict:
        """Sample a random configuration."""
        return {name: self._sample_one(space) for name, space in self.spaces.items()}

    def _sample_one(self, space) -> Any:
        """Sample a single parameter."""
        if isinstance(space, (Grid, Choice)):
            return random.choice(space.values)
        elif isinstance(space, Uniform):
            if space.log:
                return math.exp(random.uniform(math.log(space.low), math.log(space.high)))
            return random.uniform(space.low, space.high)
        elif isinstance(space, IntUniform):
            return random.randint(space.low, space.high)
        return space

    def _run_one(self, config: dict) -> dict:
        """Run a single trial."""
        try:
            metric = self.fn(**config)
            return {"config": config, "metric": metric, "status": "completed"}
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return {"config": config, "metric": None, "status": "failed", "error": str(e)}

    def _run_parallel(self, configs: list[dict], n_parallel: int) -> list[dict]:
        """Run trials in parallel."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        import cloudpickle

        fn_bytes = cloudpickle.dumps(self.fn)

        def worker(config: dict) -> dict:
            fn = cloudpickle.loads(fn_bytes)
            try:
                metric = fn(**config)
                return {"config": config, "metric": metric, "status": "completed"}
            except Exception as e:
                return {"config": config, "metric": None, "status": "failed", "error": str(e)}

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(n_parallel, mp_context=ctx) as pool:
            results = list(pool.map(worker, configs))

        return results

    def _run_bayes(self, n_trials: int, seed: int) -> list[dict]:
        """Run Bayesian optimization with optuna."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            config = {}
            for name, space in self.spaces.items():
                if isinstance(space, (Grid, Choice)):
                    config[name] = trial.suggest_categorical(name, space.values)
                elif isinstance(space, Uniform):
                    config[name] = trial.suggest_float(name, space.low, space.high, log=space.log)
                elif isinstance(space, IntUniform):
                    config[name] = trial.suggest_int(name, space.low, space.high)
            return self.fn(**config)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return [
            {"config": t.params, "metric": t.value, "status": "completed"}
            for t in study.trials
        ]


def sweep(**spaces):
    """Decorator to create a sweep from a function.

    Example:
        @sweep(
            lr=log_uniform(1e-5, 1e-3),
            batch_size=choice(16, 32, 64),
        )
        def experiment(lr: float, batch_size: int) -> float:
            trainer = Trainer(...)
            trainer.fit()
            return -trainer.best_metrics["val/loss"]

        results = experiment.run(method="bayes", n_trials=20)
    """
    def decorator(fn):
        return Sweep(fn, spaces)
    return decorator
