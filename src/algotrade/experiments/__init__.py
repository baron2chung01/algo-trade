"""Experiment runners for evaluating strategies."""

from .breakout import (
    BreakoutExperimentConfig,
    BreakoutExperimentResult,
    BreakoutOptimizationOutcome,
    BreakoutParameterSpec,
    BreakoutParameters,
    FloatRange,
    default_breakout_spec,
    generate_breakout_parameter_grid,
    optimize_breakout_parameters,
    run_breakout_experiment,
)
from .mean_reversion import (
    ExperimentSplit,
    MeanReversionExperimentConfig,
    MeanReversionParameters,
    MeanReversionExperimentResult,
    OptimizationOutcome,
    default_parameter_grid,
    optimize_mean_reversion_parameters,
    run_mean_reversion_experiment,
)

__all__ = [
    "BreakoutExperimentConfig",
    "BreakoutExperimentResult",
    "BreakoutOptimizationOutcome",
    "BreakoutParameterSpec",
    "BreakoutParameters",
    "FloatRange",
    "ExperimentSplit",
    "MeanReversionExperimentConfig",
    "MeanReversionParameters",
    "MeanReversionExperimentResult",
    "OptimizationOutcome",
    "default_breakout_spec",
    "default_parameter_grid",
    "generate_breakout_parameter_grid",
    "optimize_mean_reversion_parameters",
    "optimize_breakout_parameters",
    "run_breakout_experiment",
    "run_mean_reversion_experiment",
]
