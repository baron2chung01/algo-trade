from datetime import date, datetime, timedelta, timezone

import pytest

from algotrade.data.schemas import IBKRBar, IBKRBarDataFrame
from algotrade.data.stores.local import ParquetBarStore
from algotrade.experiments import (
    ExperimentSplit,
    MeanReversionExperimentConfig,
    MeanReversionExperimentResult,
    MeanReversionParameters,
    optimize_mean_reversion_parameters,
    run_mean_reversion_experiment,
)
from algotrade.experiments.mean_reversion import (
    HoldRange,
    OptimizationParameterSpec,
    ParameterRange,
    generate_parameter_grid,
)


def _build_bars(closes: list[float], start: date) -> IBKRBarDataFrame:
    rows: list[IBKRBar] = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += timedelta(days=1)
        timestamp = datetime(current.year, current.month,
                             current.day, tzinfo=timezone.utc)
        rows.append(
            IBKRBar(
                timestamp=timestamp,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1_000,
                average=close,
                bar_count=100,
            )
        )
        current += timedelta(days=1)
    return IBKRBarDataFrame.from_bars(rows)


def _populate_store(tmp_path, series: dict[str, list[float]], start: date) -> ParquetBarStore:
    store = ParquetBarStore(tmp_path / "bars")
    for symbol, closes in series.items():
        df = _build_bars(closes, start)
        store.save(symbol, "1d", df)
    return store


def _build_base_config(store: ParquetBarStore) -> MeanReversionExperimentConfig:
    return MeanReversionExperimentConfig(
        store_path=store.root,
        universe=["AAPL"],
        initial_cash=100_000.0,
        splits=[
            ExperimentSplit(name="train", start=date(
                2024, 1, 2), end=date(2024, 1, 6)),
        ],
        parameters=[
            MeanReversionParameters(
                entry_threshold=30.0,
                exit_threshold=70.0,
                max_hold_days=5,
                target_position_pct=0.5,
                stop_loss_pct=None,
                lot_size=10,
            )
        ],
    )


def test_run_mean_reversion_experiment_single_split(tmp_path):
    store = _populate_store(
        tmp_path, {"AAPL": [100.0, 95.0, 110.0]}, start=date(2024, 1, 2))
    config = _build_base_config(store)

    result = run_mean_reversion_experiment(config)
    frame = result.to_frame()

    assert not frame.empty
    assert set(frame["split"]) == {"train"}
    assert (frame["final_equity"] >= 0).all()
    assert (frame["total_return"] >= -1).all()
    assert (frame["trade_count"] >= 0).all()
    assert "sharpe_ratio" in frame.columns


def test_run_mean_reversion_experiment_multiple_parameters(tmp_path):
    store = _populate_store(
        tmp_path,
        {"AAPL": [100.0, 95.0, 110.0], "MSFT": [200.0, 190.0, 205.0]},
        start=date(2024, 1, 2),
    )
    parameters = [
        MeanReversionParameters(30.0, 70.0, 5, 0.5),
        MeanReversionParameters(10.0, 80.0, 3, 0.3),
    ]
    config = MeanReversionExperimentConfig(
        store_path=store.root,
        universe=["AAPL", "MSFT"],
        initial_cash=100_000.0,
        splits=[
            ExperimentSplit(name="train", start=date(
                2024, 1, 2), end=date(2024, 1, 6)),
            ExperimentSplit(name="test", start=date(
                2024, 1, 2), end=date(2024, 1, 6)),
        ],
        parameters=parameters,
    )

    result = run_mean_reversion_experiment(config)
    frame = result.to_frame()

    assert len(frame) == len(parameters) * len(config.splits)
    assert frame["params"].nunique() == len(parameters)
    assert set(frame["split"]) == {"train", "test"}
    assert (frame["max_drawdown"] >= 0).all()
    assert "sharpe_ratio" in frame.columns


def test_config_validation_rejects_duplicate_splits(tmp_path):
    store = _populate_store(
        tmp_path, {"AAPL": [100.0, 101.0]}, start=date(2024, 1, 2))
    config = _build_base_config(store)
    config.splits = (
        ExperimentSplit(name="train", start=date(
            2024, 1, 2), end=date(2024, 1, 6)),
        ExperimentSplit(name="TRAIN", start=date(
            2024, 1, 7), end=date(2024, 1, 10)),
    )

    with pytest.raises(ValueError, match="Duplicate split name"):
        config.validate()


def test_run_rejects_invalid_parameters(tmp_path):
    store = _populate_store(
        tmp_path, {"AAPL": [100.0, 98.0, 97.0]}, start=date(2024, 1, 2))
    config = _build_base_config(store)
    config.parameters = (
        MeanReversionParameters(
            entry_threshold=80.0,
            exit_threshold=30.0,
            max_hold_days=5,
            target_position_pct=0.5,
        ),
    )

    with pytest.raises(ValueError, match="entry_threshold must be less than exit_threshold"):
        run_mean_reversion_experiment(config)


def test_result_to_frame_handles_empty_result():
    result = MeanReversionExperimentResult()
    frame = result.to_frame()

    assert list(frame.columns) == ["split", "params"]
    assert frame.empty


@pytest.mark.parametrize("use_parameter_spec", [False, True])
def test_optimize_mean_reversion_parameters_returns_outcome(tmp_path, use_parameter_spec):
    start_date = date(2021, 1, 4)
    store = _populate_store(
        tmp_path,
        {
            "AAPL": [100 + i * 0.5 for i in range(900)],
        },
        start=start_date,
    )

    training_window = (start_date, start_date + timedelta(days=500))
    paper_window = (training_window[1] + timedelta(days=1),
                    training_window[1] + timedelta(days=250))

    if use_parameter_spec:
        spec = OptimizationParameterSpec(
            entry_threshold=ParameterRange(5, 10, 5),
            exit_threshold=ParameterRange(70, 80, 10),
            max_hold_days=HoldRange(3, 5, 2, include_infinite=True),
            target_position_pct=ParameterRange(10, 20, 10),
            stop_loss_pct=None,
            include_no_stop_loss=True,
            lot_size=10,
        )
        outcome = optimize_mean_reversion_parameters(
            store_path=store.root,
            universe=["AAPL"],
            initial_cash=100_000.0,
            training_window=training_window,
            paper_window=paper_window,
            parameter_spec=spec,
        )
        generated = generate_parameter_grid(spec)
        assert outcome.best_parameters in generated
        assert outcome.parameter_frame["params"].nunique() == len(generated)
    else:
        params = [
            MeanReversionParameters(5.0, 70.0, 3, 0.1),
            MeanReversionParameters(10.0, 80.0, 5, 0.2),
        ]
        outcome = optimize_mean_reversion_parameters(
            store_path=store.root,
            universe=["AAPL"],
            initial_cash=100_000.0,
            training_window=training_window,
            paper_window=paper_window,
            parameter_grid=params,
        )
        assert outcome.best_parameters in params

    assert outcome.training_metrics["final_equity"] >= 0
    assert outcome.paper_metrics["final_equity"] >= 0
    assert "sharpe_ratio" in outcome.training_metrics
    assert "sharpe_ratio" in outcome.paper_metrics
    assert not outcome.parameter_frame.empty


def test_generate_parameter_grid_supports_infinite_hold():
    spec = OptimizationParameterSpec(
        entry_threshold=ParameterRange(5, 5, 1),
        exit_threshold=ParameterRange(70, 70, 1),
        max_hold_days=HoldRange(2, 2, 1, include_infinite=True),
        target_position_pct=ParameterRange(10, 10, 1),
        stop_loss_pct=None,
        include_no_stop_loss=True,
        lot_size=10,
    )

    parameters = generate_parameter_grid(spec)

    assert any(param.max_hold_days == 0 for param in parameters)


def test_generate_parameter_grid_single_stop_value():
    spec = OptimizationParameterSpec(
        entry_threshold=ParameterRange(5, 5, 1),
        exit_threshold=ParameterRange(70, 70, 1),
        max_hold_days=HoldRange(3, 3, 1),
        target_position_pct=ParameterRange(10, 10, 1),
        stop_loss_pct=ParameterRange(7, 7, 1),
        include_no_stop_loss=False,
        lot_size=10,
    )

    parameters = generate_parameter_grid(spec)

    assert {param.stop_loss_pct for param in parameters} == {0.07}
