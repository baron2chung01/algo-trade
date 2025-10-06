# Algo-Trade

Python research and execution stack for US equities strategies routed through Interactive Brokers (IBKR) Trader Workstation.

## Quickstart

1. **Create environment**

   ```bash
   uv venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. **Configure credentials**

   - Copy `.env.example` to `.env` and fill in IBKR client / port settings.

3. **Run tests**
   ```bash
   pytest
   ```

## Repository layout (work in progress)

- `src/algotrade/` – core package (data adapters, strategies, execution).
- `tests/` – automated test suite.
- `PROJECT_PLAN.md` – project roadmap and scope.

Refer to `PROJECT_PLAN.md` for the detailed roadmap and upcoming milestones.
