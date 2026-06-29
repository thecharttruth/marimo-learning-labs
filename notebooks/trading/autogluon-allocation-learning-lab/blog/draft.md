# AutoGluon Portfolio Allocation Learning Lab

## Working Thesis

AutoML can help with the modeling part of trading research, but it does not remove the need for careful validation, allocation design, and skepticism about backtest results.

## Audience

Students who know basic Python or machine learning and want to understand how predictive models connect to actual portfolio weights.

## Outline

1. The problem: predicting returns is not the same as building a portfolio.
2. Feature engineering with ETF price history.
3. What AutoGluon automates and what it does not.
4. Why walk-forward validation matters in trading.
5. Turning forecasts into weights with mean-variance allocation.
6. Why Monte Carlo allocation can make weights less brittle.
7. How to read allocation weights and equity curves.
8. Where the notebook is still unrealistic: costs, slippage, market impact, and execution timing.
9. The no-lookahead parameter optimizer: a bounded search that selects on an inner window and reports on later splits.
10. Benchmarks (SPY, 60/40), out-of-sample gates, and prediction diagnostics: how to judge whether the model is useful.
11. Caching market data with `mo.persistent_cache` so kernel restarts stay fast without hiding stale data.

## Notes To Develop

- Explain every chart in plain language.
- Include screenshots from the notebook including the optimizer, out-of-sample gate, and diagnostics tables.
- Avoid implying the strategy is production-ready.
- Emphasize the difference between a research notebook and a trading system.
- Note that the PEP 723 script header now declares all runtime dependencies so `uv run notebook.py` installs everything in one pass.
