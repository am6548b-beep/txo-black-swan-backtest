# Project Guardrails

## Project Objective

Build a Risk-Scaled Hedge Coverage System for TXO put spread hedging.

The system should use ex-ante observable quantitative risk indicators to gradually raise or lower put spread hedge coverage. The primary goal is to reduce black-swan drawdown and blow-up risk while keeping the framework conservative, auditable, and resistant to overfitting.

## Non-goals

- Not a fixed quarterly hedge system.
- Not a rolling always-hedged system.
- Not a parameter optimization project.
- Not a crash-window fitting project.
- Not a return-maximizing options strategy.
- Not a Sharpe-maximizing or CAGR-maximizing backtest.
- Not a framework for making historical performance look better by changing rules after seeing results.

## Architecture Layers

### 1. Risk Detection

Risk detection estimates hedge need using only data observable at or before the decision date.

Core components:

- HedgeNeedScore
- ValuationRisk
- VolatilityComplacencyRisk
- MacroDemandFragility
- SupplyStressIndex
- LiquidityStress
- TrendFragility

If any component uses a proxy, such as realized-volatility VIX proxy, the report must explicitly mark the signal as lower confidence.

### 2. Hedge Coverage Targeting

Hedge coverage decisions must serve the target coverage implied by HedgeNeedScore.

Required concepts:

- HedgeNeedScore to target_hedge_coverage
- current_hedge_coverage
- hedge_gap
- annual hedge budget

Fixed quarterly insurance, rolling insurance, exit-DTE variants, and moneyness variants are not final strategies. They are diagnostics for understanding execution feasibility, coverage gaps, quote decay, and lifecycle risk.

### 3. Execution Feasibility

Execution feasibility determines whether a hedge can be safely executed. It must not decide whether the portfolio should be hedged.

Execution checks include:

- quote_quality_status
- is_tradable_quote
- liquidity filters
- expiry validity
- position lifecycle
- forced_unfilled_exit handling
- bid/ask plus slippage execution

Execution feasibility filters can block trades that are not executable, but they must not become risk-timing rules.

### 4. Diagnostics / Audit

Diagnostics exist to explain failure modes and data limitations. They must not be treated as parameter-selection engines.

Allowed audit areas:

- coverage gaps
- forced exits
- quote quality
- no-lookahead
- budget usage
- crash window diagnostics
- data availability
- lifecycle consistency
- VIX proxy and other proxy usage

## Guardrails Against Overfitting

- Do not optimize rules on 2008, 2020, 2022, or any other crash window.
- Do not adjust HedgeNeedScore weights based on test-period outcomes.
- Do not use future information in score construction, entry eligibility, or diagnostics that describe real-time decision state.
- Do not use all-period percentiles when rolling or expanding measures are required.
- Do not reinterpret a diagnostic variant as the final strategy because it improves historical crash coverage.
- Do not treat a single regime, single memory cycle, or single macro episode as sufficient validation.
- Mark any signal dependent on proxy data as lower confidence.

## Allowed Diagnostics

Allowed diagnostics include:

- quarterly insurance coverage diagnostics
- rolling replacement coverage diagnostics
- exit timing diagnostics
- moneyness tradability diagnostics
- forced exit and liquidity diagnostics
- entry-time exit feasibility diagnostics
- quote quality diagnostics
- position lifecycle diagnostics
- HedgeNeedScore and target hedge coverage diagnostics
- crash-window diagnostics that explain what happened without changing rules

These diagnostics may compare outputs, but they must not select a best variant or recommend a final parameter set.

## Disallowed Optimizations

- No auto-search for best DTE.
- No auto-search for best moneyness.
- No crash-window fitting.
- No changing rules to improve 2008, 2020, or 2022 results.
- No using close instead of bid/ask.
- No ignoring non-VALID quote failures.
- No deleting dirty data to improve results.
- No increasing leverage to improve coverage.
- No treating execution feasibility as hedge-need logic.
- No report language that labels any variant as best or recommended.

## Current Status

- Real TXO/TXF data pipeline exists.
- Quote quality gate exists.
- Position lifecycle bug fixed.
- Coverage diagnostics completed.
- Forced exit diagnostics completed.
- Rolling, quarterly, moneyness, and exit timing variants are diagnostics only.
- Need to return to HedgeNeedScore and target hedge coverage.

## Next Step

Implement HedgeNeedScore diagnostics without creating trades.

The next implementation layer should focus on:

- daily HedgeNeedScore decomposition
- target_hedge_coverage mapping
- current_hedge_coverage measurement
- hedge_gap audit
- annual budget state
- proxy confidence flags
- crash-window audit that explains gaps without tuning rules
