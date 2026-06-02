# Macro Risk Framework

## Purpose

This document defines the macro risk families the system tracks so risk detection does not collapse into a single narrative.

Risk modules can affect HedgeNeedScore only through fixed, auditable risk-score inputs. Execution remains gated by quote quality, budget, liquidity, and lifecycle checks.

## 1. Liquidity / Rates Stress

Thesis:

Rising rates, stronger USD, tighter liquidity, and funding stress compress equity valuations and raise crash risk.

Observable indicators:

- US Treasury yields
- Yield curve
- DXY
- USD/TWD
- JPY and JGB pressure
- Credit spreads
- Global liquidity proxies

Data availability:

- Partly available from public sources such as FRED and exchange data.
- Some Taiwan funding and cross-market liquidity proxies remain incomplete.

Update frequency:

- Daily for yields, FX, and market proxies.
- Weekly or monthly for some liquidity measures.

How it affects HedgeNeedScore:

- Can increase `LiquidityStress`.
- Can amplify `ValuationRisk` when valuation is high and liquidity tightens.

Backtestability status:

- Backtestable signals: US yields, DXY, USD/TWD.
- Proxy-only signals: global liquidity, JPY carry stress.
- Missing data: complete credit spread integration.

Current implementation status:

- Partial placeholders exist through risk-indicator framework.

## 2. Inflation / Stagflation Risk

Thesis:

High inflation with weak real wages reduces the ability of consumers to absorb higher electronics prices. This can turn supply-chain cost pressure into demand break.

Observable indicators:

- CPI
- Core CPI
- PPI
- Oil prices
- Real wages
- Consumer confidence
- Retail sales
- Policy rate

Data availability:

- US official macro data is generally available.
- Taiwan and regional consumer-demand proxies need source work.

Update frequency:

- Monthly for inflation, wages, retail sales, confidence.
- Daily for market-implied rate proxies.

How it affects HedgeNeedScore:

- Can increase `MacroDemandFragility`.
- Can interact with AI bullwhip risk when component costs rise.

Backtestability status:

- Backtestable signals: CPI, policy rates, retail sales.
- Proxy-only signals: real-time consumer electronics affordability.
- Missing data: regional end-demand granularity.

Current implementation status:

- MacroDemandFragility exists conceptually; data coverage remains incomplete.

## 3. Valuation / Earnings Risk

Thesis:

High valuation with earnings deterioration creates asymmetric downside when growth expectations reset.

Observable indicators:

- TWSE PE
- PB
- Dividend yield
- Earnings revision proxy
- Taiwan market cap / GDP proxy
- Margin balance
- Electronics revenue YoY / gross-margin pressure

Data availability:

- TWSE valuation data is partly available.
- Earnings revisions and gross-margin pressure need structured data sources.

Update frequency:

- Daily for market valuation where available.
- Monthly for revenue and margin proxies.
- Quarterly for earnings data.

How it affects HedgeNeedScore:

- Can increase `ValuationRisk`.
- Can raise target hedge coverage when paired with liquidity or demand fragility.

Backtestability status:

- Backtestable signals: TWSE PE, PB, dividend yield.
- Proxy-only signals: earnings revision proxy.
- Missing data: clean market cap / GDP and sector gross-margin pressure series.

Current implementation status:

- Basic valuation fields exist in the risk-indicator schema; coverage is incomplete.

## 4. Market Structure / Positioning Risk

Thesis:

Crowded positioning, unstable option skew, and poor liquidity can turn a correction into a disorderly move.

Observable indicators:

- Foreign futures net position
- Dealer option positioning
- Put/Call ratio
- TXO skew
- ATM straddle premium
- Margin balance
- Turnover and liquidity proxies

Data availability:

- Local TXO chain-derived proxies exist.
- Official institutional positioning coverage remains incomplete.

Update frequency:

- Daily for options chain and exchange positioning.
- Intraday if raw data is available.

How it affects HedgeNeedScore:

- Can increase `VolatilityComplacencyRisk` and `LiquidityStress`.
- Helps distinguish low-volatility calm from option-market stress.

Backtestability status:

- Backtestable signals: TXO local chain proxies, Put/Call ratio where available.
- Proxy-only signals: dealer gamma and detailed positioning.
- Missing data: full official long-history positioning integration.

Current implementation status:

- Local TXO-derived risk indicators are implemented.

## 5. Supply Chain / AI Bullwhip Risk

Thesis:

AI/HBM capacity absorption can distort traditional memory and component supply chains, producing false demand signals before inventory reversal.

Observable indicators:

- HBM capacity absorption
- DRAM / DDR5 prices
- PC / phone shipments
- PCB / MLCC / Driver IC revenue
- Inventory days
- Accounts receivable growth
- End-demand affordability

Data availability:

- Most indicators require manual source registry work.
- Some component revenue proxies may be available monthly.

Update frequency:

- Monthly for revenue and shipments.
- Weekly or daily for some price proxies if licensed data exists.

How it affects HedgeNeedScore:

- Scenario-aware input to `SupplyStressIndex`, `MacroDemandFragility`, or a future AI-specific risk module.
- It should adjust risk score only after data quality is audited.

Backtestability status:

- Scenario-aware signals: HBM capacity pressure, OEM panic ordering.
- Proxy-only signals: component revenue minus end demand.
- Missing data: historical HBM/DDR5 and inventory series.

Current implementation status:

- Thesis and module design added.
- First diagnostics audit reads `ai_bullwhip_indicators.csv` when present.

## 6. Geopolitical / Event Risk

Thesis:

Discrete geopolitical or policy events can cause gaps, liquidity failures, and supply-chain repricing that historical daily data may understate.

Observable indicators:

- Taiwan Strait event risk
- Iran / oil shock risk
- US sanctions / export controls
- SpaceX IPO / AI capex sentiment shock
- Tariffs and supply-chain policy changes

Data availability:

- Mostly event calendar and scenario data.
- Many sources are qualitative and require governance before ingestion.

Update frequency:

- Event-driven.
- Daily monitoring if source registry is established.

How it affects HedgeNeedScore:

- Scenario-aware event risk can raise risk sensitivity.
- It must not directly place trades.

Backtestability status:

- Scenario-aware signals: sanctions, military escalation, IPO sentiment shocks.
- Proxy-only signals: oil shock channels.
- Missing data: reliable structured event-risk history.

Current implementation status:

- `event_flag` exists; detailed geopolitical module is not implemented.

## Risk Module Priority Table

| Priority | Module | Type | Reason |
|---|---|---|---|
| 1 | Market Structure / Positioning | Backtestable / proxy | Local TXO data exists and affects execution feasibility. |
| 2 | Liquidity / Rates Stress | Backtestable | Public data is accessible and broad risk impact is clear. |
| 3 | Supply Chain / AI Bullwhip Risk | Scenario-aware | Core thesis, but data coverage must be built carefully. |
| 4 | Inflation / Stagflation Risk | Backtestable / proxy | Important demand-fragility amplifier. |
| 5 | Valuation / Earnings Risk | Backtestable / proxy | Useful when paired with liquidity and earnings pressure. |
| 6 | Geopolitical / Event Risk | Scenario-aware | Important, but hard to structure without subjective labels. |

## Data Collection Priority

1. Official VIX / TXO positioning / Put-Call ratio history.
2. TWSE valuation history.
3. FRED rates, inflation, and USD data.
4. Monthly Taiwan electronics supply-chain revenue proxies.
5. PC / phone shipments and sell-through proxies.
6. Inventory days and receivable growth by component segment.
7. Structured event calendar sources.

## Guardrails

- Do not use a single macro narrative to place trades directly.
- Do not use crash windows to reverse-engineer weights.
- Scenario-aware indicators can adjust risk scores only after data quality audit; they do not create trades.
- Execution must still pass quote quality, budget, liquidity, and lifecycle audit.
- Missing data must remain visible and must not be filled with zeros.
