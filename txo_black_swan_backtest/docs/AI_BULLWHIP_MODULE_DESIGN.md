# AI Bullwhip Module Design

## Purpose

This module defines a thesis-driven, diagnostics-only structure for AI supply-chain bullwhip risk.

It is not a parameter search system and does not generate trades. It is intended to measure whether the data needed for AI bullwhip risk detection exists and whether the three thesis components can be computed.

## Subscore A: SupplyDistortionScore

SupplyDistortionScore measures AI/HBM capacity absorption and traditional DRAM / DDR5 supply mismatch.

Candidate proxies:

- `memory_revenue_yoy`
- `dram_price_proxy`
- `ddr5_price_proxy`
- `hbm_supply_pressure_proxy`
- `memory_maker_capex_mix`
- `ai_server_revenue_yoy`

Interpretation:

- Rising HBM pressure with rising DRAM / DDR5 price proxies indicates supply displacement.
- AI server revenue growth can support the thesis only when paired with traditional memory tightness.
- This subscore is scenario-aware until data coverage is stable.

## Subscore B: FalseDemandBoomScore

FalseDemandBoomScore measures OEM panic ordering and component false prosperity.

Candidate proxies:

- `pcb_revenue_yoy`
- `mlcc_revenue_yoy`
- `driver_ic_revenue_yoy`
- `component_revenue_yoy`
- `component_revenue_growth_minus_end_demand_growth`
- `inventory_days_components`
- `accounts_receivable_growth`

Interpretation:

- Component revenue acceleration without end-demand confirmation is the core warning.
- Inventory days and receivables are critical because reported revenue can lag order quality.
- This subscore is strongest when revenue acceleration and inventory accumulation occur together.

## Subscore C: DemandBreakRiskScore

DemandBreakRiskScore measures the probability that end users cannot absorb higher PC / phone / electronics prices.

Candidate proxies:

- `pc_shipments_yoy`
- `smartphone_shipments_yoy`
- `consumer_electronics_revenue_yoy`
- `real_wage_growth_yoy`
- `consumer_confidence`
- `retail_sales_yoy`
- `cpi_yoy`
- `policy_rate`

Interpretation:

- Weak PC / phone shipments while costs rise is a direct demand-break warning.
- Real wage weakness, falling consumer confidence, and high policy rates amplify fragility.
- This subscore should remain scenario-aware when shipment and sell-through data are sparse.

## Composite Score

`AI_Bullwhip_Risk_Score = 0.35 * SupplyDistortionScore + 0.35 * FalseDemandBoomScore + 0.30 * DemandBreakRiskScore`

The weights are thesis-driven fixed weights. They are not derived from historical performance optimization.

The weights must not be changed because of 2008, 2020, or 2022 backtest results.

## Computability Rules

First version diagnostics should report:

- Field availability.
- Non-null coverage by field.
- Subscore data coverage.
- Whether each subscore is computable.
- Overall confidence.

The first version should not fill missing fields, infer data from prices, or produce trades.

## Confidence Levels

- `HIGH`: broad field coverage across all three subscores.
- `MEDIUM`: enough data to compute at least two subscores, but missing important proxies.
- `LOW`: partial coverage only.
- `MISSING`: no usable AI bullwhip indicator data.

Confidence is a data-quality label, not a forecast.

## Integration Boundary

Allowed:

- Feed future risk detection after data coverage is audited.
- Produce scenario-aware risk diagnostics.
- Mark missing proxies explicitly.

Not allowed:

- Direct trade generation.
- Crash-window threshold fitting.
- Replacement of execution quality gates.
- Filling missing fields with zeros or future information.
