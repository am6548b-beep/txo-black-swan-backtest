# AI Bullwhip Risk Thesis

## 1. Thesis Summary

The core thesis is that AI/HBM demand can distort the electronics supply chain before the broad equity market recognizes the fragility.

The chain is:

AI/HBM high-margin capacity absorption
-> traditional DRAM / DDR5 supply mismatch
-> PC / phone component cost pressure
-> OEM panic ordering
-> PCB / MLCC / Driver IC / component revenue false boom
-> end demand fails to absorb higher prices
-> inventory reversal and order cuts
-> Taiwan electronics supply-chain valuation reset.

This is not a generic AI bubble model. It is a supply-chain mismatch and bullwhip-risk model.

## 2. Why This Is Not A Normal Cycle

Normal semiconductor cycles are often inventory and capex cycles around broad end demand. The AI bullwhip thesis is different because one high-margin demand source can pull capacity, capital, attention, and pricing power away from traditional components.

The risk is not simply that AI demand slows. The risk is that AI demand remains strong while traditional downstream demand cannot tolerate the price and inventory consequences created by the AI capacity pull.

Key differences:

- HBM and AI server economics can remain strong while PC / phone economics weaken.
- Traditional DRAM / DDR5 scarcity can look like demand strength when it is partly supply displacement.
- Component makers can report strong revenue before sell-through confirms real demand.
- OEM orders can reflect fear of shortages instead of consumer demand.
- Inventory and receivables can reveal the thesis earlier than index price action.

## 3. Why Historical Backtest Is Insufficient

The full AI/HBM capacity reallocation pattern has limited historical precedent. A purely historical backtest can understate the risk because:

- HBM did not have the same supply-chain priority in older cycles.
- Modern AI capex concentration is more extreme than prior PC / handset cycles.
- Taiwan equity index exposure to AI infrastructure is structurally different from 2008, 2011, or 2020.
- Official high-frequency macro and supply-chain data are incomplete.
- Scenario-aware indicators may identify fragility before a crash window appears in price data.

Therefore this module is for scenario-aware diagnostics, not for fitting crash-window outcomes.

## 4. Observable Proxy Indicators

Supply distortion proxies:

- `memory_revenue_yoy`
- `dram_price_proxy`
- `ddr5_price_proxy`
- `hbm_supply_pressure_proxy`
- `memory_maker_capex_mix`
- `ai_server_revenue_yoy`

False demand boom proxies:

- `pcb_revenue_yoy`
- `mlcc_revenue_yoy`
- `driver_ic_revenue_yoy`
- `component_revenue_yoy`
- `component_revenue_growth_minus_end_demand_growth`
- `inventory_days_components`
- `accounts_receivable_growth`

Demand break proxies:

- `pc_shipments_yoy`
- `smartphone_shipments_yoy`
- `consumer_electronics_revenue_yoy`
- `real_wage_growth_yoy`
- `consumer_confidence`
- `retail_sales_yoy`
- `cpi_yoy`
- `policy_rate`

## 5. Scenario Model Variables

The scenario model should track:

- Supply displacement pressure from AI/HBM capacity allocation.
- Traditional DRAM / DDR5 pricing pressure.
- Component revenue acceleration not confirmed by end demand.
- OEM inventory days and component inventory days.
- Accounts receivable growth versus shipment growth.
- End-market affordability and consumer demand fragility.
- Whether upstream optimism remains high while sell-through weakens.

## 6. How It Connects To HedgeNeedScore

The AI Bullwhip module should eventually feed risk detection, not execution.

Connection path:

1. AI bullwhip indicators produce scenario-aware risk diagnostics.
2. Diagnostics can inform a fixed risk module such as `SupplyStressIndex` or a future `AI_Bullwhip_Risk_Score`.
3. HedgeNeedScore maps risk to target hedge coverage.
4. The execution layer attempts put-spread coverage only when hedge gap, budget, quote quality, liquidity, and lifecycle checks pass.

The module must not directly create trades.

## 7. Guardrails Against Overfitting

- Do not tune weights from 2008, 2020, or 2022 outcomes.
- Do not use crash windows to set thresholds.
- Do not infer missing data from later market returns.
- Do not treat scenario-aware data as a direct trading trigger.
- Do not change quote quality, budget, or lifecycle rules to improve coverage.
- Keep weights thesis-driven and fixed until a governance review changes them explicitly.

## 8. Current Missing Data

Likely missing or incomplete data:

- HBM-specific supply pressure.
- DRAM and DDR5 spot series with clean historical licensing.
- OEM inventory days by segment.
- Component-level inventory and receivable history.
- PC / phone sell-through data.
- AI server revenue and capex mix by supplier.
- Consistent monthly PCB / MLCC / Driver IC revenue data.

## 9. Next Implementation Steps

1. Establish `data/processed/ai_bullwhip_indicators.csv` format.
2. Add source registry entries for each proxy before ingestion.
3. Build a coverage audit for available fields and missing fields.
4. Compute subscore availability before computing any risk score.
5. Add scenario documentation for supply distortion, false boom, and demand break.
6. Only after data quality is documented, evaluate whether the module should feed HedgeNeedScore.
