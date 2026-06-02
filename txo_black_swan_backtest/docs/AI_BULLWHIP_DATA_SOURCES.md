# AI Bullwhip Data Sources

This inventory separates practical manual inputs from harder or paid sources. It is a data-collection guide, not a trading rule.

## P0: Manual / Free / Easier To Obtain

| indicator_name | subscore_used_by | source_candidate | frequency | availability | manual_or_auto | confidence | notes |
|---|---|---|---|---|---|---|---|
| `component_revenue_yoy` | FalseDemandBoomScore | Taiwan listed company monthly revenue, component basket | Monthly | Free public filings | Manual first | Medium | Requires basket definition and consistent constituents. |
| `pcb_revenue_yoy` | FalseDemandBoomScore | PCB company monthly revenue basket | Monthly | Free public filings | Manual first | Medium | Useful for false boom detection. |
| `mlcc_revenue_yoy` | FalseDemandBoomScore | Passive component / MLCC monthly revenue basket | Monthly | Free public filings | Manual first | Medium | Needs basket notes to avoid survivorship changes. |
| `driver_ic_revenue_yoy` | FalseDemandBoomScore | Driver IC company monthly revenue basket | Monthly | Free public filings | Manual first | Medium | Proxy for panel / consumer electronics pull-through. |
| `memory_revenue_yoy` | SupplyDistortionScore | Taiwan memory company monthly revenue basket | Monthly | Free public filings | Manual first | Medium | Captures memory cycle but not pure HBM mix. |
| `consumer_electronics_revenue_yoy` | DemandBreakRiskScore | PC / phone / consumer electronics proxy basket | Monthly | Free public filings | Manual first | Medium | Proxy for end demand and channel health. |
| `pc_shipments_yoy` | DemandBreakRiskScore | ODM / PC OEM revenue proxy if shipment data unavailable | Monthly | Free public filings | Manual first | Low-Medium | Revenue can include ASP effects; note proxy nature. |
| `smartphone_shipments_yoy` | DemandBreakRiskScore | Phone supply-chain revenue proxy | Monthly | Free public filings | Manual first | Low-Medium | Shipment proxy may lag true sell-through. |
| `inventory_days_components` | FalseDemandBoomScore | Quarterly financial statement inventory days proxy | Quarterly | Public filings | Manual first | Medium | Must be marked quarterly and lagged. |
| `accounts_receivable_growth` | FalseDemandBoomScore | Quarterly financial statement AR growth proxy | Quarterly | Public filings | Manual first | Medium | Useful for revenue quality checks. |

## P1: Semi-Public Or Requires More Cleaning

| indicator_name | subscore_used_by | source_candidate | frequency | availability | manual_or_auto | confidence | notes |
|---|---|---|---|---|---|---|
| `pc_shipments_yoy` | DemandBreakRiskScore | Public industry summaries / company shipment disclosures | Quarterly | Semi-public | Manual | Medium | Avoid unsourced articles as numeric source. |
| `smartphone_shipments_yoy` | DemandBreakRiskScore | Public industry summaries / company shipment disclosures | Quarterly | Semi-public | Manual | Medium | Complete vendor data may require paid sources. |
| `dram_price_proxy` | SupplyDistortionScore | Public DRAM spot summaries or manually maintained proxy | Weekly / Monthly | Semi-public | Manual | Low-Medium | Must document source and licensing. |
| `ddr5_price_proxy` | SupplyDistortionScore | Public DDR5 spot summaries or manually maintained proxy | Weekly / Monthly | Semi-public | Manual | Low-Medium | First version should accept manual proxy values. |
| `ai_server_revenue_yoy` | SupplyDistortionScore | AI server supplier revenue proxy basket | Monthly | Public filings with mapping work | Manual first | Medium | Needs basket governance. |
| `hbm_supply_pressure_proxy` | SupplyDistortionScore | Memory maker commentary / capex mix proxy | Quarterly | Semi-public | Manual | Low | Scenario-aware only unless source quality improves. |
| `cloud_capex_proxy` | SupplyDistortionScore | Hyperscaler capex filings | Quarterly | Public filings | Manual first | Medium | Not Taiwan-specific but relevant to AI demand pull. |

## P2: Harder Or Paid

| indicator_name | subscore_used_by | source_candidate | frequency | availability | manual_or_auto | confidence | notes |
|---|---|---|---|---|---|---|
| `hbm_asp` | SupplyDistortionScore | Specialist memory pricing providers | Monthly / Quarterly | Often paid | Manual only if licensed | High if licensed | Do not scrape or infer from headlines. |
| `dram_exchange_detailed_price` | SupplyDistortionScore | DRAMeXchange or similar detailed series | Daily / Weekly | Often paid | Manual only if licensed | High if licensed | Free summaries may not be enough for clean history. |
| `gartner_idc_complete_pc_shipments` | DemandBreakRiskScore | Gartner / IDC complete shipment dataset | Quarterly | Paid | Manual only if licensed | High if licensed | Do not use without license. |
| `company_inventory_by_segment` | FalseDemandBoomScore | Detailed company segment inventory data | Quarterly | Hard to compile | Manual | Medium-High | Useful but labor intensive. |

## Manual Ingestion Rules

- Raw manual files go under `data/raw/ai_bullwhip/`.
- Processed output is `data/processed/ai_bullwhip_indicators.csv`.
- Monthly data remains monthly.
- Quarterly data remains quarterly.
- Do not forward-fill to daily market dates in the builder.
- If daily alignment is needed later, output explicit `source_frequency`, `forward_filled`, `data_lag_days`, and `latest_available_date`.
- Missing fields remain NaN.
- No missing field is filled with zero.

## Current Priority

1. Monthly Taiwan component revenue baskets.
2. Quarterly inventory days and accounts receivable growth.
3. PC / phone end-demand proxies.
4. DRAM / DDR5 price proxies with documented source.
5. AI server revenue proxy basket.

## Guardrails

- Do not ingest paid data unless licensing is explicit.
- Do not use news articles as numeric time-series data.
- Do not tune data definitions using crash-window outcomes.
- Do not connect this module to HedgeNeedScore until data quality is audited.
- Do not create trades from AI bullwhip indicators directly.
