# Macro Raw Data Ingestion

Place macro factor CSV files in this directory:

- `data/raw/macro/macro_factors.csv`

The loader also accepts `macro.csv`. Files may be placed anywhere under `data/raw/`; the pipeline searches recursively.

## Required Column

- `date`

## Optional Macro Columns

Supply-chain distortion inputs:

- `hbm_asp_index`
- `ddr5_spot_index`
- `pc_shipments_yoy`
- `smartphone_shipments_yoy`
- `pc_sellthrough_yoy`
- `inventory_days_oem`
- `inventory_days_components`
- `pcb_revenue_yoy`
- `mlcc_revenue_yoy`
- `driver_ic_revenue_yoy`
- `unit_growth_yoy`
- `asp_growth_yoy`
- `ai_server_capex_yoy`
- `consumer_sentiment`

Macro demand fragility inputs:

- `cpi_yoy`
- `core_cpi_yoy`
- `ppi_yoy`
- `real_wage_growth_yoy`
- `consumer_confidence`
- `unemployment_rate`
- `policy_rate`
- `us10y_yield`
- `credit_card_delinquency`
- `oil_price_yoy`
- `usd_index`
- `retail_sales_yoy`

Relative strength and risk inputs:

- `sox_relative_strength`
- `tsmc_relative_strength`
- `memory_relative_strength`
- `pcb_relative_strength`
- `mlcc_relative_strength`
- `valuation_risk_index`
- `liquidity_stress_index`
- `event_flag`

Missing optional macro columns are allowed. Validation will warn when key diagnostics are unavailable. Missing values must not be backfilled with future information.

See `data/sample/macro_factors_raw_template.csv`.

## Build Process

Run:

```bash
python main.py --mode build_dataset --config config.py
```

By default, `build_dataset` will not overwrite existing `data/processed/*.csv`. To rebuild processed data from raw files, use:

```bash
python main.py --mode build_dataset --config config.py --overwrite
```

If the command reports `NO_RAW_DATA`, the output must be treated as: `This is not a real-data backtest.`

## Git Policy

Raw macro data and processed data should not be committed. Only this README and sample templates under `data/sample/` are intended to be tracked.
