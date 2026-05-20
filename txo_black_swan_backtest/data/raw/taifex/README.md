# TAIFEX Raw Data Ingestion

Place raw TAIFEX CSV files in this directory:

- `data/raw/taifex/tx_futures.csv`
- `data/raw/taifex/tx_vix.csv`
- `data/raw/taifex/txo_options.csv`

The loader also accepts common aliases such as `futures.csv`, `market.csv`, `vix.csv`, `taifex_vix.csv`, `options.csv`, and `taifex_options.csv`. Files may be placed anywhere under `data/raw/`; the pipeline searches recursively.

## Futures / Market Raw Columns

Required:

- `date`
- `tx_close`

Optional:

- `tx_open`
- `tx_high`
- `tx_low`
- `txf_close`
- `volume`

If `tx_open`, `tx_high`, or `tx_low` are missing, the processed dataset fills them from `tx_close`. If `txf_close` is missing, it may be mapped from `tx_close`.

See `data/sample/market_raw_template.csv`.

## VIX Raw Columns

Required:

- `date`
- `vix`

If VIX is missing from processed market data, validation will warn and the backtest data loader may use realized volatility as a proxy. Proxy VIX must be treated as a limitation in reports.

## Options Raw Columns

Required:

- `date`
- `expiry`
- `cp`
- `strike`
- `close`

Optional but strongly preferred:

- `bid`
- `ask`
- `volume`
- `open_interest`
- `iv`
- `delta`

If `bid` or `ask` are missing, downstream loading estimates them from `close` and sets `bid_ask_estimated=True`. If `iv` is missing, the loader attempts implied-vol inversion and sets `iv_estimated=True` when successful. If `delta` is missing, the loader estimates Black-Scholes delta and sets `delta_estimated=True` when successful. Contracts with failed IV or delta estimates are not tradable.

See `data/sample/options_raw_template.csv`.

## Build Process

Run:

```bash
python main.py --mode build_dataset --config config.py
```

By default, `build_dataset` will not overwrite existing `data/processed/*.csv`. To rebuild processed data from raw files, use:

```bash
python main.py --mode build_dataset --config config.py --overwrite
```

The command reports one or more explicit states:

- `FOUND_RAW_DATA`
- `NO_RAW_DATA`
- `USING_SAMPLE_DATA`
- `USING_EXISTING_PROCESSED_DATA`

If the status is `NO_RAW_DATA`, the output must be treated as: `This is not a real-data backtest.`

## Git Policy

Raw TAIFEX data and processed data should not be committed. Only this README and sample templates under `data/sample/` are intended to be tracked.
