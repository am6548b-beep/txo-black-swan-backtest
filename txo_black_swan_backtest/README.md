# 台指選擇權黑天鵝對沖 + 恐慌退散賣波動回測框架

這是一個保守、可驗證、可擴充的 Python MVP。它的第一目標不是找出最好看的績效，而是檢查台指選擇權策略能否降低黑天鵝期間的總資產最大回撤與爆倉風險。第二目標才是在恐慌退散後有限度收割波動率。

## 策略說明

狀態機：

1. `NORMAL`
2. `HEDGE_ON`：持有災難保險 Put Spread
3. `PANIC`：市場下跌，等待 Put Spread 停利或風控出場
4. `POST_PANIC`：等待 Iron Condor 條件
5. `SHORT_VOL_ON`：持有有限風險 Iron Condor

第一階段 Put Spread：

- 高檔、低 VIX、過熱時建立。
- 買入近價外 Put，賣出更遠價外 Put。
- 依年度避險預算與股票 beta 曝險限制口數。
- 到期前、停利、下跌幅度或殘值過低時出場。

第二階段 Iron Condor：

- 僅在 Put Spread 獲利後，或大跌後進入恐慌後狀態。
- 要求 20 日大跌、從低點反彈、VIX 跌破 5MA、近 5 日不再破低。
- Put 側較遠、Call 側略近，反映崩盤後 skew 與二次探底風險。
- 禁止單邊 rolling；虧損側必須先整組出場，再重新滿足進場條件。

## 資料格式

放在 `data/`：

- `market.csv`：`date, tx_close, tx_open, tx_high, tx_low, txf_close, volume, vix, event_flag`
- `options.csv`：`date, expiry, dte, cp, strike, close, bid, ask, volume, open_interest, iv, delta`
- `portfolio.csv` 可選：`date, stock_equity, portfolio_beta`

如果缺少 `vix`，系統會用 20 日 realized volatility proxy，並發出 warning。如果缺少 `bid/ask`，會用 `estimated_spread` 估算。若缺少 `delta`，會用 Black-Scholes 估算；若缺少 `iv`，會反推 IV，失敗則該合約不可交易。

## 如何執行

安裝依賴：

```bash
pip install -r requirements.txt
```

基本回測：

```bash
python main.py --config config.py
```

模式：

```bash
python main.py --mode put_spread_only
python main.py --mode iron_condor_only
python main.py --mode full
python main.py --mode stress
python main.py --mode walk_forward
```

## 真實資料接入流程

資料目錄：

- `data/raw/`：放真實原始 CSV，不進 git。
- `data/processed/`：pipeline 轉換後的標準格式，不進 git。
- `data/sample/`：小型格式範例，可進 git。

原始檔可使用下列檔名之一：

- 台指期 / market：`data/raw/tx_futures.csv`、`taifex_futures.csv`、`futures.csv`、`market.csv`
- 台指 VIX：`data/raw/tx_vix.csv`、`vix.csv`、`taifex_vix.csv`
- 台指選擇權：`data/raw/txo_options.csv`、`options.csv`、`taifex_options.csv`
- macro factors：`data/raw/macro_factors.csv`、`macro.csv`

建立 processed data：

```bash
python main.py --mode build_data --config config.py
```

輸出：

- `data/processed/market.csv`
- `data/processed/options.csv`
- `data/processed/macro_factors.csv`

檢查 processed data：

```bash
python main.py --mode validate_data --config config.py
```

回測資料來源順序：

1. 若 `data/processed/market.csv` 與 `data/processed/options.csv` 有有效資料，回測使用 `data/processed/`。
2. 若 processed 尚未建立，回測 fallback 到 `data/sample/`。
3. 可用 `--data-dir` 明確指定資料夾。

範例：

```bash
python main.py --mode put_spread_only --config config.py --data-dir data/processed
python main.py --mode put_spread_only --config config.py --data-dir data/sample
```

注意：pipeline 只負責欄位標準化，不會改策略規則，不會調參數，也不會把成交價改成 close。若原始選擇權缺少 bid/ask，後續 loader 仍會依保守估計 spread 補值，並在報告中保留資料品質風險。

測試：

```bash
pytest
```

## 如何解讀報告

輸出在 `reports/`：

- `summary.csv`：每組參數或模式的總表，重點看 `max_drawdown`、`minimum_free_cash`、`margin_usage_max`、`cost_drag`。
- `trades.csv`：每一腳成交紀錄，成交價使用 bid/ask 加滑價，不使用 close 當萬用成交價。
- `equity_curve.csv`：每日現金、股票、選擇權、總權益、保證金使用率。
- `regime_report.csv`：2008、2011、2015、2018、2020、2022、2024-2026 分段績效。
- `stress_report.csv`：六種假想壓力情境，加上黑天鵝延後 6/9/12 個月的保費拖累測試。
- `overfit_risk_report.csv`：粗網格鄰近參數穩定性標記。
- `charts/`：權益曲線、回撤、年度避險成本、Put Spread payoff、Iron Condor PnL 分布、保證金使用率。

請優先檢查：

- 策略是否降低總資產最大回撤。
- Put Spread 是否在崩盤期間補償股票虧損。
- Iron Condor 是否在假反彈時被反殺。
- 黑天鵝延後發生時，保費拖累是否可承受。
- 任何結果是否只靠單一年份或單一參數點。

## 避免過擬合原則

- 不自動搜尋最佳 Sharpe、CAGR 或總報酬。
- VIX 分位數只使用當日前資料。
- Walk-forward 固定為：2008-2015 calibration、2016-2019 validation、2020-2026 test。
- 參數只使用粗顆粒網格，不做細網格最佳化。
- 若單點參數表現很好但鄰近參數差，標記為 `overfit_risk = HIGH`。
- 不允許用 test 結果回頭調整參數。

## AI Supply Distortion & Bullwhip Risk

這不是單純 AI 泡沫模型，而是監控：

AI 高獲利資源虹吸導致 HBM 產能優先、傳統 DRAM 供給被擠壓，進而推升 DDR5 / DDR4、PCB、MLCC、Driver IC 等供應鏈恐慌性拉貨。若終端 PC / smartphone / 3C sell-through 無法承接高 ASP，假繁榮可能反轉為庫存與砍單長鞭效應。

新增 `data/macro_factors.csv` 支援：

- AI supply inputs：`hbm_asp_index`、`ddr5_spot_index`、`pcb_revenue_yoy`、`mlcc_revenue_yoy`、`driver_ic_revenue_yoy`、`inventory_days_oem`、`inventory_days_components`、`unit_growth_yoy`、`asp_growth_yoy`
- Demand inputs：`pc_shipments_yoy`、`smartphone_shipments_yoy`、`pc_sellthrough_yoy`、`consumer_sentiment`
- Stagflation amplifier：`cpi_yoy`、`core_cpi_yoy`、`ppi_yoy`、`real_wage_growth_yoy`、`consumer_confidence`、`unemployment_rate`、`policy_rate`、`us10y_yield`、`credit_card_delinquency`、`oil_price_yoy`、`retail_sales_yoy`

新增指標：

- `SupplyStressIndex`：0-100，監控 HBM/DDR5 價差、DDR5 加速、零組件營收加速、庫存背離、ASP 與 unit growth 背離、消費信心惡化、sell-through 轉弱。
- `MacroDemandFragilityIndex`：0-100，監控通膨、PPI、實質薪資、消費信心、失業率、利率、信用卡延滯、油價與零售銷售。
- `CombinedRiskScore = 0.4 * SupplyStressIndex + 0.3 * MacroDemandFragilityIndex + 0.2 * ValuationRiskIndex + 0.1 * LiquidityStressIndex`

新增 macro states：

- `AI_SUPPLY_DISTORTION`
- `BULLWHIP_COLLAPSE`
- `STAGFLATION_PRESSURE`
- `STAGFLATION_DEMAND_BREAK`

系統不預測崩盤時間，而是監控：

- 假繁榮
- 庫存異常
- ASP 與 unit growth 背離
- consumer demand weakening
- inventory shock risk
- AI bullwhip 與 stagflation 同時存在時的灰犀牛轉黑天鵝風險

新增報告：

- `reports/macro_dashboard.csv`
- `reports/macro_dashboard.md`
- `reports/report_audit.csv`

避免過擬合限制：

- 不用事後已知崩盤時間調整 `SupplyStressIndex` 門檻。
- 不用單一記憶體週期最佳化。
- 不用 2026-2027 假設反推參數。
- 僅使用 `LOW` / `MEDIUM` / `HIGH` / `EXTREME` 粗顆粒 regime。

## 已知限制

- 若缺乏真實 bid/ask，結果偏樂觀。
- 若缺乏完整夜盤資料，跳空風險低估。
- 若使用 Black-Scholes 估 Delta，不能完全反映台指選擇權 skew。
- 過去黑天鵝不代表未來黑天鵝。
- 回測只能驗證策略脆弱性，不能證明未來獲利。

## 開發備註

目前是 MVP，優先完成資料載入、無前視指標、合約選擇、Put Spread、Iron Condor、walk-forward、壓力測試與 pytest。後續接入真實期交所資料時，建議先強化：

- 夜盤與跳空成交模型。
- 真實保證金規則。
- skew-aware delta / IV surface。
- 可預知事件日資料來源與資料版本控管。
## Raw Data Ingestion Guide

Raw data should be placed under:

- `data/raw/taifex/` for TAIFEX futures, options, and VIX CSV files.
- `data/raw/macro/` for macro factor CSV files.

Sample raw templates are tracked under:

- `data/sample/market_raw_template.csv`
- `data/sample/options_raw_template.csv`
- `data/sample/macro_factors_raw_template.csv`

Raw and processed data are intentionally ignored by git. The sample templates and raw directory README files are allowed in git.

Accepted TAIFEX raw filenames include:

- `tx_futures.csv`, `taifex_futures.csv`, `futures.csv`, or `market.csv`
- `tx_vix.csv`, `vix.csv`, or `taifex_vix.csv`
- `txo_options.csv`, `options.csv`, or `taifex_options.csv`

Accepted macro raw filenames include:

- `macro_factors.csv`
- `macro.csv`

Required market fields are `date` and `tx_close`. Required option fields are `date`, `expiry`, `cp`, `strike`, and `close`. Required macro field is `date`.

Optional but important option fields are `bid`, `ask`, `iv`, and `delta`. If `bid` or `ask` are missing, downstream loading estimates bid/ask from `close` and marks `bid_ask_estimated=True`. If `iv` is missing, IV inversion is attempted and successful estimates are marked `iv_estimated=True`. If `delta` is missing, Black-Scholes delta is estimated and marked `delta_estimated=True`.

Build processed data with:

```bash
python main.py --mode build_dataset --config config.py
```

By default, `build_dataset` does not overwrite existing `data/processed/*.csv`. To rebuild processed files from raw data:

```bash
python main.py --mode build_dataset --config config.py --overwrite
```

The build output explicitly reports these states when applicable:

- `FOUND_RAW_DATA`
- `NO_RAW_DATA`
- `USING_SAMPLE_DATA`
- `USING_EXISTING_PROCESSED_DATA`

If the output status is `NO_RAW_DATA`, treat the result as: `This is not a real-data backtest.`
