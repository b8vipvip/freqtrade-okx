# Freqtrade + OKX + AI 策略迭代项目

## 1. 项目说明
本项目用于构建一个可 Docker 部署的 Freqtrade 工作流：

Freqtrade + Docker → OKX API → dry-run 模拟盘 → 导出 backtest-result.json → Python 脚本读取回测结果 → 调用 OpenAI API 分析 → 生成新的 Strategy.py → 重新回测。

## 2. 安全提醒
- 默认仅 dry-run（模拟盘），请勿在未充分验证前切换实盘。
- 不要上传 `.env` 到 GitHub。
- OKX API 请勿开启提现权限。
- AI 只负责分析和生成策略，不直接控制实盘交易。

## 3. 安装 Docker
请先安装 Docker Desktop（Windows/macOS）或 Docker Engine + Docker Compose（Linux）。

## 4. 初始化项目
```bash
cd freqtrade-ai
```

## 5. 复制配置文件
```bash
cp .env.example .env
cp user_data/config.example.json user_data/config.json
```

## 6. 修改 `.env`
填写 OpenAI API Key 与路径配置。

## 7. 修改 `user_data/config.json`，填入 OKX API
由于 Freqtrade 的 `config.json` 通常不会自动读取 `.env` 里的 OKX 字段，建议手动填写：
- `exchange.key`
- `exchange.secret`
- `exchange.password`

## 8. 下载历史数据命令
```bash
docker compose run --rm freqtrade download-data \
  --config user_data/config.json \
  --exchange okx \
  --timeframes 5m \
  --timerange 20240101-20260501
```

## 9. 回测 SampleStrategy 命令
```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy SampleStrategy \
  --timeframe 5m \
  --timerange 20240101-20260501 \
  --export trades \
  --export-filename user_data/backtest_results/backtest-result.json
```

## 10. AI 分析命令
```bash
python ai_tools/analyze_backtest.py
```

## 11. AI 生成策略命令
```bash
python ai_tools/generate_strategy.py
```

## 12. 一键 AI 循环命令
```bash
python ai_tools/run_ai_cycle.py
```

## 13. 回测 AI 策略命令
```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy AI_Generated_Strategy \
  --timeframe 5m \
  --timerange 20240101-20260501 \
  --export trades \
  --export-filename user_data/backtest_results/backtest-ai-result.json
```

## 14. 启动 dry-run 模拟盘
```bash
docker compose up -d
```

## 15. 查看日志
```bash
docker compose logs -f freqtrade
```

## 16. 停止
```bash
docker compose down
```


## 使用 OpenAI 中转站 API
如果你使用的是兼容 OpenAI 的中转站，请在 `.env` 中配置：

```env
OPENAI_API_KEY=你的中转站Key
OPENAI_BASE_URL=https://你的中转站域名/v1
OPENAI_MODEL=你的中转站支持的模型名
```

说明：
- `OPENAI_BASE_URL` 留空时将直连官方 OpenAI 接口。
- `OPENAI_MODEL` 仍然通过 `.env` 配置，脚本会按该值调用模型。

## 新版 Freqtrade 回测结果说明
- 新版 Freqtrade 回测结果默认会写入 `user_data/backtest_results/` 下的 `backtest-result-*.zip`。
- `ai_tools/analyze_backtest.py` 会优先读取 `.env` 中 `BACKTEST_FILE` 指向的 `backtest-result.json`。
- 如果该文件不存在，脚本会自动查找最新的 `backtest-result-*.zip`，并读取其中第一个非 `.meta.json` 的 `.json` 回测结果。

## 17. 常见问题
- 找不到 `config.json`：请先执行复制命令并确认路径为 `user_data/config.json`。
- 找不到 `backtest-result.json`：请先完成一次 backtesting 并导出结果文件。
- OpenAI API Key 未配置：检查 `.env` 是否包含 `OPENAI_API_KEY`。
- 策略类名不匹配：确认策略文件中类名与命令 `--strategy` 参数一致。
- Freqtrade 找不到策略：确认策略文件位于 `user_data/strategies/`。
- OKX API 权限错误：检查 API 是否启用交易权限、IP 白名单与 passphrase 是否正确。

## 18. 自动策略优化循环（中文友好模式）

### 18.1 目标文件说明
- 实际运行目标文件：`ai_tools/optimization_goal.json`
- 示例模板文件：`ai_tools/optimization_goal.example.json`
- 建议先复制示例，再按你的需求修改实际目标文件。
- 中文模式请保持：`"language": "zh-CN"`。

### 18.2 如何编辑 `ai_tools/optimization_goal.json`
重点字段：
- `train_period.timerange`：训练区间（例如 `20260501-20260525`）
- `validation_periods[]`：验证区间列表（建议至少 2~3 段）
- `target.min_profit_total_pct`：最低目标收益率
- `target.max_drawdown_pct`：最大回撤限制
- `target.min_trades` / `target.max_trades`：交易次数上下限
- `overfit_guard`：防过拟合规则（训练验证收益差、验证 PF、验证回撤等）

### 18.3 推荐运行
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5
```

### 18.4 使用 `--auto-approve`
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --auto-approve
```
说明：遇到风险提示时自动继续，不再人工确认。

### 18.5 使用 `--skip-download`
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --skip-download
```
说明：跳过开头的历史数据下载步骤。

### 18.6 仅验证模式
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --validation-only
```

### 18.7 promote best 策略
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --promote-best
```
说明：会先备份 `user_data/strategies/MultiCoin_AI_Strategy.py`，再用最佳策略覆盖。

### 18.8 中文交互确认选项
当未开启 `--auto-approve` 且触发风险时，可输入：
- `y`：继续
- `n`：停止
- `s`：跳过当前策略
- `b`：查看当前最佳策略
- `p`：将当前最佳策略提升为正式策略



### 18.9 启动方式补充
普通交互式运行：
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5
```

完全使用 JSON，不弹出向导：
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --no-wizard
```

全自动不中途确认：
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 10 \
  --auto-approve
```

修改后保存到 JSON：
```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --save-goal
```
