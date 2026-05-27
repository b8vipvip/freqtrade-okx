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
