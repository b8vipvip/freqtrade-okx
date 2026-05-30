# freqtrade-okx

一个面向 **OKX 现货、Freqtrade Docker、AI 策略分析与自动迭代** 的量化交易项目。项目默认以 dry-run（模拟盘）为安全运行模式，包含基础回测、AI 读取回测结果、AI 生成策略、自动多轮优化、防过拟合验证、手动 AI 任务包、运行日志归档与部署更新脚本。

> **重要声明**：本项目不构成投资建议。所有策略、回测、AI 分析结果都可能失真或过拟合。请先使用 dry-run 和小规模验证，充分确认风险后再考虑任何实盘操作。

## 目录

- [项目结构](#项目结构)
- [核心能力](#核心能力)
- [安全约束](#安全约束)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [常用 Freqtrade 命令](#常用-freqtrade-命令)
- [AI 回测分析与策略生成](#ai-回测分析与策略生成)
- [自动策略优化循环](#自动策略优化循环)
- [手动 AI 任务包模式](#手动-ai-任务包模式)
- [运行日志与独立日志仓库](#运行日志与独立日志仓库)
- [策略文件说明](#策略文件说明)
- [部署更新脚本](#部署更新脚本)
- [测试](#测试)
- [常见问题](#常见问题)

## 项目结构

```text
.
├── README.md
├── deploy_freqtrade_ai.sh
└── freqtrade-ai/
    ├── README.md
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .env.example
    ├── ai_tools/
    │   ├── analyze_backtest.py
    │   ├── generate_strategy.py
    │   ├── run_ai_cycle.py
    │   ├── auto_optimize_strategy.py
    │   ├── utils.py
    │   ├── model_config.example.json
    │   ├── model_config.json
    │   ├── optimization_goal.example.json
    │   ├── optimization_goal.json
    │   ├── strategy_blacklist.example.json
    │   ├── strategy_lessons.example.json
    │   └── strategy_memory.example.json
    ├── tests/
    │   └── test_auto_optimize_strategy_round_summary.py
    └── user_data/
        ├── config.example.json
        └── strategies/
            ├── SampleStrategy.py
            ├── AI_Generated_Strategy.py
            ├── BTC_Only_AI_Strategy.py
            └── MultiCoin_AI_Strategy.py
```

## 核心能力

1. **Freqtrade Docker 运行环境**
   - 使用 `freqtradeorg/freqtrade:stable` 镜像。
   - 通过 `docker compose` 运行下载数据、回测和 dry-run 模拟盘。
   - `user_data/` 挂载到容器内 `/freqtrade/user_data`。

2. **OKX 现货 dry-run 默认配置**
   - 示例配置默认 `dry_run: true`。
   - 默认 `trading_mode: spot`，交易币种为 USDT。
   - 示例白名单包含 BTC/USDT、ETH/USDT、SOL/USDT。

3. **AI 回测分析**
   - `ai_tools/analyze_backtest.py` 读取 Freqtrade 回测结果 JSON 或最新 zip。
   - 提取收益、回撤、交易次数、胜率、profit factor、交易对表现、退出原因等核心字段。
   - 调用 OpenAI 兼容接口生成中文策略审查报告。

4. **AI 策略生成**
   - `ai_tools/generate_strategy.py` 根据 AI 分析结果生成完整 Freqtrade 策略。
   - 默认输出到 `user_data/strategies/AI_Generated_Strategy.py`。
   - 生成代码会校验类名、`IStrategy`、指标、入场与出场函数等必要内容。

5. **自动策略优化循环**
   - `ai_tools/auto_optimize_strategy.py` 支持多轮策略生成、回测、验证、best 判断、防过拟合检查。
   - 支持训练区间、多个验证区间、可选 holdout、随机采样窗口、早停、相似策略检查。
   - 支持自动下载历史数据、跳过下载或强制下载。

6. **多 Provider / 模型池切换**
   - 支持 OpenAI 兼容接口。
   - 支持为策略顾问和代码生成分别配置 provider pool。
   - AI 调用失败时可自动切换 provider / 模型。

7. **半自动 / 手动 AI 模式**
   - 可生成任务包，让外部 Codex/AI 工具离线生成策略。
   - 再将策略文件交回本地脚本执行回测和 best 判断。

8. **部署更新保护**
   - 根目录 `deploy_freqtrade_ai.sh` 会先备份本地运行数据，再检查代码修改，最后执行 `git fetch` 与 `git pull --ff-only`。

## 安全约束

请务必遵守以下原则：

- 默认只使用 dry-run，不要在未充分验证前切换实盘。
- 不要提交 `.env`、真实交易所 API Key、真实 OpenAI/API 中转站 Key。
- OKX API 不应开启提现权限。
- 策略应保持现货、只做多、无杠杆、无马丁格尔、无无限补仓。
- 不要在策略实时交易函数中调用 OpenAI 或其他外部 AI API。
- AI 生成内容只能作为辅助，不能替代人工风控审查。

## 环境要求

- Docker Desktop 或 Docker Engine + Docker Compose
- Python 3.10+
- 可访问的 OpenAI 或 OpenAI 兼容 API
- OKX API Key（dry-run 下载/模拟盘按需配置）

Python 依赖位于：

```bash
freqtrade-ai/requirements.txt
```

安装方式：

```bash
cd freqtrade-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

### 1. 进入项目目录

```bash
cd freqtrade-ai
```

### 2. 复制环境与 Freqtrade 配置

```bash
cp .env.example .env
cp user_data/config.example.json user_data/config.json
```

### 3. 编辑 `.env`

至少填写：

```env
OPENAI_API_KEY=你的OpenAI或中转站Key
OPENAI_BASE_URL=https://your-proxy-domain/v1
OPENAI_MODEL=gpt-5.5
```

如果使用官方 OpenAI 接口，可将 `OPENAI_BASE_URL` 留空或删除。

### 4. 编辑 `user_data/config.json`

在 `exchange` 中填写 OKX 信息：

```json
{
  "exchange": {
    "name": "okx",
    "key": "你的OKX_API_Key",
    "secret": "你的OKX_API_Secret",
    "password": "你的OKX_API_Passphrase"
  }
}
```

> Freqtrade 的 `config.json` 通常不会自动读取 `.env` 中的 OKX 字段，因此建议手动填写到本地 `user_data/config.json`。不要提交这个文件中的真实密钥。

## 配置说明

### `.env.example`

主要变量：

| 变量 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI 或 OpenAI 兼容服务 API Key |
| `OPENAI_BASE_URL` | API 中转站或兼容服务地址，官方接口可留空 |
| `OPENAI_MODEL` | 基础分析/生成脚本使用的模型名 |
| `BACKTEST_FILE` | 首选回测结果 JSON 路径 |
| `AI_ANALYSIS_FILE` | AI 分析报告输出路径 |
| `AI_STRATEGY_FILE` | AI 生成策略输出路径 |
| `AI_MODEL_SWITCH_ON_ERROR` | AI 调用失败时是否自动切换模型/provider |
| `AI_MODEL_MAX_ATTEMPTS_PER_CALL` | 单次 AI 调用最多尝试次数 |
| `AI_MODEL_TIMEOUT_SECONDS` | AI 请求超时时间 |
| `STRATEGY_ADVISOR_PROVIDER_POOL` | 策略顾问角色 provider 池 |
| `STRATEGY_CODEGEN_PROVIDER_POOL` | 代码生成/修复角色 provider 池 |

Provider 池变量命名规则：

```env
STRATEGY_ADVISOR_PROVIDER_POOL=apihost_claude_opus47,deepseek_official

AI_PROVIDER_APIHOST_CLAUDE_OPUS47_BASE_URL=https://example.com/v1
AI_PROVIDER_APIHOST_CLAUDE_OPUS47_API_KEY=你的key
AI_PROVIDER_APIHOST_CLAUDE_OPUS47_MODEL=claude-opus-4-7
AI_PROVIDER_APIHOST_CLAUDE_OPUS47_TYPE=openai_compatible
```

脚本会将 provider 名称转换成大写并替换非字母数字字符，再加上 `AI_PROVIDER_` 前缀。

### `ai_tools/model_config.json`

定义不同 AI 角色的默认 provider、环境变量和模型池：

- `strategy_advisor`：策略顾问，负责分析、提出变异建议。
- `code_generator`：代码生成，负责输出策略文件。
- `code_repair`：代码修复，负责修复不可运行或不符合要求的策略。

### `ai_tools/optimization_goal.json`

自动优化的核心目标文件。重点字段：

| 字段 | 说明 |
| --- | --- |
| `strategy_family` | 策略家族名，默认 `MultiCoin_AI_Strategy` |
| `config` | Freqtrade 配置路径，例如 `user_data/config.5coins.json` |
| `timeframe` | 回测周期，例如 `5m` |
| `timerange` | 默认训练回测区间 |
| `pairs` | 优化关注的交易对 |
| `data_download` | 自动下载历史数据的交易所、周期和区间 |
| `train_period` | 训练区间定义 |
| `validation_periods` | 多个验证区间，用于防过拟合 |
| `overfit_guard` | 防过拟合约束 |
| `target` | 收益、回撤、交易次数、profit factor 等目标 |
| `constraints` | 现货、只做多、无杠杆、无马丁等硬约束 |
| `baseline` | 当前基线表现 |
| `language` | 输出语言，建议保持 `zh-CN` |

> 注意：当前 `optimization_goal.json` 默认引用 `user_data/config.5coins.json`。仓库只提供 `user_data/config.example.json`，如需运行 5 币种优化，请自行复制/创建 `user_data/config.5coins.json` 并配置对应白名单与密钥。

## 常用 Freqtrade 命令

以下命令均在 `freqtrade-ai/` 目录下执行。

### 下载历史数据

```bash
docker compose run --rm freqtrade download-data \
  --config user_data/config.json \
  --exchange okx \
  --timeframes 5m \
  --timerange 20240101-20260501
```

如自动优化需要 5m 与 1h：

```bash
docker compose run --rm freqtrade download-data \
  --config user_data/config.json \
  --exchange okx \
  --timeframes 5m 1h \
  --timerange 20260101-20260525
```

### 回测示例策略

```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy SampleStrategy \
  --timeframe 5m \
  --timerange 20240101-20260501 \
  --export trades \
  --export-filename user_data/backtest_results/backtest-result.json
```

### 回测 AI 生成策略

```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy AI_Generated_Strategy \
  --timeframe 5m \
  --timerange 20240101-20260501 \
  --export trades \
  --export-filename user_data/backtest_results/backtest-ai-result.json
```

### 启动 dry-run 模拟盘

```bash
docker compose up -d
```

### 查看日志

```bash
docker compose logs -f freqtrade
```

### 停止服务

```bash
docker compose down
```

## AI 回测分析与策略生成

### 分析回测结果

```bash
python ai_tools/analyze_backtest.py
```

可指定输入/输出：

```bash
python ai_tools/analyze_backtest.py \
  --input user_data/backtest_results/backtest-result.json \
  --output user_data/backtest_results/ai-analysis.txt
```

读取规则：

1. 优先读取 `.env` 中 `BACKTEST_FILE` 指向的 JSON。
2. 如果 JSON 不存在，自动查找 `user_data/backtest_results/backtest-result-*.zip` 中最新的 zip。
3. 从 zip 内读取第一个非 `.meta.json` 的 JSON 回测结果。

### 生成策略

```bash
python ai_tools/generate_strategy.py
```

可指定分析文件与输出策略：

```bash
python ai_tools/generate_strategy.py \
  --analysis user_data/backtest_results/ai-analysis.txt \
  --output user_data/strategies/AI_Generated_Strategy.py
```

### 一键 AI 循环

```bash
python ai_tools/run_ai_cycle.py
```

该命令会依次执行：

1. `analyze_backtest()`
2. `generate_strategy()`

生成后仍需手动运行 Freqtrade 回测验证结果。

## 自动策略优化循环

自动优化入口：

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5
```

### 推荐全自动运行

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 10 \
  --no-wizard \
  --auto-approve
```

### 跳过或强制下载历史数据

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --skip-download
```

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --force-download
```

### 增加 AI 超时时间

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --ai-timeout 600
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--goal` | 目标 JSON 文件路径 |
| `--iterations` | 最大优化轮数 |
| `--auto-approve` | 风险提示自动继续 |
| `--no-wizard` | 不弹出中文交互向导，完全使用 JSON/CLI 参数 |
| `--save-goal` | 保存向导修改后的目标文件 |
| `--config` | 覆盖 Freqtrade 配置路径 |
| `--base-strategy` | 基础策略名 |
| `--timeframe` | 覆盖 timeframe |
| `--timerange` | 覆盖训练回测区间 |
| `--skip-download` | 跳过自动下载历史数据 |
| `--force-download` | 强制重新下载历史数据 |
| `--reset-best` | 重置历史 best |
| `--force-session-best` | 更偏向本次 session 内 best 判断 |
| `--allow-near-min-trades-best` | 允许接近最小交易次数且验证较强的策略成为 best |
| `--random-sample-windows` | 额外随机采样窗口数量 |
| `--early-stop-patience` | 连续无新 best 的早停轮数 |
| `--early-stop-final-score-failures` | 连续 final_score <= 0 的早停轮数 |
| `--early-stop-duplicate-strategies` | 连续重复策略早停轮数 |
| `--print-prompt-files` | 结束时打印 Prompt 审计文件 |
| `--no-log-file` | 不保存完整终端日志 |
| `--allow-concurrent-runs` | 允许并发运行；默认禁止以避免回测 zip 污染 |

### 随机采样验证

启用随机采样时必须提供数据起止日期：

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --random-sample-windows 3 \
  --random-sample-data-start 20260101 \
  --random-sample-data-end 20260601
```

### 中文交互确认

未开启 `--auto-approve` 时，如脚本触发风险提示，通常可输入：

- `y`：继续
- `n`：停止
- `s`：跳过当前策略
- `b`：查看当前最佳策略
- `p`：将当前最佳策略提升为正式策略

## 手动 AI 任务包模式

当你不希望脚本直接调用 AI，或想使用外部 Codex/AI 环境生成策略时，可使用手动模式。

### 生成任务包

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --manual-ai-prepare \
  --manual-task-name run_custom_001 \
  --no-wizard
```

该模式只准备任务上下文，不调用 AI，不回测新策略。

### 运行外部生成的策略

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --manual-ai-run user_data/strategies/generated/YourStrategy.py \
  --manual-ai-task-dir user_data/ai_manual_tasks/run_custom_001 \
  --no-wizard
```

### 手动任务 Git 推送

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --manual-ai-prepare \
  --manual-git-push \
  --manual-git-branch ai-manual/run_custom_001 \
  --no-wizard
```

## 运行日志与独立日志仓库

自动优化脚本默认可将终端日志保存到 `user_data/logs`，并支持将日志推送到独立日志仓库。

常用参数：

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --log-dir user_data/logs \
  --push-logs-to-git \
  --log-repo-path /path/to/log-repo
```

如不需要日志文件：

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --no-log-file
```

## 策略文件说明

| 文件 | 说明 |
| --- | --- |
| `SampleStrategy.py` | Freqtrade 示例/基础策略 |
| `AI_Generated_Strategy.py` | 基础 AI 生成策略默认输出目标 |
| `BTC_Only_AI_Strategy.py` | BTC/USDT 专用现货只做多策略 |
| `MultiCoin_AI_Strategy.py` | 多币种现货只做多策略，配合自动优化目标使用 |

策略通用原则：

- 继承 `IStrategy`。
- 使用 `timeframe = "5m"` 为主周期。
- 现货只做多，`can_short = False`。
- 使用 TA-Lib / pandas 指标，不使用未来函数。
- 不在策略运行逻辑中调用 AI API。

## 部署更新脚本

根目录提供：

```bash
./deploy_freqtrade_ai.sh
```

默认变量：

```bash
REPO_DIR=/opt/freqtrade-ai
BACKUP_ROOT=/opt/freqtrade-ai-local-backups
```

脚本行为：

1. 进入 `$REPO_DIR`。
2. 备份本地运行数据，例如 `.env`、`user_data/ai_memory`、本地配置、生成策略等。
3. 检查代码文件是否存在本地修改。
4. 如无代码修改，执行：

```bash
git fetch --all --prune
git pull --ff-only
```

自定义路径示例：

```bash
REPO_DIR=/opt/freqtrade-ai \
BACKUP_ROOT=/opt/freqtrade-ai-local-backups \
./deploy_freqtrade_ai.sh
```

## 测试

项目包含自动优化 round summary 相关单元测试：

```bash
cd freqtrade-ai
python -m pytest tests/test_auto_optimize_strategy_round_summary.py
```

也可以先做语法编译检查：

```bash
python -m compileall ai_tools tests user_data/strategies
```

## 常见问题

### 1. 找不到 `user_data/config.json`

请先复制示例配置：

```bash
cd freqtrade-ai
cp user_data/config.example.json user_data/config.json
```

### 2. `optimization_goal.json` 引用了 `user_data/config.5coins.json`，但文件不存在

仓库没有提供该运行时配置。你可以：

```bash
cp user_data/config.example.json user_data/config.5coins.json
```

然后修改 `pair_whitelist`、OKX API 和其他参数。

### 3. 找不到回测结果 JSON

先执行一次回测，并导出结果：

```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy SampleStrategy \
  --timeframe 5m \
  --timerange 20240101-20260501 \
  --export trades \
  --export-filename user_data/backtest_results/backtest-result.json
```

新版 Freqtrade 也可能生成 `backtest-result-*.zip`，分析脚本会自动查找最新 zip。

### 4. OpenAI API Key 未配置

检查 `.env`：

```env
OPENAI_API_KEY=你的key
OPENAI_MODEL=你的模型名
```

如果使用中转站，还需要：

```env
OPENAI_BASE_URL=https://你的中转站域名/v1
```

### 5. Freqtrade 找不到策略

确认策略文件位于：

```text
freqtrade-ai/user_data/strategies/
```

并且策略类名与命令中的 `--strategy` 参数完全一致。

### 6. OKX API 权限错误

检查：

- API Key、Secret、Passphrase 是否正确。
- 是否启用交易权限。
- 是否误开启提现权限（不建议）。
- IP 白名单是否包含运行机器出口 IP。
- `exchange.name` 是否为 `okx`。

### 7. AI 生成策略无法回测

可检查：

- 类名是否正确。
- 是否继承 `IStrategy`。
- 是否存在 `populate_indicators`、`populate_entry_trend`、`populate_exit_trend`。
- 是否使用了 Freqtrade 当前版本不支持的字段。
- 是否导入了项目环境中不存在的依赖。

### 8. 自动优化并发运行被拒绝

脚本默认使用锁文件避免多个优化任务同时生成/读取回测 zip，防止结果污染。如确认没有任务在运行，可按脚本提示删除锁文件，或显式使用：

```bash
python3 ai_tools/auto_optimize_strategy.py \
  --goal ai_tools/optimization_goal.json \
  --iterations 5 \
  --allow-concurrent-runs
```

不建议在同一 `user_data/backtest_results` 下并发运行多个优化任务。

## 推荐工作流

```text
复制配置 -> 填写 .env 与 OKX dry-run 配置 -> 下载数据 -> 回测基线策略
       -> AI 分析回测 -> 生成策略 -> 回测 AI 策略
       -> 配置 optimization_goal.json -> 多轮自动优化
       -> 人工审查 best 策略 -> dry-run 长时间观察 -> 再决定是否实盘
```
