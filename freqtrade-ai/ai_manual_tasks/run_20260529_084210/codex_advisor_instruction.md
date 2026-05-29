你是 strategy_advisor。请读取本目录上下文文件，只输出 `generated_mutation_spec.json`，不要输出 Python 策略代码。

必须使用的上下文文件：
- `optimization_goal.snapshot.json`
- `current_best_summary.json`
- `nearest_candidate_summary.json`
- `last_run_summary.json`
- `strategy_lessons.json`
- `strategy_blacklist.json`
- `strategy_memory_excerpt.json`
- `leaderboard_recent.json`
- `pair_entry_tag_summary.json`

当前目标交易数：训练区间总交易数 25~80（不是单币种）。
当前历史 best 指标：`current_best_summary.json`，关键字段：{"total_trades": 26, "profit_total_abs": 0.3270239700000003, "profit_total": 0.00032702397000000026, "profit_total_pct": 0.03270239700000003, "profit_factor": 1.0584862288413162, "max_drawdown": 0.004564201220540596, "max_drawdown_pct": 0.4564201220540596, "winrate": 0.8461538461538461, "parsed": true, "roi_count": 19, "roi_profit_abs": 5.010045, "stop_loss_count": 4, "stop_loss_profit_abs": -5.59146959, "trailing_stop_loss_count": 3, "trailing_stop_loss_profit_abs": 0.90844856, "force_exit_count": 0, "force_exit_profit_abs": 0.0, "exit_signal_count": 0, "exit_signal_profit_abs": 0.0, "pairs": [{"key": "SOL/USDT", "trades": 7, "profit_mean": 0.005299027251842058, "profit_mean_pct": 0.53, "profit_total_abs": 1.85743962, "profit_total": 0.0018574396200000001, "profit_total_pct": 0.19, "duration_avg": "6:51:00", "wins": 7, "draws": 0, "losses": 0, "winrate": 1.0, "cagr": 0.028624382194727405, "expectancy": 0.2653485171428572, "expectancy_ratio": 100.0, "sortino": -100.0, "sharpe": 13.45009880959193, "calmar": -100.0, "sqn": 5.9125, "profit_factor": 0.0, "max_drawdown_account": 0.0, "max_drawdown_abs": 0.0}, {"key": "ETH/USDT", "trades": 5, "profit_mean": 0.004999221701330707, "profit_
nearest_candidate 指标：`nearest_candidate_summary.json`，关键字段：{"profit_total_abs": -2.9489720499999996, "profit_total_pct": -0.29489720499999994, "profit_factor": 0.47259445794464205, "max_drawdown_pct": 0.5031106831379292, "total_trades": 26, "roi_profit_abs": 2.54100102, "stop_loss_profit_abs": -5.59146959}
上轮失败模式：["交易数偏低"]
当前主要问题：["交易数偏低"]
禁止方向：["add_entry_filter"]
推荐 mutation_type：优先从 `add_entry_filter`, `tighten_entry_trigger`, `remove_bad_entry_condition`, `pair_specific_filter`, `tag_specific_filter`, `adjust_roi`, `adjust_stoploss`, `reduce_trade_frequency` 中选择一个。

硬性要求：
- 本轮只能做单点小步修改。
- 不允许生成策略代码。
- 不允许启用 exit_signal。
- 不允许做空、杠杆、加仓、马丁格尔。
- 不允许为了交易数而宽松堆叠 OR 造成高频。

只输出以下 JSON object，并写入 `generated_mutation_spec.json`：

```json
{
  "session_parent_choice": "historical_best | nearest_candidate | baseline",
  "session_parent_reason": "...",
  "mutation_type": "...",
  "goal": "...",
  "changes": [
    {
      "target": "entry_filter / roi / stoploss / pair_filter / tag_filter",
      "action": "...",
      "reason": "..."
    }
  ],
  "risk_controls": ["..."],
  "do_not_change": ["..."],
  "expected_effect": {
    "trade_count": "...",
    "profit_factor": "...",
    "drawdown": "...",
    "stoploss_loss": "..."
  }
}
```
