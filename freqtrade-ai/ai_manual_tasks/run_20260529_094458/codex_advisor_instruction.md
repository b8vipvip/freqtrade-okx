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
当前历史 best 指标：`current_best_summary.json`，关键字段：{"total_trades": 25, "profit_total_abs": -0.7226056899999994, "profit_total": -0.0007226056899999994, "profit_total_pct": -0.07226056899999994, "profit_factor": 0.9237648660953921, "max_drawdown": 0.007571279851625835, "max_drawdown_pct": 0.7571279851625835, "winrate": 0.8, "parsed": true, "roi_count": 19, "roi_profit_abs": 8.0638176, "stop_loss_count": 5, "stop_loss_profit_abs": -9.47864394, "trailing_stop_loss_count": 1, "trailing_stop_loss_profit_abs": 0.69222065, "force_exit_count": 0, "force_exit_profit_abs": 0.0, "exit_signal_count": 0, "exit_signal_profit_abs": 0.0, "pairs": [{"key": "SOL/USDT", "trades": 8, "profit_mean": 0.0034815346108549154, "profit_mean_pct": 0.35, "profit_total_abs": 1.39470155, "profit_total": 0.00139470155, "profit_total_pct": 0.14, "duration_avg": "16:07:00", "wins": 7, "draws": 0, "losses": 1, "winrate": 0.875, "cagr": 0.02142254562434598, "expectancy": 0.17433769375000002, "expectancy_ratio": 0.09214633820148221, "sortino": -100.0, "sharpe": 1.409058025384772, "calmar": 58.81058626161383, "sqn": 0.5854, "profit_factor": 1.7371707056118588, "max_drawdown_account": 0.0018878216628034732, "max_drawdown_abs": 1.89196551}, {"key": "DOGE/USDT", "trades"
nearest_candidate 指标：`nearest_candidate_summary.json`，关键字段：{"profit_total_abs": -6.3810277200000005, "profit_total_pct": -0.6381027720000001, "profit_factor": 0.4979915228436526, "max_drawdown_pct": 0.791739305281996, "total_trades": 55, "roi_profit_abs": 5.91531038, "stop_loss_profit_abs": -12.71099595}
上轮失败模式：["验证区间表现不稳定", "最差验证月份拖累整体表现", "固定止损吞噬 ROI"]
当前主要问题：["验证区间表现不稳定", "最差验证月份拖累整体表现", "固定止损吞噬 ROI"]
禁止方向：["add_entry_filter", "adjust_roi", "adjust_stoploss", "pair_specific_filter", "remove_bad_entry_condition", "tighten_entry_trigger"]
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
