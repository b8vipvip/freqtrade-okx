# 半自动 AI 策略生成任务包：run_20260529_094458

这是半自动策略生成任务包。
服务器不会在本步骤调用 AI。
请先执行 `codex_advisor_instruction.md`，生成 `generated_mutation_spec.json`。
再执行 `codex_codegen_instruction.md`，生成 `generated_strategy.py`。
生成策略后，服务器会拉取代码并执行回测。

## 推荐流程

1. 打开 Codex。
2. 让 Codex 读取本目录的 `codex_advisor_instruction.md`，只生成 `generated_mutation_spec.json`。
3. 再让 Codex 读取 `codex_codegen_instruction.md`，生成 `generated_strategy.py`。
4. 在服务器回到仓库执行：

```bash
git pull
python3 ai_tools/auto_optimize_strategy.py --goal ai_tools/optimization_goal.json --manual-ai-run ai_manual_tasks/run_20260529_094458/generated_strategy.py --manual-ai-task-dir ai_manual_tasks/run_20260529_094458
```

## Git 分支

建议分支：`ai-manual/run_20260529_094458`
