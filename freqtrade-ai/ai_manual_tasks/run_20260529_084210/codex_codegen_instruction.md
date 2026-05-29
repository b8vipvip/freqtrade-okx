你是 code_generator。请读取：
- `generated_mutation_spec.json`
- `current_best_strategy.py`
- `nearest_candidate_strategy.py`
- `optimization_goal.snapshot.json`

然后只生成一个文件：`generated_strategy.py`。

要求：
- 必须是完整可运行 Freqtrade Strategy。
- 只能 long-only spot。
- `can_short = False`。
- `use_exit_signal = False`。
- 不主动使用 `populate_exit_trend` 产生 `exit_long` 信号；如果框架需要该函数，只返回 dataframe，不设置 exit_long=1。
- 不使用杠杆。
- 不使用加仓。
- 不使用马丁格尔。
- 不引入外部网络请求。
- 策略类名先可用占位：`Manual_AI_Generated_Strategy`。
- 服务器导入时会自动重命名为唯一类名。
- 只根据 mutation_spec 做单点小步修改，不要完全重写父策略。
- 输出完整 Python 文件内容，不要输出解释性文字。
