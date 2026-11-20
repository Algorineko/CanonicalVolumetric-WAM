# a2 smoke 失败记录 (2026-11-20 ~16:2x)

现象:eval_policy_xpolicylab 连接成功(8849, ckpt=a2_s42/best.pt 已核实 smoke_ckpt_a2.txt),
但 5 个 episode 全部在第一次 action 前触发 env error:
`RuntimeError: Aborting task: 5 consecutive env errors before first action (simulator/renderer is likely wedged)`

判读:模拟器/渲染器卡死,**非策略结果,不计入 a2 成功率**。链条继续跑 a3/a3mask,结束后单独重跑 a2。
前一个 variant(a1)刚被 kill -9,可能是清理窗口/显存上下文问题;v2 脚本已等端口释放 15s。
