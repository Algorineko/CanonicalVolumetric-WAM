# FM 采样符号 bug(2026-11-20 16:5x 发现,未修复,未 commit)

## 事实(实测,repro: load a0_s42, clean_val+rand_val 前 6 批, 8 步 Euler, t=1-i/8)
- `act += dt*v_a`(现适配器/adapter 与 world_intervention.py 的写法):
  归一化单位 per-dim MSE **8.99**(垃圾,远离 GT)
- `act -= dt*v_a`(教科书 flow 方向):
  归一化单位 per-dim MSE **0.0024**(与离线 eval x0-jump 0.003-0.004 rad² 一致)
- 训练协议: x_t=(1-t)a+t·ε, v_target=ε−a(train_stage4a.py:346,350) ⇒ dx/dt=v,
  从 t=1(噪声)积分到 t=0(数据)必须 **x ← x − dt·v**。`+=` 是反向积分。

## 推论链
1. **当前** RoboTwin/XPolicyLab/policy/cvwam_fm/model.py(_infer_chunk)是 `+=`
   (动作与世界流都是,世界流是本会话 16:24 修复 noisy_world=None 时照搬
   world_intervention.py 引入)→ 当前源码闭环必然输出垃圾动作。
2. 但 a0 Clean-20 19/20、Randomized-20 19/20、smoke 5/5×2 是真实成功
   (垃圾动作不可能开微波炉)⇒ a0 筛查时段服务器运行的是**另一版(符号正确)代码**;
   之后源码被改为 `+=`(RoboTwin 非 git 仓库,无法追溯何时/如何翻转;
   __pycache__ 有 12:39 的旧 pyc)。**筛选结论本身仍成立,但必须用符号修正后的
   服务器重跑一次 a0 5-ep sanity 复核后方可继续引用。**
3. chain v3(a2/a3/a3mask,正在跑)全部用 `+=` ⇒ 其结果**无效**(预计 ~0/5),
   落盘后不得作为 variant 对比引用;恰好可作为"当前源码=坏"的闭环佐证
   (与 a0 19/20 形成对照)。
4. world_intervention.py euler_chunk 同样 `+=`(动作+世界)⇒ W0 基线动作是垃圾、
   W0 世界流演化脱流形。**L9 核心结论(存在性敏感/内容不敏感)不受影响**:
   W2=W3=W4(<1e-4)是 held-canvas 比较,不做积分,与符号无关;
   W1 是架构性阻断,亦无关。但 "W0=正常生成" 的表述与 W0-vs-W2 幅度对比
   被污染,需符号修正后重跑 W0 侧。
5. a1nofm 消融、offline eval(x0-jump,无积分)不受影响。a1 smoke 5/5 redo
   (16:18)用的是当时已坏的 `+=` 服务器?——时间线:a1 v2 smoke 16:06-16:18,
   model.py 16:24 才被我改(只加了 world 分支,`+=` 动作行原样保留)⇒
   a1 redo 也运行在坏符号下 ⇒ **a1 5/5 redo 同样无效**。同理 v1 链的 a1 4/5
   (stale a0 server)本来就无效。**唯二有效闭环数字:a0 的两次 20-ep 筛查与
   a0 smoke5(其服务器为旧正确代码)。**

## 需要的修复(1 行动作 + 1 行世界,adapter model.py::_infer_chunk)
- `act = act - (1.0 / self.n_euler) * o["v_a"]`
- `world = world - (1.0 / self.n_euler) * o["v_w"]`
- world_intervention.py euler_chunk 同改;改完先离线 sanity(复现 0.0024),
  再统一重跑全部闭环 smoke(a0/a1/a2/a3/a3mask)。**不要在 chain v3 运行中改文件**
  (会造成 a2=旧/a3=新的混合协议)。

## 状态(2026-11-20 17:0x 二次更新,autopilot 会话)——采样器假设反转

**决定性闭环实证:`-=` Euler(协议正确,离线 MSE 0.0043)闭环 0/3;`+=` 时代闭环 19/20×2 + 5/5×2。**
日志:`failures/rollout_smoke_a0_signfix_EULER_MINUS_0of3_INVALID.log`。

推论:**筛查时代(12:39 代码)的 serving 采样器不是 8 步 Euler 循环**,大概率是
单步 x0-jump(x0 = ε − v(ε, t=1),条件均值动作;对 open_microwave 这类刻板运动
足以成功)。16:24 的 world 分支编辑把采样器重写为 Euler 循环(照搬
world_intervention.py),之后的全部 Euler 闭环数字均不可比。本笔记前半部分的
"符号修复"结论只对**离线积分**成立;对 serving,采样器选择是独立且更早的问题。

处置(接管会话):
- wrapper `model.py` 增加 deploy 键 `sampler: x0jump|euler_minus|euler_plus`
  (x0jump 对 a2/a3 先去噪 world 再喂回取动作);A/B 实证裁决进行中
  (SAMPLER=x0jump a0 5-ep,`runs/stage4a/sanity_x0jump_a0.log`)。
- 同批加入 stage4b 压力注入:stress.drop_cam / occ_box / calib(cam 系失标)、
  trial 种子、voxel_manifest 键、[cvwam-stress-server] 配置打印(防静默 no-op)。
- euler_minus 的"协议正确但闭环失败"之谜(在线体素输入 vs 离线缓存分布差异?)
  列为 OPEN 问题;若 x0jump 复现 5/5,serving 采样器锁定 x0jump,谜题留档不阻塞。

## 三采样器实证矩阵(17:3x 定稿,接管会话)

| 采样器 | 离线 MSE(归一化,a0_s42,6 批) | 闭环 a0 open_microwave |
|---|---|---|
| euler_minus(协议 ODE) | **0.0043**(≈GT) | **0/3**(killed) |
| x0jump(单步 ε−v @ t=1) | 0.0076(≈GT) | **5/5 = 100%**(`rollout_results/a0_smoke5_rates_x0jump.txt`) |
| euler_plus | 8.9333(垃圾) | (v3 a2 1/2,存疑未复跑) |

解读:离线上 x0jump 与 euler_minus 都恢复 ≈GT 动作;闭环只有 x0jump 存活。
x0jump 从 (ε, t=1) 一步预测 = **条件均值动作**(平滑、保守),对 serving 输入的
轻微分布偏移鲁棒;euler_minus 沿路径迭代采样,放大 v 场误差。这与蒸馏/流模型
serving 中"单步预测比迭代采样更抗输入偏移"的经验一致。
**serving 采样器锁定 x0jump**(deploy 键可切换,默认即 x0jump)。OPEN 问题降级为
分析项:为何 euler_minus 离线好却闭环死(在线 x vs 缓存 x 的分布差?dump 诊断
`/data/cvwam_debug/online_x_baseline/`,见 stage4b)。

## 追加(16:55,协作会话接管后的新事实)
- v3 a2 在 `+=` 符号下出现 1/2 成功(16:35,seed 100001 成功)— 与"垃圾动作必然失败"
  的预期矛盾,原因未明(运气/成功判据宽松/其他),待全量落盘后再判。勿引用。
- 协作会话实测:**euler_minus(协议正确,离线 per-dim MSE 0.0043)闭环 0/3**;
  筛查时代(19/20×2)的采样器既非 euler_plus 也非 euler_minus,推测为
  单次 x0-jump(x0 = x_t − t·v̂,t=1 时等效条件均值动作,确定性)。
  model.py 已被协作会话重写为 sampler 可切换(deploy.yml 控制,默认 euler_minus)
  + stress 配置;适配器所有权归协作会话,本会话 16:24 的编辑已被取代。
- 结论修正:"当前源码=坏"表述过强,应为"当前源码的两种 euler 采样器闭环均与
  筛查时代不一致;筛查数字有效性待采样器考古/复跑确认"。
