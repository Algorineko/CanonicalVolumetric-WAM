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

## 追加 2(17:3x,协作会话 sanity 链实测,SAMPLER=x0jump)
- a0 + x0jump: **5/5** ⇒ 筛查时代(19/20×2)采样器=x0jump 假说与闭环实测吻合,
  a0 筛查数字有效性恢复。
- a2 + x0jump: **0/5** ⇒ a2 闭环问题候选(其离线 act MSE 与 a0 同级 0.003),
  与 L9/Gate "world→action 耦合失效" 方向一致;待 a3/a3mask 落盘后一并判读。
- 时间线澄清:a1 v2-smoke 5/5(16:06-18)运行于 16:24 我编辑之前的源码;
  我 16:24 Edit 的 old_string 含 `+=` 且匹配成功 ⇒ 16:24 时源码已是 `+=`;
  ⇒ `+=` 翻转发生在 16:18-16:24 之间(非我所为),16:06-18 的 a1 5/5
  运行的仍是翻转前的正确代码 ⇒ a1 redo 有效性恢复(待 sanity 链 a1 复跑确认)。
- v3 链(+=,16:30-16:35)a2 的 1/2 仍无法解释,孤例,存疑不引用。
- 本会话 v3 监控作废(旧文件名);监督权移交协作会话 sanity 链。

## 追加 3(17:2x-17:4x,autopilot 会话)——根因:盲策略(serving 输入为空)

**所有闭环数字(含筛查时代 19/20×2、a0/x0jump 5/5、a2/x0jump 0/5、baseline 20/20)
都是"盲策略"数字:serving 的在线体素 x=全零、qpos=全零(夹爪=1)。策略从未见过
观测,~100% 来自单任务无条件先验动作回放。**

证据链(`runs/stage4b/diagnose_online_x.py` + `/data/cvwam_debug/online_x_baseline/`):
- dump:三相机 depth 全有效(431-1028mm)、K/E 存在,但 x=0 非零体素、qpos=[0]×12+[1,1]。
- 离线复现:dump 的 depth/K/E 过 wrapper 建链 → 点云落 base x∈[1.33,2.17](bbox x≤1.1,
  0 部分命中);头相机算出 base (1.1,-0.032,1.35),真实锚点 (0.2,0.032,1.35)。
- 根因:eval client `convert_camera` 优先取 `extrinsic_cv`(sapien world→cam,即
  cam2world_gl 的逆)填 `extrinsics_matrix`,而契约(根 AGENTS.md)要求 cam2world_gl。
  位置列恰好 x,y 取反(逆矩阵平移列),与 embodiment 名义位姿 (-0.032,-0.45,1.35)
  对比即可判定。12:39 起的 wrapper 一直中招 ⇒ 筛查时代即盲。
- 修复(已提交):`convert_camera` 改为 cam2world_gl 优先;名义位姿过链验证
  锚点精确复现 (0.2, 0.032, 1.35)。
- 采样器矩阵重估:上表全部闭环列作废(空输入下测得)。x0jump"成功"= 先验回放;
  euler_minus"失败"= 先验 v 场迭代积分漂移。**修复后需重跑全部采样器 A/B、
  a0 基线与变体 smoke;P0 压力测试必须以修复后的 pipeline 运行**(否则扰动条件
  全部无效——盲策略根本不看相机)。
- 次要 bug:qpos 首调全零(arms drive target 未初始化;夹爪=1 正常)。每 4 步
  replan 中仅第一次受影响;后续 replan 读到真实 drive target。留观。

## 追加 3(17:5x,跨链冲突警报,已通报协作会话)
- sanity 链 a3 客户端(834457)17:29:45 起挂死(1 rate-line 后 20+ 分钟无更新),
  疑似 a3 路径(世界流+observer)在 x0jump 下 server 侧异常 → WS 无响应。
- 冲突:协作会话 stage4b 链(run_stress_chain.sh baselinefix, 端口 8850, 17:30 起)
  与 sanity 链共用 deploy.yml;17:27 起 ckpt_path=a3_s42 ⇒ 其 baselinefix
  可能实际加载 a3 权重(待其 server 日志 variant 行核实)。
- 危害:sanity 脚本清理用全局 pgrep kill,将在 a3 客户端退出时连带杀掉 8850 服务器,
  并把 deploy.yml sed 成 a3mask_s42 继续污染。已建议:ckpt 走 CLI overrides、
  按 PID 清理、杀掉挂死三件套(834457/805706/834298)后接管 a3/a3mask。
- 教训(机制层):多链并发时 ①deploy.yml 是单点竞争资源,ckpt 必须走 CLI;
  ②清理必须按 PID,禁止全局 pkill;③监控过滤器不得跨日志 grep 完成标记
  (本会话监控曾被旧 a0 日志的 SANITY_ALL_DONE 误触发提前退出)。

## 追加 4(17:36,误报更正 + a3 首个闭环成功)
- **"a3 挂死"系误报**:a3 客户端 17:27-17:32 正常跑完 5 集 = **5/5**
  (stdout 块缓冲导致 mtime/rate-line 数滞后,本会话误判 hung)。
  教训:判断客户端卡死要看 /proc/PID/io 增长或等真正超时,别只看日志 mtime。
- 闭环 x0jump smoke 现有:a0 **5/5**、a2 **0/5**、a3 **5/5**(a3mask 未跑,
  sanity 脚本在 a3mask turn 启动前被终止,deploy.yml 遗留 a3mask_s42)。
- **a2 0/5 vs a3 5/5(同协议同任务)= 两条流架构的首个闭环分化证据**:
  a2 联合栈离线动作 MSE 与 a0/a3 同级(~0.003)但闭环全败 ⇒ 其动作被
  闭环分布偏移下的 world token 污染;a3 的门控/近闭环 act←world 路径
  (a3mask 学习门 ~0.49 half-open,current_cross ~0.02)则鲁棒。
  注意:5 集为 smoke 规模,方向性证据非结论;a0 5/5 说明本任务无需世界流,
  与 L9 内容不敏感一致。
- **协作会话 baselinefix 无 ckpt 污染**(更正追加 3):其链用
  --overrides ckpt_name=runs/stage4a/a0_s42/best.pt 传参,server 日志
  variant=a0 sampler=x0jump 实证;deploy.yml 竞争只影响读它的 sanity 链。
  其链 [chain-done] baselinefix 已收尾(结果在其 rollout_baselinefix_p8850.log)。

## 追加 5(17:4x,终局:盲策略根因,协作会话发现与修复)
- **根因**:wrapper `convert_camera` 把 `extrinsic_cv`(逆矩阵)当 `cam2world_gl`
  喂入 → 在线体素全零 → **本文件此前讨论的所有闭环数字(含 a0 19/20×2 筛查、
  a2 0/5、a3 5/5)均为"无条件先验回放"**:策略从未见过真实观测,任务成功来自
  固定初始状态下开环先验 chunk 的可达性(注意 eval 配置打印 Random Light/
  Background/Table Height 全为 False/0,先验回放可高成功率,与 Randomized
  也 19/20 自洽)。修复后 dump 验证 x_nonzero=1.0、qpos 真实(协作会话实测)。
- 本文件结论降级说明:
  - 采样器考古(追加 1-2、4 的 euler 方向/x0jump 判别)仍成立——它们刻画的是
    盲模式下哪种确定性 chunk 被回放;
  - **追加 4 的"a2 0/5 vs a3 5/5 分化"降级为盲输入下的先验行为差异,方向性参考;
    world-token 耦合鲁棒性机制结论待修复后重跑**(协作会话已排期统一重测)。
- 附加确认:17:32-33 sanity 脚本清理段的全局 pkill 确实误杀了当时在跑的
  baselinefix server(追加 3 预警的风险实际发生)。后续纪律:per-chain deploy
  + 按 PID 清理;共享 deploy.yml 的 sed 仅在确认无并行链时使用。
- 现状:协作会话在跑修复输入下 baseline 20-trial(8850)+ euler_minus A/B(8851)
  + open_laptop 体素缓存 6 路 + cabinet 采集(300s SIGALRM 护栏)。
  a3mask x0jump 盲策略数字作废不补。本会话保持停机,不再触碰共享文件。

## 追加 6(17:58,autopilot 会话)——采样器矩阵定论(修复输入)

修复 extrinsics 后重测(`rollout_results/a0_smoke5_rates_euler_minus.txt`):
- **euler_minus a0 = 0/5**(修复输入下);x0jump a0 = 9/9+(20-trial 进行中)。
- **结论:serving 采样器 = x0jump**,与输入是否盲无关;euler_minus 在盲/真实两种
  输入下均闭环失败(尽管离线 MSE 0.0043≈GT)。矩阵:

| 采样器 | 离线 MSE | 盲输入闭环 | 修复输入闭环 |
|---|---|---|---|
| x0jump | 0.0076 | 5/5 | 9/9+ |
| euler_minus | 0.0043 | 0/3 | 0/5 |
| euler_plus | 8.9333 | (a2 1/2) | NOT_RUN |

- OPEN 分析项(数据在手):在线 x(实时 sim 观测)与缓存 x(导出管线)的分布差
  使迭代 ODE 漂移;`diagnose_online_x.py` 量化中。对部署含义:单步预测对此类
  分布差鲁棒;若 P2/P3 需要迭代采样,须先对齐 online/cache 通路。
- 同帧量化(17:5x):online chunk0 vs cached frame0 — 通道差 <2%,observed IoU 0.973,
  hit IoU 0.817 ⇒ serving/训练输入分布几乎一致,分布差假设排除。
  euler_minus 闭环失败的最佳解释改为**闭环协变量偏移**:模型自执行动作访问的状态
  超出专家状态分布,迭代 ODE 在偏移状态处失准并复合;单步条件均值(回归到均值)
  对此鲁棒——与模仿学习文献中 regression-vs-sampling 的鲁棒性差异一致。P2 若需
  迭代采样,考虑 DAgger 式状态覆盖或对采样轨迹加约束。

## 追加 7(20:0x,autopilot 会话)——部署计算图事实(§7 probe,只读调查)

**核心事实:v_a 读取的是"输入 noisy_world 的 token 编码",不是模型生成的 clean future。**

1. **a2**(models_stage4a.py:337-345):G_tok(当前体素)+ w_tok(noisy_world 编码)+ act_tok
   三段拼成 557 token 做 **2 层 full self-attention(无 cross-attn、无门)**;v_a 与 v_w
   是同一联合序列上的两个输出头。部署通路:当前体素+world 输入+qpos 全部经双向交互。
2. **a3/a3mask**(:346-368):世界流 wf = w_tok(noisy_world)→world_stack(2 块,
   self-attn + **对 act_tok 的 cross-attn;G_tok/当前体素从不进入世界流**);动作流
   act_h →action_stack(2 块,act←wf cross-attn,门≈0.49)。**v_w 只是同一批 wf 上的
   输出头,单次 forward 内不反馈进 v_a** ⇒ 部署单次调用时动作流只见噪声级 world tokens。
3. **plain a3 对当前体素零依赖**:v_a/v_w 完全由 (noisy_world, noisy_act, qpos, t, offset)
   决定(G_tok 计算后未使用);a3mask 当前体素仅经 current_cross(门 sigmoid≈0.019,
   三种子一致 [0.505,0.512,0.486,0.490,0.019] —— 都停在初始化附近)。
4. **W1 阻断 = 部署有效通路,但仅对 a3/a3mask**(action_stack 的 act←world 两块);
   **a2 的 block_world_to_action 标志在代码中从未被引用 = 对 a2 是空操作**,
   gate_note 里 a2 的 W1 偏差 0.053 实为"world 噪声重抽"伪影(W0/W1 各自
   randn 世界起始噪声)。a2 的 W1' (modality-embedding-only) docstring 未实现。
5. **x_current 是死参数**(签名有、体不用);a3mask act←current 读的是 x_t 的
   G_tok。cam72 部署恒 None → observer 头从未激活且**从未训练**(run_epoch 无
   observer 损失;--lam-obs 仅写入 meta)。
6. **部署 x0jump 两步调用 vs 训练分布**:第二次调用 (w0≈干净画布估计, t=1.0) 是
   训练未见组合(训练中画布干净度与 t 严格耦合);且 a3/a3mask 的 w0 不依赖当前
   观测(world_stack 不读 x)→ w0 至多是 qpos 条件的未来画布边缘分布。两次调用间
   act 不更新(第一次 v_a 丢弃、第二次 v_w 丢弃)。
7. **结论(供 §7 Two-pass 判断)**:two-pass 部署(Pass2 喂 w0)在 a3/a3mask 上
   动作流读到的仍是 world_stack 编码的 w0 token——结构上"读到了生成的未来",
   但该未来不含当前场景信息(除 a2)。**若要内容敏感,通路必须让当前体素进入
   world/future 表征(a2 结构)或动作流直接 cross-attn 未池化体素 patch(P2 候选 1)。**

## 追加 8(20:1x,autopilot 会话)——cam_t030 失败归因(P0 Failure Attribution,§3)

**方法**:ep10(在评测种子集内,seed 1000019)确定性重放,nominal vs shifted(0.30m,
dir_seed 1000 = cam_t030 同款)成对渲染,427 帧全对齐;43 帧配对分析
(`runs/stage4b/attribute_cam_t030.py`,结果 `metrics/attribution/att_{frames.csv,summary.json}`)。

| 指标 | 值 | 含义 |
|---|---|---|
| observed 格数 shift/nom | **2.07×**(15k→31k) | 偏移相机看得更多,不是更少 |
| hit IoU / observed IoU | 0.55 / 0.39 | 画布内容大幅不同 |
| 门区域观测率(shift) | **0.80** | **任务相关几何证据完好(§10.1:否)** |
| 同噪声动作 MSE(归一化) | 0.0025 | 单帧动作偏差 ~GT 量级 |
| FK EE 端差 | 2.4cm(最大 2.9cm) | 单帧小偏差 |
| 训练画布 observed 范围 | ~11-21k | **shift 31k 超上界 1.5×(分布外)** |

**归因(四类排除法)**:
- 感知信息缺失:**排除**(门 0.80、覆盖 2×);
- 部署错误:**排除**(同一 ckpt/管线在 nominal 下 20/20);
- 输入分布偏移:**成立**——画布统计量(observed 2×、hit 模式 IoU 0.55)远超训练范围;
- 闭环协变量偏移:**推定成立**——单帧动作偏差仅 2.4cm,但闭环 0/13 ⇒ 小偏差在
  自执行状态上复合(与 euler_minus 之谜同机制家族,见追加 6;直接证据采集见 §6 诊断)。

**对 P3 的含义(裁决 §10.6 前置)**:修"画布统计量鲁棒"可能不需要信念——B3(相机
增广)与输入归一化是第一梯队;B1/B2 解析融合**也会**改变画布统计量(融合后 observed
更多),**B1/B2 若在 cam_t030 恢复成功率,其机制可能不是"历史信息"而是"统计量回
到训练范围"——必须加 B3 对照剥离**(§5 的 current-only+增广强对照正是为此)。
- **cam_t030blind 终版(20:1x):5/5=100%**(偏移相机+零观测)vs cam_t030(偏移观测)0/13+。
  §10.2 定论:**偏移观测严格有害**——同条件下"不看"严格优于"看"。画布统计量 OOD
  下策略主动误动作;修复第一梯队=画布统计量鲁棒性(B3 相机增广/输入归一化),
  此结果同时是评估方法学的强论点("分布偏移下观测可以劣于盲目")。
