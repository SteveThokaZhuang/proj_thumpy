# DuplexGen / HumDial-FDBench 勘察（2026-09-23）

> **一句话**：DuplexGen **已经在盘上**（用户 2026-08-21/23 自下，657 GB 音频），
> HumDial-FDBench **不在盘上、需要下载**。
> 🔴 **但在 DuplexGen 上做任何 L1↔L3 的对比之前，必须先知道：已发布的两部分不共享文本，
> `example_id` 不是可用的 join 键**（§3，本次实测）。
>
> 上游：上个月周报结尾承诺的「验证 pipeline 扩展至 DuplexGen 及 HumDial-FD，
> 确立『元数据-音频失配』是合成数据共性缺陷还是 Behavior-SD 特有偏差」。

---

## 1. 为什么要看这两个

本项目的中心命题是「**全双工基准测的是它继承的标签，而不是它声称的行为**」，
证据链是 L1（元数据/生成器自采样）→ L2（声学派生）→ L3（人工）三级对照，
而**目前所有 L1 侧的证据都来自 Behavior-SD 一份数据**。

DuplexGen 与 HumDial-FDBench 各自补上一块：

| | 补什么 | 对应本项目的 |
|---|---|---|
| **DuplexGen** | 第二个**合成**数据集，且是**生成式**的（Behavior-SD 是拼接式） | 「是共性缺陷还是 Behavior-SD 特有」 |
| **HumDial-FDBench** | 第二个**真实人类**语料，且**自带官方交互场景标注** | CANDOR 之外的独立真实侧 |

---

## 2. 本地家底（2026-09-23 实测）

### 2.1 已有

```
/share/workspace3/shared_dataset/
├── duplexgen-corpus   62 M   2026-08-23 15:53   (owner: zhuangruicen)
├── duplexgen-spoken  657 G   2026-08-21 14:25   (owner: zhuangruicen)
├── switchboard        28 G   2026-08-23 17:51   (owner: zhuangruicen)
├── Full-Duplex-Bench 2.3 G   2026-06-29
└── behavior-sd       176 G   2026-08-20
```

来源（README 实测）：
- Hub：`DuplexGen/duplexgen-corpus`、`DuplexGen/duplexgen-spoken`；代码 `github.com/duplexgen/duplexgen-code`
- 论文：*DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues*（`arXiv:2607.26178`）
- 生成器 **Qwen3.5-122B-A10B**；音频由 **Chatterbox** TTS 渲染
- 六场景与上游（**每场景许可证不同**）：
  `TEA`=SocraticLM(Apache-2.0)、`PLN`=MultiWOZ(MIT)、`INT`=Anthropic Interviewer(CC-BY-4.0)、
  `NEG`=CraigslistBargain(MIT)、`PER`=DailyPersuasion(Apache-2.0)、`SOC`=SODA(CC-BY-4.0)

### 2.2 `duplexgen-corpus`（62 M，文本）

| 部分 | 实测 | README 声称 |
|---|---|---|
| `dialogues/<CODE>/train.jsonl` | INT 1000 行 / **1000 个唯一 id**（1 行 1 id） | 合计 **5,999**（PER 为 999，其余 1000） |
| `annotations/<CODE>/train.jsonl` | INT 20 行 | 每场景 20（合计 120） |
| `annotations/<CODE>/test.jsonl` | INT 50 行 | 每场景 50（合计 300） |

**`dialogues` 的 slot 结构**（L1 的直接来源）：

```json
{"word_index": int,
 "probs": {"floor_taking": f, "backchannel": f, "silence": f},
 "decision": "floor_taking|backchannel|silence",
 "inserted_token": str|null}
```

⇒ **生成器自己的软概率与它采样出来的决策都在**。全库 125,137 个决策槽（README 口径）。

**`annotations` 的 slot 结构**（L3 的直接来源）：

```json
{"word_index": int,
 "total_count": int,                          // 该槽的评分人数，实测 1–5
 "counts": {"silent": int, "backchannel": int, "take_floor": int},
 "probabilities": {...}}
```

INT test 实测：50 条 / **1,016 个 boundary**，`total_count` 分布
**{1:43, 2:98, 3:180, 4:250, 5:445}**，`counts` 键集合
`{backchannel,silent}` 575 / `{silent}` 273 / `{silent,backchannel,take_floor}` 98。

> 🔑 **这是本项目至今唯一一份「同一个 slot 多个真人投票」的数据**。
> F3 人工核验的硬限制正是「**单人标注、无 kappa**」（`2026-09-15_f3_human_verification_results.md`），
> 所以当时只敢写「还不能说必须改成三级标注」。**这份数据能把这个洞补上**，且**零 GPU**。

⚠️ **术语陷阱（README 明写，但极易踩）**：
`annotations` 用 `silent` / `take_floor`，`dialogues` 用 `silence` / `floor_taking`；
只有 `backchannel` 两边同名。**列名不同名 ⇒ 对着抄必错**（本项目的 `hardcoded-conclusions-escape-reproduction`
一族里「抄串一列」已经犯过）。

### 2.3 `duplexgen-spoken`（657 G，音频）

- `metadata.jsonl` **35,746 行**，字段：
  `key`（如 `INT/work_0000/var00`）、`scenario`、`license`、`example_id`、**`variant`**、
  `shard`、`duration`、`num_turns`、`n_utterances`、**`n_backchannels`**、`has_backchannels`、`n_files`
- 分片（`.tar`）：INT 208 / NEG 35 / PER 73 / PLN 45 / SOC 49 / TEA 57 = **467 个**
- **同一 `example_id` 有多个 `variant`**（前 4000 行里 628 个 id 对应 4000 个 (id,variant)）

🔑 **`n_backchannels` 是「元数据声称的 backchannel 数」** —— 这是本项目
「元数据 vs 音频实测」对照在 DuplexGen 上的现成接口。

---

## 3. 🔴 本次勘察最重要的发现：L1↔L3 在已发布文件上 **join 不通**

> ⚠️ **这一节是本次险些犯错的地方，必须留档。**

### 3.1 差点写出去的错误结论

第一次跑 join 时我拿到：

```
slot 级对齐: 人类 boundary 共 1016, 在生成对话里找到同 word_index 决策槽 124 (12.2%)
L1 (生成器自采样 decision): {'silence': 87, 'backchannel': 35, 'take_floor': 2}
L3 (人类多数票)          : {'silent': 109, 'backchannel': 12, 'take_floor': 3}
```

这张表**看着完全像一个结果**（"生成器比人类多报 3 倍 backchannel"），
**但它是废的**：那 12.2% 不是对齐，是**索引撞号**。

`word_index` 是「**角色内容内的词序号**」。内容不同 ⇒ 同一个序号指的不是同一个词。

### 3.2 实测的真相

| 检查 | 结果 |
|---|---|
| `annotations` 的 id 是否都在 `dialogues` 里 | ✅ **50/50 都在**（train 20/20） |
| 逐 turn `content` 是否相同 | ❌ **0/16, 0/13, 0/11, 0/10, 0/12, 0/14 … 全 0**（50 条全查） |
| 首轮文本能否在 1000 条 dialogues 里找到 | ❌ **0/50 命中** |
| 是不是 **id 错位**（文本在，但挂在别的 id 下） | ❌ **不是**，0 条错位 |
| 最长公共词块 | **6–9 词**（就是 "thanks for taking the" 这种普通短语）|

样例（同一个 `work_0500`）：
- `dialogues`：`Hi there! I'm Claude from Anthropic's research team. Thanks for taking the …`
- `annotations`：`Hey, thanks for chatting with me. We're just trying to learn how …`

**⇒ 同一个 `example_id` 装的是两段不同的对话。**

### 3.3 最可能的解释（**未证实**）

README 描述的流水线是「文本对话 → **转成口语风格转写** → 定位候选轮换槽 →
**用人类标注校准** → 应用校准做合成」。最可能是：
**人工标注做在「候选槽定位」那个中间产物上，而发布的 `dialogues` 是最终合成版**，
那个中间产物**没有发布**。

**⇒ 但这是推测，不能写进结论。** 待核实（§5.4）。

### 3.4 为什么这条本身就有价值

DuplexGen 的核心主张是「**human calibration 才是场景化轮换行为的来源**」。
但如果**被标注的文本与被发布的文本不是同一份**，第三方就**无法复核这个校准**。
这与本项目的主命题同源 —— **标签的来源不可追溯**。

⚠️ 表述纪律：只能说「**在已发布的这两个 part 上 join 不通**」，
**不能说「数据集有缺陷」** —— 缺的那份可能就在 `duplexgen-code` 里或 HF 的别的 revision 里。

### 3.5 方法论教训（进 claims_ledger 的候选）

> **join 之前必须先验证 join 键携带的是同一实体，不能只看键能不能对上。**
> `example_id` 对上了 50/50，但内容 0/50 —— **键命中率 100% 与实体同一性 0% 可以同时成立**。
> 这与 `hardcoded-conclusions-escape-reproduction`（先填表后核数）、
> `windowed-eval-check-gt-straddles-boundary`（诊断时打预测"时刻"而非计数）同族。
> **防线**：join 完先打一条「逐字段是否真的相等」的自检，**并且让它会炸**。

---

## 4. HumDial-FDBench 现状

**本地：不存在。** 搜索范围（2026-09-23 实测）：`/share/workspace`、`/share/workspace2`、
`/share/workspace3`、`/share/home/zhuangruicen` 四棵树**深度 3 以内**，`*humdial*` **零命中**
（同一次搜索下 `*duplexgen*` 命中 2 个，见 §2.1 ⇒ 搜索本身有效）。
⚠️ 深度 3 以外的路径未搜，**不排除更深层藏有副本**。

| 项 | 值 |
|---|---|
| 仓库 | `github.com/ASLP-lab/HumDial-FDBench` |
| 数据 | HF `ASLP-lab/HumDial-FDBench`（**只明确说 test 集可下**，5,000 条） |
| 出处 | **ICASSP 2026 HumDial Challenge 全双工交互赛道** |
| 数据性质 | **真实人类录音、双声道**，>100 小时，**中英双语** |
| 场景 | 两类 9 子场景：**Interruption**（追问 / 否定不满 / 请求重复 / 话题切换 / 沉默或停止）、**Rejection**（实时 backchannel / 停顿处理 / 第三方语音 / 对他说的话）|
| 划分 | train 8,898 / dev 1,800 / test 5,000（**不均匀**：Third-party Speech 训练集仅 120，Speech Directed to Others 训练集为 **0**）|
| 协议 | 建在 **Full-Duplex-Bench v1.5** 之上；榜单列 `Int. / Rej. / Delay(s) / D-Sco. / Final` |
| ⚠️ 缺口 | README **没有**给出逐 utterance/turn 的标注字段表、文件格式、音频布局 —— **要下下来才知道 L1 长什么样** |

🔑 **本地已有 Full-Duplex-Bench**（2.3 G，2026-06-29），而且**本项目已经把它审计过一遍**：
E2 实验（`2026-08-25_ari_e2_fdbench_audit.md`）查到 —— FD-Bench 的标签其实是 **L2**
（输入/输出两条流各自跑 Silero VAD + 区间规则，不是人工标注），且
**「同一套区间规则换片段来源，EIR 漂移 25 倍（0.018→0.449）」、10% 串扰即令指标集归零**
（`2026-08-28_story_report.md:29`、`2026-09-12_storyline_master.md:131`）。

**⇒ 这对 HumDial 是双重相关的**：① 它的评测协议**建在 FD-Bench v1.5 上**，而我们**已经知道那套指标脆**；
② 意味着 HumDial-FDBench 的「官方标签」很可能**也是 L2 而非 L3** —— 但**这是推测，
必须下下来核实它的标注字段表**（§4 表末已标：README 未给 schema）。

---

## 5. 现在能做什么

**约束**：GPU 被 C 的评估占着（job 72610，**12/14 落地**〔09-23 14:20 核〕，
剩 seed 55555 的两个，ETA 今晚 **~19:30–20:00**）。
所以下面按「要不要 GPU」分档。**5.1 已完成，5.2 是下一个。**

### 5.1 ✅ **已完成（2026-09-23）** —— DuplexGen 人类标注的**内部**一致性

> 📄 **结果全部在 → [2026-09-23_duplexgen_annotation_reliability.md](2026-09-23_duplexgen_annotation_reliability.md)**
> （本文只留指针；下面这段是最初的设想，实际做出来比设想多走了两步，
> 见该文档 §4 的槽级信号检验与 §5 的常数基线对照。）
>
> **三句话结论**：三分类 **κ = 0.0506 / AC1 = 0.3692**；
> **backchannel 的票不携带槽级信息（p = 1.0000），take_floor 携带（p = 0.0010）**；
> **一个不读上下文的逐场景常数预测器，KL 比论文报告的完整方法还低（0.115 vs 0.465，逐 turn）**。
>
> 🔴 **原设想里有两条被证伪**，见该文档：① 「`take_floor` ↔ `backchannel` 混淆矩阵」
> 没做成 —— 因为**没有 join 就没有配对**，逐槽只有边际计票，做不出混淆矩阵；
> ② 「test 300 条是主分析集」的隐含假设（train 只是附属）不成立 ——
> **train/test 的 κ 几乎一样**（0.0619 / 0.0456），主分析用的是**全量 4,639 槽**。

**这是最该先做的**，理由有三：① 不需要 GPU；② 直接补上 F3 缺的 κ；
③ **不依赖任何 join**（只用 `annotations/` 一个目录）。

- 每槽多人投票 ⇒ **Fleiss' κ**（`total_count` 1–5；**只用 `total_count ≥ 2` 的槽**，
  单标注者槽 43/1016 不能进 κ）
- **分歧率**：`counts` 极化的槽（如 5:0）vs 分裂的槽（如 3:2）各占多少
- **`take_floor` ↔ `backchannel` 混淆矩阵** —— 对应本项目「一个 yeah 跨 5 种功能」
- **场景间差异**：合作类（TEA/PLN/INT）vs 竞争类（NEG/PER）的 κ 与分歧率
- 规模：420 条 / 6 场景（test 300 条是主分析集）

### 5.2 ✅ 也要 0 GPU（只解码音频，CPU）—— 声学重叠检测

🔍 **2026-09-23 补充勘察（还没跑，但结构已查清，且比原设想更有利）**：

```
metadata.jsonl   35,746 行 / 467 个 tar 分片
字段: key scenario example_id variant shard duration num_turns
      n_utterances n_backchannels has_backchannels n_files
tar 内布局 (例 INT/work_0000/var00/):
  backchannels/01_0.wav 07_0.wav 09_0.wav 11_0.wav 19_0.wav 19_1.wav
  dialogues/dialogue.wav
  utterances/00.wav 01.wav 02.wav ... (逐 turn)
```

🔑 **`backchannels/NN_K.wav` 的命名 = 「第 NN 个 utterance 里的第 K 个 backchannel」**
（例：`19_0` 与 `19_1` = 同一个 utterance 里插了两次）。
⇒ **`n_backchannels` 是生成管线记下的「插入清单」，不是从音频里量出来的。**

**原本的假设**：「生成时插入 N 个片段」vs「混音后真的听得见这 N 个吗」是同一件事的两个记录，
可以直接对账 —— 预期这是 Behavior-SD 那个「元数据说 X、音频实测 Y」缺陷的**第二个实例**。

## 🔴 2026-09-23 探针结果：**假设被证伪了**（这一条很重要，是负结果）

仪器可行性探针见 `pilot_study/scripts/duplexgen_bc_probe.py`。**规则先定好再跑**：
每个场景取 tar 列表里**第一个**含 `dialogue.wav` 的目录（不按结果挑点）。

**口径 #1 的答案**：`dialogues/dialogue.wav` 与 `utterances/NN.wav` 都是**双声道**，
`backchannels/NN_K.wav` 是**单声道**。实测两声道相关 ≈ 0（+0.0000 ~ −0.0006）
⇒ **两个说话人各占一个声道，可以按声道分离**。这一条比原设想有利。

**试过但不可行的做法：做减法。** 若 `dialogue = Σ utterances + Σ backchannels`，
残差就该正好是 backchannel。但 utterances 在对话里的**时间位置未知**，从 t=0 顺序累加会让
残差能量爆到 backchannel 总能量的 **190 倍**（实测 78480 vs 408）—— 残差比 dialogue 本身还大。
**这是错位，不是"backchannel 不存在"。** ⇒ 减法这条路封死。

**可行的做法：归一化互相关（不需要对齐）。** 把每个 `backchannels/NN_K.wav` 拿去和
`dialogue.wav` 的每个声道做归一化互相关，看峰落在哪。

**结果（6 场景 × 各 1 个对话 = 20 个 backchannel）**：

| | L 声道峰 | R 声道峰 | 片段自身分布中位数 |
|---|---|---|---|
| **全部 20 个** | **1.0000** | 0.07 – 0.39 | 0.007 – 0.016 |

- **20/20 全部在 L 声道给出 1.0000** —— 逐样本精确相关，不是"听得出"，是**原样混入**。
- 恢复出的插入时刻与 `NN_K` 命名**严格单调一致**（TEA `09_0@51.72 / 09_1@54.30 / 09_2@55.76`；
  INT `19_0@183.20 / 19_1@184.44`）⇒ **命名约定被音频独立证实**。

**为什么"全 pan 到 L"不是伪影**（这一条差点被误读成发现）：
`utterances/NN.wav` 的声道占用是**严格交替**的（`0=L, 1=R, 2=L, 3=R, …`，20/20 无例外），
而**带 backchannel 的 utterance 全是奇数 = 全是 R**。backchannel 又全 pan 到 L。
⇒ **L 恒为"边听边给反馈"的一方，backchannel 永远在 R 说话时由 L 发出** ——
这在语义上**完全正确**，是设计使然，不是硬编码伪影。
（R 话语里那 2–9% 的 L 残留能量，正是 backchannel 本身。）

### 结论

> **DuplexGen 的生成期元数据是逐样本精确的。** 列出几个 backchannel，混音里就真有几个，
> 且声道归属在语义上正确。**Behavior-SD 那个"元数据说 X、音频量到 Y"的缺陷不能推广到这个数据集。**

这条的价值在**对照**：同一套审计方法在一个数据集上量出真缺陷、在另一个上量不出 ——
说明方法不是"总能挑出毛病"。这正是本项目要区分「共性缺陷」与「数据集特有」的方式。

### 还开着的口子（探针没有覆盖，别当成已答）

1. **n 太小**：每场景只看了**第一个**对话。20/20 说明"这个机制存在"，
   **不说明"全库 35,746 条都如此"** —— 要报 base rate 必须按功效配样本量（本项目老账
   `power-analysis-before-ablation-runs`）。
2. **`has_backchannels=false` 的条目**没查：那里应当**没有** backchannel，
   是零模型的天然素材（"没混进去时"的峰分布长什么样）。
3. **C.3 那个孤儿 n（1395）与本审计的 3244/4639 对不上**（见 §5.1 遗留），仍未解释。
4. **判据线仍未定**：上面 1.0000 与中位数 0.01 之间差两个数量级，这一例不需要判据线；
   但要对全库判"在/不在"，必须对 (2) 的零模型取分位（老账
   `threshold-must-match-null-model`）。**探针故意没设阈值**，免得拍出来的数被后续引用。

### 5.2-formal 设计草图（**尚未冻结、尚未跑**）

探针买到的只是"这个机制存在"。要把它变成能引用的话，缺的是**全库上的 base rate 有没有反例**。
下面这些**必须在跑之前写死并单独成文**，现在只是草图：

- **主读数 = 「列出且判为在场」的比例**（不是均值、不是 p 值）。
- **判据线**：从零分布取高分位。零模型 = **把对话 A 的 backchannel 与对话 B 的 dialogue 声道
  做同样的互相关**（B 取 A 的**下一个**对话 —— 确定性规则，不随机挑，免得又添一路方差源）。
  零分布与备择分布共用同一套解码/相关流程，差异只在"是不是同一个对话"。
- **抽样**：不用随机种子，直接取两个场景各自**前 2 个 tar 的全部对话**（预估 ~500–1000 条）。
- **n 由什么定**：主读数在真值 p=1 附近，均值没有信息量，**信息全在反例数**。
  0 反例时报「n 条中 0 例，95% 上界 ≈ 3/n」；要把上界压到 0.5% 以下需 **n ≥ 600**。
- **先写死不可判读情形**：某对话若两声道相关 > 0.5（不是双人分声道），**剔除并计数**。
- **不做**：不报均值、不跨场景合并后再挑场景、不引用单次结果。

⚠️ 探针已经在 6 个对话上看过结果，所以正式版**不能再声称是盲的** —— 这句话要写进正式预登记。

### 5.2b 「重叠」到底有多少？——**同一个数据，两套仪器差 6 倍**（2026-09-23）

探针（§5.2）顺带暴露一件事：`utterances/NN.wav` 的声道占用是**严格交替**的
（0=L, 1=R, 2=L, … 无例外）⇒ **轮次层面没有重叠**。可这个数据集叫 **full-duplex**，
而全双工的核心恰恰是**重叠说话**。所以问：`dialogue.wav` 里究竟有多少时间双方同时出声？

#### 三套仪器，三个数

| # | 仪器 | 脚本 | 重叠占比 | 判据 |
|---|---|---|---|---|
| 1 | 每声道**自己的 Otsu 阈值**判活跃 | `duplexgen_overlap_probe.py` | **0.17–1.74%** | ❌ 错 |
| 2 | NCC 定 BC 时刻 + 对方声道的**分位** | `duplexgen_bc_timing.py` | — | ✅ 20/20 BC 落在对方说话时 |
| 3 | **精确数字零**（样本 ≠ 0 即有人） | `duplexgen_overlap_exact.py` | **2.02–6.71%** | ✅ 与 #2 一致 |

#### 仪器 1 为什么错（两层，都是我这边的）

1. **根本不需要阈值。** 波形逐 200 ms 打出来看，说话人不在说话时该声道是
   **精确的 `0.00000`（−240 dB）**：没有混响、没有串音、没有本底噪声。
   这是**干净的拼接**，不是模拟的声学场景。「有没有人」于是是个**确定事实**，不是估计量。
2. **Otsu 恰好漏掉 backchannel。** Otsu 的分割点在「响亮语音 vs 静音」上，而 backchannel
   是短的轻声，落在**它自己说话人**分布的第 34–86 分位 ⇒ 顶不过阈值。
   ⇒ 仪器 1 量到的"重叠"只剩轮次交接的碎渣（如 PER：1 个 BC 却数出 5 段"重叠"）。

#### 仪器 2 自带零模型（不用另造）

窗口**随机摆放**时，「窗口内中位能量」落在全段分布的分位数期望就是 **50%**。
⇒ **判据线是 50，不是拍的。** 实测 20/20 高于 50%（合并中位 **72.4%**）
⇒ 随机摆放下 P = 2⁻²⁰ ≈ 1e-6。

#### 仪器 3 的结果

| 场景 | 同时非零 | BC 窗口被重叠覆盖 | 重叠中来自 BC | 非 BC 重叠 |
|---|---|---|---|---|
| INT | 2.69% | 94% | 74% | 0.71% |
| NEG | 2.48% | 93% | **100%** | 0.00% |
| PER | 2.02% | 95% | 33% | 1.34% |
| PLN | 3.60% | 95% | 84% | 0.57% |
| SOC | 6.71% | 91% | 84% | 1.06% |
| TEA | 5.20% | 87% | 82% | 0.95% |

> 「BC 窗口被重叠覆盖」的**天花板不是 100%**：BC 自己那个声道也只有 93–96% 的样本非零
> （片段两端近乎为零）⇒ 观测到的 87–95% **已经贴着天花板**。
> 换句话说：**只要 backchannel 在响，对方就在说话。**

**第三份独立证据**（本次顺带撞上的）：BC 文件名是 `<轮次号>_<序号>.wav`
（`01_0 07_0 09_0 11_0 19_0 19_1`），而探针**测出**的时刻
（21.78 / 70.55 / 89.30 / 108.61 / 183.20 / 184.44）与之**单调一一对应**。
⇒ 命名、声道归属、音频时刻三者互相印证，又一次支持「元数据是逐样本精确的」。

#### 🔴 两个我自己写错、且从汇总统计里看不出来的仪器 bug

1. `duplexgen_bc_timing.py` 里把秒换成帧下标时写成 `t0*sr/1000/HOP_MS` ——
   左结合 ⇒ 除以 10000 而非 hop 的 240 个样本，**差 24 倍**，量的是早了 21 秒的错误窗口。
   输出全是 0.0%/74.9%/98.7% 这类**合法百分位**，肉眼无从察觉；
   `t1 ≤ 总时长` 之类的边界 assert **也拦不住**。改法是**从样本直接切片**（结构上做不出下标错）。
2. `duplexgen_overlap_exact.py` 拿 `spans_of()`（丢掉短于 60 ms 的段，为了让打印好看）
   的结果去和 BC 求交，得到「BC 被覆盖 15%」，而逐 BC 直接数是 98.9–100% —— **差 6 倍**。
   原因：`both` 是样本级掩码，BC 窗口里 L 有约 5% 散点为零 ⇒ 掩码碎成几十片 ⇒ 被过滤器整个丢掉。
   ⇒ 纪律：**过滤可以用于「打印好看的段」，绝不能用来算总量。**

**逮住这两个 bug 的不是汇总统计，是**：(a) 逐单元打原始量（每个 BC 的非零样本占比），
(b) 两套**互相独立**的仪器对同一件事给出矛盾答案。两次的汇总统计都长得很合理。

#### 结论（含边界）

> **DuplexGen 的 backchannel 是真的全双工重叠**：20/20 落在对方说话时，
> 混音是样本级精确拼接，重叠压倒性地由 backchannel 造成
> （74–100% 的重叠时间落在 BC 窗口；PER 的 33% 是例外，它有 1.34% 的轮次交接渗漏）。
> **加上 §5.2 那条，「元数据 vs 音频」在这个数据集上被两套独立方法各查一遍，都查不出缺陷。**

**不成立的**（别外推）：n = 6，每个场景只看了**第一个**对话。表里 SOC 6.71% vs PER 2.02%
这类**场景间差异没有确立**，base rate 更没有。要引用必须走 §5.2-formal 的样本量。

### 5.3 ⏳ 需要 GPU（等 C 跑完）—— L2 标注器

- 项目现成的 X2-Turn（帧级）+ SoulX（块级）概率融合管线跑 DuplexGen 音频 ⇒ L2
- 之后可做 **L1/L2/L3 三源两两 κ**，与 Behavior-SD 那份「三源 κ 0.06–0.37」并排
- ⚠️ **但 L2↔L3 的 join 同样受 §3 限制**（除非 §5.4 解决）；
  **L1↔L2（都在 spoken 的 key 上）不受影响**，先做这条

### 5.4 🔴 需要先核实（阻塞 L1↔L3，但 **不阻塞 5.1/5.2**）

1. 拉 `github.com/duplexgen/duplexgen-code`，找**候选槽定位的中间产物**是否落盘
2. 查 HF `DuplexGen/duplexgen-corpus` 的**别的 revision / 别的 config** 有没有那份文本
3. 查 `duplexgen-spoken` 的 `.tar` 里**是否含转写文本**（`n_files` 28 vs `n_utterances` 20 + 6 BC ⇒ 可能只有 wav）
4. 都不行 ⇒ **问作者**
5. 仍然不行 ⇒ 退路是**只在 spoken 侧**做（L1 的 `n_backchannels` ↔ L2 声学），放弃 L1↔L3

### 5.5 HumDial-FDBench

先**下载 test 集**（5,000 条），下下来第一件事是**摸清它的标注字段表**（README 没给），
再决定它接 L1 还是接 L3。**在摸清 schema 之前不要写任何实验设计。**

---

## 6. 复现本次勘察

（脚本都在 `proj-thumpy/pilot_study/scripts/`，**不是** docs 那一侧。全部 0 GPU，
纯读盘；一律经 `srun --overlap --jobid=<j> -n 1` 跑，别在 login 上跑。）

```bash
D=/share/workspace3/shared_dataset/duplexgen-spoken
P=/share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study/scripts

# §5.2 探针：BC 是否混进混音（故意不设阈值）
$P duplexgen_bc_probe.py --tar $D/shards/INT/INT-00000.tar --member INT/work_0000/var00

# §5.2b 三套仪器（同一件事，三个数）—— 按顺序跑一遍就能看到 §四.21 那张表
$P duplexgen_overlap_probe.py --tar $D/shards/INT/INT-00000.tar --member INT/work_0000/var00  # ① Otsu，错
$P duplexgen_bc_timing.py     --tar $D/shards/INT/INT-00000.tar --member INT/work_0000/var00  # ② 分位数
$P duplexgen_overlap_exact.py --tar $D/shards/INT/INT-00000.tar --member INT/work_0000/var00  # ③ 精确零

# 六条对话的 (member, tar) 由每场景第一条的 metadata 决定：
$P - <<'PY'
import json
seen={}
for line in open("/share/workspace3/shared_dataset/duplexgen-spoken/metadata.jsonl"):
    r=json.loads(line)
    seen.setdefault(r["scenario"], r)
for sc in sorted(seen):
    print("%s\t%s" % (seen[sc]["key"], seen[sc]["shard"]))
PY
```

```bash
cd /share/workspace3/shared_dataset

# 家底
du -sh duplexgen-corpus duplexgen-spoken switchboard Full-Duplex-Bench
ls duplexgen-corpus/annotations/*/ ; ls duplexgen-spoken/shards/*/ | head

# join 检验（会打印 §3.2 那张表）
python3 - <<'PY'
import json, difflib
base="duplexgen-corpus"; sc="INT"
ann=[json.loads(l) for l in open(f"{base}/annotations/{sc}/test.jsonl")]
dlg={json.loads(l)["example_id"]: json.loads(l) for l in open(f"{base}/dialogues/{sc}/train.jsonl")}
print("id 命中:", sum(a["example_id"] in dlg for a in ann), "/", len(ann))
for a in ann[:5]:
    d=dlg[a["example_id"]]
    n=min(len(a["history"]),len(d["history"]))
    same=sum(1 for ha,hd in zip(a["history"],d["history"]) if ha["content"]==hd["content"])
    print(f"  {a['example_id']}: content 逐 turn 相同 {same}/{n}")
PY
```

---

## 7. 与总索引的关系

- 本文只做**勘察与可行性**，**不产生任何实验结论**。
- §3 的教训（join 键命中 ≠ 实体同一）**待进 `2026-09-22_claims_ledger.md` 的「方法论教训」**。
- 5.1 / 5.2 出结果后另开文档；索引登记在 `syllabus.md`。
- 上游动机：上个月周报的「下一步」（见 `2026-09-22_pilot_study_overview.md` §6 的对外叙事）。
