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

**这正是本项目主命题的第二个实例**，而且**比 Behavior-SD 那边更干净**：
- Behavior-SD 那边是「元数据声称有 X」vs「音频里量到 Y」，两边是不同的东西；
- 这里是「**生成时插入了 N 个片段**」vs「**混音后的 `dialogue.wav` 里真的听得见这 N 个吗**」
  —— 同一件事的两个记录，**可以直接对账**。

**代价小**：不必跑 657 G，抽几个 tar（每个 ~1.5 G）解出 wav 即可。
**要定的口径**（跑之前写死）：
1. `dialogue.wav` 是单声道还是双声道？**若是单声道，就无法按说话人分离声道** ——
   那 `utterances/*.wav` 的逐 turn 序列就是唯一的「谁在说」依据（需用能量/相关性把
   `backchannels/NN_K.wav` 在 `dialogue.wav` 里定位）。
2. 「重叠」的判据必须**先对零模型**（本项目老账 `threshold-must-match-null-model`）：
   判据线不能拍，要先量「纯静音/纯单说话」时的 RMS 分布。
3. **抽样量要按功效算**，不能先跑再看。

- 在 `duplexgen-spoken` 上跑**上个月那套声道级 RMS 能量检测**（抽样，不必跑 657 G）
- 对照 `metadata.jsonl` 的 `n_backchannels` / `has_backchannels`
- **这就是「元数据说 X、音频实测 Y」在第二个合成数据集上的复现** ——
  直接回答「是共性缺陷还是 Behavior-SD 特有」，而且**用的是本项目自己的方法**，不是新工具

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
