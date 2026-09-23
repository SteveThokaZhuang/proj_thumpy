# HumDial-FDBench test 集：字段表勘察（recon §5.5 的答案）

> **一句话**：它**不是**双工素材 —— 音频是**单声道 16 kHz**，而且**只有被标注的那个人**
> （助手的声音根本不在文件里，标注区间外是**数字静音**）。
> 但它自带**两个文本源**（参考文本 + 词级 ASR 假设），在**同一批音频**上
> **逐字相同只有 50.6%** ⇒ 这是一份现成的 **L-参考 ↔ L-ASR** 对照，4172 条。
>
> 日期：2026-09-23 ｜ 状态：✅ 勘察完成 ｜ 全 0 GPU，纯读盘
> 上游：[2026-09-23_duplexgen_humdial_recon.md](2026-09-23_duplexgen_humdial_recon.md) §5.5
> ⚠️ **本文只做勘察，不含任何实验设计**（recon §5.5 的明文要求）。

---

## 1. 数据从哪来 + 完整性核对

| 项 | 值 |
|---|---|
| 来源 | HF `ASLP-lab/HumDial-FDBench`（test 集） |
| 落地 | `/share/workspace3/shared_dataset/humdial-fdbench/Humdial-Track2-Test.zip` |
| zip 字节 | **1,818,079,900** |
| 条目 | **23,417**（HF 清单 `ziplist.json`） |
| 解压后 | **6.91 GB** |

🔴 **`download.log` 里只有一行 `rc=7`（curl 连不上）—— 那不是一次成功的记录。**
zip 是后来另一个进程写的，**没有任何日志为它背书**。所以先核对了完整性
（`scripts/humdial_zip_verify.py`），**不采信日志**：

| 判据 | 结果 |
|---|---|
| P1 中央目录条目数 == 清单条目数 | ✅ 23417 == 23417 |
| P2 逐条 `(name, usz)` 比对 | ✅ 只在清单 0 / 只在 zip 0 / 大小不符 0 |
| P3 清单最远偏移 ≤ 实际字节 | ✅ 1,815,285,904 ≤ 1,818,079,900（尾隙 2,793,996 = 中央目录） |
| P4 抽 20 条真读 + CRC | ✅ 20/20 通过 |

⚠️ **仍未做全量 CRC。** 结构自洽 + 抽样通过 ≠ 每个字节都对；
若某条结论依赖具体某个文件，读它时会自然校验。

---

## 2. 字段表（全部由 zip 字节现算，说明文字一律未采信）

全库 **9098 个 json**，只有 **2 种**顶层形状：

### 2.1 分段族 `{final_duration, speech_segments}` —— 5198 条

```json
{"final_duration": 29.12,
 "speech_segments": [{"xmin": 0.01, "xmax": 8.69,
                      "text": "What are the most effective strategies to improve job satisfaction in a remote work environment?"},
                     {"xmin": 13.69, "xmax": 19.11,
                      "text": "How often should these video check-ins ideally be scheduled?"}]}
```

### 2.2 词级族 `{text, chunks}` —— 9098 条

```json
{"text": "What are the most effective strategies to improve job satisfaction in a remote work environment?",
 "chunks": [{"text": "What", "timestamp": [0.24, 0.56]}, …]}
```

### 2.3 🔴 全库递归键扫描：**没有任何 speaker / role / spk 字段**

`humdial_content_probe.py` 的 N5 遍历了全部 9098 条 json 的**所有**嵌套键：

| 族 | 键（n） |
|---|---|
| plain (4172) | `final_duration` / `speech_segments` / `[].xmin` / `[].xmax` / `[].text` |
| timestamp (7444) | `text` / `chunks` / `[].text` / `[].timestamp` |
| add (1026) | 同 plain |
| add_timestamp (1654) | 同 timestamp |

⇒ **HumDial 不提供逐轮说话人。** 这一条是后面所有判断的前提。

### 2.4 音频：**9098 个 wav 全部 ch=1 / sr=16000 / bits=16**

逐个 wav 读 `fmt` 块（**不是抽样**）—— 结果**只有一种格式**，没有例外。

---

## 3. 文件名结构与四族的关系

命名式：`[clean_]{DDDD}_{NNNN}[_add]{.wav | .json | _timestamp.json}`

### 3.1 全库 id 形态（P0）

| clean | add | 有的族 | id 数 |
|---|---|---|---|
| ✗ | ✗ | seg + ts + wav | **4172** |
| ✗ | ✓ | seg + ts + wav | **827** |
| ✓ | ✗ | ts + wav | **3272** |
| ✓ | ✓ | ts + wav | **628** |
| ✓ | ✓ | seg + ts + wav | **199** |

逐目录看（`humdial_pair_probe.py` 的按目录表）：

- 每个 `场景 × 语言` 恰好 **300 条主版录音**（`others_talk_to_user_after/before` 各 150，
  `talk_to_others` 99/100）⇒ 合计 **4999**。
- **每一条主版录音都恰好有一个 `speech_segments` json**（逐目录都是 300/300）。
- ⚠️ index 探针报的「plain 4172 / add 1026」是**文件数**；id 数是 4172 / 827，
  差的 **199** 是 `talk_to_others` 里的 `clean_X_add.json`（那一族主版文件名本身就带 `_add`）。
  **文件数 ≠ id 数**，两个都要报，只报一个会看起来自相矛盾。

### 3.2 `clean_` 是什么 —— 🔴 逐样本钉死

**先说我上一版错的结论**：我用**全部** speech_segments 造掩码去量 clean，
得到「区间内逐样本 maxΔ 高达 0.98、逐帧能量相关只有 0.43」，据此写了
**「clean 是另一路音频」**。**那条结论是错的** —— 掩码是脏的。

摊开一个具体 id 就看到了（`humdial_group_dump.py`）：

```
0001_0004_timestamp.json        text = "…remote work environment? How often should these
                                        video check-ins ideally be scheduled?"   (24 chunks)
clean_0001_0004_timestamp.json  text = "…remote work environment?"                (15 chunks)
```

⇒ **clean 版只有用户的第一轮，第二轮（要求重复那句）没了。**
而主版掩码把第二轮也算进「用户区间」，clean 在那一段上本来就是静音 ——
**差异是我的掩码造出来的，不是数据的。**

**逐段重测**（`humdial_clean_resolve.py`，12 个 id，每场景取第 1 个）：

| | 主版高能帧 | clean 高能帧 | 逐样本 maxΔ | 逐帧能量 r |
|---|---|---|---|---|
| **第 1 段**（n=12） | 84.8% | 82.3% | **中位 0.000000** | **1.000** |
| **第 2 段及以后**（n=13） | 79.7% | **0.0%** | — | — |

⇒ ✅ **`clean_X.wav` = `X.wav` 的同一段音频，用户第一轮之后被置成数字静音。**
11/12 个 id 的第 1 段 maxΔ **精确为 0**、能量相关**精确为 1.000**。

⚠️ **1 个例外**：`cn_test_nondev/talk_to_others/0003_0002_add` 的第 1 段
maxΔ 0.335、clean 高能 58.5% vs 主版 89.3%。**是个例外就报成例外**，
不做平均、不四舍五入掉。

### 3.3 `_add` 是什么 —— 同一 id 下的**另一条对话**

| 量 | 值 |
|---|---|
| 能配成对的 | **287** 对 |
| 规范化文本相似度 | 中位 **0.254**，均值 0.245，范围 [0.000, 0.623] |
| 完全一样（>0.99） | **0 / 287** |
| 毫不相干（<0.3） | **172 / 287** |

具体一例：

```
cn_test_nondev/ask/0003_0006.json       [0.00, 4.89] 在游戏设计中，如何平衡道具的攻击力和使用频率？
                                        [9.89,15.92] 那如果道具既有高攻击力又频繁掉落，该怎么调整平衡？
cn_test_nondev/ask/0003_0006_add.json   [0.00, 4.14] 家庭医生签约后能提供哪些具体的健康管理服务？
                                        [9.14,12.16] 那健康评估一般包括哪些检查项目？
```

⇒ **`_add` 与主版共享编号但内容完全无关**（wav 时长也差 3.76 s）。
「_add」应理解为「同一编号下追加的另一条」，**不是同一条的增补版**。

### 3.4 官方标签是**两个源**，不是一份标注的两种渲染

把 `speech_segments[].text` 拼起来与 `chunks.text` 比（规范化后，4172 条可配对的）：

| | 值 |
|---|---|
| **逐字相同** | **2111 / 4172 = 50.6%** |
| 不相同 | 2061 |

不相同的那些，**相似度极高**（0.95–0.997），差的全是**识别式错误**：

| `speech_segments`（参考） | `chunks.text`（假设） |
|---|---|
| …delivery **services**? | …delivery **service**? |
| a garbage **sorting** app | a garbage **shorting** app |
| without **sounding** … | without **something** … |
| **say** that again | **said** that again |
| participate **in** World Animal Day | participate **in on** World Animal Day |

⇒ 🔴 **`chunks` 是 ASR 输出（带词级时间戳），`speech_segments` 是参考文本。**
**同一批音频上的两个来源**，不是同一份标注的两种粒度。

---

## 4. 🔴 音频里到底有谁 —— 助手**不在文件里**

`humdial_energy_probe.py`，20 ms 帧 / 5 ms 跳，逐帧 RMS：

| 区域 | 主版帧 RMS |
|---|---|
| `speech_segments` **内** | 中位 **0.017 – 0.067** |
| `speech_segments` **外** | 中位 **0.00000**，p90 0.00001，p99 0.0001，**max 0.0006** |

区间外高能帧占比（三个阈值下）**一律 0.00%**。

⇒ **`X.wav` 只含被标注的那个人的语音，其余是数字静音。**
「空隙」不是助手在说话，**空隙是空的** —— 助手那一侧的音频**没有随数据发布**。

⚠️ **顺带**：第三方的场景（`others_talk_to_user_*` / `talk_to_others`）里，
被标注者可能不止一人，而**没有说话人字段**可以区分。这一点要在任何后续使用中写明。

---

## 5. 🔴 推翻 recon §4 的断言

recon §4 那张表是从 **GitHub/HF 的说明文字**抄的。现算之后：

| recon §4 断言 | 实测 | |
|---|---|---|
| 「真实人类录音、**双声道**」 | **单声道** 16 kHz 16-bit，9098/9098 无一例外 | 🔴 **假** |
| 「test 5,000 条」 | 主版录音 **4999** 条（+4099 条 clean 孪生 + 词级 json 9098） | 量级对，口径要说清 |
| 「两类 **9** 子场景」 | 9 个场景 × 2 语言 = **18 个目录** | ✅ 真 |
| 「README 没给字段表」 | 确实没给；**本文 §2 现算出来了** | ✅ 真 |
| 「官方标签可能是 L2 而非 L3」（§4 的推测） | **推测方向对，但结论更具体**：官方给的是**参考文本 + 词级 ASR** 两个源，且**不含声学双工信息** | 见 §6 |

这正是 memory `config-differs-from-config-file` 那一族：
**上游说明文字也是待验证断言。** 本次「双声道」三个字差点让整个可用性判断反过来。

---

## 6. 这对本项目意味着什么（**只陈述可用性，不写设计**）

### 6.1 它**不能**做什么

- ❌ **不是双工素材**：助手侧音频不存在，**声学重叠/backchannel 检测做不了**。
  （这堵死了「拿 HumDial 当第二个 overlap 语料」这条路。）
- ❌ **没有逐轮说话人**：§2.3 全库键扫描已证。
- ❌ **不是人工多层投票**：没有 DuplexGen 那种 `counts` / `total_count`，**算不出 κ**。

### 6.2 它**能**做什么

- ✅ **真实的轮次边界**：4999 条真人录音，用户各轮的 `xmin/xmax` 是现成的；
  空隙位置**就是**助手在说话的位置（只是听不到）。**真人双工轮次的时序**是真信号。
- ✅ **同一音频上的两个文本源**（§3.4）：4172 条 **参考文本 ↔ 词级 ASR**，
  逐字相同仅 **50.6%**。这是**现成的 L-参考 vs L-ASR 对照**，
  与本项目「三源两两 κ」那条线直接同族（Behavior-SD 那份是三源 κ 0.06–0.37）。
  🔻 **但已量过时间层面，口径要收窄**（见下方 ⚠️）。
- ✅ 规模大、双语、0 GPU（纯读盘即可复算）。

> ⚠️ **2026-09-23 补充（时间一致性已测，见 [timing 预登记](2026-09-23_humdial_timing_prereg.md)）**：
> 上面那条「两个文本源」的乐观**要下调一档**。预登记后实测（全量 4172 条，0 GPU）：
> **两源在时间上大体一致 —— 帧级 F1 = 0.9125**（precision 0.98 / recall 0.85），
> 残余差异**约 2/3 是粒度差**（R 是"轮次"、A 是"词"，轮内停顿处 A 没有词）、
> **约 1/3 是边界内缩**（每轮 ~0.21 s，方向与"ASR 时间戳偏内"的预测同号）。
> ⇒ **准确的说法是**：HumDial 给的是 **「同分段、不同词」** 的对照
> （**词汇**层面只对上一半，**时间**层面大体一致），
> **不是**"两个互相打架的标签"。**它不能用来证明"标签在时间上就对不上"。**
> ⚠️ 另外：**本文没有裁判谁对** —— 50.6% 里那 49.4% 谁对谁错，一个字都没说。

### 6.3 对 recon §5.5「接 L1 还是接 L3」的回答

**都不是。** 按本项目自己的分层口径：

- **L1（元数据/生成器自采样）**：无 —— 这是真人录音，不是合成的。
- **L2（声学派生）**：`chunks` 的词级时间戳**是**声学派生物（ASR 强制对齐/识别）。
- **L3（人工）**：`speech_segments` 的**参考文本是人工/脚本给出的**，
  但**没有多标注者、没有票**，所以**不是** DuplexGen 意义上的 L3（多数票）。

⇒ **正确的说法是：HumDial 提供「参考文本(L3 级但无票) ↔ ASR 假设(L2)」这一对，
在真人的轮次时序上。** 它接的不是三源体系里的某一层，而是**补上了
「同一音频、两个文本源」这个此前没有的对照轴**。

🔻 **事后限定（2026-09-23，时间一致性实测之后）**：这条对照轴是**词汇的**，不是**时间的**
—— 两源帧级 F1 = 0.9125、残余差异主要是粒度差（详见 §6.2 的 ⚠️ 块）。
⇒ 引用时请写「**同分段、不同词**」，**不要**写成"两个标注源互相矛盾"。

---

## 7. ⚠️ 本次我自己犯的错（诚实清单）

按项目惯例逐条列出，**这些都是「看起来像结果」的坏数**：

1. 🔴 **正则把汉字删光 ⇒ 恒真比较。**
   `NORM = re.compile(r"[^a-z0-9]+")` 会把**整个中文**规范化成空串，
   于是所有 `cn_*` 两两「相等」。症状是 Q1b 报出
   「逐字相同 **1635/3272 = 正好 50.0%**」—— **正好一半**就是因为语料一半中文。
   我当时觉得「这个数长得像配对错位」，但**先怀疑的是配对、不是规范化**，绕了一圈。
   ⇒ 改用 `str.isalnum()`（对 CJK 返回 True）。**恒真检查的典型形态。**
   🔴 **修好后 Q1b 是 `8 / 3272 = 0.24%`** —— 从「正好一半」塌到「几乎没有」。
   ⚠️ 但**同一个 bug 也污染了 Q2**，而 Q2 修好后**基本没动**（仍 **50.6%**，
   因为那个 50.6% 是**真发现**：参考文本 ↔ ASR 假设）。
   ⇒ **两个量一个是全假、一个恰好是真，修完之后数的变化幅度完全不同却都"合理"**
   ⇒ **不能靠「数变了没、变了多少」判断修没修对**，要看这个比较按定义在比什么。
2. 🔴 **用脏掩码造出了差异再当成发现。**
   拿**全部** speech_segments 当掩码量 clean，得到「逐样本 maxΔ 0.98、相关 0.43」，
   据此写「clean 是另一路音频」。真相是 clean **本来就只保留第一轮**，
   第二轮静音是**定义**，不是差异（memory `threshold-must-match-null-model` 第五面：
   「定义域里按定义恒定的那一段会被读成发现」）。**摊开一条数据就破了。**
3. ⚠️ **过滤器制造空表，空表被读成结论。**
   Q1c 里写了 `len(a1) != len(a2): continue`，而两者时长比是 1.0004（差 ~12 ms），
   于是**所有样本被筛光**，打印出「相等 0 / 不相等 0」——**长得像结果**。
   发现是因为我同时打印了分母。
4. ⚠️ **路径没拼目录。** Q2/Q3 用 `"%s.json" % stem` 去全路径集合里查，
   全部落空 ⇒ 报出「缺一侧 4172」这个**假数字**。
5. ⚠️ 格式串 `%d` 吃到 list（N2 崩了）—— 小错，但**崩之前已经把半张表打出来了**，
   差点当成完整结果读。

⇒ 五条里 **1、2 两条是「坏数长得像发现」**，正是 memory
`sd-from-few-points-manufactures-anomalies` / `tiebreak-default-read-as-evidence` 那一族。

---

## 8. 复现命令

脚本都在 `proj-thumpy/pilot_study/scripts/`（**不是** docs 那一侧）。全部 0 GPU，
纯读盘；一律经 `srun --overlap --jobid=<j> --ntasks=1` 跑，**别在 login 上跑**。

```bash
D=/share/workspace3/shared_dataset/humdial-fdbench
Z=$D/Humdial-Track2-Test.zip
P=/share/home/zhuangruicen/miniconda3/envs/fd_analysis/bin/python
cd /share/workspace3/zhuangruicen/proj-thumpy/pilot_study

# ① 完整性（结构 + 抽样 CRC）—— 别信 download.log
srun --overlap --jobid=<j> --ntasks=1 -c 2 \
  $P scripts/humdial_zip_verify.py --zip $Z --list $D/ziplist.json

# ② 字段表 / 目录清单 / 音频格式（全量，不抽样）
srun --overlap --jobid=<j> --ntasks=1 -c 2 \
  $P scripts/humdial_schema_probe.py --zip $Z

# ③ 文件名结构与四族关系（含递归键扫描）
srun --overlap --jobid=<j> --ntasks=1 -c 2 \
  $P scripts/humdial_content_probe.py --zip $Z --dir en_test_nondev/ask

# ④ Q1/Q2/Q3：clean 归属、两源文本、_add 关系
srun --overlap --jobid=<j> --ntasks=1 -c 4 \
  $P scripts/humdial_pair_probe.py --zip $Z --audio-n 14

# ⑤ 摊开看具体一条（统计解释不了时就做这个）
srun --overlap --jobid=<j> --ntasks=1 -c 2 \
  $P scripts/humdial_group_dump.py --zip $Z --dir en_test_nondev/ask --mode differ --k 2

# ⑥ 能量包络：区间外到底有没有人
srun --overlap --jobid=<j> --ntasks=1 -c 4 \
  $P scripts/humdial_energy_probe.py --zip $Z --n 14

# ⑦ 逐段钉死 clean（本轮的定案仪器）
srun --overlap --jobid=<j> --ntasks=1 -c 4 \
  $P scripts/humdial_clean_resolve.py --zip $Z --n 12
```

产物在 `pilot_study/real_data/results/humdial/`。

---

## 9. 与总索引的关系

- 本文**只做勘察，不产生任何实验结论**，也**不含实验设计**（recon §5.5 的明文要求）。
- 已登记进 `2026-09-22_claims_ledger.md`：
  - **确证**：第 **28** 条（音频里只有被标注者）+ 第 **29** 条（两个文本源 50.6%）。
  - **证伪**：§二 第 **17** 条（recon §4 的「双声道」）。
  - **方法论**：§四 第 **27** 条（「正好一半」是恒真事实的投影）。
    ⚠️ 精确说：§7 的**第 1/2/3 条**（三个"长得像结果"的坏数）进了方法论；
    第 4/5 条（路径没拼目录、格式串崩溃）是**低级 bug**，
    只在本节留档，**不另立条目** —— 别把 bug 清单和方法论混为一谈。
- recon 文档 §4 表末「要下下来才知道 L1 长什么样」→ **本文即答案**；
  `2026-09-23_duplexgen_humdial_recon.md` §5.5 已加指针。
- 索引登记在 `syllabus.md`。
