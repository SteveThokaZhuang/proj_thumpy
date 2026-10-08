# fd_activities —— 全双工行为定义：文献汇编

> **建立**：2026-10-08
> **用途**：收录已发表工作对「全双工行为（full-duplex behaviors / activities）」的**定义**——
> 即：它们说 full-duplex 是什么、把哪些行为单列成类（backchannel / interruption / turn-taking / barge-in / …）、
> 每一类用什么**判据**操作化（阈值 / 字段 / 标注协议 / 模型 / 判官）。
> **服务对象**：论文相关工作节；以及 `2026-09-27_behavior_annotation_design.md` §2「六个体系」表的**可溯源扩展版**。
> **收录标准**：只收**已发表**（含 arXiv 预印本）工作；本项目自己的口径单列 §5，不混入。

## 证据分级（每条都标，别混用）

| 标记 | 含义 |
|---|---|
| 【原文✓】 | 引用抄自论文全文（HTML/PDF 文本），**关键句已本地回核**，节/表号已记 |
| 【抓取】 | 经自动抓取摘录，**引用可能非逐字**，待人工回核 |
| 【转述】 | 二手转述（项目文档 / 搜索摘要 / 上游 README），未经论文原文核实 |
| 【未取得原文】 | 抓取失败或未找到，条目留空待补 |
| 🔴 | 已发现的**冲突或反直觉**事实（如：同一行为两个文献阈值不同；搜索摘要的编号是幻觉） |

> ⚠️ **编号核实纪律**：arXiv 编号从搜索/摘要来的一律先核（标题对得上才算）。
> 本次已抓到两条实例：搜索摘要把 survey `2609.30798` 的一个小节标题
> 「The Full-Duplex-Bench family」当成了论文标题；`2602.05105` 被指为 FDB-v3
> 实为无关论文 *GAMMS*（survey 自己也在 §2 警告过这一条）。

---

## 0. 速查表（一览）

> 逐篇详情见 §1–§4；本表只给「行为类别 + 判据一句话」，**引用一律以详录为准**。

| 文献 | 年 / venue | 行为类别（它定义的） | 关键判据（一句话） | 地基 | 节 |
|---|---|---|---|---|---|
| FD-Bench | 2025 / Interspeech | 无行为分类学；插话五类 A/D/F/R/S；**backchannel 未定义** | VAD(0.5) 时间戳 + 自定义指标（EIR/NIR…） | 规则 + 时长 | §1.1 |
| FDB v1.0 | 2025 / ASRU | pause handling / backchanneling / smooth turn-taking / user interruption | **bc = 短于 1 秒且少于 2 词**；takeover 二元 | **时长阈值**（+ 仅 bc 时机用人工 GT） | §1.2 |
| FDB v1.5 | 2026 / ICASSP | user interruption / user backchannel / talking to others / background speech | GPT-4o 四分类 Respond/Resume/Uncertain/Unknown + stop/response latency | 语义判官（只吃 ASR 文本）+ VAD 时长 | §1.3 |
| FDB v2 | 2026 / ACL | 无行为清单；任务族 × TT/IF/TS 评分 | 1–5 分（Gemini）；backchannel 是**考官**的施压手段 | LLM 主观评分 | §1.4 |
| FDB v3 | 2026 / (WIP) | 五类 disfluency：false starts / self-corrections / fillers / pauses / hesitations | Δt<0 即 Interruption；无显式"自然时序"秒数 | 规则日志 + LLM 判内容 | §1.5 |
| DuplexAct-Bench | 2026 / (预印本) | **6 类**：Agent Interruption / User Interruption / Interruption Resistance / Active Silence / Proactive Initiation / Agent Backchannel | **逐试次标注区间**（无全局常数）；BAS 用 1s 窗 + σ=1.0s | 人标区间 + LLM 判 + 模型投票 | §1.6 |
| HiPLEX | 2026 / (预印本) | 沿用 FDB v1 四类，改造为 RL 奖励分量 | **≤1s = short / >1s = sustained**；bc 匹配上限 1s | VAD 片段时长 + 距离罚项 | §1.7 |
| **CANDOR** | 2023 / Sci Adv | backchannel（算法+词表）；**interruption 无独立字段** | **Backbiter 三规则**：≤3 词 & >50% bc 词 & 非禁词开头 | 词表 + 规则 | §2.1 |
| **Behavior-SD** | 2025 / NAACL | **4 维度 × 3 档**（V/F/B/I）+ utterance 级三分 | 时间由**高斯采样**（0.4/0.2/0.45s）；interruption 七子类 | 生成端元数据（L1） | §2.2 |
| **DuplexGen** | 2026 / EMNLP | floor_taking / backchannel / silence（四套命名） | 人标 5 票软标签（κ=0.05 为项目自算） | **人标（仅文本）** | §2.3 |
| **HumDial-FDBench** | 2026 / ICASSP Challenge | 场景级：Interruption 5 子 + Rejection 4 子 | DeepSeek-V3 判四类；S_Total = 0.4/0.4/0.2 | LLM 判官 + VAD | §2.4 |
| Moshi | 2024 | **无**（架构级） | — | — | §3.1 |
| **dGSLM** | 2023 / TACL | 提出 bc/int **结构判据后拒绝实现** | IPU>200ms；"will not attempt to extract here" | 结构 | §3.2 |
| LSLM | 2024 | 仅 interruption（IRQ） | 停说 ∈ **[0, 2μ]（1s）** | 时长阈值 | §3.3 |
| Freeze-Omni | 2024 | 3 个 chunk 级状态 | **无判据**；标签构造未交代 | — | §3.4 |
| **X2-Turn** | 2026 | **5 state**（含 backchannel） | **无阈值**；Qwen3.5-Plus LLM 语义标注，80ms | LLM 标注（语义） | §3.5 |
| **SoulX-Duplug** | 2026 | **5 state**（含 user_backchannel） | bc 只有一句语义定义、**零判据** | LLM 标注（Qwen2.5-72B） | §3.6 |
| Fun-Audio-Chat | — | 无 | — | — | §3.7 |
| **TACT** | 2026 / SLT | **6 意图 + 4 场景（两套正交）** | bck onset 律 ex-Gaussian(−0.32,0.18,0.20)s；α=0.73 | 人标 + 分布 | §4.2.1 |
| "WildTurn"（数据集） | 2026 / ACM MM | NTT / ITT / BC / BI / NA | ITT = onset 在用户话轮结束前；66 词词典 | **VAD+词典（全自动）** | §4.2.2 |
| Instruct-FD | 2026 | 5 动作 | judge：bc <4 词；GT ±2s 容差 | LLM 选址 + 对齐回填 | §4.2.3 |
| **Duplex Cue** | 2026 | **2 轴 3×3**（cue intent × response） | 语义定义；复标一致 81.7%，分歧多在 bc–int 边界 | 人标 | §4.2.4 |
| BC-head | 2026 | bc onset（帧级） | 合格帧（对方说 & 己方静默）+ τ；onset F1 ±0.16s | 帧级 + 人标（α=0.42） | §4.2.5 |
| **Cathcart 2003** | 2003 / EACL | continuers | acknowledge move 过滤；**无时长门槛** | 词表 | §4.1① |
| **Liesenfeld 2022** | 2022 / Interspeech | response tokens | top decile 频次 + 同说话人 ≥2 连发；**明确拒绝词表** | 序列结构 | §4.1② |
| **VAP / Ekstedt 2022** | 2022 / Interspeech | bc vs SHORT/LONG shift | **≤1s 且前静默 1s / 后静默 2s、对方先有 VA** | 时长 + 邻域静默 | §4.1③ |
| **Uro 2024** | 2024 / LREC-COLING | 六分 floor-taking | 人标；**25% 平滑换手有重叠 / 19% 违规无重叠** | 功能 / 意图 | §4.1④ |
| **TurnBench** | 2026 | BC / INT / Turn / Non-floor-taking INT | bc = "does not claim the floor, **regardless of duration**" | 功能 / 意图 | §4.1④ |
| **Paierl 2025** | 2025 / Languages | hearer response tokens | 🔴 **项目旧引用已撤回**（未测 bc 时长，见 §4.1④） | 词表式语义定义 | §4.1④ |

---

## 1. Benchmark 类

### 1.1 FD-Bench（arXiv:2507.19040）

- **引用信息**：Yizhou Peng, Yi-Wen Chao, Dianwen Ng, Yukun Ma, Chongjia Ni, Bin Ma, Eng Siong Chng.
  *FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems.*
  Interspeech 2025. arXiv:2507.19040.
- **取证方式**：【抓取】2026-10-08 抓 `arxiv.org/abs/2507.19040` + `arxiv.org/html/2507.19040v1`（自动摘录，待回核 PDF）

**它的 full-duplex 定义**（两个维度，非行为清单）：

> "This pipeline defines full-duplex performance across two key dimensions: interruption handling and response quality."

**行为类别**：它**没有**总的行为分类学（taxonomy）。最接近的是「用户插话类型」（Table 1）五类：

> "Affirmative Acknowledgment, Denial and Discontent, Further Inquiry, Requiring a Repeat, and Topic Shift"

（缩写 A/D/F/R/S；一次对话最多 4 次插话，类别可共现。）

**🔴 backchannel 未被定义**：backchannel 只在 Introduction 的列举里出现一次
（"interruptions, affirmations, backchanneling, and topic shifts"），
**全文没有 backchannel 的形式化定义、也没有 backchannel 专项指标**；
最接近的类别是插话类型里的 Affirmative Acknowledgment。
⇒ 若把 FD-Bench 当「评估了 backchannel 的 benchmark」引用，是**过度解读**。

**操作化判据**（原文引用，节号见表注）：

- 评估侧 VAD：> "We apply Silero-VAD with a threshold of 0.5 to get timestamp information"（两侧都取时间戳）
- ASR：> "Whisper-large-v3-turbo to perform ASR with a beam-size of 5"
- 指标（§2.3，**论文只给文字定义、无公式**）：
  > "SRRate (measures the odds of Success-Replies per User-non-interrupt-inquiries)"
  > "SIRate (measures the rates of Success-Interrupts per User-interrupt-inquiries)"
  > "SRIRate (measures the odds of Success-Replies-to-Interrupts per SIs)"
  > "EIRate (measures the odds of Early-Interrupts per User-inquiries)"
  > "NIRate (measures the rates of Noise-Interrupts per Noise-gaps between user inquiries)"
  > 时延指标 "IRD(interrupt-response-delay), FSED(first-speech-emit-delay), ERT(early-reply-time), and EIT(early-interrupt-time)"（取中位数）
- 难度分级由插入静音时长控制："Easy, Medium, and Hard" = "6–10 seconds, 4–6 seconds, and 2–4 seconds"
- 规模：> "over 40 hours of generated speech, with 293 simulated conversations and 1,200 interruptions"

**⚠️ 与本地 E2 审计的关系**：本项目 `2026-08-25_ari_e2_fdbench_audit.md` 复现了 "FD-Bench 的 interval rules"
并测得 EIR 25× 漂移——注意那套 interval 规则来自 **FD-Bench 代码**（`third_party/FD-Bench`），
论文正文里**并无**最小语音/静音时长、hangover 之类的分段规则描述（本次抓取核实：absent）。
⇒ 引用"FD-Bench 的规则"时要注明是**代码口径**还是**论文口径**。

---

### 1.2 Full-Duplex-Bench v1.0（arXiv:2503.04721）★ 本簇的"行为清单"源头

- **引用信息**：Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopal Anumanchipalli,
  Alexander H. Liu, Hung-yi Lee. *Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken
  Dialogue Models on Turn-taking Capabilities.* arXiv:2503.04721v3（v1 2025-03-06；v3 2025-08-16）。
  **Accepted by ASRU 2025**。https://arxiv.org/abs/2503.04721
- **取证方式**：【原文✓】抓取 arXiv HTML 全文（`/tmp/fdb_lit/test_2503.txt`），关键句本地回核

**行为类别（4 类）**（Fig. 2 + Abstract）：

> "a benchmark that systematically evaluates key interactive behaviors: pause handling, backchanneling, turn-taking, and interruption management."

**① Backchannel —— 全文最关键的操作化定义**（🔴 作者自己承认这是简化）：

> "Generally, backchanneling refers to short utterances produced by the listener while the speaker is talking."

> "In this work, we classify a speech segment as backchanneling if it meets the following criteria:
> **(1) it has a short duration of less than 1 second; and (2) it contains fewer than two words.**
> This ensures that speech is delivered at a reasonable pace and that brief utterances do not interrupt
> the current speaker's turn. **The concept of backchanneling differs across the literature [42, 43],
> so developing a comprehensive detector is beyond the scope of this paper and is left for future work.**"

**② Takeover（TO）—— 二元变量**：

> "A takeover occurs when the model effectively assumes control of the conversation, dominating the turn
> and granting minimal opportunity for the user to speak. In this work, takeover is treated as a binary
> variable. **If the model merely responds with silence or a backchannel, no takeover is deemed to have
> occurred.** Conversely, any other non-silent speech that is not a backchannel indicates the model's
> attempt to take over."

- TOR = "the average value of the binary TO variable across the dataset"

**逐类判据**：

| 行为 | 判据（原文） | 地基 |
|---|---|---|
| Pause Handling（§III-B-1） | "Can the model recognize when the other speaker is still holding the turn and understand that it should not take over?" → 用 TOR，"A lower TOR signifies better pause management" | 时长（数据构造：pause ∈ 0.4–1.0s；turn > 5s） |
| Backchanneling（§III-B-2） | 三指标：TOR（lower better）、Backchannel Frequency（"events per second"）、**JSD**（模型 bc 时机 vs 人类 bc 分布，"JSD ranges from 0 (perfect alignment) to 1 (complete divergence)"） | 时长阈值 + **人工 GT**（ICC 数据集） |
| Smooth Turn Taking（§III-B-3） | "averaged response latency, the time (in seconds) between the end of the user's speech and the start of the model's response"，"The latency is calculated only when TO equals 1" | 时长 |
| User Interruption（§III-B-4） | TOR（"ideally TOR = 1"）+ GPT-4o Score（"coherence, relevance, and adaptability" 0–5）+ Latency After Interruption | LLM 判官 + 时长 |

**数据构造阈值（§III-C，原文）**：

- Pause: "select turns containing an internal pause between **0.4 and 1.0 seconds**"；"We exclude turns shorter than **5 seconds**"；
  并"remove cases where the speech immediately before or after the pause consists of backchannel responses"
- Smooth turn-taking: "retain only those where all gaps are **less than 0.4 seconds**"；"each turn to be longer than **4 seconds**"；"we append 5 seconds of silence at the end of the input stream"
- Backchannel GT（ICC）: "responses from **118 native speakers**"；"segmenting the audio into **200-ms time windows** and normalizing the backchannel counts, we generate a ground truth backchannel distribution Q"
- Synthetic（interruption / pause）: "the interrupting speech is played around **7 seconds** after the preceding utterance"；"we append 15 seconds of silence after the interruption"

**样本量（Table II）**：Pause(Candor) 216、Smooth TT(Candor) 119、Backchannel(ICC) 55、User Interruption(Synthetic) 200、Pause(Synthetic) 137。

**地基小结**：**时长/结构阈值当家**（<1s & <2 词；0.4/1.0/5s 等都是构造用常数）；
唯一的人工 GT 是 backchannel 的**时机分布**（且来自另一个语料 ICC，非本 benchmark 自标）。
转写用 Nvidia parakeet-tdt-0.6b-v2（词级时间对齐）。

---

### 1.3 Full-Duplex-Bench v1.5（arXiv:2507.23159）

- **引用信息**：Guan-Ting Lin, Shih-Yun Shan Kuan, Qirui Wang, Jiachen Lian, Tingle Li,
  Shinji Watanabe, Hung-yi Lee. *Full-Duplex-Bench v1.5: Evaluating Overlap Handling for
  Full-Duplex Speech Models.* arXiv:2507.23159v4（v1 2025-07-30；v4 2026-04-26）。
  **Accepted by ICASSP 2026**。代码/数据：github.com/DanielLin94144/Full-Duplex-Bench
- **取证方式**：【原文✓】抓取 arXiv HTML 全文，关键句本地回核

**行为类别（4 类 overlap 场景）**：

> "The benchmark simulates four representative overlap scenarios: **user interruption, user backchannel, talking to others, and background speech.**"

**逐类：Capability Tested + Data（原文）**：

- **User Interruption**："**Reactive turn-yielding and semantic repair.** When a user barges in (e.g., "Wait, I want to …"), an effective system must cede the floor rapidly and address the new query."／"We synthesize **200** contextually relevant interruptions using the same speaker voice as the initial query, with no acoustic channel differences."
- **User Backchannel**："**Filtering non-floor-taking cues.** Listeners often produce short affirmations (e.g., "uh-huh") to signal engagement, not to take the turn. The model must ignore these cues to maintain conversational flow."／"We synthesize **99** backchannel utterances (e.g., yeah, right, mm-hmm) from a curated list"
- **Talking to Others**："**Addressee detection and social appropriateness.** ... the model must recognize that it is not the intended recipient and gracefully resume."／声学仿真："volume reduced by 8 dB, a high-shelf filter attenuating frequencies above 4 kHz by 5 dB, and two reflections added at 45 ms (–6 dB) and 120 ms (–12 dB)"
- **Background Speech**："**Robustness to ambient acoustic interference.**"／"reduce the volume by 15 dB, apply a 3 kHz low-pass filter, and add echo with a 100 ms delay (–10 dB)"

**🔴 判官 —— GPT-4o 四分类（本簇最重要的分类协议）**：

> "we classify its post-overlap event response into one of four categories using **GPT-4o**:
> • **Respond**: Addresses the content of the overlapping speech.
> • **Resume**: Ignores the overlap and continues its prior utterance.
> • **Uncertain**: Expresses confusion (e.g., "Could you repeat that?").
> • **Unknown**: Produces an irrelevant response or remains silent."

> "Crucially, the text-based GPT-4o evaluator operates solely on ASR transcripts from (Parakeet-TDT) ...
> **strictly separating the evaluation modality (text) from the generation modality (audio) to minimize bias.**"

**时长度量（式 1、2，Silero-VAD 检测）**：

> "**Stop latency** is the interval from the onset of overlapping user speech to the moment the model stops speaking: t_stop = t_model_stop − t_user_start."
> "**Response latency** is the interval from the end of the overlapping speech to the model's next utterance: t_resp = t_model_start − t_user_end."

**场景化期望（Table 1，即"判据表"）**：

| Scenario | Expected | Stop Latency | Response Latency |
|---|---|---|---|
| User Interruption | Respond | Low | Low |
| User Backchannel | Resume | High | Low |
| Talking to Others | Resume | High | Low |
| Background Speech | Resume | High | Low |

**全文唯一的显式数值判据**（§4.2）：

> "Timing-wise, effective human-like repair typically requires **t_resp ≤ 1.5 s**."

**地基小结**：声学仿真 + **语义判官（GPT-4o，只吃 ASR 文本）** + VAD 时长 + 会话分析惯例
（bc 不夺话的依据是 "In conversation analysis, generic backchannels (e.g., "uh-huh", "yeah")
conventionally invite the speaker to continue rather than take the floor"）。
**无人工行为标签**——全自动。

---

### 1.4 Full-Duplex-Bench v2（arXiv:2510.07838）※ 搜索摘要曾漏报

- **引用信息**：Guan-Ting Lin, Shih-Yun Shan Kuan, Jiatong Shi, Kai-Wei Chang, Siddhant Arora,
  Shinji Watanabe, Hung-yi Lee. *Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for
  Duplex Dialogue Systems with an Automated Examiner.* arXiv:2510.07838v2（v1 2025-10-09）。
  **Accepted by ACL 2026**。**取证**：【原文✓】HTML 全文

**行为类别：不是"行为清单"**，而是「任务族 × 三个评分维度」：

- 任务族（§2.3）：**Daily / Correction / Entity Tracking / Safety**（Safety 含 11 类政策场景）
  > Correction: "self-repairs that occur mid- or cross-turn ... (e.g., "I want a cold coffee" → "Oh, please make it hot")"
- 评分维度（§2.4，判官 = **Gemini 2.5 Flash**，1–5 分）：
  > "**Turn-Taking Fluency (TT, per event)**: Evaluates how natural and well-timed the Evaluatee's responses are at each turn-taking event, **including overlaps and handoffs**."
  > "**Multi-Turn Instruction Following (IF, per event)**"；"**Task-Specific Metric (global)**"

**🔴 注意方向**：v2 里 backchannel 是**考官（Examiner）的行为**（施压手段），不是被评测的指标：

> Fast pacing: "The Examiner may proactively advance the dialogue by speaking immediately upon stage
> completion or **to provide backchannels**, even if the Evaluatee is still talking."
> Slow pacing: "The Examiner replies only after an end-of-turn (EOT) or a long pause; **no barge-in**."

**地基**：LLM 主观 1–5 分（无 v1 的 TOR/时长阈值）。

---

### 1.5 Full-Duplex-Bench v3（arXiv:2604.04847）※ 搜索摘要曾漏报

- **引用信息**：Guan-Ting Lin, Chen Chen, Zhehuai Chen, Hung-yi Lee. *Full-Duplex-Bench-v3:
  Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency.*
  arXiv:2604.04847v1（2026-04-06）。**无 venue**（comment 为 "Work in progress"）。**取证**：【原文✓】HTML 全文

**行为类别：五类 disfluency（§3.2 原句）**：

> "Each recording is annotated for **five disfluency categories**, each targeting a distinct failure mode:
> **false starts** (abandoning an intent for a new one) ...; **self-corrections** (updating parameters
> mid-sentence) ...; **fillers** (e.g., um, uh) ...; **pauses** (mid-utterance silences) and
> **hesitations** (filler–repetition combinations) test end-of-turn detection robustness."

**判据（§4.1，全为规则/日志，无人工标注）**：

> "**Turn-take rate** measures the fraction of turns with a natural-timing response.
> Base latency is Δt = t_agent_start − t_user_end; **if Δt < 0, the event is an Interruption**."
> 三段延迟分解："**First Response Latency** ... **Tool Call Latency** ... **Task Completion Latency**"
> "We also report the **Filler Rate**: the fraction of scenarios where the agent emits a content-free
> filler sentence (e.g., "Sure, let me look that up") before the substantive response"

**🔴 两个洞**：(a) **没有 backchannel 类别**（DuplexAct 的 Table 1 把它记为 AB ✗）；
(b) "natural-timing response" **未给出显式秒数阈值**（原文未定义，非"我没找到"）。

数据：**100 recordings from 12 speakers**；"we capture 30 seconds of each speaker's actual ambient
environment rather than appending digital silence"；21 scenarios 含 self-correction。

---

### 1.6 DuplexAct-Bench（arXiv:2609.39446）★ 本簇行为分类最全

- **引用信息**：Keyue Xing, Wentao Ding, Mengmeng Wang, Wenming Tu, Zilong Zheng, Yipeng Kang（BIGAI）.
  *DuplexAct-Bench: Broadening Full-Duplex Speech Evaluation toward Proactive Interaction across
  Diverse Behavioral Requirements.* arXiv:2609.39446v1（2026-09-30）。**无 venue**。
  Project page: alitaxky.icu/DuplexAct-Bench/。**取证**：【原文✓】HTML 全文

**行为类别（6 类，§1 原句）**：

> "**Agent Interruption**: intervening during an ongoing user turn；
> **User Interruption**: yielding when interrupted；
> **Interruption Resistance**: maintaining an ongoing activity despite non-disruptive user input；
> **Active Silence**: withholding speech when silence is appropriate；
> **Proactive Initiation**: initiating speech without an explicit request；
> **Agent Backchannel**: providing brief floor-preserving responses during the user's turn"

**三种情境条件**：

> "**Pre-session**, where the behavioral requirement is established through a persistent profile before
> interaction; **In-session**, where it is explicitly introduced during the streamed interaction; and
> **No-explicit**, where no behavioral instruction is given and the appropriate behavior must be inferred
> from the semantic or acoustic context."

规模：**1,290** streaming trials（英/中）、30 scenarios、12 systems，用户音频 ~80-ms 块、RTF 1.0。

**逐类 Timing 成功判据（§2.2.2）—— 关键设计：阈值是逐试次标注的区间，不是全局常数**：

> "We therefore define **behavior-specific Timing success criteria**... For latency-based metrics, `L`
> is defined reasonably for each behavior...; failures are assigned the corresponding evaluation interval."

- Agent Interruption: "Success requires the first semantically valid interruption to occur **after the annotated earliest valid interruption point and before the end of the user audio**."
- User Interruption: "Success requires both **yielding within the designated yield interval** and **initiating a semantically valid response ... within the response interval**. We define `L = L_yield + L_resp`"
- Interruption Resistance: "Success requires maintaining the intended ongoing activity, or satisfying the scenario-specific recovery requirement, under non-disruptive user input."
- Active Silence: "Success requires **silence throughout the annotated evaluation interval**."
- Proactive Initiation: "Success requires the first semantically valid proactive utterance to **fall within the annotated valid region**."
- **Agent Backchannel**: "each trial is divided into **consecutive 1-s windows**。For Pre-session and In-session trials, windows overlapping predefined backchannel positions serve as ground truth. For No-explicit ... windows overlapping dataset-provided backchannels serve as ground truth, while **Qwen3.8-Omni-Flash-Realtime, Doubao-Seed-2.1-Lite, and MiniCPM-o-4.5 independently label the same windows** for additional opportunities." → Gaussian 平滑 σ=1.0s → **BAS (Backchannel Alignment Score)** = ∫min(q_M,q_ref)dt / ∫max(q_M,q_ref)dt

**主指标 BCR**（§3.1）：

> "the Behavioral Correctness Rate (BCR), defined as the fraction of trials with **Content ≥ 2.5** that also satisfy the corresponding Timing criterion."

（Content = GPT-4o 三维 0–5 + AnyAudio-Judge 的 Prosodic Appropriateness 0–1。）

**🔴 它给的第三方覆盖矩阵（Table 1，对本项目做评估盘点极有用）**：
维度：AI=Agent Interruption, UI=User Interruption, IR=Interruption Resistance,
AS=Active Silence, PI=Proactive Initiation, AB=Agent Backchannel。
✓=系统性覆盖；∘=子集/近似；✗=无显式评估。

| Benchmark | AI | UI | IR | AS | PI | AB |
|---|---|---|---|---|---|---|
| Talking Turns | ∘ | ∘ | ∘ | ∘ | ✗ | ∘ |
| **FDB v1** | ✗ | ✓ | ✗ | ∘ | ✗ | ∘ |
| **FDB v1.5** | ✗ | ✓ | ∘ | ✗ | ✗ | ✗ |
| FD-Bench | ∘ | ✓ | ∘ | ✗ | ✗ | ✗ |
| FLEXI | ∘ | ✓ | ∘ | ∘ | ✗ | ∘ |
| HumDial-FD | ✗ | ✓ | ∘ | ∘ | ✗ | ✗ |
| **FDB v2** | ∘ | ∘ | ∘ | ✗ | ✗ | ✗ |
| **FDB v3** | ∘ | ✗ | ✗ | ∘ | ✗ | ✗ |
| DuplexSLA | ✗ | ✓ | ∘ | ∘ | ✗ | ✗ |
| DSB-IFEval | ∘ | ✓ | ∘ | ∘ | ∘ | ∘ |
| DuplexAct-Bench | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> "✓ denotes systematic coverage ...; ∘ coverage under a subset of these conditions or closely related
> coverage; ✗ no explicit evaluation."

（表内 Talking Turns / FLEXI / DuplexSLA / DSB-IFEval 是**另一批 benchmark**，本文档暂未单列，列入 §7 待补。）

---

### 1.7 HiPLEX（arXiv:2610.07727）※ 方法论文；把 FDB v1 四行为**分量级再操作化**

- **引用信息**：Kyudan Jung, Hyunsin Park, Yoonhyung Lee, Jinhwan Park, Jinhyeok Yang, KiHyun Nam,
  Jaegul Choo, Jinkyu Lee. *HiPLEX: Hierarchical Policy Factorization for Full Duplex Speech
  Language Models.* arXiv:2610.07727v1（2026-10-06），cs.SD，34 页。**无 venue**。**取证**：【原文✓】HTML 全文

**它转述的 FDB v1 四行为**（Sec. 4）：

> "We evaluate on Full-Duplex-Bench v1 ..., which measures **four full-duplex behaviors: holding
> silence during hesitation, backchanneling without taking the floor, taking turns after user yield,
> and responding to user interruptions.**"

**训练侧行为集合**（Sec. 3.1，73,561 annotated interaction windows）：

> "**turn taking, pause handling, backchanneling, and interruption handling.** These windows specify
> when the model should acknowledge the user, respond after a user turn, remain silent during a user
> hesitation, or yield to an interruption and re-enter after it ends."

**🔴 统一的"地基"——短/长片段划分（Appendix D）**：

> "Silero VAD extracts speech spans U from the generated audio. Let S be the speech episodes obtained
> by merging these spans separated by **at most 1 s**. **Episodes of duration at most 1 s are short;
> longer episodes are sustained.**"

- **Backchannel 分量**："机会窗口 W 与**短片段** B 匹配；F1（匹配上限 **1 s**）、Timing（exp[−2·min_b d(b,W_j)]）、
  Miss、False alarm、Overlong（"**Sustained episodes incur** r^(long) = −0.5·1[|L|>0]"）"
- **Interruption 分量**："r^(yield) = −min(1, D_U(β₀+0.16, β₁) / 1 s)"，"The **0.16 s grace is two audio frames**"
- **Pause 分量**：Intrusion 二元 −1[∃s: |s| ≥ 1s]；Speech fraction
- **Turn taking**：τ = user turn end，t⋆ = 第一个 sustained episode onset ≥ τ；onset σ=0.8s、κ=4s

**它对 FDB v1 指标的两条批评（对本项目评估设计极有参考价值）**：

> "we found **two of the v1 metrics to be degenerate under the official protocol**"
> "The **official smooth-turn TOR is nearly saturated** for both methods and therefore provides limited
> evidence about turn-boundary behavior."
> "Official scores largely ask whether an event occurred, whereas **human timing is distributed**. ...
> **Thus a model can pass a binary rule yet answer consistently late or acknowledge too often.**"
> → 改用 Wasserstein-1 距离比模型与人类（Seamless–Fisher）的时序分布

**合成的 pause 任务里"开场问候"会被计成 takeover**（Appendix I）：

> "the annotated hesitation sits only a second or two into a short clip, so it **coincides with the
> conversation-start boundary**. ... with the serving stack's silence warm-up removed, even the base
> model greets and scores **0.985**, and **that greeting, over the synthetic clip, is counted as a takeover.**"

---

## 2. 语料 / 标注体系类

### 2.1 CANDOR（Science Advances 2023）★ 本项目主力语料

- **引用信息**：Andrew Reece, Gus Cooney, Peter Bull, Christine Chung, Bryn Dawson, Casey Fitzpatrick,
  Tamara Glazer, Dean Knox, Alex Liebscher, Sebastian Marin. *The CANDOR corpus: Insights from a large
  multimodal dataset of naturalistic conversation.* **Sci Adv 9(13), eadf3197 (2023-03-31)**.
  DOI 10.1126/sciadv.adf3197；预印本 arXiv:2203.00674。
- **取证**：【原文✓】预印本 PDF 全文（116 页含 SOM），本人回核 Backbiter 三规则与两张词表

**行为类别**：**backchannel 有独立的算法 + 词表**；**interruption 没有独立标注字段**
（只有 WSO 这个概念，且两处出现均非定义句）。

**bc 定义（预印本 §1.3, p.29 原句）**：

> "Back-channels—the short words and utterances that listeners use to signal speakers **without taking
> over the floor** (e.g., 'yeah,' 'mhm,' 'exactly')—represent a third basic mechanical feature of conversation…"

**🔴 判定 = Backbiter 算法三条规则（pp.30–31 原句，这是 CANDOR 真正的操作化定义）**：

> "Backbiter creates two transcript entries for each speaking turn: first, the words uttered by that
> turn's active speaker; second, the back-channel phrases, if any, uttered by that turn's listener.
> Backbiter employs **three basic rules** to identify and reclassify utterances as back-channel turns:
> **(1) a back-channel turn must be three words or fewer; (2) a back-channel turn must contain >50%
> back-channel words** (e.g., 'yeah,' mhm,' 'exactly'); and finally **(3) a back-channel turn must not
> start with a prohibited word**, such as 'I'm…'."

**词表（SOM pp.87–88，全部列出）**：

- `backchannel_CUES`（30 词）：a, ah, alright, awesome, cool, dope, e, exactly, god, gotcha, huh, hmm,
  mhm, mm, mmm, nice, oh, okay, really, right, sick, sucks, sure, uh, um, wow, yeah, yep, yes, yup
- `NOT_backchannel_CUES`（15 词）：and, but, i, i'm, it, it's, like, so, that, that's, we, we're, well, you, you're

**🔴🔴 论文中没有任何时长阈值**（无 0.3s / 0.5s 之说）。项目 §5.3 的 `dur>0.3` / `dur>0.5` 是
**本项目自定采样口径**，此前一直被列在"CANDOR 行"里 —— 引用时必须写成
"本项目在 CANDOR 上的采样口径"，**不能写成 CANDOR 的定义**。

**interruption 的实际地位**（两处原句，均非定义句）：

> 正文 §"Turn exchange"："…overlap (between-speaker overlap), and within-speaker overlap (WSO),
> (when one speaker starts speaking in the middle of another speaker's turn, **such as in the case of
> an attempted interruption or back-channel**)"
> SOM："…classified as either a Gap …, Pause …, Overlap …, or **WSO (within-speaker overlap, an
> interruption)**."

**字段出处辨析**：`backchannel_start` / `backchannel_stop` 等是 **ConvoKit 的字段**
（"The start time of the first backchannel during this turn."），**不是论文定义的字段**——
论文只定义算法，字段是工具层。⇒ 引用"`backchannel_start` 非空即 bc"时要说清是 **ConvoKit 口径**。

**其他事实**：Gaps/Overlaps 近似正态、中位数 380ms / −410ms；gaps 52.1% / overlaps 47.9%；
median between-speaker interval 80ms。

---

### 2.2 Behavior-SD（NAACL 2025）★ 项目 L1 合成数据来源

- **引用信息**：Sehun Lee\*, Kang-wook Kim\*, Gunhee Kim. *Behavior-SD: Behaviorally Aware Spoken
  Dialogue Generation with Large Language Models.* **NAACL 2025 (Long), pp. 9574–9593**.
  DOI 10.18653/v1/2025.naacl-long.484。数据：HF `yhytoto12/behavior-sd`；模型 BeDLM。
- **取证**：【原文✓】ACL PDF 全文抓取，本人回核 Table 1 与高斯参数段

**行为类别（Table 1，原句）——不是二/三分类，而是「四个行为维度 × 三档等级」**：

> "(V) **Verbosity** The length of utterances during a speaker's turn."
> "(F) **Filler words** The occurrence of filler words (e.g, 'like,' 'um', 'you know') during a speaker's turn."
> "(B) **Backchannels** The frequency of backchannel responses (e.g, 'yeah,' 'uh-huh') during the other speaker's turn."
> "(I) **Interruptions** The frequency of interrupting the other speaker's turn to begin speaking."
> 表题："…**Each behavior type is classified into three levels: none (0), moderate (1), and frequent (2).**"

外加 **utterance 级三分**：Backchannel / Interrupt / None。

**🔴 utterance 三分的时间判据 = 高斯采样，不是硬阈值**（§3.2原句）：

> "The previously generated utterances are categorized as Backchannel, Interrupt, or None. …
> none-type utterances are appended with an inter-utterance gap sampled from **N(0.4s, 0.2s)**,
> backchannels follow after a brief delay sampled from **N(0.2s, 0.02s)**, and interrupts overlap the
> preceding speech by a duration drawn from **N(0.45s, 0.05s)**."

（参数 "derived from statistics in a real-world spoken dialogue corpus (Reece et al., 2023)" = CANDOR。
注意：三个均值 **Int 0.45s / BC 0.2s 都 < 0.5s**。）

**bc 的生成机制**（§3.1）：

> "Backchannels typically occur at natural pause points in dialogue, often referred to as
> **backchannel opportunity points (BOPs)**."（GPT-4o-mini 检测 BOP、GPT-4o 生成 bc）

**interruption 的七子类（§3.1 原句）**：

> "We further categorize interruptions into **seven types: agreement, disagreement, floor taking,
> tangentialization, clarification, assistance, and topic change**, as follows Goldberg (1990)."

**声道**："two-channel audio, where each channel corresponds to a single speaker" ——
**论文没有说哪路是 L/R**；"spk0=L spk1=R" 是项目实测（932/932 可分离）。

**🔴 论文中没有 0.5s / 0.1s 阈值**（全文 grep 只命中一个表号）：项目表里 Behavior-SD 行的
"realized >0.5s→Int、0.1–0.5s→BC" 是**项目自己的 realized 分箱**，不是论文定义。

规模：abstract "over 100K spoken dialogues (2,164 hours)"；108,174 dialogues / 1,091,323 utterances。
地基：**生成端元数据（L1）**，行为由 LLM 按采样 trait 等级生成、时间由高斯采样放置；
人**只出现在评测端**（§7.1 三名人类评估者，5 点 Likert）。

---

### 2.3 DuplexGen（arXiv:2607.26178v2，EMNLP 2026）★ 唯一「人标 + 仅文本」的体系

- **引用信息**：Takyoung Kim, Kang-wook Kim, Sang Hoon Woo, Julia Hirschberg, Gunhee Kim,
  Dilek Hakkani-Tür. *DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues.*
  arXiv:2607.26178v2（v2 2026-08-29；arXiv 列表注明 EMNLP 2026）。
  **本地有一手材料**：`pilot_study/refs/duplexgen_2607.26178v2.pdf` + 文本抽取 `duplexgen.txt`（本项目所存）。
- **取证方式**：【原文✓】本人逐字回核本地 PDF 文本抽取（行号即 `duplexgen.txt` 行号）
- **本项目已做的二次分析**：`2026-09-23_duplexgen_annotation_reliability.md`（可靠性）、
  `2026-09-23_duplexgen_humdial_recon.md`（产物结构）。**引用其结论前先读那两份的"待核"标记。**

**行为类别（三分类，🔴 四套命名并存——跨产物对齐前必须先查名字）**：

| 语境 | silence 侧 | backchannel 侧 | floor 侧 |
|---|---|---|---|
| 数据 `annotations/*.jsonl` | `silent` | `backchannel` | `take_floor` |
| 数据 `dialogues/*.jsonl` | `silence` | `backchannel` | `floor_taking` |
| 人类标注界面（§C.2） | `Continue`（= Silence） | `Backchannel` | `Take Floor` |
| LLM 标注 prompt（§B.3） | `silence` | `backchannel` | `floor_taking` |

**§3 的两类定义（原句，`duplexgen.txt:303-335`）**：

> "We focus on two linguistically expressible turn-taking behaviors, **floor-taking** and **backchanneling**,
> because they represent the fundamental dichotomy of listener responses (i.e., **actively claiming versus
> passively acknowledging the conversational floor**) and are both expressible in text, making them directly
> actionable for LLM assistants. Grounded in conversation-analytic theory (Sacks et al., 1978), floor-taking
> refers to entering at or near transition-relevance places, **including competitive entries like interruptions**.
> Backchanneling (e.g., "uh-huh") **displays attention without claiming the floor**."

**🔴 人类标注指南（§C.2，原句）里的三动作 —— 注意 "Take Floor" 的覆盖范围**：

> **"Backchannel"**: "The AI provides intermediate feedback signals (e.g., "um-hum", "right") without taking the floor."
> **"Take Floor"**: "The AI interrupts the user (**barge-in**) **or** responds at a verbal completion point to speak."
> **"Continue"**: "The AI maintains silence and continues listening."
> "If no action is needed, it is considered as **Silence** (Continue)."

⇒ 在 DuplexGen 的标签空间里，**打断（barge-in）与正常接话是同一个标签 `Take Floor`** ——
它没有单独的 interruption 类。这一点与 CANDOR / Behavior-SD / 本项目 L2 的"bc / int / none 三分类"**结构性不同**。

**生成侧的 bc 词表（§B.5 insertion prompt，原句）**——词表型操作化的一个实例：

> "# Backchannel definition: -A very short listener vocalization showing attention/understanding
> -It does NOT take the floor and does NOT change the topic -The speaker continues immediately after it"
> "# Output constraints: -**1–2 words only** (3 only if absolutely necessary) -lowercase -no emojis..."
> "# Decision rubric (choose the mildest that fits): -ack/continue: mhm, mm-hmm, uh-huh, yeah, right, okay
> -mild surprise/interest (new/unexpected info): oh, wow, really -confusion (ONLY if unclear/contradictory): huh?"
> "# Default: -If uncertain, output "mhm"." ／ "Allowed outputs: mhm | mm-hmm | uh-huh | yeah | right | okay | oh | wow | really | huh?"

**标注协议（§3.2.3，原句）**：

> "we present text chunks incrementally"（避免 retroactive annotation bias）
> "We recruit **248** English-speaking participants ... via Prolific"
> "**Because turn-taking is inherently subjective, each sample is annotated by five participants**,
> allowing human judgments to be treated as distributions rather than single decisions."
> "Rather than reducing human annotations for each slot to a majority-vote label, we use
> **empirical annotation counts as a soft label** … a hard label discards information about human disagreement."

**可靠性（🔴 本项目自算，非论文自报——论文全文未报任何 IAA）**：

- Fleiss' κ = **0.0506**；Gwet's AC1 = 0.3692；边际 p_j = silent 0.6520 / bc 0.2621 / take_floor 0.0858
  （`2026-09-23_duplexgen_annotation_reliability.md:116`）
- 二值塌缩 κ = 0.005（含 0）（`2026-09-22_claims_ledger.md:162-165`）
- 项目已核实的负结果：**论文全文没有任何 inter-annotator agreement 统计**（`:107-108`）

**作者自述的边界（Limitations 节原句）**：

> "DuplexGen calibrates turn-taking actions from **incrementally revealed transcripts** and therefore
> **does not directly condition on prosody, pause duration, intonation, speech rate, overlap, or visual
> cues.** Although the downstream listening study evaluates acoustically realized outputs (§4.3.3),
> **it does not establish agreement between text-based and audio-conditioned preference judgments.**"

**地基小结**：**人标（仅文本、无音频、增量呈现、每人 5 票软标签）** ——
本汇编里唯一的非时长/非规则体系；但 κ≈0.05 表明"意图标注"在这个形式下不可靠。
校准集规模：120 对话（20/task，Table 4）。

---

### 2.4 HumDial-FDBench（ICASSP 2026 Challenge，arXiv:2604.21406）

- **引用信息**：Chengyou Wang, Hongfei Xue, Guojian Li, Zhixian Zhao, Shuiyuan Wang, Shuai Wang,
  Xin Xu, Hui Bu, Lei Xie. *Full-Duplex Interaction in Spoken Dialogue Systems: A Comprehensive Study
  from the ICASSP 2026 HumDial Challenge.* arXiv:2604.21406（2026-04-23）。
  数据：GitHub `ASLP-lab/HumDial-FDBench`（只放 test 集）。
- **取证**：【原文✓】arXiv 全文 + GitHub README；发布数据的格式结论由**项目全量审计 + agent 独立抽查**双重确认

**行为类别 = 场景级（不是帧级行为标签）**：

- **Interruption 五子场景**：Follow-up Question / Negation or Dissatisfaction / Repetition Request /
  Topic Switch / Silence or Stop
- **Rejection 四子场景**：User Real-time Backchannels / Pause Handling / Third-party Speech /
  Speech Directed to Others

**user backchannel 的唯一定义（§3.1 原句）**：

> "involve **short acknowledgments** (e.g., 'uh-huh,' 'yeah') that **should not interrupt an ongoing response**."

**判据（§4）**：ASR 文本 → **DeepSeek-V3 判四类**（Respond / Resume / Uncertain / Unknown）；
中断场景只有 Respond 算对、拒绝场景只有 Resume 算对；延时用 Silero-VAD 边界；
总分 **S_Total = 0.4 × S_Int + 0.4 × S_Rej + 0.2 × S_Delay**。

**🔴 "dual-channel" 声称 vs 发布数据**：论文/README 称 "high-quality **dual-channel** dataset"，
但**发布的 test 包实测为单声道**且只含被标注一方：
项目全量审计（9098/9098 wav 全 ch=1 / sr=16000，`2026-09-23_humdial_schema.md:74-77`）
＋ agent 本次独立抽查（wav 头 + 40 个 json 的键集合）**双重确认**。
⇒ 这是"描述物 ≠ 被描述物"的又一实例；**引用它的"dual-channel"前必须先说这是论文声称**。





## 3. 全双工系统类

> **本节的总结论（先读这条）**：系统论文里的 "full-duplex" **绝大多数是架构级定义**
> （"同时听与说 / 没有话轮边界"），**行为级定义出现在另一条独立的线上** ——
> turn-state 预测模块（§3.5 X2-Turn、§3.6 SoulX-Duplug）与评测基准（§1）。
> 四个经典系统（Moshi / dGSLM / LSLM / Freeze-Omni）**都没有可用的 backchannel 行为分类**；
> 其中 dGSLM 是**提出了结构判据又明确拒绝实现**的那一篇（§3.2 的红条）。

### 3.1 Moshi（arXiv:2410.00037v2）

- **引用**：Défossez, Mazaré, Orsini, Royer, Pérez, Jégou, Grave, Zeghidour. *Moshi: a speech-text
  foundation model for real-time dialogue.* arXiv:2410.00037v2（v2 2024-10-02）。【原文✓】HTML 全文
- **full-duplex 定义 = 纯架构级**：
  > "it always listens and always generates sound, either speech or silence—real-time conversational LLM."
  > "these models cannot handle full-duplex communication, **where there is no boundary between speaker
  > turns**, as any side of the conversation can be active at any time."
- **行为分类：无**。"backchannel" 全文仅 3 次，且都不是标签体系：
  一处动机句（"non-interrupting interjections such as 'OK' or 'I see'"）、两处是**造训练数据的提示词**
  （"Use some backchanneling." / "Use a lot of backchanneling."）。
- **评估**：借 dGSLM 的 turn-taking 指标（IPU / gap / pause / overlap，见 §3.2），
  加 DialoGPT 困惑度；Fisher 1000 条 10 秒 prompt。
- **Inner Monologue 不是行为概念**：是文本-音频对齐技术（text token 作音频 token 的前缀）。

### 3.2 dGSLM（TACL 2023）★ 提出了结构判据、又明确拒绝实现的那一篇

- **引用**：Nguyen, Kharitonov, Copet, Adi, Hsu, Elkahky, Tomasello, Algayres, Sagot, Mohamed,
  Dupoux. *Generative Spoken Dialogue Language Modeling.* **TACL 2023, 11:250–266**；
  arXiv:2203.16502v2。【原文✓】TACL PDF + ar5iv，两版 §3.4 逐字一致
- 🔴 **它全文从不自称 "full-duplex"**（TACL PDF grep "duplex" = 0 命中）；
  是后来者追认的（Moshi §2.3："The only previous full-duplex dialogue system is dGSLM"）。
- **结构定义（§3.4, p.254）**：IPU = "continuous stretch of speech in one speaker's channel, delimited by
  a VAD silence of **more than 200ms** on both side"；silence 分 gap（异说话人）/ pause（同说话人）；
  turn = 同一说话人相邻 IPU 被 pause 分开后重组。
- 🔴🔴 **最关键的一句（原文）——结构式 bc/int 判据被提出后当场放弃**：

  > "Overlap could also theoretically be **subdivided into backchannel** (when it is rather **short IPU
  > contained within an IPU of the other speaker**) **and interruption** (when it **starts within an IPU
  > of the other channel and continues after its end**), but **the exact definition is dependant on
  > high-level linguistic features, which we will not attempt to extract here.** In our analysis, we will
  > therefore tally the distribution of duration of IPUs, gaps, pauses and overlaps..."

  ⇒ 这是"时长/结构判据**已知不足**"的最早自述之一（2023），与项目 §5.5 的 F3 反例（NONOVER 的 bc）
  和 §1.2 FDB 的自述（"differs across the literature... left for future work"）构成同一条证据链。
- 另有一处动机性 bc 定义（引 Yngve 1970; Schegloff 1982）：
  "content-neutral verbal information (e.g., 'hmm', 'yeah') or non-verbal vocalization (e.g., laughter),
  used to convey a listening attitude (back-chanelling)"。
- 评估：Turn-taking Event Statistics（pyannote VAD）、Event Consistency、WPM/LPM/FWR/FTO、语义 PPL/VERT。

### 3.3 LSLM（arXiv:2408.02622v1）

- **引用**：Ma, Song, Du, Cong, Chen, Wang, Wang, Chen. *Language Model Can Listen While Speaking.*
  arXiv:2408.02622v1（2024-08-05，唯一版本；⚠️ 未核实到 venue）。【原文✓】HTML 全文
- **架构级定义（§2.2，电信视角 simplex/half/full 三级分层）**：
  > "Full duplex SLMs ... have the capability to **listen and speak simultaneously**, allowing for
  > turn-taking whenever a human interrupts the machine."
- **行为分类：无类别清单**，只有 interruption 一个二值事件，用 IRQ 特殊 token 操作化（§4.3）：
  > "we add an **interruption token IRQ** to the tokenizer vocabulary to allow the model to terminate
  > early if turn-taking occurs."
- **判据（阈值型，§6.1 原句）**：
  > "A successful turn-taking is defined as the model stopping speaking **within the [0, 2μ] interval
  > (1 second in our setting)** after the interruption begins." → 四格（TP/FN/FP/TN）→ P/R/F1。
- 全文无 backchannel / barge-in。

### 3.4 Freeze-Omni（arXiv:2411.00774v5）

- **引用**：Wang, Li, Fu, Shen, Xie, Li, Sun, Ma. *Freeze-Omni: A Smart and Low Latency Speech-to-speech
  Dialogue Model with Frozen LLM.* arXiv:2411.00774**v5**（2024-12-08）。
  ⚠️ **必须引 v5**（v1 与 v5 的 §2.4 措辞不同）。【原文✓】HTML 全文（v1 + v5 对读）
- 🔴 **全文 grep "full-duplex" = 0 命中**（v1、v5 都是 0）；它只用 "duplex dialogue"。
- **三个 chunk 级状态（§2.4 原句）**：
  > "**Three states are defined here, state 0** indicates that the current LLM can continue to receive
  > speech, and **state 1 or 2** indicates that the current chunk is the end of the speech. **State 1**
  > means that the user will interrupt the dialogue and the LLM will perform a new generate stage, and
  > **state 2** means that there is no need to interrupt the dialogue."
- **无 backchannel 类别、无阈值判据**；且**状态标签如何构造、由谁标注——论文未写**（原文缺项）。
- 评估只有延迟（statistical / non-statistical latency，约 160–320ms），无行为基准。

### 3.5 X2-Turn（arXiv:2608.10878v3）★ 本项目 L3 主标注器

- **引用**：Kaiqi Fu, Rime Wen, Altman Lin, Shawn Qin, Roy Gan, Hao Wang, Qian Wang.
  *X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction.*
  arXiv:2608.10878v3（v3 2026-09-08）。【原文✓】HTML 全文
- **五个 turn state token（§2.2 逐字）**：

  > "We define **five** turn state tokens to represent the evolving state of a user turn:"
  > - "`<|idle|>` represents **user silence**, corresponding to non-speech segments derived from forced alignment."
  > - "`<|noidle|>` indicates **active speech without semantic content** (e.g., the initial syllables of an utterance)."
  > - "`<|incomplete|>` denotes active speech containing **partial** semantic content."
  > - "`<|complete|>` signifies active speech with **complete** semantic content."
  > - "`<|backchannel|>` captures **user backchannel signals or filler words** (e.g., 'um', 'ah')."

- 🔴 **它显式与 SoulX-Duplug 的定义分道（§2.2 原句）**：
  > "**Unlike the definition adopted in prior work [27], our `<|noidle|>` state refers specifically to
  > early active speech for which sufficient semantic content has not yet been observed. It does not
  > represent non-speech or background noise.**"
  ⇒ 引用两家的状态名时**不可混为一谈**。
- **判据来源 = 语义描述 + LLM 标注，无阈值**：标注器 "Qwen3.5-Plus as an LLM annotator to perform
  word-level semantic turn state labeling"；强制对齐给时间戳；帧率 **80ms**。
- ⚠️ **论文 5 类 vs 官方推理接口 6 类**：README 的六状态
  `idle / noidle / speaking / turn_end / backchannel / uncertain`（`speaking`=`incomplete`、
  `turn_end`=`complete` 是别名；**`uncertain` 只在推理接口，论文未定义**）。
  本项目 CANDOR-FD 的 6 类（§5.6）即抄自接口层 —— **引用时要说清是接口还是论文**。

### 3.6 SoulX-Duplug（arXiv:2603.14877v1）★ 本项目 L3 补充标注器

- **引用**：Yan, Chen, Liu, Ma, Lin, Wen, Xie, Wu, Liang, Zhao, Feng, Qian, Meng, Dai, Yin, Tao, Xie,
  Yu, Wang, Chen. *SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime
  Full-Duplex Speech Conversation.* arXiv:2603.14877v1（2026-03-16）。
  arXiv comments："submitted to Interspeech 2026, under review"。【原文✓】HTML 全文
- **五个 state token（§3.2 逐字）**：
  > "We define five state tokens to model the interaction dynamics in full-duplex spoken dialogue..."
  > - "`<|user_idle|>` indicates that the current audio chunk contains no semantic content, such as silence or noise."
  > - "`<|user_nonidle|>` denotes that the chunk contains semantically meaningful speech."
  > - "`<|user_backchannel|>` **represents user backchannel behavior.**"
  > - "`<|user_complete|>` indicates that the user's utterance is semantically complete and the assistant may take the turn."
  > - "`<|user_incomplete|>` represents that the user pauses but his/her utterance is semantically incomplete so the assistant should wait."
- 🔴 **backchannel 标签只有一句语义定义、零操作化判据**（无时长/词数阈值、无清单）——原文缺项。
  标注器："State labels are annotated using **Qwen2.5-72B-Instruct**."
- **评估用 FDB 指标**（§5.2）：User Backchannel → Resume Rate (RsR)；Interruption → Respond Rate (RpR) + Stop Latency。
- ⚠️ **命名三套并存**：论文（`user_*` 五 token）／README（`"idle","nonidle","speak","blank"`）／
  training-code 分支（项目转述为 `wait`，见 §7 待核）。引用状态名时必须注明取自哪一层。

### 3.7 Fun-Audio-Chat（QwenAudio）

- README 里**没有任何** backchannel / interruption / turn-taking 的行为类别定义，
  只有泛化的 "full-duplex" 宣传语与 benchmark 名（OpenAudioBench、VoiceBench 等）。
  仓内有一份 `Fun-Audio-Chat-Technical-Report.pdf`（**尚未被项目引用过**），若要收录需先读该报告。
- 出处：`third_party/Fun-Audio-Chat/README.md`（本项目本地）。



## 4. backchannel / interruption 专文 与 2026 新动向

### 4.1 backchannel / interruption 行为定义专文

> **本小节的组织方式**：按**判据地基**分四条路线 —— ① 词表／② 序列结构／③ 时长+邻域静默／
> ④ 功能意图（人标）。同一行为在这四条路线上的定义**互不兼容**，这是本汇编最重要的对照之一。
> **取证**：全部抓取原文全文（PDF 解码）；Cathcart / Liesenfeld / Uro / Lebourdais / Paierl 的关键句
> 已本地逐字回核，TurnBench 为两次独立抓取一致（未落盘）。

**① 词表路线 —— Cathcart, Carletta & Klein (2003)**（"continuers"）

- *A Shallow Model of Backchannel Continuers in Spoken Dialogue.* EACL 2003, pp. 51–58（HCRC Map Task 语料）。
  【原文✓】作者自存 PDF 全文（venue/页码来自二手书目，PDF 本身未印）
- 定义："the class of backchannel utterances, **with minimal content**, used to clearly signal that the
  speaker should continue with her current turn"
- 判据：**acknowledge move 过滤** —— "filtered ... by removing any that contained content words or words
  that generally convey acceptance such as *alright*"；Table 1 实测词频（right 29% / okay 14% / mm-hmm 11%…）
- **无时长门槛**；与 floor-taking 的边界是**位置性的**（不引发 turn exchange）。
- 唯一时长相关的数是**插入判据**：停顿 >900ms 即插入（P22/R59/F32）——是停顿阈值，不是 bc 时长。

**② 序列结构路线 —— Liesenfeld & Dingemanse (2022)**（"response tokens"）

- *Bottom-up discovery of structure and variation in response tokens ('backchannels') across diverse
  languages.* Interspeech 2022, pp. 1126–1130（16 语言档案语料）。【原文✓】ISCA PDF 全文
- 🔴 **明确拒绝按形式（词表）检索**："**we do not search corpora for forms that sound like (or are
  translated as) 'yeah' or 'hmm'**. Instead we define structural facts about how turns follow one another"
- 判据 = 序列结构：**(i) 频次 top decile；(ii) 至少一次出现在同说话人 ≥2 连发中**
- 副作用（Limitations，原句）："we **removed response tokens that occur in overlap** … this means we
  **exclude 25 to 45% of response tokens per corpus**" ⇒ 四分之一的 bc 直接落在重叠里，被该研究排除。

**③ 时长 + 邻域静默路线 —— Ekstedt & Skantze (2022)（VAP）** ★ 本项目做帧级建模时最可直接复用的

- *Voice Activity Projection: Self-supervised Learning of Turn-taking Events.* Interspeech 2022,
  pp. 5190–5194；arXiv:2205.09812。【原文✓】ar5iv 全文
- **bc 定义（§3(c) 逐字）**："We identify BCs as **short and isolated VA segments**"：
  "it **cannot be longer than bc-duration (1s)**"；邻域静默 "**pre-silence (1s)** and **post-silence
  (2s)**"；"a BC must be **preceded by VA from the other speaker**"
- **bc vs 夺话的判别（§3(d) SHORT/LONG 任务）**：取 VA 段 onset，判它是"**SHORT（= BC）**"还是
  "**LONG（proper SHIFT）**"；评估区只覆盖 onset 后 **200ms**。
- ⇒ 这是"**时长 + 邻域结构**"最完整的操作化；WildTurn 的 66 词词典即引自这一系（Ekstedt & Skantze 2022）。

**④ 功能/意图人标路线（四篇，含一条对项目文档的撤回）**

- **Uro et al. 2024**（LREC-COLING 2024, pp. 1225–1232；法语媒体语料）——六分 floor-taking：
  Smooth / Back / **CompI（竞争性打断）/ CoopI（合作性打断）** / CompIA / CoopIA（attempt 类）。
  - 🔴🔴 **两条对本项目 L2 口径直接成反例的实测**（§3.1 逐字）：
    > "**25% of the transitions annotated as smooth occur within overlapping speech intervals** …
    > Likewise, **19% of labeled violations were realized without being associated with overlapping speech.**"
    （表：Back n=524 其中 80% 在重叠内；Smooth n=929 其中 25%。）
    ⇒ "有重叠 = 打断 / 无重叠 = 非打断"这类判据**两头都漏**。
  - 信度：TRP Fleiss' κ=0.75；**Backchannel κ=0.56**；interruption/non-interruption κ=0.5；
    "the type of violation seems more complicated to classify"。
- **Lebourdais et al. 2024**（LREC-COLING 2024, pp. 1959–1968）——interruption 定义为**重叠的子类**
  （West & Zimmerman 1975："a subclass of overlapped speech ... with a speaker change that occurs before
  an arbitrary distance to the end of the complete proposition"）；四类：Backchannel / Anticipated turn
  taking / Complementary information / Interruption。作者自述："the definition of interruption in our
  study corresponds to our annotators' notion of interruption, **which is clearly not universal**"。
- **TurnBench（Jiang et al. 2026, arXiv:2608.25218）** ★ 2026 年口径最规范的一篇
  - Table II 逐字：**Backchannel** = "Listener vocalization that **does not claim the floor, regardless of
    duration**."；Interruption = "Listener vocalization that **takes the floor** from the current speaker
    mid-turn."；另有 **Non-floor-taking Interruption** = "Interruption attempt that does not take the
    floor, **indistinguishable from Interruption at onset**."（该类别作**排除区间**处理）
  - 🔴 **"起点不可判定"的明文（§VII）**："at its onset an interruption is indistinguishable from a
    backchannel, so speed costs FPR or recall."（§IV-A 脚注：标签 "dependent on future information
    revealed seconds later"）
  - 信度：三名标注员、帧级 100ms、pairwise Cohen's κ 0.77–0.80 / Fleiss' κ 0.78、onset 边界 F1 0.94–0.96
    （±200ms）、85.8% 事件过金标筛。
  - 事件量：32,820 = TURN 11,791 + BC 7,917 + INT 1,151（+ 8,197 EOT 锚点 / 4,254 mid-turn pause 负例）。
  - ⚠️ **注意它和 FDB v1 的对立**：TurnBench 明说 bc "regardless of duration"，FDB v1 说 bc "<1s 且 <2 词"
    —— 同一年（2025/2026）两个 benchmark 对同一行为给出**互相排斥的判据哲学**。
- 🔴 **Paierl et al.（2025，不是 2024）—— 本项目一处引用经核实不成立**：
  - 实际引用：Michael Paierl, Anneliese Kelterer, Barbara Schuppler. *Distribution and Timing of Verbal
    Backchannels in Conversational Speech: A Quantitative Study.* **Languages (MDPI) 10(8):194, 2025**
    （Published 2025-08-15）。DOI 10.3390/languages10080194。【原文✓】MDPI PDF 全文
  - **撤回理由**（§3.2.1 逐字）："Specifically, we measure **the durations of all PCOMPs (duration)**
    and **the time gap between hearer response tokens and preHRTs (time_gap)**" ——
    **全文没有测量 bc token 自身的时长**；其分布是 **time_gap**（前句结束→bc 起点）：
    "right-skewed, with most values falling within the range of −0.5 to 0.8 s and a median around 0.14 s"。
  - ⇒ `docs/pilot_study/2026-08-23_ari_exp_guidance.md:92` 的「**BC 时长多集中在 0.2–0.5s**｜Paierl」
    **文献不支持**（该行是项目里唯一一条"阈值有文献出处"的记载）。年份也须从 2024 改为 2025。
  - 该文 bc 定义（Table 1 的 `hrt`）："hearer response token; usually short backchannels, continuers,
    acknowledgments, etc., that **do not contain a (new) proposition of their own and do not take up
    the turn**"（语料：GRASS 奥地利德语 12 段/70 分钟，hrt 547 次）。

**⑤ 一个否定结论 + 一条二手转述**

- **不存在 Interspeech / ICASSP 官方的 backchannel detection 挑战赛**（多次检索无果）。
  当前的"benchmark 级"标签定义由 FDB（§1.2）与 TurnBench 承担，二者皆非 challenge。
- Skantze (2017) 的"**<500ms vs >2500ms**"短/长切分 —— **仅经 IWSDS 2025 综述转述，未取得原文**，
  引用前必须回核（记入 §7）。
- 未覆盖：Levitan 2011（能量比）、Ward 2019（F0 随动）、Ruede 2017 —— 属韵律特征依据，
  不在本次取证范围；项目里已有引用但**均未标待核**，同样记入 §7。



### 4.2 2026 新动向（五篇，均已取得原文全文）

> ⚠️ **本节三条同时是「对项目已有记录的更正」** —— 项目 `2026-10-03_description_as_training_target.md:349-356`
> 的核实表把这三篇列为「仅搜索摘要级、写进论文前必须核实」。以下为其核实结果。

#### 4.2.1 TACT（arXiv:2609.27372，IEEE SLT 2026）★ 项目标记为"优先于一切"的那篇

- **引用信息**：Kian Shamsaie, Iman Modarressi. *Neither Silence nor Overlap Is Failure:
  Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models.*
  arXiv:2609.27372v1（2026-09-23）。**Accepted to IEEE SLT 2026**（arXiv comments 字段）。
- **取证**：【原文✓】HTML 全文抓取 + 关键句本地回核

**它有两套正交分类法（🔴 勿混）**：

1. **六类潜在意图** 𝒵 = {ans, ref, rhe, hold, bck, abn}（§III-C）：
   > "Each decision point is annotated with a latent intent z over 𝒵={ans, ref, rhe, hold, bck, abn}:
   > immediate-answer-seeking, reflective, rhetorical, floor-holding, cooperative-incoming-inviting,
   > and turn-abandonment ... **bck covers continuers, collaborative completions, and urgent
   > clarifications, whose normative onset lies in overlap.**"
   - 分布：ans 31.6% / hold 18.2% / ref 16.1% / **bck 15.3%** / abn 9.6% / rhe 9.2%
2. **四类场景**（沿用 FDB v1）：pause handling / backchanneling / smooth turn-taking / user interruption
   - 🔴 **Table I caption 自带警告**："These are scenario types, distinct from the six latent intents of Fig. 2;
     **in particular the Backch. scenario is not the bck intent.**"

**判据（§IV）**：threshold-weighted CRPS；偏移域 𝒯 = [−T_a, T_h]，**T_a = 2s（预视域）、T_h = 5s**。
**对本项目最有用的一条 —— bck 的 onset 参考律**：
> ex-Gaussian (μ, σ, τ) = **(−0.32, 0.18, 0.20) s**，"the bck mode of **−0.19 s** places **72%** of onset
> mass **inside the ongoing turn**"；no-response 质量 β_bck = 0.45

**✅ IAA：报了（这正是项目要的答案）**——§III-C：
> "**Three to five annotators (median three)**, shown full context, profile, and human continuation, label
> each episode ... **Agreement is Krippendorff α=0.73 for intent (nominal), 0.84 for onset windows,
> and 0.79 for overlap acceptability** [62]."

作者自述其限："the six-class taxonomy discretizes a continuum and **α=0.73 leaves label noise**"（§VI-F）。

**效度**：与人类判断 Spearman ρ=0.81（对比 binary composite 0.46、latency 0.39）。
规模：9,728 episodes / 73.2 h，5 语料（CANDOR 2,560 / SSSD 2,304 / otoSpeech 1,792 / Seamless 1,536 / AMI-IHM 768 + 合成 768）。

- 🔴 **更正项目记录**：项目文档问的是"κ 多少"——**应为 Krippendorff α，不是 Cohen's κ**；
  标注者 **3–5 人、中位 3 人**。

#### 4.2.2 "WildTurn"（数据集名；论文名另有其名）

- **引用信息**：Tianrui Pan, Qinglin Zhang, Chong Deng, Luyao Cheng, Qian Chen, Wen Wang, Jie Tang,
  Gangshan Wu, Jie Liu（Alibaba Token Foundry）. *Enabling Proactive Spoken Turns via a Generalized
  Style-Aware Full-Duplex Framework.* arXiv:2608.28630v1（2026-08-04）。**Accepted by ACM MM 2026**。
- **取证**：【原文✓】HTML 全文 + 关键句本地回核

- 🔴 **更正项目记录**：**"WildTurn" 是这篇论文构建的数据集，不是论文名**。
  若按 "WildTurn (arXiv:2608.28630)" 写进论文引用，标题会错。

**五类动作（§1/§3.2 原句）**：**NTT（Normal Turn Taking）/ ITT（Interruptive Turn Taking）/
BC（Backchanneling）/ BI（Barge In）/ NA（No Action）**：
> "NTT, ITT, and BC trigger SpeechLLM to generate speech responses, **NA instructs SpeechLLM to maintain
> its current state** (i.e., either continue speaking or remain silent), and **BI instructs SpeechLLM to
> stop generating speech tokens**."

**评估期判据（§4.2，原句）**——NTT/ITT 是同一 response 决策按 onset 位置二分：
> "A response prediction is categorized as **ITT if its first onset i* occurs before the end of the user
> turn (i* < n), and as NTT otherwise** … This onset-based criterion ensures that **each user turn yields
> exactly one turn-taking decision, while BC may occur multiple times or not at all.**"

**bc 词典（§3.3）**：
> "BC is identified with a **lexicon-based procedure using a 66-entry English backchannel lexicon
> expanded from (Ekstedt and Skantze, 2022)**."（**正文未列出条目**；要拿到须追 Ekstedt & Skantze 2022）

**⭐ 标注是全自动的（§3.3）**：
> "**each 40 ms audio chunk is annotated with one action label** ... **NTT, ITT, and BI are detected
> directly from VAD-based (Silero) state changes, whereas BC is identified with a lexicon-based procedure**"

- 🔴 **所以它没有人工标注、全文不报 IAA**（agent 对 raw HTML 穷举 `krippendorff|kappa|inter-annotator|agreement` 为 0 命中）。
  项目若计划写"与 WildTurn 的标注一致性对比"，**对象不存在**；能对比的只有它的 judge-vs-human 相关性
  （BC ρ=0.719 / ITT ρ=0.637，§5 Ablation 4）。

- 另：风格切分 5 类 turn-taking 风格（Patient … Assertive）× 5 类 backchannel 风格（High/Low × Early/Late + No BC），
  按 ITT/NTT 比、latency 分位阈值切分（Table 1）。数据 2,981 h（Seamless + Fisher + 少量 Behavior-SD 合成），86,430 samples。

#### 4.2.3 Instruct-FD（arXiv:2607.20460）

- **引用信息**：Yuzhi Tang, Wentao Ma, Xiling Zhao 等 20 人（Boson AI）.
  *Instruct-FD: Can Your Full-Duplex Speech System Follow Turn-Taking Instructions?*
  arXiv:2607.20460v1。⚠️ abs 页写 "Submitted on 15 May 2026"，与 ID 前缀 2607 不一致（页面原文如此，如实记录）。
- **取证**：【原文✓】HTML 全文（含附录 A.13）+ 关键句本地回核

**五类动作（§2.1 原句，逐条）**：

> "(1) **Backchannel**: the model provides **brief acknowledgments during user speech at appropriate
> moments (e.g., pauses, confirmation cues) without taking the floor.**"
> "(2) **Interrupt**: the model proactively takes the floor when an explicit trigger is present
> (e.g., contradiction, safety-critical signal), with both timely entry and functionally relevant content."
> "(3) **Listen**: the model remains silent during user speech, suppressing backchannels and interruptions
> until clear turn completion."
> "(4) **Continue**: when the user produces a short continuer (e.g., "mm-hmm") during model speech, the
> model maintains the floor and completes its thought rather than yielding."
> "(5) **Acknowledge**: when the user barges in with a correction or redirect during model speech, the
> model recognizes the interruption and adapts accordingly."

**🔴 判官 rubric 里 bc 的边界更紧（§A.11）**："A brief acknowledgment (**<4 words**) produced DURING the
other…"；"**If an utterance flows into a full response, it is an early turn-take, not a backchannel.**"

**GT 构造（§2.2，三步）**——文本 LLM 离线选址、再回填音频时间戳：
> "An LLM turn agent ... **inserts structured inline markers (e.g., `<backchannel: PHRASE>`,
> `<interrupt: CONTENT>`) into the previous speaker's text at the intended cut-in point.**" → Gemini TTS 合成
> → "**ASR and forced alignment are then applied to recover word-level timestamps and locate the exact
> insertion point ... yielding ground-truth tag timestamps.**"

**±2 s 的确切含义（§A.13）**：> "**Ground-truth backchannel timestamps are selected by a text LLM given the
full conversation transcript, with a ±2 s tolerance window to absorb natural timing variation.**"

- ⚠️ **论文里"2s"还有另外两种含义，引用时不可混**：judge 判据的 2s 是**合格线**
  （"An overlap of <2s after USER barges in is acceptable reaction time... **The 2s limit is STRICT and
  NON-NEGOTIABLE.**"）；人类实验的 2s 是**决策采样间隔**。

**63.7% 的确切语境（§5）**：**是人类被试的准确率、且是下界**：
> "Participants achieved **89.6% case accuracy** ... **Backchannel is the most challenging action (63.7%)**"
> "**Backchannel accuracy should therefore be interpreted as a lower bound on human competence at
> identifying appropriate backchannel moments.**"（§A.13）

两个复合成因：**每例含 3–4 个决策点、任一处抢答即整例失败**；以及**离线标注 vs 在线感知的错配**。
人类实验规模：11 名参与者、1,199 cases / 4,031 decision points。
- 判官效度（§A.5）：180 条人工判 → judge 88.9% 准确率（**这是 judge 效度，不是 IAA**；全文未报 IAA）。

#### 4.2.4 Duplex Cue（arXiv:2609.13117）★ 两轴分类法；"bc 与 int 的边界"直接被量化的一篇

- **引用**：Yunqi Lu, Tyler Baumgartner, Nikhil Johri, Brandon Tai, Candice Fan, Luc Debaupte,
  Ruben Aguilar, Bill Wang, Yi Zhong. *Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping
  Speech in Full-Duplex Agents.* arXiv:2609.13117v1（2026-09-11）。性质：评测（含 PersonaPlex 案例研究）。
- **取证**：【原文✓】HTML 全文

**两轴独立标注（§3 原句）**：

> "We refer to B's contribution as the **cue** and A's subsequent behavior as the **response**. Each
> event is described by **two separate labels**: B's cue intent and A's observed response."

**Cue intent 三类（§3.1 原句）**：

> "A **Backchannel** supports A's ongoing speech **without calling for a change in its content or direction**.
> A **Collaboration** offers an answer, correction, clarification request, constraint, or completion to
> help shape A's ongoing utterance **without claiming the floor**. An **Interruption** attempts to
> **take or retain the floor** from A."

**Observed response 三类（§3.2 原句）**：

> "**Continued** describes A proceeding as if B had not contributed. **Adapted** describes A responding to
> B within the ongoing turn through acknowledgment, rewording, or substantive revision. ... **Yielded**
> describes A handing over the floor in response to B."

- 3×3 矩阵；它明确批评既有体系缺"继续但修正"这一格（"Its response axis ... has no value for continuing with revision."）
- 🔴 **对人最有用的一条 —— bc/int 边界上的人标分歧被直接量化**：复标 60 条一致性
  **49/60（81.7%）**，且 "**Seven of eleven disagreements crossed the Collaboration–Interruption
  boundary.**"
- 语料：80 段双声道英语自由对话、20.32 小时、39 名被试；cue 库存 2,591 条
  （1,896 Backchannels / 361 Collaborations / 334 Interruptions）。

#### 4.2.5 Controlling Backchannels in Streamable Full-duplex Models（arXiv:2609.29418）

- **引用**：Maike Züfle, Peter Polák, Sefik Emre Eskimez, Jan Niehues, Peter Bell, Ondřej Klejch.
  *Controlling Backchannels in Streamable Full-duplex Models.* arXiv:2609.29418v1（2026-09-24）。
  【原文✓】HTML 全文
- **bc 的帧级 onset 操作化（§2.1 原句）**：
  > "At each frame, the model predicts whether the current point in the interlocutor's speech is an
  > appropriate moment for a backchannel, i.e., a moment at which a human would produce one."
- **"合格帧"定义（§2.1）——本项目若做帧级评估可直接复用**：
  > "the head is trained using Focal Loss evaluated only over **eligible frames** 𝒯_valid where
  > **the partner is speaking and the agent is silent**. All frames during agent speech or mutual
  > silence are excluded from 𝒯_valid."（基率："onsets account for **∼1%** of conversational frames"）
- 阈值化：
  > "a backchannel is triggered whenever p_t ≥ τ, giving control over backchanneling frequency
  > independently of placement quality."
- 评测：**TurnBench**（"human-human conversations with backchannel annotations from **three annotators
  per speaker (majority vote)**"，取三人一致分段 12 段 = 137 分钟、574 个 bc）；
  onset F1 容差 "±0.16s around the annotated onset"（两帧）；人工评分 Krippendorff α = 0.42。



## 5. 本项目口径（对照用，**非已发表**）

> ⚠️ 本节全部是**本项目自己的定义**，列在这里只为让 §6 对照表一行对齐。出处给 file:line，改口径前先改这里。

### 5.1 三层标注来源框架（L1 / L2 / L3）

论文主框架：**L1** 生成端 GT（合成基准，构造精确）／**L2** 声学派生（VAD/ASR 时间戳 + 硬阈值规则）／
**L3** 模型标注器（turn-detection 模型输出帧级状态）。
出处：`docs/paper/full_duplex_label_audit_icassp.md:17-23`。

### 5.2 L2 时长口径（`0820guidance.md:318-320`）

| 行为 | 判据 |
|---|---|
| Realized **Interruption** | overlap **> 500ms** |
| Realized **Backchannel** | **100ms < overlap ≤ 500ms** |
| Realized **None** | overlap ≤ 100ms |

（实现：`0820guidance.md:328-333`；说话人分离处 ">100ms 视为有效重叠"，`:184`。）

- 🔴 **已被自有实证反例化**：F3 的 `clip_010`（`grade=BC_PURE`、flag `NONOVER`）是**没有 overlap 的 bc**，
  而 L2 规则要求 overlap 是必要条件 ⇒ 此口径的"必要条件"已被自己的数据打穿
  （`2026-09-27_behavior_annotation_design.md:164-168`、claims_ledger 方法论第 14 条 `:383-385`）。

### 5.3 CANDOR 采样口径（**本项目自定，非 CANDOR 官方定义**）

| 行为 | 判据（采样时） |
|---|---|
| BC | `transcript_backbiter.csv` 的 `backchannel_start` **非空** |
| Int | `transcript_audiophile.csv` 中 `overlap=True` 且 **dur > 0.3** |
| None | `overlap=False` 且 **dur > 0.5** |
| （不采样） | `overlap=True` 且 `dur ≤ 0.3` 的歧义短重叠；与 BC 窗口 ±0.3s 内一致的 turn |

出处：`pilot_study/scripts/ari_extract_candor.py:130`（`ok = ov and dur > 0.3`）、`:132`、`:109`（±0.3s 排除）。
⚠️ `overlap` / `backchannel_start` 本身是 **AWS Transcribe 的 L2 标签**（非人工），
CANDOR 官方对这两列的原始定义**尚未取得**（见 §7）。

### 5.4 模型侧标签空间（F8 / F8c）

`listen` / `speak` / `backchannel`，**1 秒块互斥单标签**；X2-Turn 软概率 → `states_to_words` 阈值化，
**优先级 bc > speak > listen**（`bc_tau=0.2`、`speak_tau=0.5`）。
出处：`pilot_study/scripts/ari_f8_build_dataset.py:31-49`；200ms 版本 `2026-08-28_f7b_framewise.md`。

### 5.5 F3 行为分级（6 grade + 3 flags，**本项目标注实验自建**）

- `grade`：`BC_PURE`（不干扰说话人的 bc）/ `BC_MID`（**会干扰说话人的 bc = bc-interruption**）/
  `INTR` / `TURN`（有 bc 词但无重叠，单纯话轮转换）/ `MIXED`（一段里真 bc + interruption 并存）/ `NOISE`
- `flags`：`NONOVER`（没有重叠但仍是 bc）/ `INTAUD`（文本有 bc 词但听不出声音）/ `INTENT`（意图与实际效果不一致）
- 实测（n=47）：BC_PURE 9 / BC_MID 3 / MIXED 7 / INTR 13 / TURN 13 / NOISE 2；flags：NONOVER 6 / INTAUD 3 / INTENT 1

出处：`pilot_study/scripts/ari_f3_analyze.py:30-40`（定义原文）；分布 `2026-09-27_behavior_annotation_design.md:143-157`。
⚠️ 已知缺陷：`raw` 取自脚本字面 dict 而非 `judgments2.csv` 的 `notes` 列，已漂 1 条（`:194-208`）。

### 5.6 CANDOR-FD（**本项目自建基准**，不是外部文献）

（2026-10-08 网上核实：搜索无任何外部同名工作，确认是项目内部名。）

- 6 类 `idle / noidle / speaking / turn_end / backchannel / uncertain`、80ms 帧、硬标签 + 并行软概率。
- 🔴 **该 6 类清单实为 X2-Turn 的类别集**（见 §3.x），项目把它打包成基准
  （四层构成：AWS(L2) / realized(声学) / X2-Turn(L3 主标注) / SoulX(L3 补充) / fused）。
- 出处：`2026-08-27_f2_candor_fd.md:20-24`；规模 100 会话 / 6,570 个 AWS BC 窗口（`:2`）。
- ⇒ `2026-09-27_behavior_annotation_design.md` §2 表把 CANDOR-FD 与其他五个体系并列时，
  **要注明它是本项目自建、判据继承自 X2-Turn**，否则会被读成"外部体系"。



## 6. 横向对照

### 6.1 backchannel 判据对照（跨 15 个体系）

| 体系 | bc 判据 | 地基 |
|---|---|---|
| CANDOR（Backbiter） | ≤3 词 **且** >50% 为 bc 词 **且** 不以禁词开头 | 词表（30 词）+ 三条规则 |
| FDB v1.0 / HiPLEX | **<1 秒**（HiPLEX：≤1s short episode）；FDB 另加 **<2 词** | 时长（+词数） |
| DuplexGen（生成侧） | 允许表 10 词：mhm\|mm-hmm\|uh-huh\|yeah\|right\|okay\|oh\|wow\|really\|huh?（1–2 词） | 词表 |
| WildTurn | 66 词英文词典（Ekstedt & Skantze 2022 扩编） | 词表 |
| Instruct-FD（judge） | "<4 words produced DURING the other's speech" | 词数 + LLM 判官 |
| BC-head | 帧级 onset：partner speaking & agent silent 的合格帧 + 阈值 τ | 帧级 + 人标（α=0.42） |
| TACT | `bck` 意图 = continuers + collaborative completions + urgent clarifications；onset 参考律 ex-Gaussian(−0.32, 0.18, 0.20)s | 人标（α=0.73）+ 分布 |
| Duplex Cue | 语义："supports ongoing speech **without calling for a change** in content or direction" | 人标（复标 81.7%） |
| **SoulX-Duplug** | 🔴 **无判据**（"represents user backchannel behavior"） | LLM 语义标注 |
| **X2-Turn** | 🔴 **无阈值**（"user backchannel signals or filler words"） | LLM 语义标注（Qwen3.5-Plus） |
| dGSLM | 提出"**短 IPU 被含于对方 IPU 内**"→ **拒绝实现** | — |
| 本项目 L2（§5.2） | 100ms < overlap ≤ 500ms | 时长 |
| 本项目 CANDOR 采样（§5.3） | `backchannel_start` 非空（ConvoKit 字段） | 工具层派生的 L2 |
| Paierl 2025 | 🔴 **无 bc 时长可引**（项目旧引用经核实不成立，§4.1④） | 词表式语义定义 |

### 6.2 interruption 判据对照

| 体系 | int 判据 | 地基 |
|---|---|---|
| **CANDOR** | 🔴 **无独立标注字段**（只有 WSO 提法"an interruption"） | — |
| FDB v1.0 | takeover 二元：非 silence 非 bc 的出声即 takeover | 时长 + 规则 |
| FDB v1.5 | Respond 四分类（GPT-4o，只吃 ASR 文本） | 语义判官 |
| FDB v3 | **Δt < 0**（agent 开口早于 user 说完） | 日志规则 |
| Behavior-SD | 生成参数 **N(0.45s, 0.05s)** 重叠；七子类（agreement … topic change, Goldberg 1990） | 生成端 |
| LSLM | 停说落在 **[0, 2μ]（=1s）** 内算成功 | 时长阈值 |
| Freeze-Omni | state 1 = 用户打断 | chunk 状态（判据未交代） |
| DuplexGen | **Take Floor = barge-in ∪ 正常接话（同一标签）** | 人标 |
| WildTurn | ITT = response 的 onset **在用户话轮结束前** | 位置规则 |
| Instruct-FD | Interrupt = 有显式触发（矛盾/安全信号）时夺话 | 语义 |
| 本项目 L2 | overlap > 500ms | 时长 |
| dGSLM | "starts within an IPU ... and continues after its end"（未实现） | — |

### 6.3 「无判据」清单（只有类别名/语义描述，无操作化）

Fun-Audio-Chat（无）、Moshi（无行为层）、
**SoulX-Duplug 的 bc**、**X2-Turn 的 bc**（判据外包给 LLM 标注器）、
Freeze-Omni 的三状态（标签构造原文未交代）、
FDB v3 的 "natural-timing response"（无秒数）、
HumDial 的 user backchannel（只有一句语义）、CANDOR 的 interruption（无字段）。

⇒ **"定义了行为类别" ≠ "定义了行为判据"**：引用任何体系前先分清它给的是哪一层。

### 6.4 结构性判据「被提出又放弃」的三次记录 + 本项目一次

| 时间 | 体系 | 发生了什么 |
|---|---|---|
| 2023 | dGSLM | 提出 bc/int 的 IPU 结构判据 → "**we will not attempt to extract here**"（依赖高层语言特征） |
| 2025 | FDB v1 | 自己用 `<1s & <2 词`，同时承认 "**differs across the literature ... left for future work**" |
| 2025 | FDB v1.5 | 放弃纯结构，改用 **GPT-4o 语义判官** |
| 2026 | 本项目 | L2 的 0.5s 口径被 F3 的 **NONOVER 反例**打穿（无 overlap 的 bc 存在，§5.5） |

### 6.5 时长阈值的"各家各法"（同一个行为，没有一个共同的数）

- bc 时长：FDB v1 **<1s**／HiPLEX **≤1s**／VAP **≤1s**（+邻域静默 1s/2s）／BC-head onset 容差 **±0.16s**；
  CANDOR 与 TurnBench 则**完全不用时长**（词表制 / 明文 "regardless of duration"）——
  🔴 原以为可引的 Paierl「0.2–0.5s」经核实**不成立**（其数不是 bc 时长，§4.1④）
- int 时长：Behavior-SD 生成均值 **0.45s**／LSLM 判据 **1s**／本项目 L2 **0.5s**／
  FDB v1 构造 pause 用 **0.4–1.0s**
- ⇒ 与 `2026-09-27_behavior_annotation_design.md` §3.2「**同一行为三个体系三个秒数、无一论证**」
  的结论一致；本节的贡献是把**出处全集**补齐了（此前只有三行转述）。

### 6.6 人标可靠性：形式决定成败？（本汇编里四个可比的数）

| 体系 | 形式 | 可靠性 |
|---|---|---|
| DuplexGen | 3 选 1、仅文本增量、5 票 | **Fleiss' κ = 0.0506**（项目自算） |
| TACT | 6 意图、给全上下文 + 人类续说 + 合格窗口 | **Krippendorff α = 0.73** |
| Duplex Cue | 2 轴、复标 60 条 | **49/60 = 81.7%** |
| BC-head 的 TurnBench | 3 人、多数票 | onset 一致；人工评分 α = 0.42 |

⇒ 支持"**问题可能出在标注形式而不是意图本身**"这一假设（09-27 文档 §3.3 的推论）：
带上下文与参考律的 TACT（α=0.73）远好于三选一的 DuplexGen（κ=0.05）。
⚠️ 但四个数**不是同一口径**（κ/α/复标率），**不可直接排大小**——引用时只能并列陈述。



## 7. 未决与待核清单

**A. 引用前必查（已知有坑）**

- [ ] 🔴 **编号幻觉两例**：`2609.30798` = survey（"The Full-Duplex-Bench family" 只是它 §7 的小节标题）；
  `2602.05105` = *GAMMS*（多智能体仿真器），**不是** FDB-v3；FDB-v3 真身 = `2604.04847`
- [ ] FD-Bench 的 interval rules：**代码（`benchmarking.py`）vs 论文口径**的准确定位（论文无分段规则描述）
- [ ] SoulX-Duplug 命名三套并存：**论文**（`user_idle/nonidle/backchannel/complete/incomplete`）vs
  **README**（`idle/nonidle/speak/blank`）vs **training-code 分支**（项目转述为 `wait`）——须回代码核实
- [ ] WildTurn 的 66 词词典**条目本身**（须追 Ekstedt & Skantze 2022 原文）
- [ ] Skantze (2017) 的 "<500ms vs >2500ms" —— 仅经 IWSDS 2025 综述**转述**，须取原文

**B. 未取到原文（HF 不可达 / 反爬拦截 / 无公开全文）**

- [ ] X2-Turn `uncertain` 的语义（HF model card 不可达；"低置信回退"仅为搜索摘要）
- [ ] Behavior-SD 的 HF 数据卡 schema（HF 不可达）
- [ ] CANDOR 官方 **Sci Adv SOM**（Cloudflare 拦截；本次用 arXiv 预印本 SOM 替代，两版正文一致）
- [ ] HumDial 官方标注字段表（README 无 schema）
- [ ] Yngve (1970)（backchannel 术语原始出处，无公开全文）

**C. 仅摘要级、未读正文（若要收录须补）**

- [ ] MTR-DuplexBench（Zhang et al.）；Talking Turns / FLEXI / DuplexSLA / DSB-IFEval（DuplexAct Table 1 所列）
- [ ] SID-Bench（2603.24144）、StepAudio 3 Realtime、SteerDuplex、MultiTalk、NemotronLabs VoiceChat、
  Realtime-Venus、M3-DuplexBench、FastTurn、DuplexDrama 等（agent 从 arXiv 摘要筛出的候选）
- [ ] Levitan 2011（能量比）/ Ward 2019（F0 随动）/ Ruede 2017 —— 项目已引用但本次未核

**D. 本地文档同步更正（✅ 2026-10-08 已全部写回源文档）**

- [x] `2026-08-23_ari_exp_guidance.md:92` + 参考文献 3：Paierl 行 —— 年份 2024→**2025**，
  「BC 时长 0.2–0.5s」**加删除线并标注撤回**（§4.1④）
- [x] `2026-09-27_behavior_annotation_design.md` §2 表：CANDOR 行补 Backbiter 三规则并注明
  `dur>0.3`/`dur>0.5` 是**本项目采样口径**；Behavior-SD 行注明 `0.5s/0.1s` 是**本项目 realized 分箱**、
  论文侧是 4 维度 × 3 档 + 高斯采样；CANDOR-FD 行注明**本项目自建、类目继承自 X2-Turn 接口**（§2.1/§2.2/§5.6）
- [x] `2026-10-03_description_as_training_target.md` 核实表：TACT 补 **α=0.73**（注明是 α 非 κ）；
  WildTurn 行补**论文名更正**（数据集≠论文名）与**无人工标注、不报 IAA**；Instruct-FD 行补"2s 三种含义"
  （§4.2）
- [x] `2026-08-26_ari_qwen_advice_1.md` 风险表下方加更正注：「FD-Bench 代码不开源」不成立
  （本地 `third_party/FD-Bench/` 有完整 `benchmarking.py`），原建议表保持原样不动

## 8. 专题：全双工行为的定义该更「语义」吗？

> **由来**（2026-10-08，用户提出）：「我感觉全双工行为是不是大家习惯用**声学信息**来界定了，
> 但根据**我的实验**这个定义应该更**语义**一点。」本节：§8.1 校准前提（用本汇编 §6 的实测，不用印象）；
> §8.2 文献支持度（专门检索）；§8.3 Claude 的评估与可操作建议。
>
> **约定**：为精确起见，本节把「声学」推广为**形式（form）**——含**声学时序**（时长/重叠/VAD）
> 与**词汇形式**（词表/词数）两类，二者都是"从可观测表面定类"；与之相对的是**功能（function）**——
> 意图 / uptake（对方如何接） / 语义角色。
> **用户实验的所指**（本汇编内部出处）：F3 的 47 条自由描述撑破三分类（`grade` 6 值 + `flags` 3 个，
> 其中 **NONOVER 6/47≈13%** 是无 overlap 的 bc、`INTENT` 是意图≠效果，见 §5.5）；
> 同一个 `yeah` 两种功能（clip_052 vs clip_049）；ARI 实验里人类标签在五个声学特征上
> **ARI≈0.058**（近随机）而合成标签 0.309 靠 silence shortcut（`docs/paper/full_duplex_label_audit_icassp.md` §3.1）。

### 8.1 前提校准：文献实际用什么定义？（本汇编实测，不用印象）

把本汇编里**所有给出 bc 事件判据**（或明说无判据）的体系按路线归类：

| 路线 | 体系 | 计数 |
|---|---|---|
| **形式-声学时序** | FDB v1（<1s 且 <2 词）、HiPLEX（≤1s）、VAP（≤1s+邻域静默）、BC-head（帧级 onset+τ）、**本项目 L2**（overlap 100–500ms） | 5 |
| **形式-词汇** | CANDOR（Backbiter 三规则）、DuplexGen 生成侧（10 词）、WildTurn（66 词）、FDB v1.5（99 条合成列表）、Instruct-FD judge（<4 词）、**本项目 CANDOR 采样**（ConvoKit 字段派生） | 6 |
| **功能/语义（人标）** | TurnBench（"regardless of duration"）、Duplex Cue、**TACT**（bck 意图）、Uro、Lebourdais、Paierl、Cathcart（功能定义+词表过滤，混合） | 7 |
| 序列结构（既非声学也非语义） | Liesenfeld & Dingemanse（form-agnostic） | 1 |
| **无判据**（类别名/语义描述，操作化外包） | SoulX-Duplug、X2-Turn（LLM 语义标注、判据不公开）、Freeze-Omni、dGSLM（提出又拒绝）、HumDial（一句语义描述，无操作化） | 5 |

**读表三条结论**：

1. **「形式」合计 11 个 vs「功能」7 个 —— 用户的方向判断成立，但有一处要修正**：
   ① 词表（6）与时长（5）**几乎各占一半**，所以更准确的叫法是「**形式**」而不是「声学」——
   词汇形式同样是表面特征，且**词表本质上是把"功能"外包给了一个词形假设**（"mhm 一定不是夺话"）。
   ② 「无判据」那 4 个是**隐性形式**：X2-Turn / SoulX 名义上让 LLM"按语义"标，但判据不公开、
   不可审计 —— 它到底是语义还是隐式的表面启发式，**文献层面不可知**。
2. **语义已经进场，但进的是另一个位置**：FDB v1.5 / HumDial / FDB v2 / DuplexAct 都用了语义判官
   （GPT-4o / DeepSeek-V3）——但判的是「**系统响应**」对不对，不是「**事件**是什么」。
   ⇒ 准确的说法是：**大家愿意用语义判断"做得好不好"，却不愿用语义定义"这是什么"**。
3. **功能式定义不是没人做，它的分布很有规律**：7 个功能式全部落在**对话分析传统**
   （Cathcart / Uro / Lebourdais / TurnBench / Paierl）与 **2026 新工作**（TACT / Duplex Cue）里；
   而**工程系统与 benchmark 侧**（FDB 家族、FD-Bench、LSLM、Freeze-Omni、Moshi、本项目 L2）
   **近乎全是形式或无定义**（DuplexAct 的 bc 事件层是个例外：它用人标位置当 GT，但**没有公开的判据文字**）。
   用户假设中"大家习惯用声学"在**工程/评测文献**里成立，
   在**对话分析文献**里不成立 —— 这两个圈子此前基本不互引，本汇编第 §4.1 是它们第一次被并排放。

### 8.2 文献支持度（专门检索，2026-10-08）

> 检索口径：① 经典对话分析线（§8.2.0，9 查 5 中）；② 2024–2026 计算线（§8.2.1 支持 / §8.2.2 削弱）。
> 全部条目抓取原文；14 篇 arXiv 编号经 API 反查标题核对，**无一编号幻觉**。
> 关键句本地逐字回核：SID-Bench / ECHO / NaturalTurn / 2610.03078（其余为抓取原文，未逐句二次核对）。

#### 8.2.0 经典线：对话分析与早期计算语言学（9 查 / 5 取得全文）

> **取证**：PDF 全文抽取 + 关键句本地逐字回核（Schegloff / Goodwin / Jurafsky / Allwood 已核）。
> **4 篇未取得**（closed access，均为付费墙）：Yngve 1970、Jefferson 1984、Drummond & Hopper 1993、
> Bavelas et al. 2000。系统性障碍留档：**本机无法访问 archive.org**（v4/v6 连接被阻断）、
> Cloudflare 系站点（academia.edu / CORE / T&F 等）一律 403 —— 后续补这三篇需机构订阅。

- 🔴 **反直觉发现（先泼一盆冷水）**：**Yngve (1970)（术语"backchannel"的原始出处）不能当"功能优先"的源头引**。
  原文未取得（CLS 会议论文集无公开全文），唯一可得的转引（Wikipedia，未核页码）显示其定义是**信道式**的：
  "the person who has the turn receives **short messages** such as 'yes' and 'uh-huh' **without relinquishing the
  turn'" —— "短"与社会信道隐喻本身就是形式式的。反而有二手源提到他讨论过"较长评论也可属 back channel"
  （p.574 "back-back-channel"），若属实，说明**他自己也知道"短"这条形式判据不成立** —— 补核到原文前，两边都别引。

- **① Schegloff (1982)——经典线最强正面引文**（*Discourse as an interactional achievement*, GURT 1981, pp.71–93）
  - **功能式定义（p.81）**：
    > "It takes the stance that the speaker of that extended unit **should continue talking** … 'Uh huh', etc.
    > exhibit this understanding, and take this stance, **precisely by passing an opportunity to produce a full
    > turn at talk**. When so used, utterances such as 'uh huh' may properly be termed '**continuers**'."
    （定义落在"接受方**接下来做什么**"，不是时长/词表/重叠。）
  - **明说聚合式/注意力式描述不足以定类（p.79）**：
    > "the characterization of the class as signalling attention, interest, or understanding **appears
    > equivocal**"；"**does not help discriminate** 'uh huh' from any other talk"
  - **明说脱离序列环境就什么都看不出（p.86）**：
    > "disengaging the listener behavior from its local sequential context **not only undercuts the possibility
    > of understanding what it is doing**; it can remove an important basis for understanding what is going on
    > in the discourse itself."（例子：同一个 'mm hmm' 可能同时在**扣住一个笑**）
  - p.88："**It is not that there is a direct semantic convention in which 'uh huh' equals a claim or signal of
    understanding.**"
- **② Goodwin (1986)——"形式相似、功能不同"的直接论证**（*Between and within*, Human Studies 9:205–217）
  > "Though assessments and continuers occur in roughly the same environment … **the detailed sequential
  > treatment each receives reveals that they are in fact being treated as different types of phenomena**."（p.207）
  > "**In this they resemble continuers**"（p.214，指两者都可短、可非词汇、可手势化）——
  > 所以**不能靠形式区分**；区分靠说话人怎么对待它："she treats it **precisely as a signal to continue**"（p.209）。
- **③ Jurafsky et al. (1998)——计算语言学侧的硬数据**（ACL/COLING Workshop, pp.114–120）
  - **"yeah" 同时是四类 DA 各自的最常见词形**（Table 6：Agreements 36% / Continuers 27% / Incipient Speaker 59% / Yes-Answer 56%）
  - **改标实验**：只看文本标注时，**38% 的改标是 continuer → agreement**（Table 7）——"词形/文本形式无法决定功能类别"的定量证据。
  - 并**统计复现了 Jefferson (1984)**（Jefferson 原文未取得，此为转引）：
    > "uh-huh is **twice as likely as yeah** to be used as a continuer, while yeah is **three times as likely as
    > uh-huh** to be used to take the floor."
- **④ Allwood, Nivre & Ahlsén (1992)——功能分类学的源头**（*Journal of Semantics* 9(1):1–26）
  - 语言反馈 = 四类基本交际功能的信息交换：**contact / perception / understanding / attitudinal reactions**。
  - 同一形式的极性翻转（Table 1）：对 "It isn't raining" 回 "yes"，**"ambiguous between rejection (yes it is)
    and acceptance (yes you are right)"** —— 功能由前序话语决定。
  - ⚠️ 作者自留件首页印 1993，**正确年份是 1992**（笔误）。
- **⑤ Stolcke et al. (2000)——早期计算侧承认形式不够**（*Computational Linguistics* 26(3):339–373）
  > "some utterances are **inherently ambiguous based on words alone** … the distinction between
  > **BACKCHANNELS and AGREEMENTS** … which **share terms such as right and yeah**."；消歧靠上下文
  > （"the following utterance is a YES-ANSWER or NO-ANSWER"）。
- ⚠️ **两处使用警告**：**Bavelas et al. (2000) 别当"形式不足"的引文** —— 仅取得摘要，且其
  generic/specific 之分**恰恰是按形式/内容耦合度划的**（"点头、mhm" vs "皱眉、惊呼"）；
  **Jefferson 1984 / Drummond & Hopper 1993 只能经 Jurafsky 1998 转引**（原文付费墙）。

**§8.2.0 小结**：经典线的结论与计算线（8.2.1）**同构** —— **形式不足以定类，须看序列位置与接受方的后续处置**。
差别在时间：CA 传统 1982 年就把这句话说全了，工程界 2026 年才在 benchmark 上重新发现它
（TurnBench "regardless of duration" / SID-Bench "intent"）。

#### 8.2.1 支持「应更语义」的近期工作（9 篇，按论证力度排序）

**① SID-Bench（ICME 2026，arXiv:2603.24144）—— 论证最外显的一篇** ★
- 口径：*Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model*（Xia, Mu, Shi, Xu, Xie；Qwen Team 实习工作）。
- **明文意图式定义（§II-B 原句）**：
  > "We define a true **Interruption** as an event where a speaker begins their turn with a **new communicative
  > intent that semantically warrants** the other speaker to yield their turn. In contrast, a **Backchannel**
  > is an utterance used to **acknowledge, agree, or show continued attention while not altering the
  > conversational goal**."
- **明文判声学不足（§V-C 原句）**：> "**relying solely on acoustic energy is an inadequate foundation**
  for intelligent interruption handling."；VAD 基线被称 "**trigger-happy**"（FIR 平均 **0.840**，
  对话场景 >0.90）；其自身 APT 0.711s vs 最好基线 2.129s。
- ⚠️ **必须一起引用的三条限制**：**全文零 IAA**（无标注者信度统计，grep 验证）；标签由 **LLM 生成**
  （Qwen-max/plus 插 `<break>` + Kaldi 强制对齐）；**被测模型与造标签模型同源**（Qwen 系），
  且作者自承 "**correlating our automated metrics with human perceptual judgments**" 是 future work。
- 判定：**支持**（但它是"用语义替代声学"的一次主张，不是已验证的结论）。

**② TACT（IEEE SLT 2026，arXiv:2609.27372）—— 有对照数的最强同向证据**（详见 §4.2.1）
> "TACT agrees with human judgments at **Spearman 0.81 versus 0.46 for binary metrics**."
（且用同一语料 CANDOR；IAA：Krippendorff α=0.73。）

**③ ECHO（arXiv:2609.17360）—— 实验设计本身就是论证**
- *Same Words, Different Actions: Paired Turn-Taking Evaluation under Rewritten Dialogue Contexts*（Zhao, Cai 等）。
- **同一段重叠转写、只换前文语境，正确动作相反**（一个应 Yield、一个应 Keep）：
  > "Reliable full-duplex interaction therefore hinges on deciding **whether overlapping speech warrants a
  > Yield or a Keep, not on detecting that speech occurred**."
- 顺带命中本项目的"恒真判据"问题：> "a system with a **constant action preference scores well on any
  interruption-only test**" —— 实测三个系统有 **Yield bias**（Easy Turn 99.09% / SoulX-Duplug 89.80% 判 Yield）。
- 判定：**支持**（形式信息被完全控制后系统仍判错 ⇒ 决策信息在语境层）。

**④ 自动标注流水线（arXiv:2610.03078）—— 本批最直白的一句** ★
> "**Hence, backchannels cannot be identified using duration or a fixed lexical dictionary alone, as their
> classification depends on conversation context.**"（§1；其 bc 定义为功能性：
> "a listener contribution that provides feedback to the current speaker without attempting to take the floor"）
- ⚠️ 代价同时暴露：其语境式判据的实测 **F1 = 0.621**（与人标对照见其 §3.4）。
- 判定：**支持**（且是"形式不够"的直接证词）。

**⑤ Semantic VAD（Tencent AI Lab，arXiv:2502.14145）—— 按意图二分 barge-in**
- `Real INT`（用户意图改变话题 ⇒ 该停）vs `Fake INT`（非破坏性 ⇒ 该继续说，**例子含 back-channeling**）。
- 原文批静音阈值："acoustic VAD ... can only predict `<|S-S|>` — **assuming the user has finished speaking**"。
- 判定：**支持**。

**⑥ GaMMA 标注协议（SIGDIAL 2025）—— 存在性证明：功能性标注可达 κ 0.75**
> "Current computational methods, such as speech diarization, **VAD**, and ASR, **lack robustness in detecting
> socially meaningful conversational structures like TCUs, backchannels, or failed floor transfers**"
> "… **OpenAI's advanced voice mode interrupts on the user's backchannel speech**"
- 其 backchannel 二元标注 **F1 = 0.832 / κ = 0.754**（但逐被试 0.645–0.865；定稿后为单人标注）。
- 判定：**支持**（并给出"功能性定义也能标到 κ≈0.75"的存在性证明）。

**⑦ Wang et al. 2024（ICASSP，arXiv:2401.14717）—— 声学最弱、语义最强的那个类**
- 三分任务实测：Backchannel 类的 AUC —— HuBERT（声学）**0.6455** vs GPT2（文本）**0.7744**；
  作者归因：> "'Turn-taking' and 'Continuing speech' are strongly cued by **intonation and duration**,
  whereas '**Backchannel' is possibly more related to syntactic and semantic information**"。
- ⚠️ 同一篇里"声明语义、操作化词表"（"the 20 most frequent one and two-word phrases"）——
  它本身就是 §8.2.3「声明-操作化缺口」的样本。
- 判定：**支持但带限定**（bc 受益于文本，但仍是最难类：0.79 < Turn 0.91）。

**⑧ Duplex Cue（arXiv:2609.13117）—— 把"效果/uptake"提升为一等判据**（详见 §4.2.4）
- ⚠️ 但自曝一处循环性：> "Cue intent **was not labeled blind** to Speaker A's subsequent response …
  some portion of the 68.2% adaptation rate on Collaboration cues is **definitional rather than an
  independent behavioral finding**" ⇒ **"语义/效果式"定义同样可能自证**。

**⑨ JAL-Turn（arXiv:2603.26515）—— 双向证据**：其 bc 类失败归因支持语义
（"the **intrinsically context-dependent nature of backchannels**"），但其框架同时反驳"语义即够"
（"systems rely **solely on acoustic or semantic cues** → suboptimal"）。

#### 8.2.2 削弱/反例线（3 篇，必须一起收录）

**① NaturalTurn（Sci Rep 15:39155, 2025；Cooney & Reece —— 即 CANDOR 作者）★ 最强反例**
- 用**纯形式**判据（cue list + ≤3 词 + 禁始词，同 Backbiter 三规则），且**明文把"不做语义"当优点**：
  > "**By deliberately stopping there and eschewing prosodic features, gaze trackers, sequence models, etc.**,
  > NaturalTurn remains **deterministic, transparent, and scalable**, able to segment millions of turns
  > **without extra annotation, training data, or feature engineering**."
- 它也批评过粗糙 VAD（"insert spurious boundaries at the listener's 'mhm'"）——**所以它不是"声学派"，
  它是"形式 + 可扩展派"**。
- 判定：**削弱**，但削弱的是"语义更对"，不是"声学更对"——它揭示的是**取舍**：可扩展性 vs 语义保真。

**② VAP（Ekstedt & Skantze 2022，Interspeech）**：批评阈值式 turn-taking 之后，
**以语音活动形状定义 bc**（"shorter, isolated VA from the other speaker, which roughly corresponds to
the phenomenon called backchannels"），并**自监督建模** —— 高引工作证明形式定义可规模化落地。
- 判定：**削弱**。

**③ Less can be More（arXiv:2609.11066, 2026-09）—— 设计最干净的严格反例** ★
- 同训练条件穷举 7 个模态子集：> "the **acoustic–prosodic combination achieves the best balance** …
  **Adding text increases premature detections without improving performance** …
  **turn-taking is primarily conveyed through intonation and silence patterns rather than semantic completeness**."
  （text-only 平均每句 5 次误触发；所有含文本配置的 FA 都更高，McNemar p<10⁻⁹）
- ⚠️ **转引必须注明的任务差异**：它是**单人 EOT**（判断"这个说话人说完没有"），
  **不是双人重叠/bc 判定**；且限"in-domain two-speaker English telephony"；融合方式有自陈缺陷。
- 判定：**削弱**（对本项目立场是**局部**反例：它反驳的是"EOT 需要语义"，不是"bc/重叠判定需要语义"）。

#### 8.2.3 一个比「大家都用声学」更精确、也更难反驳的表述 ★

把 8.2.1 与 8.2.2 并排后，浮现出一个**两条线都同意的结构性事实**：

> **多数工作的「声明定义」早已是功能性/语义的，形式性恰恰落在「操作化」层 —— 二者之间的缺口才是问题所在。**

- 声明语义、操作化形式的样本（本汇编内就有 4 个）：Wang 2024（"without signaling an intent to take a
  turn" → 20 个高频短语）；Easy Turn（"should not interrupt the system's speech output" → `<2s` 筛选）；
  FDB v1.5（"should not interrupt an ongoing response" → 99 条合成词表）；HumDial（同上，无操作化）；
  **本项目 L2**（行为语义 → overlap 100–500ms）。
- 这个表述的优势：**14 篇里没有任何一篇能反证它**；且 NaturalTurn 那类"有意为之"的立场
  反而**证明该缺口是自觉的取舍**（可扩展性/确定性），而不是疏漏 —— 这正是本项目可以正当地去
  填的位置：**不是"你们不懂语义"，而是"这个取舍的代价从未被量化过，我们来量化"**。
- ⚠️ 附一处**命名撞车**（引用时必查）：*Less can be More* 的 **APT = Acoustic–Prosodic–Text**（架构名），
  与 SID-Bench 的 **APT = Average Penalty Time**（度量名）**完全同形异物**。

### 8.3 Claude 的评估

> 以下为【评估】级内容（非文献引用、非实验结论），按项目惯例与证据分开列。

#### (1) 结论先说：方向支持，但建议改一处表述、加一处区分

**改一处表述**：不写"文献都用形式定义、所以定义应更语义"，改写成
**「声明定义早已语义化，形式性落在操作化层；这个缺口的代价从未被量化」**（§8.2.3）。
理由：① 更准确（8.2.1 里 Wang 2024 / Easy Turn / FDB v1.5 / HumDial 全是"声明语义、落地形式"）；
② **抗反驳** —— NaturalTurn（CANDOR 作者自己）与 *Less can be More* 从反例变成论据：
它们证明这个缺口是**自觉的取舍**（可扩展性 / 声学已够用），而取舍的代价没人量过；
③ 不与 TACT 撞车（TACT 在"意图条件化评估"上已占位；我们的位置是"**缺口 + 可靠性审计**"）。

**加一处区分**（我认为这是目前讨论里所有人都在混的一层）——**【评估】定义层与检测层是两个问题**：

| | 问题 | 本项目证据 | 文献证据 |
|---|---|---|---|
| **定义层** | "什么算 bc / interruption" | **支持语义/功能**：F3 的 NONOVER 13%（无重叠的 bc 存在）、INTENT（意图≠效果）、同一个 yeah 两种功能；ARI 人标 0.058（声学五特征恢复不出人类标签） | 支持：CA 传统 + §8.2.1 九篇 |
| **检测层** | "怎么自动识别出它" | L3 软分数对齐最好（window AUROC 0.667），但硬阈值下优势消失 | **混合**：bc 靠语义（Wang 2024: 0.65→0.77）、EOT 靠声学（*Less can be More*） |

⇒ **项目的立场应明确落在「定义层」**：不是"模型该多用语义特征"（那会被 *Less can be More* 直接顶回来，
而且它反驳的正是检测层），而是"**行为类别本身不该由形式特征来定义**"。
工程文献偏爱形式判据，很大程度上是把**检测的便利**误当成了**定义的依据** —— 形式特征好算、好复现、
好自动化（NaturalTurn 的辩护成立），但"好算"不是"对"。**这个混淆正是本项目审计的对象。**

#### (2) 支持你方向的证据（你的实验 + 文献，分别对应）

- 你说"根据我的实验"—— 对应关系是：**F3 的 47 条描述**证明现有形式边界**装不下**真实行为
  （这是定义层证据，最硬）；**ARI 0.058** 证明人类标签**不在**那五个声学特征里（这是定量证据）；
  **L3 软分数最好**证明"模型标注器比 VAD 规则更接近真实重叠"（这是检测层证据，但方向一致）。
- 文献里最该引的三条：**TACT**（0.81 vs 0.46，同一语料 CANDOR）、**SID-Bench**（"relying solely on
  acoustic energy is an inadequate foundation"）、**2610.03078**（"cannot be identified using duration
  or a fixed lexical dictionary alone"）。

#### (3) 必须一起带上的风险（否则会被审稿人打回）

1. **运行时可得性（最硬的技术约束）**：`effect/uptake` 类语义（"对方是否被抢走话轮"）**只在未来可知**
   （TurnBench 明文 "dependent on future information revealed seconds later"）。
   ⇒ 语义式定义**必须分 intent / effect 两栏**；流式训练目标只能挂 **intent**（onset 时可见），
   effect 归评估侧。此推导已写在 `2026-10-03` 文档 §6.2，本汇编与其一致。
2. **可靠性不是自动的**：DuplexGen（三选一）κ=0.05 vs TACT（带上下文+参考续说+合格窗口）α=0.73 ——
   差距来自**引出形式**，不是"语义 vs 形式"。⇒ 走语义路线必须**同时**升级引出形式，否则重蹈 DuplexGen。
3. **同源与循环风险**：SID-Bench 用 LLM 标签、且被测模型与造标签模型同族；Duplex Cue 自曝 intent 标注
   **没做盲**（看到对方回应后标）⇒ "效果式"定义容易自证。**本项目 F3 的独特性**恰好在这：
   描述是**人写的、且写在分类之前**（09-27 文档 §6 的设计原则）—— 这是可辩护的，要守住。
4. **代价与定位**：NaturalTurn 的辩护（百万级、零标注、确定性）在**生产管线**里是对的。
   ⇒ 语义式定义的正确定位是**评估与审计**（小规模、高保真），不是替代生产判据——
   这与论文 no-shortcut protocol 的定位天然一致。
5. **局部反例要如实承认**：*Less can be More* 证明**单人 EOT** 任务上语义不加分。
   ⇒ 我们的主张必须限定在"**双人重叠中的行为归类（bc vs int）**"，不能扩大到所有 turn-taking 子任务。

#### (4) 可操作建议（按性价比）

1. **论文表述**按 (1) 改写（半天工作量，收益最大）。
2. **把"缺口"变成可测命题**：挑 3–4 个"声明语义、操作化形式"的系统（Wang 2024 / Easy Turn /
   FDB v1.5 / 本项目 L2），在同一批事件上比较"它们的操作化判据"与"语义判据"给出的排序差异 ——
   本项目 E1/E2 的审计方法可直接搬，**零新标注**。
3. **语义侧可靠性**按 TACT 式设计（上下文 + 参考 + 窗口），并守住"**描述先于分类**"（F3 的教训）；
   每轴单独算 κ（补 09-27 §5.2 的洞）。
4. **可证伪点（建议预登记）**：① 若在改进形式下 intent 标注可靠性仍低于形式判据的重复性 ⇒ 支点消失；
   ② 若 intent-conditioned 与形式式评估在**系统排序**上 Spearman > 0.9（即"缺口没有代价"）⇒ 论点不成立；
   ③ 反面参照：TACT 的 0.81 vs 0.46 是我们期望看到的那种分离度。

#### (5) 一句话总结

**你的直觉经得起文献检验，但"大家用声学"要改写成"大家的声明早已语义、操作化仍是形式代理"；
真正的贡献点不是"语义更好"（有人已经在做了），而是"这个缺口的代价可以被量化，且我们已经在量"**
—— F3 的 47 条描述 + ARI 0.058 + 本节的缺口清单，就是这条论点的三根桩。

## 更新日志

- 2026-10-08 建档；§1.1 FD-Bench（本人抓取）；§1.2–1.7 FDB 家族 + DuplexAct + HiPLEX（agent 抓取 HTML 全文，关键句本地回核）；
  核实并记录两条 arXiv 编号幻觉。
- 2026-10-08（同日续）：补齐 §2（CANDOR / Behavior-SD / DuplexGen / HumDial）、§3（系统类 7 篇）、
  §4.1（四条判据路线 + TurnBench + Paierl 撤回）、§4.2（2026 新动向 5 篇）、§5（本项目口径）、§6（横向对照）。
  **共 7 条对项目已有记录的更正**：① Paierl 引用撤回；② CANDOR 论文无时长阈值；③ Behavior-SD 论文无 0.5s/0.1s 阈值；
  ④ TACT 报 α 非 κ；⑤ WildTurn 是数据集不是论文名；⑥ WildTurn 无人工标注无 IAA；⑦ X2-Turn 论文 5 类 vs 接口 6 类。
  7 条更正已同步写回源文档（§7 D 已核销），并随首次提交入库。
- 2026-10-08（同日续二）：新增 **§8 专题「定义该更语义吗」** —— §8.1 前提校准（24 体系按判据路线归类）；
  §8.2 文献支持度（经典线 9 查 5 中 + 计算线 14 篇，含最强反例 NaturalTurn 与 *Less can be More*，
  及 §8.2.3 的表述修正：**「声明已语义、操作化仍是形式代理，缺口代价未量化」**）；
  §8.3 评估（定义层/检测层区分 + 5 条风险 + 3 条可证伪点）。
