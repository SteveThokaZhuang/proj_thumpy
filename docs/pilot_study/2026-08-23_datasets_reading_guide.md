# Switchboard / CANDOR / Fisher 数据集读取指南

> 探索日期：2026-08-23
> 数据位置：`/share/workspace3/shared_dataset/{switchboard, CANDOR, Fisher}`
> 说明：本文档基于对共享数据目录的只读检查整理（未解压、未修改原目录）。三个数据集均**不需要解压**即可读取。

---

## 1. Switchboard（switchboard-speechlaugh 预处理版）

**位置**：`/share/workspace3/shared_dataset/switchboard/`
**来源**：HuggingFace 数据集 `hhoangphuoc/switchboard`（git 仓库克隆，remote 指向 `hf-mirror.com/datasets/hhoangphuoc/switchboard`），是原始 Switchboard 语料（LDC97S62）为笑声/言语笑声识别任务做的预处理版本，pretty_name 为 **switchboard-speechlaugh**。

### 1.1 目录结构

```
switchboard/
├── README.md          # HF 数据集卡片（含 dataset_info、split 统计、笑声统计）
├── .git/ .gitattributes
└── data/              # 75 个 parquet 文件（28 GB）
    ├── train-00000-of-00054.parquet ... train-00053-of-00054.parquet   (54 个, 185,402 条)
    ├── validation-00000-of-00006.parquet ... validation-00005-of-00006.parquet (6 个, 20,601 条)
    └── test-00000-of-00015.parquet ... test-00014-of-00015.parquet     (15 个, 51,501 条)
```

### 1.2 读取方法（无需解压）

parquet 是自包含列式文件，直接读：

```python
import pandas as pd
# 按 split 读全部文件（多文件用 glob）
df = pd.read_parquet("/share/workspace3/shared_dataset/switchboard/data/train-*.parquet")
# 或 pyarrow
import pyarrow.parquet as pq
tbl = pq.read_table("/share/workspace3/shared_dataset/switchboard/data/validation-*.parquet")
```

若装有 `datasets` 库，也可直接从目录加载（README 中含 HF dataset_info）：
```python
from datasets import load_dataset
ds = load_dataset("/share/workspace3/shared_dataset/switchboard")
```

**schema（3 列）**：

| 列名 | 类型 | 含义 |
|---|---|---|
| `audio` | struct<bytes: binary, path: string> | 音频本体（WAV bytes 已嵌入 parquet）+ 文件名 |
| `sampling_rate` | int64 | 恒为 16000 |
| `transcript` | string | 转写文本 |

### 1.3 音频 ↔ transcript 对应关系

- **一一对应**：每行 = 一个音频片段 + 其转写，同一行内直接对应，无外键。
- `audio.bytes` 是完整的 16 kHz 单声道 16-bit WAV（可由 `wave` 模块直接解析，已用 `wave.open(io.BytesIO(bytes))` 验证 framerate=16000）。
- `audio.path` 形如 `sw02440B_400303_401980375.wav`，编码 `sw<会话号><声道A/B>_<数字>_<数字>.wav`。注意：**后两个数字与音频实际时长无一致换算关系**（抽样检查中甚至出现负跨度），推断是源标注流程遗留的编号，不能当作时间戳使用。
- 数据集本身未提供原始 LDC `sw*.sph` 全音频；如需原始语料需另从 LDC/Google Drive 获取（README 中有 gdown 链接）。

### 1.4 transcript 记录的信息

- 纯文本转写（小写、无标点），词数 1–49，均值约 10。
- 特殊标记：**`<LAUGH>`**（纯笑声片段，如整条 `'<LAUGH>'`）；抽查 13,736 行中仅出现 `<LAUGH>` 一种尖括号标记。
- **笑声 vs 言语笑声（speech-laugh）的区分方式**：README 给出各 split 的 laughter / speechlaugh 统计（train: 16044/9586，val: 1845/1133，test: 4335/2775）。由于转写中只有 `<LAUGH>` 一种标记，speech-laugh 应指"笑声出现在言语片段中"（`<LAUGH>` 与词语混在同一 utterance 内），纯 laughter 指整条只有 `<LAUGH>`（该推断需与数据构建方确认）。
- 平均每条 1–6 秒，适合做笑声/言语笑声二分类或 ASR 任务。

**原文示例**（`validation-00000-of-00006.parquet` 第 38 行，含 `<LAUGH>` 的言语笑声片段，完整不截断）：

| 字段 | 值 |
|---|---|
| `audio.path` | `sw03491B_1038215_11522175.wav` |
| `sampling_rate` | `16000` |
| `transcript` | `matter of fact he looks just as about as old and the uh next generation as he does in the latest star trek <LAUGH> imagine that <LAUGH>` |

---

## 2. CANDOR（Conversational, Affective, Naturalistic, Dyadic, Online Recordings）

**位置**：`/share/workspace3/shared_dataset/CANDOR/`
**来源**：BetterUp Labs 发布的真实 dyadic 线上对话语料（Science Advances 2023，1,600+ 场对话，2020 年 COVID 期间录制）。官方申请入口：betterup-data-requests.herokuapp.com。

### 2.1 目录结构（无需解压即可用的部分）

```
CANDOR/
├── urls.txt            # 官方 S3 下载链接（processed_media_part_001~XXX.zip，签名 URL 已于 2024-10-31 过期）
├── download.sh         # 用 wget 批量下载上述 zip 的脚本（自动跳过已存在文件）
├── CANDOR.zip          # 2.4 GB 自建补充包（非官方发布物，见 2.5）
└── files/<session_uuid>/          # 1656 个会话目录，每个会话：
    ├── metadata.json              # 会话元数据（speakers、channels、视频文件、时长等）
    ├── survey.csv                 # 每会话 2 行（每位说话人一行）的问卷数据
    ├── audio_video_features.csv   # 每秒 2 行（每声道/每说话人）的音视频特征
    ├── processed/
    │   ├── <uuid>.mp3             # ★ 音频文件（双声道 64kbps MP3, 48kHz, 时长≈会话时长）
    │   ├── channel_map.json       # 声道映射 {"L": user_id_A, "R": user_id_B}
    │   └── thumbnail.png
    └── transcription/
        ├── transcribe_output.json      # ★ AWS Transcribe 原始输出
        ├── transcript_cliffhanger.csv  # ★ 粗粒度 turns（每行一个长 turn）
        ├── transcript_audiophile.csv   # ★ 细粒度 turns（短 utterance 级）
        └── transcript_backbiter.csv    # ★ turns + backchannel（附和语）分析
```

### 2.2 读取/解压方法

- **核心数据（files/ 下所有内容）已解压就绪**，直接读文件即可，无需任何解压操作。
- `CANDOR.zip` 如需使用，将其解压到**自己的临时目录**（不要解到共享目录）：
  ```bash
  unzip /share/workspace3/shared_dataset/CANDOR/CANDOR.zip -d <你的目录>/candor_extract
  ```
  内含 `CANDOR/whisper/*.npy`（1656 个）与 `CANDOR/laughs/*.npy`（1656 个），文件名与会话 uuid 对应。
- 官方原始发布（urls.txt 中的 zip）**当前链接已过期**，需走 BetterUp 官方申请流程获取；本地 `files/` 即官方 zip 解压后的 processed 媒体。

### 2.3 音频 ↔ transcript 对应关系

- **声道即说话人**：`processed/<uuid>.mp3` 是双声道混音（已验证 MPEG frame header channel_mode=stereo）；`channel_map.json` 把声道 L/R 映射到 `user_id`，`metadata.json` 的 `speakers[]` 给出每个说话人的 `channel` 字段（"L"/"R"）。
- **时间戳对齐**：三个 transcript CSV 的 `start`/`stop` 列是**相对音频开头(0s)的秒数**，可直接切音频；`audio_video_features.csv` 的 `timedelta` 列也是秒（每说话人每秒一行）。
- 转写 CSV 中 `speaker` 列直接写 `user_id`，与 `channel_map.json` 对应。
- 原始视频（mkv）未包含在本地 files/ 中（metadata 中 `files[].filename` 仅作记录）。

### 2.4 transcript 记录的信息

三个 CSV 均为 turn 级转写（AWS Transcribe 产出的说话人分离文本），列含义：

| 列 | 含义 |
|---|---|
| `turn_id` | turn 序号 |
| `speaker` | 说话人 user_id |
| `start` / `stop` | 该 turn 起止时间（秒） |
| `utterance` | 转写文本 |
| `interval` | 与上一 turn 的间隔（负值=重叠/插话） |
| `delta` | 本 turn 时长 |
| `questions` | 问句数量 |
| `end_question` | 是否以问句结尾 |
| `overlap` | 是否与对方重叠 |
| `n_words` | 词数 |
| `backchannel*`（仅 backbiter 版）| 附和语（嗯/对）检测：内容、数量、说话人、起止时间 |

三版差异：`cliffhanger` 把连续短句合并成长 turn；`audiophile` 拆到 utterance 级（更细）；`backbiter` 额外带 backchannel 分析。`transcribe_output.json` 是 AWS Transcribe 原始 JSON（`results.transcripts[].transcript` 全文）。

**原文示例**（会话 `0020a0c5-1658-4747-99c1-2839e736b481`，两版 CSV 各一条完整 turn，utterance 全文不截断）：

`transcript_audiophile.csv` 的 turn 3（注意 `interval` 为负值 = 与上一 turn 重叠，`overlap: True`）：

```text
turn_id: 3
speaker: 5a73899f9cdd1800017786f0
start: 200.74   stop: 201.76
utterance: hey I'm gone.
interval: -0.21999999999999886   delta: 1.0199999999999818
questions: 0   end_question: False   overlap: True   n_words: 4
```

`transcript_backbiter.csv` 的 turn 0（带 backchannel 列；本 turn 的 `stop` 为 200.96，因合并了后续所有短 turn）：

```text
turn_id: 0
speaker: 5fa072f4f4aa580b63834357
start: 4.34   stop: 200.96
utterance: Mhm. Mhm. Just, mm. And Uh huh, mm. Mhm. Mhm. What? Mhm. Uh huh. Mhm. Yeah. Mhm. Mhm. Yeah. Oh. Oh yeah. Mhm. Yeah. Yeah. Great. Really? Yeah. Yeah. Yeah. Yeah. Yeah. Mhm. Yeah. Mhm. Yeah. Mhm. Yeah. Oh
backchannel: Yeah   backchannel_count: 1
backchannel_speaker: 5a73899f9cdd1800017786f0
backchannel_start: 198.34   backchannel_stop: 198.66
interval: (空)   delta: 196.62   questions: 2   end_question: False   overlap: False   n_words: 39
```

另有 `survey.csv`：两位说话人各自的问卷（affect/arousal、记忆、好感度、孤独感、人格 BFI 等心理量表，字段约 240 个）——不是转写，但对会话分析很有用。

### 2.5 CANDOR.zip 补充包（whisper/ 与 laughs/）内容说明

抽查结果（以会话 `f5880665-...` 为例）：

- `whisper/<uuid>.npy`：**object 数组，形状 (2,)，每个元素是一个 dict**：`{'text': <该声道 Whisper ASR 全文>, 'segments': [...], 'language': 'en'}` —— 即双声道各一份 Whisper 转写。
- `laughs/<uuid>.npy`：**float64 数组，形状 (2, N)**（N≈72k–73k，随会话变化），内容数值很小（首元素≈42.985、其余≈±几），两行相关性 0.66，疑似**双声道音频片段**（约 1.5s @48kHz）或笑声相关特征序列；首元素在所有抽查会话中均≈42.985（仅第 5 位小数不同），语义待确认。
- ⚠️ 该 zip 是数据整理方自建的加工产物（2024-11 打包），不在官方发布物中；`laughs/` npy 的确切语义未见文档，**使用前建议谨慎核实**（可对照 whisper dict 的 segments 时间戳）。

---

## 3. Fisher（Fisher English Training Speech + Transcripts）

**位置**：`/share/workspace3/shared_dataset/Fisher/`
**来源**：LDC Fisher 英语电话对话语料（Part 1: LDC2004S13 音频 / LDC2004T19 转写；Part 2: LDC2005S13 / LDC2005T19），2002–2003 年采集，11,699 段电话对话，每段最长 10 分钟，双通道（每通道一个说话人）。

### 3.1 目录结构

```
Fisher/
├── 000/ 001/ ... 112/        # 113 个编号目录，共 11,279 个 wav（按 fe_03_ID 前三位归目录）
│   └── fe_03_00005.wav       # ★ 每段对话一个文件（NIST SPHERE 格式，见 3.3）
├── Missing/                  # 420 个 wav：下载时"缺位"的文件，按 ID 应收进对应编号目录（与编号目录零重叠，全量 11699 = 11279 + 420）
└── transcription/
    ├── LDC2004T19/           # Part 1 转写（5850 段）
    │   ├── LDC2004T19.tgz    # ★ 原始 LDC 发布包（76 MB，内容即 fe_03_p1_tran/）
    │   └── fe_03_p1_tran/
    │       ├── doc/          # 说明文档 + 元数据表（fe_03_readme.txt, bbn_trans_readme.txt, *.tbl, fe_03_topics.sgm）
    │       └── data/
    │           ├── trans/    # 5850 个 fe_03_XXXXX.txt（LDC 统一格式，含 BBN 转写转换版）
    │           └── bbn_orig/ # BBN/WordWave 原始 5 文件结构（5076 段 BBN 转写的对话）
    └── LDC2005T19/           # Part 2 转写（05851–11699，5849 段）
        ├── LDC2005T19.tgz    # 原始发布包（77 MB）
        ├── data/             # ★ 整理方解压的副本（与 fe_03_p2_tran/ 内容几乎重复）
        ├── fe_03_p2_tran/data/  # 官方结构（trans/ + bbn_orig/）
        ├── doc/ index.html
```

### 3.2 读取/解压方法

- **全部已解压就绪**，无需解压。`*.tgz` 只是官方发布包归档（内部结构 = 已解压的 `fe_03_pX_tran/`），常规使用直接读 `trans/` 与 `bbn_orig/` 即可。
- 注意 LDC2005T19 下 `data/` 与 `fe_03_p2_tran/data/` 内容基本重复（整理时解压产生了双份），**读取时取其中一份即可**（如 `fe_03_p2_tran/data/trans/`，与 Part 1 布局一致）。
- **读取音频不能用标准 wave 模块**（文件名为 .wav 实为 NIST SPHERE 格式）。可用 `sph2pipe`（LDC 工具）或 python `nist-sphere` / `soundfile`（libsndfile 支持 NIST）：
  ```python
  import soundfile as sf
  data, sr = sf.read("/share/workspace3/shared_dataset/Fisher/000/fe_03_00005.wav", dtype="float32")
  # data: (N, 2) —— 第 0 列 = 声道 1 (A)，第 1 列 = 声道 2 (B)
  ```
  NIST 头已验证：8 kHz、2 声道、8-bit ulaw、sample_count 5,759,712（≈12 min，含静音尾段）。

### 3.3 音频 ↔ transcript 对应关系

- **文件 ID 即对应关系**：音频 `fe_03_00005.wav` ↔ 转写 `data/trans/<前3位>/fe_03_00005.txt`，一一对应（11,699 段对话都有转写）。
- **声道 ↔ 说话人**：每段对话音频含 2 个声道（NIST 头 channel_count=2）；LDC 统一格式转写中用 **A:（声道 1）/ B:（声道 2）** 标记说话人；每行是 `开始秒 结束秒 A/B: 文本`。
- **BBN 原始结构（bbn_orig/）的对齐方式**：每个会话 5 个文件——`originals/*.txo`（WordWave 人工转写原文，带标点/大小写，说话人标 L:/R:）+ `auto-segmented/*.trn`（SNOR 格式转写，带 utterance ID）+ `auto-segmented/*.ana`（分段：`-c <声道> -f <起-止样点>`）+ `rejected/*.trn` + `rejected/*.ana`。`.ana` 中 `-c 1`/`-c 2` 对应声道 1/2（即 A/B），样点偏移按 8 kHz 换算为秒；`.trn` 与 `.ana` 通过 utterance ID（如 `fe_03_00007-A-0001`）配对。

### 3.4 transcript 记录的信息

**LDC 统一格式**（`data/trans/`，转写方含 LDC 与 BBN 两类）：
- 文件头两行：`# fe_03_XXXXX.sph`（会话 ID）+ `# Transcribed at the LDC` / `# Transcribed by BBN/WordWave`（转写方）。
- 每行一个 utterance：`start end A|B: text`（秒级时间戳 + 声道/说话人 + 小写无标点文本）。
- **`(( ))`**：转写不确定内容（LDC 版=听不清的部分；BBN 转换版=被对齐评分拒绝的 segment）；`(( ))` 内可为空。
- 无标点、无大小写；无专门的笑声标签（BBN 的 SNOR 转写中有 `[LAUGHTER]`、`[NOISE]`、`[MN]` 等标记；`.txo` 原文有 `[LAUGH]`、`[MN]` 及标点、大小写）。

**原文示例**（LDC 版，`data/trans/000/fe_03_00001.txt`，全文 576 行，头两行为文件头；完整 utterance 行不截断）：

```text
# fe_03_00001.sph
# Transcribed at the LDC

84.86 87.33 A: who turned me onto this he works for (( )) and he
```

**BBN 原始结构对照**（`bbn_orig/024/originals/fe_03_02413.txo`，同一语料的另一种转写格式：带标点、大小写，说话人标 `L:`/`R:`，`[?]` = 转写不确定）：

```text
L:  Hi, my name's Candy.
R:  Hi Candy.  My name is Cindy.
R:  Okay.  Um, I'm from Min[?] Missouri.
```

**元数据表**（`doc/` 下，均 CSV 风格，首行为列名）：
- `fe_03_p1_calldata.tbl`（及 Part 2 同款）：每段通话一行——CALL_ID、DATE_TIME、TOPICID（对应 topics.sgm 的话题）、SIG_GRADE（信号质量 0–4）、CNV_GRADE（对话质量 0–4）、A/B 双方 PIN、性别.方言（m/f.a/o）、加密电话号码、电话类型（手机/无绳/座机）。
- `fe_03_pindata.tbl`：每个参与者一行——PIN、性别、年龄、受教育年限、母语、成长地、参与通话数及通话ID_声道列表。
- `fe_03_topics.sgm`：话题定义（id + 标题 + 引导问题），如 ENG01 "Professional Sports on TV."。
- `file.tbl` / `doc_filelist_tbl.txt`：bbn_orig 文件清单及说明。
- 注意：**Part 1 与 Part 2 的 pindata/calldata 表是分开的**（Part 2 在 `LDC2005T19/doc/` 下）。

---

## 4. 速查对比

| | Switchboard（speechlaugh） | CANDOR | Fisher |
|---|---|---|---|
| 需要解压？ | 否（parquet 直接读） | 否（files/ 已就绪） | 否（trans/ 已就绪） |
| 音频格式 | 16kHz 单声道 WAV（嵌入 parquet） | 48kHz 双声道 MP3（L/R=两说话人） | 8kHz 双声道 NIST SPHERE（ch1/ch2=两说话人） |
| 音频↔文本对应 | 行内直接对应 | channel_map + 秒级时间戳 | 文件名 ID 一一对应 + 秒级时间戳 |
| transcript 粒度 | 短片段级（1–6s，笑声事件切分） | turn 级（3 个版本） | utterance 级（自动切分） |
| 特殊标记 | `<LAUGH>`（笑声/言语笑声） | 无特殊标记（backchannel 单独列） | `(( ))`、`[LAUGHTER]`、`[NOISE]`（BBN 版） |
| 说话人信息 | 无（仅声道 A/B 在文件名中） | user_id（问卷/特征表可关联人口学+心理测量） | PIN（可关联性别/年龄/教育/母语） |
| 会话数 | 185,402+20,601+51,501 条片段 | 1,656 场对话 | 11,699 段通话 |

---

## 5. 注意事项

1. **本次探索未修改共享目录**：全程只读；抽查 npy 时解压到 `/tmp` 的临时文件已删除，未在原目录留下任何文件。
2. CANDOR 官方 S3 签名链接已过期（2024-10-31），需要原始 mkv 视频需走官方申请；本地只有 mp3 音频。
3. Fisher 的 `.wav` 是 NIST SPHERE 而非 RIFF WAV，用 `wave`/`torchaudio`(soundfile 后端支持) 读取会报错，需 libsndfile 系工具。
4. switchboard 的路径数字不能当时间戳；CANDOR zip 内 `laughs/*.npy` 语义未经官方文档证实。
