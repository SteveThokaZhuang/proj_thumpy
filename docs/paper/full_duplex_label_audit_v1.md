# Who Labels the Labels? Auditing Annotation Layers in Full-Duplex Dialogue Evaluation

**Draft v1 — 2026-08-26** · target: Interspeech/ICASSP 2027 · 8 pages, 8 figures + 1 table
*All numbers in this draft are final experimental values from the pilot-study reports; figure files are in `docs/pilot_study/figures/`.*

---

## Abstract

Full-duplex spoken dialogue models are evaluated on turn-taking behaviors — interruptions and backchannels — whose ground-truth labels must come from somewhere. In practice they come from one of three layers: generation-side metadata (synthetic benchmarks), acoustic-derived pipelines (VAD/ASR timestamps plus threshold rules), or, increasingly, turn-detection models used as annotators. Each layer has its own failure mode, and none has been audited. Using matched channel-level acoustic features and a balanced, session-level bootstrap protocol (ARI, K=3), we show that synthetic data exhibits 5× higher acoustic–label consistency than human conversation (ARI 0.309 vs 0.058, Δ=−0.251, p<0.001) — but 84% of this advantage is carried by a single silence shortcut: removing the log energy-ratio feature collapses the synthetic ARI to 0.050, and no prosodic feature survives ablation. The same finding propagates to benchmark metrics: FD-Bench-style VAD interval rules shift EIR 25× when the segment source changes (0.018→0.449), and 10% channel crosstalk collapses the entire metric set. Three label sources on the same real-audio events agree at κ=0.06–0.37 — the label itself is a degree of freedom. Finally, we calibrate a turn-detection model (X2-Turn) on synthetic gold labels and apply it to real conversations: frame-level AUC 0.636 and a window-level AUROC of 0.667 against stereo-realized overlap — substantially better aligned with acoustic reality than any acoustic-derived label source. We distill these results into a *no-shortcut evaluation protocol* for full-duplex benchmarks.

## 1. Introduction

Full-duplex spoken dialogue systems (FD-SDS) promise natural, barge-in-capable conversation, and their evaluation increasingly centers on *interactional* behaviors: successful interruption handling, backchannel timing, turn-boundary latency (FD-Bench; X2-Turn; SoulX-Duplug). Every such metric presupposes labels: which spans of the audio are backchannels, which are interruptions, which turns were handled successfully. Where do these labels come from? Three sources dominate:

- **L1 — generation-side ground truth.** Synthetic evaluation data is generated from scripts (GPT-written dialogues, TTS rendering) whose metadata specifies every event precisely. Perfect in principle; available only for synthetic audio, whose acoustic properties differ from real conversation in ways that matter.
- **L2 — acoustic-derived labels.** Real recordings are labeled by toolchains: VAD/ASR timestamps plus hard-coded interval rules (e.g., FD-Bench's Silero-VAD streams with 0.5s/2.5s thresholds), or ASR-turn annotations (e.g., AWS-Transcribe-derived backchannel analyses). Works on real audio, but the labels inherit every error of the toolchain.
- **L3 — model annotators.** Turn-detection models trained for real conversations (X2-Turn, SoulX-Duplug) emit frame-level turn states including backchannel, and can annotate arbitrary real audio — LLM-assisted labeling by another name.

This paper audits the three layers with one consistent protocol. For each layer we ask the same question: **how consistent are its labels with the acoustics of the labeled events themselves?** We measure acoustic–label agreement with channel-level prosodic features (log energy ratio, F0 correlation, F0 slope, spectral centroid, voiced ratio) extracted from true-stereo recordings, and cluster with balanced sampling and session-level bootstrap (B=500, K=3, permutation baselines).

Three results structure the paper. First (§4), the L1 audit: synthetic metadata labels are 5× more acoustically consistent than human-derived labels — but the advantage is 84% a *silence shortcut* (the other speaker's channel being quiet), with no prosodic content. Second (§5), the L2 audit: the metrics built on acoustic-derived labels are determined by the toolchain, not by model behavior — three label sources agree only weakly on the same events (κ=0.06–0.37), and FD-Bench-style metrics drift 25× across segment sources and collapse under 10% crosstalk. Third (§6), the L3 audit: a turn-detection model calibrated on synthetic gold labels transfers to real audio with frame-level AUC 0.636 and tracks *realized* overlap (window AUROC 0.667) better than any acoustic-derived source — making model annotators the most viable label source for real data, provided their label space is mapped explicitly during calibration. We close (§7) with the *no-shortcut evaluation protocol* (Table 1).

Our contributions: (i) the first unified audit of the three annotation layers in FD-SDS evaluation; (ii) quantification of the silence shortcut and its downstream effect on benchmark metrics; (iii) a calibrated model-annotator pipeline for real conversations, with evidence for how its labels should be fused.

**[Fig. 1]** — the three-layer framework (file: `figures/fig1_label_taxonomy.png`).

## 2. Related Work

**Full-duplex benchmarks.** FD-Bench [1] evaluates FD-SDS with generated TTS conversations and metrics including SRR/SIR/EIR/NIR computed from Silero-VAD timestamps on separate input/output streams via hard threshold rules; we replicate these rules faithfully in §5. Its turn-taking scores are proxied by interval rules over synthetic, channel-clean audio — the assumption our audit targets.

**Turn-detection models.** X2-Turn [2] is a 4B frame-synchronous streaming ASR + turn-state model emitting, every 80ms, one of six states (idle/noidle/speaking/turn_end/backchannel/uncertain) with probabilities. SoulX-Duplug [3] is a 0.6B text-guided streaming state predictor emitting, every 160ms, user-side states including backchannel, with a cascaded external ASR. Both were designed as runtime components; we repurpose them as annotators (§6).

**Backchannel acoustics.** Backchannel tokens are brief listener responses with characteristic prosody — falling pitch, reduced energy relative to the host speaker [4,5] — motivating our feature set. CANDOR [6] provides 1,656 naturalistic dyadic video-call conversations with AWS-derived turn annotations; we use it as the real-audio testbed.

## 3. Label Taxonomy and Audit Protocol

**Three layers.** L1 (generation GT) exists only for synthetic data and is exact by construction. L2 (acoustic-derived) applies to any audio but inherits toolchain noise: VAD threshold effects, ASR segmentation errors, and hard rule sensitivity. L3 (model annotators) applies to any audio, produces frame-level soft labels, and can be calibrated against L1 on synthetic data. Fig. 1 summarizes each layer's claim and audited failure mode.

**Data.** Synthetic side: Behavior-SD (TTS-rendered dyadic dialogues with full metadata: nested backchannels with timestamps; 5,857 files). Real side: CANDOR (1,656 sessions; backbiter/AWS-derived BC windows and turn-level overlap flags). Both are true stereo with one speaker per channel — enabling channel-level acoustic analysis that mixed-mono pipelines cannot perform.

**Events and features.** We extract three event classes (BC / Int / None) with identical channel-level definitions on both datasets. Five features per event window, computed on the event speaker's channel relative to the partner's channel over the same window: log10 energy ratio, F0 correlation (jointly voiced frames), F0 slope (Hz/s), spectral centroid, voiced ratio. Duration is deliberately excluded (circular with label definitions). Feature extraction details follow [4,5].

**Agreement metric.** Balanced 1:1:1 sampling (N=1,000/class), session/file-level bootstrap (B=500), K=3 KMeans, ARI against labels; permutation baseline ≈ 0 throughout. This is the *acoustic–label consistency* (ALC) of a label layer. For label-source comparisons we additionally use Cohen's κ on matched events and frame/window-level AUROC.

## 4. Audit I — Synthetic Ground Truth (L1)

**Result 1: the direction is reversed from naive expectation.** Matched-protocol ALC is 0.309 [0.287, 0.332] for synthetic metadata labels vs 0.058 [0.043, 0.076] for human-derived labels (Δ=−0.251 [−0.281, −0.223], p<0.001; Fig. 2–3). Synthetic labels are far *more* acoustically recoverable — because the synthetic rendering makes them so, not because human labels are meaningless.

**Result 2: the advantage is a silence shortcut.** Feature ablation (Fig. 4): the log energy-ratio feature alone reproduces 0.314 of the 0.309 full-set ALC; removing it collapses ALC to 0.050 (chance). Every prosodic feature is at or near chance on both datasets (event-channel prosody: 0.043 synthetic / 0.018 human); F0 features *hurt* clustering (no-F0 ALC 0.337 > full). The cross-dataset distribution shift is dominated by a single term: the None class's energy ratio shifts by Cohen's d=1.57 — in synthetic audio, the partner's channel is truly silent during non-overlap turns; in real recordings, crosstalk keeps it audible. A classifier trained on synthetic features transfers to real audio at 0.447 macro-F1 (in-domain 0.718), and removing the energy ratio sends the transfer to chance (0.344) — the shortcut is real, load-bearing, and non-transferable.

**Result 3: interactional state is absent from synthetic prosody.** Within-category tests (A1/A2): for a fixed event type, whether the token *actually* overlaps host speech (stereo-realized) is barely decodable from the token's own prosody — event-channel-only AUC 0.58 (BC) / 0.53 (Int), pitch-slope Cohen's d=0.002; only substantial overlaps (>10–30% of the window) leave a prosodic trace (AUC rising to 0.64/0.73). Human BCs are prosodically flat and text-independent (median slope −7.7 Hz/s), while synthetic BC prosody follows lexical templates, including rising "mhm"/"really" (+74/+87 Hz/s) — pragmatically wrong for acknowledgment tokens. The synthetic renderer produces backchannel-shaped tokens whose interactional property (concurrency with the host) is not written into their sound.

**[Fig. 2]** ALC comparison (CANDOR vs Behavior-SD, with CIs). **[Fig. 3]** bootstrap Δ distribution. **[Fig. 4]** feature ablation.

## 5. Audit II — Acoustic-Derived Labels and Benchmark Metrics (L2)

**Result 4: the label itself is a degree of freedom.** We ran three toolchains on the *same* CANDOR events: AWS-derived labels, Silero-VAD interval rules (FD-Bench parameters), and channel-level stereo-realized overlap. Pairwise agreement is weak: κ=0.37 (AWS vs VAD), κ=0.06 (AWS vs realized), κ=0.24 (VAD vs realized). The AWS Int/None boundary cuts through acoustic reality: 62% of AWS-Int events show only brief (0.1–0.5s) true overlap and 36% none at all, while 29% of AWS-None events contain brief both-active spans. VAD detects only 37% of AWS backchannel windows, and only 44.5% of them occur during host speech. (Fig. 5.)

**Result 5: benchmark metrics inherit the toolchain.** Replicating FD-Bench's interval rules on 1,857 synthetic files with four segment sources — metadata GT, VAD on clean channels, VAD with 10%/30% injected crosstalk — the same rules produce wildly different metric values: EIR shifts 25× (0.018 with GT vs 0.449 with VAD; per-file correlation 0.17), SIR drifts 0.15 (corr 0.57), NIR anticorrelates (−0.13). With 10% crosstalk the VAD stream count collapses from 6.5 to 1.0 rounds per file and SIR/NIR degenerate to zero — the "separate clean streams" assumption is load-bearing, and real recordings exceed 10% crosstalk. (Fig. 6.) The synthetic timestamps themselves are honest (99.6% of GT interruptions show real both-active overlap), so the instability lives in the *label-derivation pipeline*, not in the data.

**Implication.** Reported SIR/EIR/NIR values on synthetic benchmarks measure the VAD pipeline's segmentation behavior at least as much as the model under test; on real audio the same metrics are not even computable.

**[Fig. 5]** label-source consistency matrix. **[Fig. 6]** FD-Bench rule audit under crosstalk.

## 6. Audit III — Model Annotators (L3)

**Calibration on synthetic gold (E3).** Running X2-Turn per channel on 30 Behavior-SD files (164 gold BC windows): the frame-level backchannel probability ranks gold-BC frames at AUC 0.784 — the state concept is right — but hard event-level matching yields F1 0.22. The diagnosis is *label-space mismatch*, not incompetence: 64% of its backchannel firings are brief acknowledgment tokens inside the speaker's own turn (linguistically valid backchannels that the metadata's cross-speaker event definition excludes), and forcing the "host must be speaking" prior collapses F1 to 0.028 — consistent with 44–54% of metadata BCs being rendered into pauses. Calibration therefore must include an explicit label-space mapping; we adopt frame-level probabilities and window-coverage metrics as the calibrated protocol. (Fig. 7.)

**Application to real audio (E4).** On 20 CANDOR sessions (1,532 AWS BC windows): frame-level AUC 0.636±0.079 for gold windows, pipeline sanity AUC 0.695 for speech, and — the key result — **window-level AUROC 0.667**: the model's maximum backchannel probability inside a labeled BC window predicts whether that window contains real stereo overlap. This is far above every acoustic-derived source's alignment with realized overlap (AWS κ=0.06; VAD κ=0.24; §5). The synthetic-calibrated annotator degrades mildly out-of-domain (0.78→0.64), its label-space mismatch largely vanishes on real short-turn audio (5.6% vs 64%), and it covers 43% of AWS BC windows at τ=0.1. (Fig. 8.)

**Two annotators (E5).** SoulX-Duplug (cascaded SenseVoice ASR) on the same windows: coverage 0.37 vs X2-Turn 0.27; inter-annotator κ=0.21. Both track real overlap (any-hit windows 0.63–0.73 realized overlap vs 0.50 unhit), but their *intersection* (0.64) does not beat the better single annotator (SoulX-only 0.73) — consensus-by-intersection fails; fusion should use union plus probability averaging.

**Implication.** A synthetic-gold-calibrated turn-detection model is the most acoustically faithful label source currently available for real full-duplex conversations, provided (i) calibration includes label-space mapping, and (ii) multiple annotators are fused by union/probabilities rather than hard intersection.

**[Fig. 7]** calibration curves. **[Fig. 8]** real-audio annotation results.

## 7. Discussion

**Table 1 — The no-shortcut evaluation protocol.** For any turn-taking metric built on acoustic-derived labels we recommend reporting four companion numbers: (1) *shortcut ablation*: the metric computed without the energy-ratio/other-channel-silence features (on synthetic data this collapses ALC 0.31→0.05); (2) *segment-source sensitivity*: the metric drift when the VAD/ASR toolchain or its thresholds change (EIR drifts 25× in our replication); (3) *crosstalk stress*: degradation at 10%/30% channel leakage (full collapse at 10%); (4) *label-source consistency*: pairwise κ among the label sources used (0.06–0.37 on real audio).

**Limitations.** The three label layers are not semantically identical (L1 cross-speaker events vs L3 linguistic tokens), which our protocol handles by reporting both event- and frame-level agreement. CANDOR's AWS labels are themselves L2 and noisy; our human-side ceiling (≈0.07 after channel cleaning) is conditional on the five literature features. The annotator results use 20 real sessions; ASR dependence of both annotators is unmeasured (an oracle-ASR ablation is planned). KMeans/balanced-N settings are fixed. All code and data products are available for reproduction.

**Future work.** Probe-fine-tuning annotators on synthetic gold; probability-level fusion of multiple annotators; extending the audit to other benchmarks; oracle-ASR attribution.

## 8. Conclusion

Full-duplex benchmarks measure the label they inherit, not only the behavior they claim. We audited the three available label layers with one protocol: synthetic ground truth is exact but shortcut-carried (84% silence shortcut, no prosodic content); acoustic-derived labels are toolchain-determined (κ 0.06–0.37 across sources, 25× metric drift, collapse under 10% crosstalk); and calibrated model annotators are the most acoustically faithful option for real audio (frame AUC 0.636, overlap AUROC 0.667) when their label space is mapped explicitly. We contribute the audit evidence and the no-shortcut evaluation protocol (Table 1) as a first step toward trustworthy evaluation of full-duplex dialogue systems.

---

## References (working list)

1. FD-Bench: A Full-Duplex Benchmarking Pipeline, arXiv:2507.19040.
2. X2-Turn: Frame-synchronous streaming ASR and turn-state prediction, arXiv:2608.10878.
3. SoulX-Duplug: Plug-and-Play Streaming State Prediction for Full-Duplex Speech Conversation, arXiv:2603.14877.
4. Levitan et al., "Entrainment in Speech Preceding Backchannels," ACL 2011.
5. Ward, "Backchannel Facts," 2019 (cs.utep.edu/nigel/bc).
6. Reece et al., The CANDOR corpus, Science Advances 2023.
7. Hubert & Arabie, "Comparing Partitions," Journal of Classification 1985 (ARI).
8. Silero VAD (github.com/snakers4/silero-vad).

---

**Figure inventory** (all in `docs/pilot_study/figures/`): Fig1 `fig1_label_taxonomy.png`; Fig2 `ari_comparison.png`; Fig3 `delta_hist.png`; Fig4 `b_ablation.png`; Fig5 `e1_label_matrix.png`; Fig6 `e2_fdbench_audit.png`; Fig7 `e3_calibration.png`; Fig8 `e4_candor.png`. Supporting: `a2_threshold_auc.png`, `confusion.png`, `pca_scatter.png`, `e5_dual_annotator.png`, `e_prosody_text.png`, `d_human_baseline.png`, `c_transfer.png`.
