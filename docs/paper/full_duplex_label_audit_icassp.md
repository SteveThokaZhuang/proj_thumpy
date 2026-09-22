# Full-Duplex Benchmarks Measure the Label They Inherit: Auditing Annotation Layers in Turn-Taking Evaluation

**ICASSP 2027 version — 4 pages + references.**
*Scope narrowed per reviewer feedback: the label audit, the no-shortcut protocol,
and the L3 annotator evidence. The training flywheel is future work (one sentence
in the conclusion). All numbers are final experimental values from the pilot-study
reports; figure files are in `docs/pilot_study/figures/`.*

---

## Abstract

Full-duplex spoken dialogue systems are evaluated on turn-taking behaviors whose
ground-truth labels must come from somewhere: generation-side metadata (synthetic
benchmarks), acoustic-derived pipelines (VAD/ASR timestamps plus threshold rules),
or turn-detection models used as annotators. We audit these three layers with one
protocol — channel-level acoustic features, balanced session-level bootstrap, and
adjusted Rand index (ARI). Two findings follow. First, synthetic data exhibits 5×
higher acoustic–label consistency than human conversation (ARI 0.309 vs 0.058),
but 84% of this advantage is a silence shortcut: removing the log energy-ratio
feature collapses the synthetic ARI to chance, and no prosodic feature survives
ablation. Second, the same shortcut propagates into benchmark metrics: FD-Bench-style
VAD interval rules shift EIR 25× when the segment source changes, and 10% channel
crosstalk collapses the metric set; three label sources on the same real-audio events
agree only at κ=0.06–0.37. Finally, a turn-detection model calibrated on synthetic
gold transfers to real conversations with frame-level AUC 0.636 and tracks realized
overlap with window AUROC 0.667 — the best-aligned label source for real audio at the
soft-score level. We distill these results into a *no-shortcut evaluation protocol*.

## 1. Introduction

Every full-duplex evaluation metric presupposes labels: which spans are backchannels,
which are interruptions, which turns were handled well. Three sources dominate:
**L1** generation-side ground truth (synthetic benchmarks, exact by construction);
**L2** acoustic-derived labels (VAD/ASR timestamps + hard threshold rules, applicable
to real audio but toolchain-dependent); **L3** model annotators (turn-detection models
emitting frame-level states). Each layer has its own failure mode, and none has been
audited. Our claim: *full-duplex benchmarks measure the label they inherit, not only
the behavior they claim.*

Contributions: (i) a unified audit of the three annotation layers on matched events;
(ii) quantification of the silence shortcut and its propagation into benchmark metrics;
(iii) evidence, at the soft-score level, that model annotators are the most
acoustically faithful label source currently available for real conversations;
(iv) the no-shortcut evaluation protocol (Table 1).

**[Fig. 1]** The three annotation layers and their audited failure modes
(`fig1_label_taxonomy.png`).

## 2. Audit Protocol

**Data.** Synthetic side: Behavior-SD (TTS-rendered dyadic dialogues with full
metadata; 5,857 files). Real side: CANDOR (1,656 naturalistic dyadic video-call
conversations with AWS-derived turn annotations). Both are true stereo with one
speaker per channel, enabling channel-level analysis.

**Events and features.** Three event classes (backchannel / interruption / none)
defined identically on both datasets. Five features per event window, computed on
the event speaker's channel relative to the partner's channel over the same window:
log energy ratio, F0 correlation, F0 slope, spectral centroid, voiced ratio.
Duration is excluded (circular with label definitions).

**Agreement metric.** Balanced 1:1:1 sampling (N=1,000/class), session-level
bootstrap (B=500), K=3 KMeans, ARI against labels; permutation baselines ≈ 0.
For label-source comparisons: Cohen's κ on matched events and frame/window AUROC.

## 3. Auditing L1 and L2

### 3.1 Synthetic ground truth is shortcut-carried

Matched-protocol ARI is 0.309 [0.287, 0.332] for synthetic metadata labels vs
0.058 [0.043, 0.076] for human-derived labels (Δ=−0.251 [−0.281, −0.223], p<0.001):
synthetic labels are far *more* acoustically recoverable, but the advantage is a
shortcut. The log energy-ratio feature alone reproduces 0.314 of the 0.309 full-set
ARI; removing it collapses the synthetic ARI to 0.050 (chance). Every prosodic
feature is near chance on both datasets. The cross-dataset distribution shift is
dominated by one term: the None class's energy ratio shifts by Cohen's d=1.57 — in
synthetic audio the partner's channel is truly silent during non-overlap turns.
Moreover, within a fixed event type, whether a token *actually* overlaps host speech
(stereo-realized) is barely decodable from the token's own prosody (event-channel
AUC 0.58/0.53; pitch-slope d=0.002); only substantial overlaps leave a trace
(AUC 0.64/0.73). Interactional state is largely absent from synthetic prosody.

**[Fig. 2]** ARI comparison with bootstrap CIs + feature ablation
(`ari_comparison.png`, `b_ablation.png`).

### 3.2 Acoustic-derived labels are toolchain-determined

Three toolchains on the same CANDOR events — AWS-derived labels, Silero-VAD interval
rules (FD-Bench parameters), channel-level stereo-realized overlap — agree weakly:
κ=0.37 (AWS vs VAD), κ=0.06 (AWS vs realized), κ=0.24 (VAD vs realized). 62% of
AWS-Int events show only brief (0.1–0.5s) true overlap and 36% none; 29% of
AWS-None events contain brief both-active spans. Replicating FD-Bench's interval
rules on 1,857 synthetic files with four segment sources (metadata GT, VAD on clean
channels, VAD with 10%/30% crosstalk), EIR shifts 25× (0.018→0.449), and 10%
crosstalk collapses the metric set (stream count 6.5→1.0, SIR/NIR → 0). The synthetic
timestamps themselves are honest (99.6% of GT interruptions show real overlap):
the instability lives in the label-derivation pipeline.

**[Fig. 3]** Label-source consistency matrix + FD-Bench rule drift
(`e1_label_matrix.png`, `e2_fdbench_audit.png`).

## 4. Model Annotators (L3)

Calibration on synthetic gold exposes a label-space mismatch: 64% of the annotator's
backchannel firings are brief acknowledgment tokens inside the speaker's own turn
(linguistically valid, but excluded by the metadata's cross-speaker event definition),
and forcing the "host must be speaking" prior collapses event F1 to 0.028. Calibration
must therefore include an explicit label-space mapping; we adopt frame-level
probabilities and window-coverage metrics.

On real audio (20 CANDOR sessions, 1,532 gold windows): frame-level AUC 0.636 for
gold windows, and window-level AUROC 0.667 — the maximum backchannel probability
inside a labeled window predicts whether that window contains real stereo overlap.
This is far above every L2 source's alignment with realized overlap. The advantage
holds out-of-domain (Fisher telephone speech, 8 kHz, 54 conversations: speaking AUC
0.844, backchannel AUC 0.624), and two annotators (X2-Turn, SoulX-Duplug) fuse best
by probability
weighting (window AUROC 0.652 vs 0.620/0.570 alone) rather than hard intersection.
However, at the hard-threshold level the advantage largely disappears (κ vs realized
0.12 for the annotator vs 0.16 for VAD): the value of L3 lies in its soft scores.
Finally, ASR quality is only weakly coupled to state quality on both synthetic
(r=−0.27, n.s.) and real data (r=+0.31, likely an interaction-density confound).

**[Fig. 4]** L3 on real audio: frame AUC, window AUROC, and cross-domain results
(`e4_candor.png`; Fisher table in text).

## 5. The No-Shortcut Evaluation Protocol

**Table 1 — Companion numbers for any turn-taking metric built on acoustic-derived
labels.** (1) *Shortcut ablation*: recompute without the energy-ratio/other-channel-
silence features (collapses synthetic ALC 0.31→0.05); (2) *segment-source sensitivity*:
metric drift under toolchain/threshold changes (EIR drifts 25×); (3) *crosstalk stress*:
degradation at 10%/30% channel leakage (full collapse at 10%); (4) *label-source
consistency*: pairwise κ among the label sources used (0.06–0.37 on real audio).

**Limitations.** The three layers are not semantically identical (L1 cross-speaker
events vs L3 linguistic tokens); our protocol reports both event- and frame-level
agreement. CANDOR's AWS labels are themselves L2 and noisy; the human-side ARI ceiling
(≈0.07 after channel cleaning) is conditional on the five literature features. Human
verification of the annotator on a sample of windows is ongoing.

**Conclusion.** Full-duplex benchmarks measure the label they inherit. Synthetic
ground truth is exact but shortcut-carried; acoustic-derived labels are toolchain-
determined; model annotators are the most acoustically faithful option for real audio
at the soft-score level. The no-shortcut protocol (Table 1) is a first step toward
trustworthy evaluation — and, we conjecture, toward labels that can teach models the
interactional behaviors the benchmarks claim to measure.

---

## References

1. FD-Bench: A Full-Duplex Benchmarking Pipeline, arXiv:2507.19040.
2. X2-Turn: Frame-synchronous streaming ASR and turn-state prediction, arXiv:2608.10878.
3. SoulX-Duplug: Plug-and-Play Streaming State Prediction for Full-Duplex Speech
   Conversation, arXiv:2603.14877.
4. Reece et al., The CANDOR corpus, Science Advances 2023.
5. Levitan et al., "Entrainment in Speech Preceding Backchannels," ACL 2011.
6. Hubert & Arabie, "Comparing Partitions," J. Classification 1985.
7. Silero VAD (github.com/snakers4/silero-vad).
