H3 council result — clustering and speaker count

**Exploratory result: no material, reliable improvement supports replacing frozen ZS\*.**
The combined inner-fold 1-SE selection retains the original global-threshold AHC in all five
folds, reproducing **26.97% DER at collar 0, 25.13% at collar 0.25, and Σ|count error| = 192**.
All 40 leave-one-recording-out checks within those five inner pools also retain it. H3's
premise—heterogeneous embedding distance scales—is verified, but the implication that
NME-SC or simple recording normalization should reliably fix CONF is not supported here.
NeMo NME-SC substantially worsens confusion despite improving speaker counts. A smaller,
uncertain normalization signal merits preserving as a research candidate, not promoting
to a new default.

The strongest favorable alternative is **quantile normalization selected by inner argmin:
26.01% / 24.15% DER, count error 128**. Its collar-0 difference is −0.96 percentage points,
descriptive paired-bootstrap 95% CI **[−4.07, +0.71]**, and it improves only **2/10 recordings**.
The comparable family-specific 1-SE version is 26.89%, and the baseline remains eligible
under the combined 1-SE rule. This is a localized effect, not an absence of any signal:
train_001 improves 45.35 → 28.16%, while train_013 improves 56.90 → 53.66%; seven other
recordings worsen and one ties. Do not summarize this run as either “adaptation works” or
“all adaptive methods fail.”

**Evaluation and scope.** All aggregate numbers in this report are exploratory looks at the
same ten recordings, including the remeasured baseline. The baseline itself is the previously
frozen second-look ZS* comparator. I used cached zero-shot s20 tracks/embeddings at θ_act=0.40,
the original exclusive decoding, and gap-fill 5 s throughout. No boundary snapping, new neural
inference, fine-tuning, GPU, model download, repository edit, git operation, or subagent was used.
There are 49 prespecified configurations (45 deployable configurations, including the duplicated
baseline/global-0.60 control, and four diagnostics): 490 RTTMs, scored at both collars. All were
generated and hashed before the first new score. A later, explicitly post-look diagnostic
adds 20 RTTMs to enforce oracle K on the emitted timeline, as explained below.

Outer folds are exactly F1 001+008, F2 011+009, F3 013+005, F4 004+002, F5 003+006.
Each candidate family and the combined deployable pool select on the other eight recordings'
pooled collar-0 DER only. The 1-SE rule is the existing phase-3b rule: each candidate is eligible
when its pooled loss is within the sample standard deviation of its per-recording loss
differences from the argmin, divided by √8. Simplicity order was written before scoring.
No outer reference was used in deployable clustering or normalization; per-recording unlabeled
adaptation is allowed. These zero-shot, label-free methods need no supervised refit inside
the inner folds. Speaker-count errors and collar-0.25 results were not selection objectives.
The search design still reflects prior looks, so this is not a new held-out performance claim.

Scoring uses bnd.score, whole-file UEM, skip_overlap=False, and the repository's treatment
of same-label overlaps. Collars are **total width**: 0.25 removes ±0.125 s at each original
reference boundary. Reference totals are 27,856.00 s at collar 0 and 27,218.25 s at collar 0.25.
Counts are unique speakers in the full emitted RTTM and are independent of scoring collar.
MISS and FA can change slightly when identities change because same-speaker gap-fill depends
on clustering; those changes do not imply new acoustic speech detection.

**Primary family comparisons, inner 1-SE selection.** Every row is exploratory.


| Method | Collar 0: DER [MISS/FA/CONF] % | Collar 0.25: DER [MISS/FA/CONF] % | Σ absolute count error |
|---|---:|---:|---:|
| Frozen global AHC / global-threshold retune / combined selection | 26.97 [6.90/6.85/13.22] | 25.13 [6.35/5.85/12.93] | 192 |
| Per-recording quantile normalization | 26.89 [6.90/6.85/13.13] | 25.04 [6.35/5.85/12.83] | 128 |
| Per-recording median normalization | 26.87 [6.89/6.85/13.12] | 25.04 [6.34/5.85/12.85] | 169 |
| NeMo NME-SC | 36.83 [6.19/7.16/23.48] | 35.08 [5.67/6.14/23.27] | 83 |
| NME-SC + same-window Hungarian assignment | 38.21 [6.91/6.86/24.44] | 36.49 [6.36/5.86/24.26] | 96 |
| Cosine mixture HMM family (selects loop=0) | 27.39 [7.07/6.90/13.41] | 25.55 [6.52/5.91/13.12] | 175 |
| HMM positive-loop ablation | 29.23 [6.29/7.16/15.78] | 27.38 [5.77/6.14/15.46] | 179 |
| Overlap-context must-link hints | 38.34 [6.86/6.86/24.61] | 36.68 [6.31/5.86/24.51] | 158 |


**Argmin sensitivity, still selected on inner recordings only.** These are not all-ten
best-grid picks. The difference between family argmin and combined argmin matters: selecting
the best family separately after inspecting its outer outcome would add selection bias.


| Method | Collar 0: DER [MISS/FA/CONF] % | Collar 0.25: DER [MISS/FA/CONF] % | Σ absolute count error |
|---|---:|---:|---:|
| Global threshold, argmin | 28.76 [6.90/6.85/15.01] | 26.94 [6.35/5.85/14.74] | 160 |
| Quantile normalization, argmin | 26.01 [6.90/6.85/12.25] | 24.15 [6.35/5.85/11.94] | 128 |
| Median normalization, argmin | 28.25 [6.89/6.85/14.50] | 26.42 [6.34/5.85/14.22] | 123 |
| NME-SC, argmin | 39.33 [6.24/7.07/26.02] | 37.60 [5.71/6.05/25.84] | 72 |
| NME-SC + same-window assignment, argmin | 40.52 [6.92/6.86/26.74] | 38.82 [6.37/5.86/26.59] | 85 |
| Cosine HMM, argmin | 27.98 [7.19/7.03/13.76] | 26.14 [6.67/6.02/13.44] | 148 |
| Combined deployable pool, argmin | 28.04 [6.89/6.85/14.30] | 26.21 [6.34/5.85/14.01] | 134 |


The median-normalization 1-SE difference is −0.10 pp, CI [−0.66, +0.35], wins 4/10;
quantile-normalization 1-SE is −0.08 pp, CI [−3.63, +2.46], wins 2/10. NME-SC 1-SE is
**+9.86 pp**, CI [+4.66, +14.68], wins 1/10. The HMM family is +0.42 pp,
CI [−0.20, +1.29], wins 4/10. The combined argmin reaches 28.04%, worse than the baseline.
Intervals are 50,000 paired recording resamples, seed 20260925, with choices treated as fixed;
they omit search-design and selection uncertainty and are descriptive only.

**What was actually implemented.** The global retune preserves bnd.link exactly and searches
θ_clu={0.50,0.55,0.60,0.65,0.70,0.75,0.80}, baseline-first simplicity. Quantile normalization
uses D'=(D−q10)/(q90−q10), where quantiles come from strong-track cross-window pair distances
within each recording; thresholds are {0.35,0.45,0.55,0.65,0.75,0.85}. Median normalization
uses D'=D/q50 with thresholds {0.55,0.65,0.75,0.85,0.95}. These are positive affine transforms:
they change the recording's operating threshold, not its AHC merge order. The same transform
is used for weak attachment, and the original cannot-link/NaN policies are retained.
Independent validation reproduced every normalized partition by running the original linker
at its equivalent raw threshold.

NME-SC uses the repository's offline_clustering.py directly: min-max cosine affinity,
NMESC eigengap/neighbor search, then SpectralClustering, with max_num_speakers=40,
max_rp_threshold in {0.05,0.10,0.15,0.25}, majority count on/off, search volume 30,
nme_mat_size=512, ten k-means trials, parallelism=False, cuda=False. All 248–427-track
recordings are below the matrix-size limit; NME subsampling is not responsible for these
failures. It clusters every finite track, including finite weak tracks, and retains NaN
singletons. The primitive implementation was checked against SpeakerClustering.forward_unit_infer
on train_001 and produced the identical partition. Native spectral clustering does not enforce
the AHC hard cannot-links; the second family assigns each window's finite tracks to distinct
fixed spectral centroids with Hungarian assignment, adding dummy singleton states only when
there are fewer spectral states than tracks. It also performs poorly, so simply restoring
the constraint does not rescue this experiment.

The HMM is explicitly **not genuine VBx**: it uses cosine emissions and ordinary EM, no PLDA
or Bayesian speaker-factor updates. It initializes states from baseline finite-cluster
centroids, uses the ordered exclusive core-track events with duration weights clipped to
[0.1,2] in units of 1.5 seconds, five EM iterations, concentration {10,25,50}, and
T=loop·I+(1−loop)·1·πᵀ with loop {0,0.5,0.9}. Missing-embedding events have uniform emissions.
It does not enforce same-window cannot-links or model elapsed silent-gap length; a cached
track embedding may be reused for several events. The chosen family configuration has
loop=0 and concentration 25 in every fold, so even its small changes cannot be credited to
temporal smoothing. Restricting to positive loops selects loop=0.5, concentration 25 and
gives 29.23% DER. The parameter-free overlap-hint experiment uses the existing adjacent-context
activity-IoU matching (IoU≥0.5 and ≥0.5 s activity) plus baseline AHC. Its 38.34% result argues
against treating those context matches as reliable hard must-links here.

The fold choices are auditable below. Counts in parentheses are LORO recoveries out of eight.


| Family | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|

| all_deployable_1se | baseline (8/8) | baseline (8/8) | baseline (8/8) | baseline (8/8) | baseline (8/8) |

| norm_quantile_1se | norm_quantile_t0p45 (7/8) | norm_quantile_t0p55 (8/8) | norm_quantile_t0p45 (7/8) | norm_quantile_t0p45 (7/8) | norm_quantile_t0p45 (7/8) |

| norm_quantile_argmin | norm_quantile_t0p45 (7/8) | norm_quantile_t0p45 (7/8) | norm_quantile_t0p45 (7/8) | norm_quantile_t0p45 (8/8) | norm_quantile_t0p45 (8/8) |

| norm_median_1se | norm_median_t0p75 (8/8) | norm_median_t0p75 (8/8) | norm_median_t0p75 (8/8) | norm_median_t0p75 (8/8) | norm_median_t0p75 (8/8) |

| nme_1se | nme_rp0p1_maj0 (6/8) | nme_rp0p1_maj0 (7/8) | nme_rp0p15_maj0 (7/8) | nme_rp0p1_maj0 (7/8) | nme_rp0p1_maj0 (6/8) |

| hmm_1se | hmm_loop0_kappa25 (8/8) | hmm_loop0_kappa25 (8/8) | hmm_loop0_kappa25 (7/8) | hmm_loop0_kappa25 (8/8) | hmm_loop0_kappa25 (7/8) |


**Oracle diagnostics: speaker count is not an identity oracle or a mathematical DER ceiling.**


| Method | Collar 0: DER [MISS/FA/CONF] % | Collar 0.25: DER [MISS/FA/CONF] % | Σ absolute count error |
|---|---:|---:|---:|
| AHC: stop strong hierarchy at reference K; baseline weak policy | 26.55 [6.90/6.85/12.79] | 24.69 [6.35/5.85/12.49] | 77 |
| All-finite unconstrained AHC, θ=0.60; nearest-time NaN attachment | 27.94 [6.58/6.98/14.38] | 26.10 [6.04/5.97/14.09] | 145 |
| All-finite unconstrained AHC, reference K; same NaN policy | 30.11 [6.21/7.19/16.71] | 28.29 [5.69/6.15/16.44] | 17 |
| Post-look: output-visible finite AHC, θ=0.60; nearest-time NaNs | 28.62 [6.57/7.00/15.06] | 26.78 [6.02/5.99/14.77] | 133 |
| Post-look: output-visible finite AHC, exact reference output K | 28.61 [6.34/7.14/15.12] | 26.79 [5.82/6.12/14.85] | 0 |
| Outer-label oracle: best raw threshold per recording within tested grid | 25.35 [6.90/6.85/11.60] | 23.50 [6.35/5.85/11.30] | 146 |
| Full reference track-identity oracle, original implementation | 16.08 [6.04/6.97/3.06] | 14.18 [5.51/5.96/2.71] | 35 |


Stopping the original strong-track hierarchy at true K gives only −0.42 pp,
CI [−3.73,+1.81], wins 2/10; weak/NaN identities still cause count error 77. This is the
closest controlled test of “replace the threshold by a perfect count.” The unconstrained
all-finite oracle produces exactly K **track clusters**, but context-only tracks can form
clusters that never emit an owned frame: its output count error is 17. Its frozen artifact
ID `oracle_k_exact` refers to that finite-track count, **not exact output count**.

To remove this ambiguity, a post-main-grid diagnostic clusters only finite tracks that
win at least one exclusive output frame, attaches visible NaN tracks to the nearest visible
finite track in time, and emits the fixed winner-track timeline. Stopping this AHC at true
K achieves **zero output count error but 28.61% DER**, versus 28.62% and count error 133 for
its matched θ=0.60 control. Both drop hard cannot-links and change the weak/context policy;
they must not be represented as clean improvements to the original baseline. The follow-up
uses no oracle identities and no tuned hyperparameter, but was designed after seeing the
count mismatch, is labeled post-look, and is excluded from every deployable selection.

Even an outer-label oracle choosing each recording's best raw threshold from the tested
seven-value grid gives 25.35%, only 1.62 pp below baseline. This is a restricted-grid
diagnostic envelope, not a bound over all possible thresholds. The full identity oracle
still gives 16.08%, preserving the 10.89 pp linking gap. Much of that gap needs better
assignments or revised merge geometry, not merely a better stopping count.

**Per-recording evidence.** Entries are collar-0 DER% / emitted speaker count. Quantile here
is its inner-argmin version; median and NME use their 1-SE versions. Every row is exploratory.


| Recording | Reference K | Frozen AHC | Quantile argmin | Median 1-SE | NME 1-SE | Strong-AHC oracle K |
|---|---:|---:|---:|---:|---:|---:|

| train_001 | 11 | 45.35 / 40 | 28.16 / 24 | 44.90 / 31 | 44.34 / 10 | 28.33 / 21 |

| train_002 | 20 | 38.07 / 30 | 39.65 / 26 | 36.07 / 28 | 45.86 / 9 | 39.65 / 26 |

| train_003 | 21 | 13.51 / 20 | 13.54 / 31 | 13.59 / 27 | 29.01 / 7 | 13.75 / 25 |

| train_004 | 10 | 11.39 / 35 | 11.92 / 42 | 11.82 / 41 | 29.60 / 17 | 11.84 / 21 |

| train_005 | 19 | 30.78 / 58 | 31.89 / 63 | 31.96 / 59 | 51.76 / 14 | 35.73 / 35 |

| train_006 | 17 | 42.31 / 68 | 42.53 / 29 | 40.63 / 54 | 44.75 / 16 | 42.74 / 29 |

| train_008 | 11 | 7.24 / 10 | 7.39 / 12 | 7.24 / 10 | 7.69 / 7 | 7.39 / 12 |

| train_009 | 21 | 36.58 / 35 | 38.86 / 23 | 36.22 / 31 | 37.99 / 4 | 39.77 / 26 |

| train_011 | 18 | 28.59 / 19 | 28.59 / 19 | 28.59 / 19 | 37.74 / 7 | 28.59 / 19 |

| train_013 | 21 | 56.90 / 42 | 53.66 / 28 | 57.23 / 36 | 71.14 / 9 | 53.68 / 32 |


There is no simple relationship between count accuracy and useful diarization. train_004
has 35 output speakers against 10 yet only 11.39% DER; train_011 has 19 against 18 yet
28.59%. Across the corpus, 357 baseline output identities comprise 278 containing strong
tracks, 35 from finite weak tracks only, and 44 from NaN tracks only. The cache actually
has 121 NaN tracks, but only 44 emit speech. The weak-only and NaN-only output accounts for
47.06 s and 88.54 s respectively: many identities for just 135.60 seconds. This explains
why easy reductions in count can have little effect on DER. It also prevents attributing
all fragmentation to NaNs: most output identities still contain strong tracks.

NME's adaptive count can collapse rare speakers: with its default-like rp=0.15, train_001
estimates two finite-embedding clusters, while rp=0.05 estimates twelve (reference eleven).
For the selected NME system, train_009 emits four speakers against twenty-one and train_013
nine against twenty-one. The cap is forty, so this is not the model's eight-channel limit.
Inference: the dominant graph communities need not correspond to every annotated speaker,
particularly with unequal speaker representation and noisy, variable-duration track vectors.
That interpretation is consistent with the counts and CONF increase; it is not a measured
causal decomposition of spectral-clustering failure.

**Verified embedding evidence, independent of deployable selection.** Reference diagnostics
label the actual crop intervals, retain strong finite tracks with ≥80% reference coverage
and ≥90% reference-speaker purity, and compare pairs whose crop-weighted temporal centers
are at least 300 seconds apart. This avoids overlapping context duplicates. It retains
2,120/3,061 strong tracks, so it favors easier speech and is not an estimate of all-track
quality. Of the strong tracks, 554 fail the purity threshold and 588 fail coverage; these
groups overlap. Pooled pair AUC is 0.9440; requiring at least four usable seconds per track
raises it to 0.9677. Same/different pair medians are 0.213/0.848 for the clean set and
0.169/0.827 for the longer subset. These are descriptive, reference-filtered comparisons,
not new embedding-extraction experiments or independent pair-level statistical samples.


| Recording | Clean / long-track N | Same-speaker median | Same-speaker p90 | Different-speaker median | AUC | Same median, ≥4 usable s | AUC, ≥4 usable s |
|---|---:|---:|---:|---:|---:|---:|---:|

| train_001 | 159 / 127 | 0.589 | 0.875 | 0.906 | 0.912 | 0.515 | 0.951 |

| train_002 | 105 / 88 | 0.444 | 0.821 | 0.887 | 0.920 | 0.386 | 0.947 |

| train_003 | 268 / 241 | 0.154 | 0.385 | 0.752 | 0.962 | 0.142 | 0.988 |

| train_004 | 293 / 254 | 0.182 | 0.571 | 0.839 | 0.976 | 0.162 | 0.997 |

| train_005 | 233 / 179 | 0.328 | 0.721 | 0.873 | 0.919 | 0.151 | 0.951 |

| train_006 | 256 / 212 | 0.557 | 0.818 | 0.900 | 0.925 | 0.497 | 0.945 |

| train_008 | 274 / 250 | 0.113 | 0.280 | 0.936 | 0.997 | 0.104 | 0.998 |

| train_009 | 128 / 111 | 0.439 | 0.694 | 0.866 | 0.969 | 0.400 | 0.994 |

| train_011 | 265 / 245 | 0.147 | 0.354 | 0.756 | 0.954 | 0.137 | 0.958 |

| train_013 | 139 / 121 | 0.422 | 0.751 | 0.890 | 0.951 | 0.386 | 0.976 |


The specific scale mismatch survives this cleaner filter: train_006 versus train_011 is
0.557 versus 0.147 (3.78×); restricting to ≥4 usable seconds gives 0.497 versus 0.137
(3.62×). Their different-speaker medians vary much less, 0.900 versus 0.756 (1.19×), so the
whole distance distribution is not simply stretched by one common factor. The earlier
council's 0.55/0.16 observation came from a different diagnostic pool;
the present figures independently verify it on exactly the requested zero-shot cache.
But a recording does not have one homogeneous same-speaker scale. Among long clean tracks
with at least five distant same-speaker pairs, individual speakers' medians range
0.110–0.508 within train_011 and 0.106–0.577 within train_005. Pair-weighted and speaker-balanced
medians differ: train_011 changes from 0.147 to 0.262, and train_003 from 0.154 to 0.335.
The clean distant-pair same-speaker fraction varies from 8.2% to 34.4% across recordings.
Thus unsupervised pair-distribution quantiles mix speaker prevalence, speaker identity,
and acoustic variation; they are not direct estimates of an invariant “within-speaker scale.”

Temporal evidence also survives the ≥4 s clean filter. Among 40 within-recording speakers
with at least five pairs in both the 60–300 s and ≥900 s separation bins, 34 have higher
far-pair medians; the median far-minus-near difference is +0.055 cosine distance.
This supports nonstationarity beyond crop length, without identifying its acoustic cause.

**Inference about causes and fixes.** The measured short-track penalty, impurity/coverage
failures, heterogeneous speaker-specific scales, and growing long-range distance together
argue against a single recording-wide multiplicative correction. Plausible acoustic causes
include changing scene/channel conditions, background music/noise, emotional voice changes,
and phonetic sampling in short excerpts; these are plausible mechanisms, not causally
isolated measurements from this run. Bengali/domain mismatch is also plausible, but the
good clean-pair discrimination rejects a blanket claim that WeSpeaker cannot distinguish
these speakers. Some impurity flags also reflect coarse, pause-inclusive references rather
than certain acoustic contamination.

Better crops are a justified experiment: long clean tracks show better separation, and
the current usable mask means one Nemotron channel is active while the others are below
threshold, not that the waveform contains a clean isolated voice. However, filtering
existing tracks by duration/purity is not proof that re-extracting different crops fixes
DER; it selects different examples and speakers. Background removal or tighter activity
masking could help, hurt, or alter the benchmark convention. Within-speaker scale variation
persists after the longer-track filter, so “just use longer crops” is not established.

**Track-level embeddings are already used.** bnd/embed.py normalizes each crop vector,
averages them with crop-duration weights, then normalizes the track vector. Crops use the
whole local core+context track, with true-length extraction and no zero-padding. Averaging
short crop vectors is not mathematically equivalent to embedding one long contiguous or
concatenated waveform; that remains a distinct experiment. Longer correctly linked tracks
could reduce variance, but pooling a track containing two people entrenches an identity
mixture. This task therefore supplies no evidence that the vague change “use track-level
embeddings” would fix anything—the existing system already does it.

A different open embedder is feasible but untested here. [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
is a reproducible open control with VoxCeleb1+2 training; sharing that source domain gives
no reason to assume automatic Bengali-drama robustness. Also, **VBx itself is not gated**:
the [original public VBx recipe](https://github.com/BUTSpeechFIT/VBx/blob/master/README.md)
and a [matched ResNet101 PLDA model](https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/models/ResNet101_16kHz/plda)
exist. Their transform/PLDA cannot be treated as calibrated for the cached WeSpeaker
coordinates. Testing that complete open extractor+PLDA stack would require new embeddings;
alternatively, fitting a backend for WeSpeaker would require proper inner training/validation
splits. Neither was run, and this cosine-HMM failure does not refute genuine VBx.

**Verified evidence and reproducibility.** All statements above have local artifacts.



The scoring contract is [/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/score.py:1](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/score.py:1), with collar construction
at [score.py:222](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/score.py:222). Original hard constraints and weak/orphan attachment
are at [link.py:294](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/link.py:294) and [link.py:405](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/link.py:405).
Crop construction is at [infer.py:186](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/infer.py:186), usable-channel masking at
[infer.py:234](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/infer.py:234), and actual track embedding aggregation at
[embed.py:78](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/embed.py:78). Existing 1-SE selection is at [p3b.py:977](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/p3b.py:977).
NeMo cosine normalization is at [offline_clustering.py:424](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/third_party/nemo-speech/nemo/collections/asr/parts/utils/offline_clustering.py:424),
NMESC at [line 873](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/third_party/nemo-speech/nemo/collections/asr/parts/utils/offline_clustering.py:873), and
SpeakerClustering integration at [line 1182](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/third_party/nemo-speech/nemo/collections/asr/parts/utils/offline_clustering.py:1182).

[protocol.md](/tmp/bnd_council3/clustering-and-count/protocol.md) records the pre-score grid and rule.
[experiment.py](/tmp/bnd_council3/clustering-and-count/experiment.py) implements generation, scoring, selection, and reference diagnostics.
[manifest.json](/tmp/bnd_council3/clustering-and-count/manifest.json) contains code/cache hashes and all 490 main RTTM hashes;
[run.log](/tmp/bnd_council3/clustering-and-count/run.log) records command output. [metrics.csv](/tmp/bnd_council3/clustering-and-count/metrics.csv) is a tidy
table of all summary and supplemental metrics; [results.csv](/tmp/bnd_council3/clustering-and-count/results.csv) preserves
the original console-style table with an initial disclosure line.
[summary.json](/tmp/bnd_council3/clustering-and-count/summary.json), [selections.json](/tmp/bnd_council3/clustering-and-count/selections.json),
and [all_grid_exploratory.json](/tmp/bnd_council3/clustering-and-count/all_grid_exploratory.json) retain every reported comparison
and every grid outcome. The `rttm/`, `labels/`, `scores/`, and `details/` directories contain
candidate predictions, assignments, raw scorer components, and adaptive counts/thresholds.
[embedding_diagnostics.csv](/tmp/bnd_council3/clustering-and-count/embedding_diagnostics.csv) and
[embedding_diagnostics.json](/tmp/bnd_council3/clustering-and-count/embedding_diagnostics.json) contain the crop/pair measurements;
[supplement.json](/tmp/bnd_council3/clustering-and-count/supplement.json) contains within-speaker time comparisons, HMM ablation,
count origins, and the explicitly reference-selected threshold diagnostic.
[oracle_output_count.json](/tmp/bnd_council3/clustering-and-count/oracle_output_count.json) and its separate freeze manifest
document the post-look exact-output-K check. [sources.md](/tmp/bnd_council3/clustering-and-count/sources.md) separates primary
external sources from the measured local results. A standalone [figure](/tmp/bnd_council3/clustering-and-count/h3_diagnostics.png)
and [PDF](/tmp/bnd_council3/clustering-and-count/h3_diagnostics.pdf) visualize scale variation and the count/DER tradeoff.

[validation.json](/tmp/bnd_council3/clustering-and-count/validation.json) confirms all ten baseline RTTMs are byte-identical
to frozen ZS*, source/cache hashes are unchanged, predictions match their pre-score hashes,
fold pools are disjoint, and every normalizer matches original AHC at the equivalent raw
threshold. The count-stop merge order matches original AHC; a brute-force HMM path enumeration
agrees with forward-backward to 1.11e−16. Baseline 26.9698808% and full identity-oracle
16.0794443% independently reproduce the published rounded values. These checks do not
validate every possible alternative implementation or turn this into a held-out study.

Run the main experiment with the repository's NeMo interpreter using
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHONDONTWRITEBYTECODE=1 nice -n 10 .../third_party/nemo-speech/.venv/bin/python experiment.py all`.
Then run `validate.py` and `oracle_output_count.py` with the same environment. The supplemental
plot/report uses `supplement.py` in the existing embed-venv because the NeMo venv lacks matplotlib;
this changes no scores or selected predictions. Set `MPLCONFIGDIR=/tmp/bnd_council3/clustering-and-count/mplconfig`.
Finally run `build_report.py`. Computational stages used one CPU worker with at most four compute
threads; GPU consumption was zero. [reproduce.sh](/tmp/bnd_council3/clustering-and-count/reproduce.sh) runs this sequence in a fresh child
directory, avoiding reuse of scored artifacts. [environment.json](/tmp/bnd_council3/clustering-and-count/environment.json) records
the main scoring environment's Python and package versions.



**Dependencies, risks, and open decisions.** This role has no dependency on another council
role's measurements and made no changes to their work. Medium-session tracks or a different
embedder would change the clustering input and require rerunning the comparisons; these
results are specific to s20, activity 0.40, gap-fill 5. Gated model access is not required
for this completed experiment. The normalizers are deliberately simple, the spectral grid
is finite, and the temporal model is not PLDA-VBx; negative results do not rule out those
larger alternatives. Reference-based crop filters and oracle counts are diagnostics only.
The ten repeatedly inspected recordings cannot support a fresh generalization claim.

The practical recommendation is to keep the frozen ZS* default and report the localized
quantile-normalization signal transparently. Open decisions for the owner/orchestrator are
whether speaker count should be optimized as a separate application requirement, and whether
to preregister a fresh experiment on cleaner/longer tracks or a matched open extractor+PLDA
backend with genuinely new held-out labels. This run provides no basis to spend the remaining
training budget on the assertion that eigengap counting alone will close the linking gap.

