# How Shooti works

Three diagrams: what happens to one photo at runtime, how advice is produced
without rules, and how the grader was trained.

---

## 1. Two judges, one measurement layer

This is the whole story of the project. v1 and v2 measure the photo the *same
way* — the difference is entirely in who decides what the measurements mean. v1
applies thresholds a human wrote; v2 applies weights learned from 20,437
human-rated photos. Only one of them tracks human judgment.

```mermaid
flowchart TB
    P["Photo: camera or upload"]
    D["EXIF-aware decode to BGR<br/>shooti/loader.py"]
    P --> D

    subgraph MEAS["Measurement layer, shared by v1 and v2"]
        direction LR
        Y["YuNet DNN face detector<br/>box + 5 landmarks<br/>227 KB ONNX"]
        S["Gradient-energy saliency<br/>fallback when no face<br/>reports none if frame is flat"]
        H["Hough transform<br/>horizon angle + row<br/>gives roll AND camera pitch"]
    end

    D --> Y
    D --> S
    D --> H

    RAW["Raw measurements<br/>subject box, eye level, head yaw,<br/>horizon, balance, areas"]
    Y --> RAW
    S --> RAW
    H --> RAW

    subgraph V1PATH["v1: hand-written judge"]
        direction TB
        T["Hand-set tolerances<br/>THIRDS_TOL 0.055, TILT_TOL 1.5 deg,<br/>HEADROOM 3 to 14 pct"]
        PEN["Penalty per violated rule"]
        SC1["score = 100 minus sum of penalties"]
        T --> PEN
        PEN --> SC1
    end

    subgraph V2PATH["v2: learned judge"]
        direction TB
        FEAT["18-d feature vector<br/>same numbers, no thresholds<br/>grader/features.py"]
        HEAD["Learned head<br/>530 to 256 to 64 to 10<br/>153,034 params, 0.61 MB"]
        DIST["Softmax over 10 rating bins<br/>a distribution, not a number"]
        FEAT --> HEAD
        HEAD --> DIST
    end

    RAW --> T
    RAW --> FEAT

    CLIP["Frozen CLIP ViT-B/32<br/>512-d image embedding<br/>151M params, never trained"]
    D --> CLIP
    CLIP --> HEAD

    OUT2["Expected value = predicted rating<br/>Std dev = vote spread<br/>is this photo divisive?"]
    DIST --> OUT2

    R1["SRCC -0.006 vs humans<br/>indistinguishable from noise"]
    R2["SRCC 0.682 vs humans<br/>80.1 pct good/bad accuracy"]
    SC1 --> R1
    OUT2 --> R2

    classDef bad fill:#5b1a1a,stroke:#c0392b,color:#ffffff
    classDef good fill:#14432a,stroke:#27ae60,color:#ffffff
    classDef frozen fill:#1e3a5f,stroke:#3498db,color:#ffffff
    class R1,SC1 bad
    class R2,OUT2 good
    class CLIP frozen
```

**Why a distribution instead of a score.** On AVA, "6.0 because everyone said 6"
and "6.0 because half said 3 and half said 9" are very different photographs. The
second is divisive, and a coach should say so rather than average it away.
Training uses earth-mover distance rather than cross-entropy, because bin 3 is
closer to bin 4 than to bin 9 and the loss should know that.

---

## 2. Advice without rules: counterfactual search

v1 produced advice by asserting a rule was broken. v2 asks a different question —
*if I actually reframed the shot this way, would humans rate it higher?* — and
answers it by trying.

This is what fixes the original complaint. A centered or symmetric photo that
cannot be improved by shifting simply produces no shift suggestion. No rule needs
special-casing for it.

```mermaid
flowchart TB
    A["Photo as shot"]
    GEN["Build 14 candidate reframings<br/>8 pans, 2 tighter crops, 4 rolls"]
    A --> GEN

    CROP["Crop each candidate<br/>window clamped inside the frame<br/>rolls crop to 74 pct scale"]
    GEN --> CROP

    VERIFY{"Any synthetic pixels?"}
    CROP --> VERIFY
    REJECT["Would invalidate the score.<br/>Never grade reflected or black borders."]
    VERIFY -->|"if yes"| REJECT

    BATCH["Grade all 15 frames in one batch<br/>as-shot plus 14 candidates<br/>full pipeline from diagram 1"]
    VERIFY -->|"verified: no"| BATCH
    A --> BATCH

    DIFF["gain = candidate score minus as-shot score"]
    BATCH --> DIFF

    GATE{"gain >= 0.04?"}
    DIFF --> GATE

    SHOW["Show the crop, the predicted gain,<br/>and the camera move it implies.<br/>e.g. tilt down 7 pct of frame, +0.07"]
    QUIET["Say nothing.<br/>No reframing scored higher.<br/>v1 would still complain here."]
    GATE -->|"yes"| SHOW
    GATE -->|"no, for all 14"| QUIET

    classDef good fill:#14432a,stroke:#27ae60,color:#ffffff
    classDef warn fill:#4a3a12,stroke:#f39c12,color:#ffffff
    class QUIET good
    class REJECT warn
```

**Two things verified rather than assumed:**

- **The model responds to reframing.** Mean spread of predicted scores across the
  15 candidates was 0.51, never flat. Had it been near zero, the model would be
  blind to framing and every suggestion meaningless.
- **No synthetic pixels are ever graded.** Rolled candidates crop to 74% scale,
  checked to stay inside the frame.

**The limitation this design cannot escape:** only crops *inside* the existing
frame can be tested, because pixels outside it were never captured. So "step
back" or "zoom out" is never suggested — not because it wouldn't help, but
because it cannot be measured from one photo.

---

## 3. Training, and the split discipline

```mermaid
flowchart TB
    HF["AVA subset on Hugging Face<br/>trojblue/AVA-aesthetics-10pct-min50-10bins<br/>every photo has 50+ human votes"]
    SHARD["Parquet shards, read directly<br/>streaming stalled on this repo<br/>--cleanup keeps peak disk at 0.5 GB"]
    HF --> SHARD

    PREP["prepare.py, 10 worker processes<br/>CLIP embed + geo features + v1 rule score<br/>measured 70 img/s"]
    SHARD --> PREP

    NPZ["cache/ava_*.npz<br/>50 MB total for 25,547 images<br/>images themselves never kept"]
    PREP --> NPZ

    DUP{"Dataset ships train, validation, test"}
    NPZ --> DUP
    CHECK["Checked: test is a byte-identical copy<br/>of validation. All 5,110 ids match."]
    UNUSED["So test is unused.<br/>It carries no extra information."]
    DUP --> CHECK
    CHECK --> UNUSED

    SPLIT["Carve the selection set out of TRAIN<br/>so validation stays untouched"]
    DUP --> SPLIT
    FIT["fit: 17,371 images"]
    SEL["epoch selection: 3,066 images"]
    SPLIT --> FIT
    SPLIT --> SEL

    TRAIN["Train head. AdamW, EMD loss,<br/>cosine LR. Seconds per run."]
    FIT --> TRAIN
    SEL --> TRAIN

    CKPT["checkpoints/grader_both.pt<br/>0.61 MB, committed to the repo"]
    TRAIN --> CKPT

    HELD["Report on validation: 5,110 images<br/>used for NEITHER fitting NOR selection"]
    CKPT --> HELD
    NUM["SRCC 0.682, PLCC 0.689, acc 80.1 pct<br/>vs v1 rules: SRCC -0.006"]
    HELD --> NUM

    classDef warn fill:#4a3a12,stroke:#f39c12,color:#ffffff
    classDef good fill:#14432a,stroke:#27ae60,color:#ffffff
    class CHECK,UNUSED warn
    class HELD,NUM good
```

**Why the selection set is carved out of train.** Picking the best epoch on a set
and then reporting that same set's score is optimistic. The obvious fix — report
on the dataset's `test` split — does not work here, because that split is a
duplicate of `validation`. So the selection set comes out of `train`, and
`validation` is untouched until the final number. Train and validation are
genuinely disjoint (overlap: 0, verified).

This mattered. The first numbers reported were selection-set numbers presented as
held-out. Correcting it moved SRCC from 0.684 to 0.682 — a small change, but the
claim was wrong before.

---

## The ablation, and a negative result

| Channels into the head | SRCC | Accuracy |
|---|---|---|
| geometry only (18 features) | 0.166 | 71.4% |
| CLIP only (512-d) | **0.682** | 80.1% |
| CLIP + geometry (530-d) | 0.672 | **80.6%** |

The 18 hand-engineered geometric features **add nothing on top of CLIP** — the
combined model grades slightly worse than CLIP alone. That is a negative result
about the feature engineering in this project, and it is reported rather than
buried.

`both` is still the default for *advice*, for a measured reason: it is more
responsive to reframing (mean score spread 0.51 vs 0.44), because geometry gives
the model an explicit handle on the thing the advice is about. Better advice,
marginally worse grading — a real tradeoff, not a free win.

---

## Where each piece lives

| Stage | File |
|---|---|
| Decode | `shooti/loader.py` |
| Face + saliency detection | `shooti/subject.py` |
| Horizon / pitch | `shooti/rules.py` (`detect_horizon`) |
| v1 hand-written judge | `shooti/rules.py` |
| Measurements as features | `shooti/grader/features.py` |
| Frozen CLIP | `shooti/grader/embed.py` |
| Learned head + EMD loss | `shooti/grader/model.py` |
| Training cache | `shooti/grader/prepare.py` |
| Training + ablation | `shooti/grader/train.py` |
| Inference | `shooti/grader/grade.py` |
| Counterfactual advice | `shooti/grader/advise.py` |
| v1 UI / v2 UI | `app.py` / `app2.py` |
| Browser verification | `scripts/drive_app.py` |
