# IBF_UB Technical Review: Evidence, Baselines, Use Cases Evaluation and Future Ideas.

This is a technical review and ablation study for the methods discussed in [Information as Structural Alignment: A Dynamical Theory of Continual Learning](https://arxiv.org/abs/2604.07108).

The technical discussion focuses on Domain III, the CIFAR-100 experiment, because it is the implementation closest to a standard continual-learning benchmark. Unlike RRW, which is a synthetic mechanism-confirmation domain, and Chess, which depends on a Stockfish oracle and a supplied chess encoder, CIFAR-100 uses a familiar high-dimensional classification task, fixed train/test data, Task-IL and Class-IL metrics, and baselines that can be re-run on the same frozen features. For that reason, it is the strongest place to evaluate whether IBF is practically competitive with simpler continual-learning methods such as replay.

The other implemented domains are summarized in [Other IBF use cases in the implementation and paper](#52-other-ibf-use-cases-in-the-implementation-and-paper).

## 1. Problem Formulation

Domain III is a **continual image-classification** experiment on Split-CIFAR-100.

- `N_TASKS = 20`
- `CLASSES_PER_TASK = 5`
- each task therefore contains about `2500` training images

The task is still standard classification: **input image in, class label out**. What changes over time is the set of classes presented to the learner. The model is trained on Task 0, then Task 1, and so on through Task 19.

The notebook does not train end to end from raw pixels. Instead it uses a fixed representation pipeline:

1. extract a frozen `64D` feature vector with `ViT-B/16 + PCA`
2. train a small `100`-class scoring head on those frozen features and then freeze it
3. learn a separate memory layer that adds local corrections

The notebook therefore differs from a standard classifier in where adaptation happens:

- standard classifier: `image -> network -> class scores`
- notebook: `image -> frozen feature -> frozen base score -> memory correction -> class score`

The reported score for class `c` is:

$$
\mathrm{score}_c(x)=\log R_{\mathrm{field}}(x,c)+\Delta R(x,c)
$$

and the prediction is the class with the largest score:

$$
\hat{c}(x)=\arg\max_{c \in \mathcal{C}} \mathrm{score}_c(x).
$$

Here:

- $R_{\mathrm{field}}(x,c)$ is the frozen base-classifier probability for class $c$.
- $\Delta R(x,c)$ is the memory-based local correction.


The main evaluation metrics are:

- **Task-IL**: classify using only the 5 classes of the current task
- **Class-IL**: classify over all classes seen so far
- **BT**: backward transfer, used here mainly as a forgetting measure

One important framing point is that the scoring head is pretrained on a balanced sample from **all 100 classes** and then frozen. In the saved main run this uses `10` examples per class for `10` epochs. This is therefore not a pure class-incremental-from-scratch setting. It is better understood as **continual adaptation on top of a supplied universal prior**.

## 2. Mechanism and ML Correspondence

### 2.1 Base Scoring and Local Correction

For each candidate class `c`, the notebook builds a `68D` representation

$$
z_{\mathrm{pair}}(x,c)=\left[z_{\mathrm{image}}(x)\,;\,z_{\mathrm{class}}(c)\right]
$$

where:

- $z_{\mathrm{image}}(x)\in\mathbb{R}^{64}$ is obtained by passing image $x$ through a frozen ImageNet-pretrained ViT-B/16 encoder, taking the resulting `768D` embedding, and projecting it with the fixed unsupervised PCA map to `64D`.
- $z_{\mathrm{class}}(c)\in\mathbb{R}^{4}$ is a fixed hand-crafted class code:

$$
z_{\mathrm{class}}(c)=s\begin{bmatrix}
\frac{c}{N_{\mathrm{classes}}}\\
\frac{c\bmod 10}{10}\\
\frac{\lfloor c/10 \rfloor}{10}\\
c\bmod 2
\end{bmatrix},
\qquad s=\mathrm{MOVE\_SCALE}\;(=25.0).
$$

and computes a local correction by comparing that vector to stored value-memory centers:

$$
\Delta R(x,c)=\sum_i \mathrm{gate}_i\, v_i\,
\exp\!\left(-\frac{\left\|z_{\mathrm{pair}}(x,c)-z_i\right\|^2}{2\sigma_i^2}\right)
$$

- $\mathrm{gate}_i\in\{0,1\}$ is the read-gate for memory $i$: it is `1` when that memory is allowed to contribute, which in the notebook means either "same current context" or "crystallized and crucible-verified"; otherwise it is `0`.
- $v_i$ is the signed value stored in memory $i$, i.e. the amplitude of that memory's additive score correction. Positive $v_i$ pushes class $c$ up locally; negative $v_i$ pushes it down.


**In standard ML interpretation of the mechanism. It behaves like a kernel or RBF-style residual layer on top of a frozen classifier.**

### 2.2 Value and Agency Memories

The notebook maintains two memory populations.

**Value memories** live in the `68D` image-class space and locally raise or lower class scores. In standard terms they are closest to:

- RBF centers
- prototypes
- support points in a kernel expansion
- dictionary atoms

**Agency memories** live in the `64D` image space and modulate decision sharpness rather than class identity directly. Their closest ML analogues are:

- adaptive inverse temperature
- confidence control
- local calibration

In the saved CIFAR results, agency is nearly neutral; the main action is in the value-memory channel.

### 2.3 Relation to Existing ML Families

The closest literature families are:

- **RBF networks**
- **kernel methods**
- **prototype-based classifiers**, especially distance-to-class-prototype methods (e.g., [Learning Prototype Classifiers for Long-Tailed Recognition](https://arxiv.org/abs/2302.00491))
- **memory-based continual learning**

The method is less well described as:

- **k-means**, because centers are created from supervised error rather than unsupervised clustering
- **EM for mixture models**, because there is no global likelihood objective or E/M alternation
- **plain fine-tuning**, because adaptation is offloaded to a memory system rather than only to shared weights

From a classification perspective, a compact mapping to existing terminology is:

**A frozen-feature classifier with a growing local-kernel memory layer for continual learning**

## 3. Training Dynamics and Practical Implications

### 3.1 Update Rule

The continual-learning phase can be summarized as follows:

```text
for each task:
    for each epoch:
        for each training image x:
            z_img = frozen_feature(x)
            true_class = label(x)

            for each candidate class c in current task:
                z_pair = [z_img ; class_feature(c)]
                R_field = frozen_classifier_prob(x, c)
                delta_R = local_value_memory_correction(z_pair)
                predicted_error = clip(1 - (R_field + delta_R), 0, 1)

                target_error = 0 if c == true_class else 1
                D = target_error - predicted_error
                D = D - running_mean(D)

                update nearby value memories in 68D
                update nearby agency memories in 64D

        decay, merge, crystallize, verify, or dissolve memories
```

The sign convention is slightly unintuitive because the internal target is written as an error-like quantity rather than a direct class score. Practically, the effect is simple:

- if the true class is underscored, future `delta_R` ($\Delta_R$) is pushed up
- if a wrong class is overscored, future `delta_R` ($\Delta_R$) is pushed down

So `delta_R` ($\Delta_R$) is best read as **a learned local residual correction to the frozen scorer**.

### 3.2 What IBF Still Stores

IBF does not avoid storing old information. It avoids storing **raw past examples directly**.

The distinction is:

- classic replay mechanisms store old frozen feature vectors and labels explicitly
- IBF stores learned memory objects instead

In Domain III those memory objects include:

- value memories in `68D` image-class space
- agency memories in `64D` image space
- metadata such as widths, strengths, contexts, update counts, and state flags

From this point of view IBF is therefore better described as **a compressed memory system**, not a no-memory system.

### 3.3 Runtime Cost

The main computational cost comes from repeated local similarity computations:

- every training image is paired with each candidate class of the current task
- each image-class pair is compared against a growing memory bank
- each epoch also runs decay, merge, crystallization, and verification steps

The saved main run `full_42` takes about `15.0` hours on an `H100 80GB card...`. In rough terms, the notebook performs on the order of

- `20 tasks x 2500 images/task x 50 epochs x 5 class candidates`

or about `12.5` million update events, each of which scans a growing bank of centers. This is the main practical disadvantage of the approach.

**In comparison, the methods in the aligned comparison scripts (see next section) take on the order of a few minutes, mostly because they do not have to do repeated similarity computations against a growing memory bank.**

## 4. Benchmarks Against Standard Methods

### 4.1 Compared Methods

Two scripts were used for aligned comparison on the **same frozen `ViT-B/16 + PCA` features**.

`compare_cifar100_continual.py` evaluates ordinary classifier heads:

- `linear`: a single linear classifier on top of the frozen features.
- `mlp`: a small neural-network classifier head.
- `deepmlp`: a deeper neural-network classifier head.

each under:

- `finetune`: ordinary sequential neural-network fine-tuning. The head is trained on Task 0, then Task 1, and so on, using only the current task's data at each step. There is no explicit old-task memory, replay buffer, or parameter-protection term, so the method is expected to forget when new tasks overwrite the head.
- `replay`: the same head architecture, but with a bounded memory buffer of old examples. The implementation uses reservoir sampling, so the buffer remains an approximately uniform sample of the stream seen so far. During later tasks, each current-task minibatch is trained together with a sampled replay minibatch from the buffer, using the same cross-entropy objective. In other words, replay preserves old decision boundaries by periodically showing the classifier old feature vectors and labels again.

`compare_cifar100_frozen_features.py` evaluates the notebook-style baseline family:

- `mlp`: the notebook-style neural baseline trained sequentially on the frozen features without replay. This is essentially the current-task-only fine-tuning baseline in the notebook's feature space.
- `replay`: the same MLP baseline augmented with a reservoir replay buffer. After each task, examples from that task are added to the buffer; during later training, sampled old examples are mixed into the loss so the model continues to rehearse earlier classes.
- `ewc`: [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796). EWC does not store and replay old examples during ordinary updates. Instead, after a task it estimates which model parameters were important for that task using a Fisher-information approximation, stores the old parameter values, and adds a penalty when later training moves important parameters too far away. It is a parameter-protection baseline rather than a memory-replay baseline.

The replay implementation was aligned across the scripts to use proper **reservoir sampling** with matched feature-mode defaults:

- `batch_size = 64`
- `replay_batch_size = 64`
- `weight_decay = 0.0`

### 4.2 Aligned Results

The result tables use the evaluation metrics proposed in the original IBF CIFAR-100 evaluation:

- **Task-IL**: task-incremental accuracy. At test time, the evaluator is told which 5-class task the image belongs to, and the model only chooses among those task classes.
- **Class-IL**: class-incremental accuracy. At test time, the model must choose among all classes seen so far, without being given the task identity. This is usually the stricter continual-learning metric.
- **BT**: backward transfer. Here it measures how much performance on earlier tasks changes after later tasks are learned. Values closer to zero mean less forgetting; more negative values mean stronger forgetting.

Bold values mark the best value in each table column. For `BT`, higher is better, so values closer to zero are preferred. For `time`, lower is faster.

#### A. Running the original IBF repo, CIFAR-100 notebook, and reading `CIFAR-paper-results.json`:

| method | Task-IL | Class-IL | BT | time |
| --- | ---: | ---: | ---: | ---: |
| `IBF full_42 (linear readout)` | `0.8394` | **`0.5137`** | `-0.0853` | **`13.5h`** |
| `IBF full_42 (log readout)` | **`0.9026`** | `-` | **`-0.0039`** | **`13.5h`** |

The `log` readout uses `log(R_field) + delta_R`, so the cleaner direct comparison is the **linear** notebook readout.

#### B. Running our ablation script `compare_cifar100_continual.py`:

| method | Task-IL | Class-IL | BT | time |
| --- | ---: | ---: | ---: | ---: |
| `linear/finetune` | `0.4439` | `0.1479` | `-0.5420` | **`0.4m`** |
| `linear/replay` | **`0.9451`** | **`0.6582`** | **`-0.0151`** | `2.5m` |
| `mlp/finetune` | `0.4074` | `0.1289` | `-0.5811` | `2.3m` |
| `mlp/replay` | `0.9278` | `0.6328` | `-0.0322` | `3.9m` |
| `deepmlp/finetune` | `0.3254` | `0.0499` | `-0.6673` | `1.8m` |
| `deepmlp/replay` | `0.8844` | `0.6026` | `-0.0711` | `2.7m` |

#### C. Running our ablation script `compare_cifar100_frozen_features.py`:

| method | Task-IL | Class-IL | BT | time |
| --- | ---: | ---: | ---: | ---: |
| `mlp` | `0.3473` | `0.0642` | `-0.6421` | **`1.5m`** |
| `replay` | **`0.8702`** | **`0.5820`** | **`-0.0797`** | `4.3m` |
| `ewc` | `0.2950` | `0.0552` | `-0.6996` | `3.7m` |

### 4.3 Interpretation

The aligned comparison changes the practical conclusion because the strongest results do not come from the original IBF CIFAR-100 run.

Within the original notebook results, the `log` IBF readout gives the best reported `Task-IL` (`0.9026`) and `BT` (`-0.0039`), but it does not report `Class-IL`. The `linear` IBF readout is the cleaner full comparison row because it reports all three metrics: `Task-IL = 0.8394`, `Class-IL = 0.5137`, and `BT = -0.0853`.

In the first ablation script, `compare_cifar100_continual.py`, the clear performance winner is `linear/replay`:

- best `Task-IL`: `0.9451`
- best `Class-IL`: `0.6582`
- best `BT`: `-0.0151`

**This means that, on the same frozen `ViT-B/16 + PCA` feature representation, a simple linear classifier with a replay buffer beats the original IBF linear readout on every directly comparable metric. It also beats the IBF log readout on `Task-IL`, although IBF log still has the best reported `BT` among the rows shown and lacks a reported `Class-IL` value.**

In the second ablation script, `compare_cifar100_frozen_features.py`, the best notebook-style baseline is also replay:

- `replay`: `0.8702 / 0.5820 / -0.0797`
- original IBF linear: `0.8394 / 0.5137 / -0.0853`

**So even the more conservative notebook-style replay baseline exceeds the original IBF linear result on `Task-IL`, `Class-IL`, and `BT`, while running in minutes rather than hours.**

The failure cases are also informative. Current-task-only fine-tuning performs poorly across all head types, and EWC performs poorly in the notebook-style baseline script. The useful pattern is not "larger neural head wins"; the replay ordering is:

- `linear/replay > mlp/replay > deepmlp/replay`

This suggests that the frozen representation is already highly separable. Extra head capacity mainly adds instability, while replay directly preserves old decision boundaries by retaining a small memory of old feature vectors and labels.

The practical baseline to beat is therefore not plain fine-tuning or EWC. It is simple replay, especially `linear/replay`. Under this aligned comparison, IBF remains an interesting local-memory mechanism, but it is not the strongest CIFAR-100 method among the tested alternatives.

## 5. Glossary and Other IBF Use Cases

### 5.1 Glossary and mapping

| Notebook term | Meaning here | Closest ML wording |
| --- | --- | --- |
| task | one block of 5 CIFAR classes | task in continual learning |
| Task-IL | classify using only current task classes | task-incremental evaluation |
| Class-IL | classify over all seen classes | class-incremental evaluation |
| `R_field` | frozen base-classifier probability | base classifier score |
| `delta_R` | memory-based local correction | residual kernel / prototype correction |
| value memory | local score-correction point in 68D | RBF center / prototype / support point |
| agency memory | local decisiveness memory in 64D | adaptive inverse temperature / calibration term |
| `k_eff` | decision sharpness parameter | inverse temperature |
| `sigma` | local width of a memory | kernel bandwidth / local spread |
| crystallization | make memory persist more | consolidation |
| dissolution | weaken memory after contradiction | pruning / de-protection |
| gate | decide whether an old memory can contribute | context-based masking |
| `D` | discrepancy signal used for updates | supervised residual / error signal |

### 5.2 Other IBF use cases in the implementation and paper

The paper and repository implement two additional domains before CIFAR-100. They matter for the theory, but they play different methodological roles from Domain III.

- **Domain I: Rotating Rules World (RRW)**: RRW is a synthetic controlled environment, not a practical deployment use case. Inputs are `4D` vectors, actions are represented by `4D` action embeddings, and IBF operates over the combined `8D` state-action space. The correct action is generated by an analytic scoring rule across three phases (`A -> B -> C`). Phase B deliberately reverses part of Phase A, so the setup directly tests whether IBF can preserve useful memories while silencing or dissolving memories that become contradictory. This is best read as a mechanism-confirmation test for crystallization, gating, Crucible verification, and backward transfer.
- **Domain II: Chess**: Chess tests whether the same mechanism can support strategic action selection under a more complex external evaluation. Positions are sampled from Lichess games and grouped into three contexts: materially imbalanced, quiet/balanced, and restricted-mobility positions. A frozen CNN maps each board to an `8D` representation; each candidate move is represented as the resulting board embedding plus `4D` move features, giving a `12D` move space. Training uses Stockfish depth 4 to produce the discrepancy signal, while evaluation uses Stockfish depth 8 centipawn scores. This is more realistic than RRW, but still oracle-driven and dependent on the supplied chess encoder.
- **Why CIFAR-100 is the most relevant practical test**: CIFAR-100 is the closest of the three to a standard continual-learning benchmark. It uses a familiar image-classification setup, fixed train/test data, Task-IL and Class-IL metrics, and baselines that can be re-run on the same frozen features. That makes the replay comparison in this note more probative for practical methodology than RRW's synthetic mechanism test or Chess's Stockfish-mediated strategic evaluation.

## 6. Future Use Cases

The compressed-memory framing points to a narrower and more defensible set of possible future use cases than generic benchmark superiority. These should be read as hypotheses for future systems, not as claims established by the CIFAR notebook.

- **Streaming or continual deployment**: this is the most plausible deployment setting. If a strong frozen representation already exists, IBF can absorb recurring rare cases online without maintaining an explicit replay dataset during each update.
- **Privacy- or governance-constrained settings**: storing learned centers may be operationally easier than storing raw past examples, although the current notebook is not automatically privacy-preserving because the stored centers are still data-derived and may leak recoverable information.
- **Storage-constrained deployment**: a compressed memory bank could be useful if it remains substantially smaller than a replay buffer, but this notebook does not establish that advantage.
- **Structured rare-case adaptation on top of a supplied universal prior**: this is the clearest mechanism-level fit. The mechanism is most plausible when a strong base model already handles the common cases, but repeatedly fails on low-frequency patterns that recur with recognizable structure. Plausible examples include:
  - **Rare manufacturing defects**: a visual-inspection model may classify common defects well but miss a defect morphology that appears only occasionally. IBF-style local memories could store corrections around the feature-space neighborhood of that morphology, so future examples with the same visual structure receive a local score adjustment.
  - **Medical subtypes**: a diagnostic model may have a useful general representation but underperform on a rare subtype that has consistent imaging or tabular signatures. A memory layer could learn local residual corrections for that subtype without retraining the whole model, assuming the subtype recurs often enough to form stable memories.
  - **Recurring sensor or scanner artifacts**: this is plausible only when the artifact is systematic rather than random. For example, a specific scanner might produce a faint stripe, dead-pixel cluster, calibration drift, or repeating noise pattern that pushes the frozen model toward the wrong class. If that pattern is represented in the frozen features and repeatedly co-occurs with the same kind of prediction error, IBF could create local memories near those feature-space regions. Later, when a similar artifact appears again, the read path would add a local `delta_R` correction for the affected class scores. This would not help much for one-off noise, changing sensor failures, or artifacts that the encoder does not preserve.

## 7. Ablation Conclusions Compared With Original IBF

- The original IBF CIFAR-100 notebook remains an interesting theory-driven local-memory implementation; its clean full-metric row is `0.8394 / 0.5137 / -0.0853` and about `13.5h`.
- Our first ablation shows `linear/replay` is the strongest tested method overall: `0.9451 / 0.6582 / -0.0151`, while running in minutes (`<5m`) rather than hours.
- Our second ablation also beats original IBF linear on `Task-IL`, `Class-IL`, and `BT` (`0.8702 / 0.5820 / -0.0797`).
- Plain fine-tuning and EWC are weak baselines here; the real practical baseline to beat is replay on the same frozen features.
- The CIFAR evidence therefore supports IBF as an alternative compressed/local-memory mechanism, not as a clearly superior continual-learning method over simple replay.
