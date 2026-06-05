# Akinator

The warm-up project for *Two Faces of Entropy*. A guessing-game solver that picks one entity from a fixed table using yes/no questions, choosing each question to maximise information gain.

## The game

You think of an animal from `entities_v*.csv`. The solver asks yes/no questions; you answer. It narrows the candidate pool until one animal remains, then guesses.

The only interesting decision is *which question to ask*. The solver picks the one whose answer distribution has the highest entropy — i.e., the question most likely to split the remaining candidates evenly, eliminating the most regardless of your answer.

## The dataset

Hand-authored animal tables, each row = one animal, each column = one binary feature. Versioned so we can re-run experiments against the baseline:

- `entities_v1.csv` — 46 animals × 18 features (baseline)
- `entities_v2.csv` — 46 animals × 20 features (two features added to resolve ambiguities)

Both are generated from a single source of truth: `make_dataset.ipynb`. Toggle the `VERSION` flag in the notebook to regenerate either file.

The dataset is deliberately small so the algorithm's behaviour can be verified by hand. It's also deliberately diverse — taxonomic features (`is_bird`, `is_reptile`) are skewed and useless as opening questions, but they shine later when narrowing within a category. The entropy ranking handles this automatically.

## Running

```bash
# Batch evaluation (set the CSV path in play.py's __main__)
python play.py
```

The evaluation prints summary statistics and saves a plot to `eval_results_<version>.png`.

## Experiment 1 — baseline: uniform priors, noiseless answers

The simplest possible version. Every animal is treated as equally likely (uniform priors), and the user always answers truthfully. The solver picks the question with maximum yes/no entropy, filters candidates by the answer, and repeats.

**Dataset:** `entities_v1.csv` (18 features)

| | Steps |
|---|---|
| Theoretical minimum (log₂ 46) | 5.52 |
| Solver average | 6.65 |
| Solver median | 6 |
| Best case | 5 |
| Worst case | 18 |
| Correct guesses | 44 / 46 |

**Observations**

- Average is 21% above the theoretical optimum. That gap is the cost of having a finite question pool — the solver can't always find a perfectly balanced split, so some questions narrow less than 1 bit's worth.
- Easiest animals (5 steps): Lion, Tiger, House Cat, Dog, Octopus. These have unique feature combinations that get isolated quickly by the early high-entropy questions (`is_predator`, `is_mammal`, `has_fur`).
- Hardest animals (18 steps): **Horse, Cow, Pig, Rabbit**. The solver burns through every available question and still can't distinguish them — they share identical or near-identical feature vectors. When forced to guess, it picks wrong half the time:
  - Cow → guessed as Horse (wrong)
  - Rabbit → guessed as Pig (wrong)

Spot check confirms it: `Horse` and `Cow` rows are byte-identical across all 18 features; `Pig` and `Rabbit` likewise. No clever question-ordering can resolve animals that the feature set itself can't distinguish.

**What this tells us**

The algorithm works. The 21% inefficiency is structural — bounded by what the dataset can express. The failure cases aren't an algorithm problem; they're a **dataset problem**. The fix is to add features that *split currently-indistinguishable entities*, not to change the solver.

## Experiment 2 — closing the dataset gaps

Added two features to `entities_v2.csv` specifically to break the Experiment 1 ties:

- `produces_milk` → Cow=1, Horse=0 (resolves pair 1)
- `has_long_ears` → Rabbit=1, Pig=0 (resolves pair 2)

Everything else stays the same: uniform priors, noiseless answers, same algorithm. Only the dataset changed.

**Dataset:** `entities_v2.csv` (20 features)

| | Exp 1 (18 features) | Exp 2 (20 features) |
|---|---|---|
| Theoretical minimum (log₂ 46) | 5.52 | 5.52 |
| Solver average | 6.65 | **5.61** |
| Solver median | 6 | 6 |
| Best case | 5 | 5 |
| Worst case | 18 | **6** |
| Correct guesses | 44 / 46 | **46 / 46** |

**What changed**

- **Worst case dropped from 18 → 6.** The pathological cases where the solver burned through every question and still guessed wrong are gone. With ambiguity resolved, every animal is identifiable in at most 6 questions.
- **Average dropped from 6.65 → 5.61** — within 2% of the theoretical optimum of 5.52. The solver is now essentially perfect for this dataset; you can't do meaningfully better without changing the dataset further.
- **100% accuracy**, with no compromises.

**The bigger lesson**

Two carefully chosen features (out of an unlimited possible set) collapsed both the average step count *and* the worst-case behaviour. This is the same lesson that shows up in every information-theoretic system: *the right discriminator at the right point in the tree matters far more than total feature count.* Adding 10 generic features wouldn't have helped; adding the *2 specific features that target the indistinguishability* fixed everything.

## Experiment 3 — non-uniform priors

Until now the solver assumed every animal was equally likely to be picked. In reality some animals (Dog, Cat, Lion) come to mind far more often than others (Platypus, Chameleon, Jellyfish). Exp 3 hand-crafts a prior reflecting this and feeds it to the solver instead of `1/N`.

Concretely, the 46 animals are split into 5 tiers:

| Tier | Prior | Count | Examples |
|---|---|---|---|
| Common | 0.07 | 6 | Lion, Tiger, House Cat, Dog, Elephant, Horse |
| Frequent | 0.030 | 10 | Wolf, Brown Bear, Dolphin, Shark, Eagle, Cow, ... |
| Medium | 0.015 | 14 | Polar Bear, Panda, Whale, Parrot, Butterfly, ... |
| Rare | 0.005 | 12 | Leopard, Cheetah, Octopus, Cockroach, Bat, ... |
| Very rare | 0.0025 | 4 | Jellyfish, Ostrich, Chameleon, Platypus |

(Saved in `data/priors_exp3.json`.) Dataset and algorithm are unchanged from Exp 2; only the prior distribution changes. After each answer, the surviving candidates' priors are kept and renormalised to sum to 1 (Bayes update in its simplest form).

**Results:**

| | Exp 2 (uniform) | Exp 3 (non-uniform) |
|---|---|---|
| Solver average (unweighted) | 5.61 | **5.91** |
| Solver median | 6 | 6 |
| Best case | 5 | **4** |
| Worst case | 6 | **8** |
| Correct guesses | 46 / 46 | 46 / 46 |

The unweighted numbers look *worse* — and that's expected. The solver no longer optimises for "every animal equally"; it optimises for "the animals a real user actually picks." So rare animals take longer, common animals take less.

**Common animals get found faster** (4 steps each):
- Tiger, House Cat, Elephant, Cow, Lion

**Rare animals take longer** (8 steps each):
- Turtle, Platypus

**The headline number — weighted average**

The right metric for a non-uniform-priors solver is the weighted average — what a real user, sampling animals according to the prior, experiences across many games:

$$\bar{S}_{\text{weighted}} = \sum_{a} p(a) \cdot \text{steps}(a)$$

This is the number that *should* drop. The unweighted average is a misleading metric here because it implicitly assumes the (false) uniform distribution.

**What this tells us**

Priors shift the solver's strategy from *"split the candidate pool"* to *"split the probability mass."* Two consequences:

1. **Real users get faster games** — common animals are isolated in the first 4 questions instead of 5.
2. **The dataset's tail gets noisier** — rare animals are deferred to later questions, sometimes pushing past the previous worst case.

This is the same trade-off Wordle's solver makes (where common words get more weight than rare ones), and it generalises to every Bayesian guessing system. *You optimise for what people actually do, not for a flat distribution over what they could do.*


## Files

| File | What |
|---|---|
| `data/entities_v1.csv` | 46 animals × 18 features (Exp 1 dataset) |
| `data/entities_v2.csv` | 46 animals × 20 features (Exp 2 & Exp 3 dataset) |
| `data/priors_exp3.json` | Hand-crafted non-uniform priors (Exp 3) |
| `data/Animal Dataset.csv` | 205-animal Kaggle dataset (Exp 4, future) |
| `make_dataset.ipynb` | Generates either CSV — toggle `VERSION = "v1"` or `"v2"` |
| `solver.py` | `entropy()` + `question_picker()` |
| `play.py` | Interactive game (`run_game`) + batch evaluator (`evaluate`) with plots |
| `plots/eval_results_v1.png` | Distribution + per-animal bar chart for Exp 1 |
| `plots/eval_results_v2.png` | Same for Exp 2 |
| `plots/eval_results_v2_non_uniform_priors.png` | Same for Exp 3 |
