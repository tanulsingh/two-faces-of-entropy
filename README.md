# Two Faces of Entropy

Building information-theoretic guessing-game solvers — Akinator, and (coming soon) an LLM 20 Questions agent — to internalise entropy, information gain, and Bayesian belief update.

📖 **Full story, learning notes, and writeup →** [bits-and-surprise on my site](https://tanulsingh.github.io/projects/two-faces-of-entropy)

---

## What's here

- **[`akinator/`](./akinator)** — A guessing-game solver over a hand-authored animal table, built in the same arc as Grant Sanderson's [Wordle video](https://www.youtube.com/watch?v=v68zYyaEmEA) but on a smaller, predictable domain. 46 animals, 18 binary features, three experiments (uniform priors → fixed dataset → non-uniform priors). See [`akinator/README.md`](./akinator/README.md) for the experimental results.
- **LLM 20 Questions** — *coming soon.* The natural extension: swap the deterministic answerer for an LLM (noisy oracle), keep the same engine, add Bayesian belief update. Targeting Kaggle's [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions) competition.

## Quick start

```bash
cd akinator
pip install -r ../requirements.txt
python play.py
```

## Why this Project exists?

This Project exists to admire the beauty of Cross Entropy , Entropy and Uncertainty and how elegantly they summarise this feeling in mathematical form. We get handed information-theoretic tools in ML as if they're just formulas to plug in. They're not — they're a whole way of thinking about uncertainty, observation , and belief update that goes way beyond loss functions. The best way I know to actually internalise something like that is to build things where the formulas have to carry real weight. If a bot can't guess "Beyoncé" in seven questions , the formula failed *me* — not the other way around.

If you came here just for the code : it's all in the project folders. If you came to chase the same thread I did : start with the four links above , then poke around.
