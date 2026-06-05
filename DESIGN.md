# Akinator — Design Decisions

## 1. Dataset
- **Domain**: Animals
- **Size**: ~30–50 entities × 15–25 binary features
- **Authoring**: hand-written CSV — no scraping, no LLM generation
- **Rationale**: I should know every entity by heart so I can verify
the table by eye and feel which features split the population well.

## 2. Stop condition
Pick one (delete the other):
- **Probability-based**: stop when the top entity has posterior
probability > 0.9
- **Entropy-based**: stop when the belief entropy drops below 0.5 bits

## 3. Interaction
- CLI prompt, yes/no/maybe input
- One question at a time, show running probability of top guess
- Print final guess + entropy curve when done