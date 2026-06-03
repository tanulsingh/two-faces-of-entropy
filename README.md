# Two Faces of Entropy

*Information theory, two ways — and what happens when you stop nodding politely at it and actually use it.*

---

## The story

As an ML engineer, I used to reach for `cross_entropy_loss` the way I trust the default Adam hyperparameters — full confidence, zero scrutiny. I knew the formula, I knew how it punishes wrong predictions and rewards confident-correct ones. The number goes down, the model gets better, what's not to like. But I never *really* understood it. I didn't even know why I was choosing it in the first place when there are so many other losses out there doing apparently the same job (how is this one any different?). What me and cross-entropy had was a very surface-level relationship, and I wanted something more intimate, you know — because I'm stuck with this thing for life :-p

For that I needed to know it better and get some answers:

- What does a loss value of `3.34` actually *mean*?
- How do I interpret cross-entropy *physically* — not as a number on a tensorboard, but as a quantity in the world?
- Can I observe cross-entropy outside of an ML training loop?

So I went looking. The first thing that gave me real ground was **Chapter 5.5 ("Maximum Likelihood Estimation") and Chapter 6.2.1.1 ("Learning Conditional Distributions with Maximum Likelihood") of [Goodfellow, Bengio & Courville's *Deep Learning*](https://www.deeplearningbook.org/)** . Those two chapters walked me through the whole chain : minimising KL divergence between the empirical data distribution and the model distribution is mathematically the same as maximising log-likelihood , which collapses neatly into cross-entropy. Reading them back-to-back , "why cross-entropy?" stopped being a mystery — it's just what you get when you take MLE seriously. I felt like I finally *knew* cross-entropy , but our friend here is way more mysterious than I thought.

The math part was covered but I still wanted the bigger picture — surprisal, entropy, KL divergence — not just CE sitting in isolation. That's when I found [Artem Kirsanov's "The Key Equation Behind Probability"](https://www.youtube.com/watch?v=KHVR587oW8I) which builds the entire stack from one simple intuition (*how surprised would you be?*) all the way up through cross-entropy and KL divergence. Cleanest derivation of the whole family I've seen, and the kind of video you want to re-watch a week later just to feel smart again. This is the point where I started falling for my dearest Cross Entropy. 

I now felt very comfortable with the **probabilistic view**. Hubris achieved , Life was good. Then last week, while I was starting to write the first post of my new blog series, [The Loss Landscape of LLM Training](https://tanulsingh.github.io/blog/llm-pretraining) , I tripped over the line **"language models are compressors"** . I read it five times trying to figure out which word was the typo , then went looking for an explanation — only to find out that cross-entropy is *literally* the number of bits per symbol your model would use to compress the data. Same formula. Completely different universe. I sat with that one for a while , had a laugh or two , you sly little fox , how much more efforts do you wanna take?

That was the second half : the **information-theoretic view**. Chris Olah's [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) made the connection sing , and Grant Sanderson's [Solving Wordle using Information Theory](https://www.youtube.com/watch?v=v68zYyaEmEA) made it feel like a tool I could actually pick up and use rather than just admire from a distance.

You see the beauty of this thing? How should one not fall in Love? Two completely different stories about the exact same equation. Once you see both , you can never go back to seeing only one. This repo is what came out of chasing that thread — instead of writing yet another blog post explaining entropy (the resources below already do it much better than I ever could) , I wanted to actually *apply* it. Build real things where information theory is the engine , and see whether the intuition holds up when it has to drive code that works.

Spoiler : it does. Mostly.

---

## Where to start

If like me you also want a deeper relationship with Cross Entropy , Entropy and Uncertainty in general , here's what I would do , read these in order.

### The probabilistic view

1. **[*Deep Learning* — Chapter 5.5 + 6.2.1.1](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville). The formal derivation — KL divergence ↔ negative log-likelihood ↔ cross-entropy , all in a few pages. If you want the proof, this is the proof.
2. **[The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I)** — Artem Kirsanov. The intuitive companion to the math. Builds surprisal → entropy → cross-entropy → KL divergence from a single intuition (*how surprised would you be?*). Watch this either before or after the book chapters — they reinforce each other beautifully.

### The information-theoretic view

3. **[Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)** — Chris Olah. The single best visual introduction to all of this. Reframes the same formulas through codes and bits — entropy as "optimal codeword length", cross-entropy as "what you pay when you use the wrong codebook." Once you've read this , the compression view will feel obvious in hindsight (it really isn't).
4. **[Solving Wordle using Information Theory](https://www.youtube.com/watch?v=v68zYyaEmEA)** — Grant Sanderson (3Blue1Brown). Turns "information gain" into something you can *feel* by watching it solve a game in real time. Also the direct inspiration for two of the projects in this repo.

After those four , you'll feel the love I am feeling for our dearest Cross Entropy and Uncertainty

---

## The projects

Each project applies the same engine (entropy, information gain, Bayesian update) to a different problem of increasing nastiness.

- **[`akinator/`](./akinator)** — the warm-up. A 20-questions-style solver over a small hand-authored celebrity table. The cleanest possible demonstration of "pick the question with maximum expected information gain." If you can build this , you understand the algorithm.
- **[`wordle/`](./wordle)** — the canonical example, popularised by 3B1B. An optimal Wordle solver with frequency-based priors and the full 243-pattern entropy machinery. Same algorithm as Akinator but with significantly more engineering attached.
- **[`llm-20-questions/`](./llm-20-questions)** — the capstone. An agent for Kaggle's [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions) competition that uses the same engine but with an LLM as a very noisy oracle. Everything that *can* go wrong here does , which is kind of the whole point.

More to come as I find new things to chase. Each project has its own README with the algorithm , the result , and whatever surprised me along the way.

---

## Why this repo exists

This repo exists to admire the beauty of Cross Entropy , Entropy and Uncertainty and how elegantly they summarise this feeling in mathematical form. We get handed information-theoretic tools in ML as if they're just formulas to plug in. They're not — they're a whole way of thinking about uncertainty, observation , and belief update that goes way beyond loss functions. The best way I know to actually internalise something like that is to build things where the formulas have to carry real weight. If a bot can't guess "Beyoncé" in seven questions , the formula failed *me* — not the other way around.

If you came here just for the code : it's all in the project folders. If you came to chase the same thread I did : start with the four links above , then poke around.
