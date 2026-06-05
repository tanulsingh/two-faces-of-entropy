# Two Faces of Entropy

*Information theory, two ways — and what happens when you stop nodding politely at it and actually use it.*

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

## Where to start

If like me you also want a deeper relationship with Cross Entropy , Entropy and Uncertainty in general , here's what I would do , read these in order.

### The probabilistic view

1. **[*Deep Learning* — Chapter 5.5 + 6.2.1.1](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville). The formal derivation — KL divergence ↔ negative log-likelihood ↔ cross-entropy , all in a few pages. If you want the proof, this is the proof.
2. **[The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I)** — Artem Kirsanov. The intuitive companion to the math. Builds surprisal → entropy → cross-entropy → KL divergence from a single intuition (*how surprised would you be?*). Watch this either before or after the book chapters — they reinforce each other beautifully.

### The information-theoretic view

3. **[Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)** — Chris Olah. The single best visual introduction to all of this. Reframes the same formulas through codes and bits — entropy as "optimal codeword length", cross-entropy as "what you pay when you use the wrong codebook." Once you've read this , the compression view will feel obvious in hindsight (it really isn't).
4. **[Solving Wordle using Information Theory](https://www.youtube.com/watch?v=v68zYyaEmEA)** — Grant Sanderson (3Blue1Brown). Turns "information gain" into something you can *feel* by watching it solve a game in real time. Also the direct inspiration for two of the projects in this repo.

After those four , you'll feel the love I am feeling for our dearest Cross Entropy and Uncertainty


## The Project

### Akinator - Inspired from Wordle Solution but Easier

If you've watched Grant Sanderson's [Wordle video](https://www.youtube.com/watch?v=v68zYyaEmEA), you've already seen the canonical demo of information theory in action — entropy picking the optimal guess at every step, the solver narrowing 13,000 candidate words down to one. He even walks through the subtle moments — like when the solver gets down to two candidate words and has no way to break the tie except guessing one and hoping, then upgrades the solver with word-frequency priors so it learns to prefer the more common candidate. If you haven't watched it yet, go do that first. It's twenty minutes and it'll change how you think about guessing games forever.

I tried implementing it myself. And the *theory* clicked instantly — but the *implementation* didn't. Wordle's feedback isn't binary; every guess produces one of 243 possible colour patterns (5 squares × 3 colours each). So every entropy calculation is a 243-bucket sum, every belief update is a 243-pattern filter, and the algorithm hides under the combinatorial machinery. You can verify it works statistically, but you can't *feel* it work — the steps are too dense to follow by hand, the bookkeeping too heavy to debug by inspection.

So instead of redoing Wordle (Grant already does it definitively), I built **Akinator** — same engine on a smaller, predictable domain. 46 animals, 18 binary features, yes/no questions only. The whole entropy table fits on a screen, you can compute any step on a napkin, and you can *see* why the solver picks what features at each step

I deliberately built in the same arc as Grant's video, just slower and more visible at every step:

- **Start with uniform priors** — every animal equally likely. The solver works, the math holds, ~5.6 questions average. But two pairs of animals (Horse/Cow, Pig/Rabbit) have *identical feature vectors*, and the solver can't break the tie no matter what it asks — the same dead-end Grant hits with two-word ambiguity in Wordle.
- **Fix the dataset.** Add two features that target the indistinguishability directly. The dead-ends vanish.
- **Add non-uniform priors** — Just like in Wordle not all the candidate words can be answer and some occur more than the other , animals also differ in popularity and common animals might be the answer in real life than rarer ones. Lion gets a higher prior than Platypus, because that's what a real player would pick. Watch the solver's strategy shift from *"split the candidate pool"* to *"split the probability mass."* Common animals get found in 4 questions instead of 5. The same move Grant makes when he adds word-frequency priors from a Google dataset.

Three experiments, written up in [`akinator/README.md`](./akinator/README.md), each with the numbers and the surprises. The point is not the bot — it's that *every* concept in information theory has a small, predictable version you can verify by hand before you trust it at scale.

**If you want the deeper learning loop**: watch Grant's video, then try implementing Wordle yourself. When the 243-pattern machinery slows you down (it will), come read through Akinator. Both projects does the same thing; one just hides nothing.

### What's next — LLM 20 Questions *(coming soon)*

The natural extension is to swap the deterministic answerer for an LLM — noisy, sometimes wrong, candidate space unbounded. The same engine should still work but with Bayesian belief update instead of hard filtering. That's the bridge from Akinator (clean) to real-world systems (messy). Building this next as an agent for Kaggle's [LLM 20 Questions](https://www.kaggle.com/competitions/llm-20-questions) competition.


## Why this Project exists?

This Project exists to admire the beauty of Cross Entropy , Entropy and Uncertainty and how elegantly they summarise this feeling in mathematical form. We get handed information-theoretic tools in ML as if they're just formulas to plug in. They're not — they're a whole way of thinking about uncertainty, observation , and belief update that goes way beyond loss functions. The best way I know to actually internalise something like that is to build things where the formulas have to carry real weight. If a bot can't guess "Beyoncé" in seven questions , the formula failed *me* — not the other way around.

If you came here just for the code : it's all in the project folders. If you came to chase the same thread I did : start with the four links above , then poke around.
