# CRANE theory: memory, thought and value

This document describes the idea behind CRANE: what the system is supposed to be, why I think it can work, and how far the current code is from it. The code in this repository is an early scaffold; this is the target.

## 1. The core claim

In today's neural networks, memory is baked into thinking. Knowledge and reasoning live in the same weights, so a sentence like "I went to the lake" is pushed through the entire network, through everything it knows about Michael Jackson, Shakespeare and tax law, and only a small part of that computation is actually about lakes.

I don't think that is how cognition works. Thinking and memory are different processes:

- **Thinking** is the process: it holds the current situation, compares, infers, decides.
- **Memory** is what surfaces during thinking. The meaning of the current thought acts as a trigger, activates a region of memory, and activation spreads along associations: lake, then fish, then the morning I went fishing and it was cold, then what usually happens next.

CRANE separates the two. The network learns *how to think*. A graph memory holds *what was experienced and what follows from what*.

## 2. What a memory is

A memory is not a word and not a definition. It is an **event-like structure**: objects, the situation they were in, what happened, and what followed. Memory is the set of links between objects, events and consequences.

Concretely, the graph should hold three kinds of units:

| Unit | Example | Role |
|---|---|---|
| Object | lake, fish, cold water | things that take part in events |
| Event | "fishing at the lake at dawn, nothing was biting, I got cold" | one concrete experience |
| Situation | "fishing" | a generalised script built from many similar events |

And typed, weighted links between them:

| Link | Meaning |
|---|---|
| `INVOLVES` | event → objects that took part |
| `LOCATED_AT`, `BEFORE` | context: where, in what order |
| `CAUSES` / `LEADS_TO` | event → what followed from it |
| `INSTANCE_OF` | event → the situation it is an example of |
| `ASSOCIATED_WITH` | weak learned association, the current catch-all |

Because memory stores consequences, recalling it gives the thinking process **predictions**: in situations like this, this is what usually happens next.

## 3. Value: good, bad, painful, pleasant

Every situation is evaluated: good or bad, painful or pleasant, important or not. That evaluation is what fixes things in memory. In the brain this is done chemically (dopamine, noradrenaline, stress hormones); in CRANE it is a signal attached to events and used when updating links.

- Each event carries **valence** (from bad to good, -1..+1) and **intensity** (how much it mattered, 0..1).
- Links are strengthened by co-activation **multiplied by how much the outcome mattered and how surprising it was**. Things that fired together in an unexpectedly good or bad situation are linked most strongly. Links that never matter fade (decay) and are eventually pruned.
- **Emotion follows from memory.** When a situation activates memories, their expected valence is a prediction of whether what comes next will be good or bad. That prediction is the system's emotional state; its change over time gives analogues of surprise, relief or disappointment. The state is fed back into reasoning as context, so thinking depends on it.

I use "emotion" in a functional sense: a value prediction built from experience that changes how the system reasons and what it remembers. Whether such a system feels anything is a separate philosophical question that this project does not try to answer.

## 4. The loop

1. Input text is encoded; the salient parts of it form a query: the meaning of the current thought.
2. The query activates similar events and situations in memory; activation spreads along links (1–2 hops), weighted by link strength.
3. A significance filter keeps the few memories that matter most, including the expected consequences reached through `CAUSES` links and their valence.
4. The retrieved memories are fused into the current representation; the reasoner forms a thought, a confidence, and an expected outcome.
5. If not confident, the thought becomes part of the next query and the loop repeats (thinking longer about harder things).
6. When the real outcome is known, the **prediction error** (expected vs actual outcome and value) updates memory: confirmed links grow, wrong ones weaken, a genuinely new experience becomes a new event node.
7. The final thought is translated into language. Language is an interface to thought, not the thought itself.

## 5. Why I think this is plausible

None of the pieces are new on their own; the combination is.

- **Thought without language.** People with severe aphasia still solve logic and arithmetic problems, and the brain's language network is separate from the network used for reasoning (Fedorenko and colleagues). In ML, reasoning in continuous latent space instead of tokens is an active direction (e.g. Coconut, Meta, 2024).
- **Two memory systems.** Complementary Learning Systems theory (McClelland, McNaughton, O'Reilly, 1995): the hippocampus stores episodes quickly, the cortex learns general structure slowly. In CRANE the graph is the fast episodic store and the network weights are the slow learner.
- **Episodic memory as events.** Tulving's episodic memory (what, where, when, with whom); event segmentation theory (Zacks): experience is cut into events with boundaries; scripts (Schank and Abelson): generalised situations like "going to a restaurant".
- **Spreading activation.** Collins and Loftus (1975): concepts form a network and activation spreads to neighbours, confirmed by priming experiments.
- **Commonsense as event → consequence.** ATOMIC (Sap et al., 2019; ATOMIC 2020) and COMET encode knowledge as if-then links between events, causes and effects.
- **Value shapes memory.** Emotional arousal strengthens hippocampal encoding via the amygdala; dopamine signals reward prediction error (Schultz); Damasio's somatic markers: past valuations attached to situations guide decisions; Barrett's theory of constructed emotion: emotions are predictions built from past experience.
- **Prediction error as the learning signal.** Predictive processing (Friston, Clark): the brain predicts and learns from the mismatch.
- **Knowledge does not have to live in weights.** RETRO (DeepMind, 2021): a model with retrieval matched models roughly 25 times larger. Mixture of Experts and interpretability work (sparse features, sparse autoencoders) show that only a small part of a network is relevant for a given input, even when the whole network is computed.

Honest counterpoints:

- Transformers already behave as associative memories (attention is closely related to modern Hopfield networks; feed-forward layers act as key-value stores).
- Explicit-memory architectures (Neural Turing Machine, DNC) were elegant and still lost to plain transformers plus scale.

So CRANE is unlikely to beat a same-size transformer on next-token perplexity. Its claim is narrower and testable: **on tasks where experience and consequences matter, an explicit event memory with value-weighted links should help.** Examples: learning new facts after training without retraining, predicting what happens next in a familiar situation, not forgetting old knowledge when new is added.

## 6. Where the current code stands

The current scaffold implements the loop with a generic vector memory. Known gaps against this document:

1. **Memory units have no meaning.** Training starts from an empty graph and writes one "thought" vector per step from an untrained network; there are no objects, events or consequences. This is why the database fills with noise.
2. **One link type, no valence.** All links are `ASSOCIATED_WITH` with a single weight; events carry no value.
3. **Retrieval and memory are not aligned or trained.** The query (mean of salient slots) and the written thought vectors live in different spaces, and retrieval is not differentiable, so nothing teaches the network what to store or how to search.
4. **Confidence is not trained.** It gates both writing and stopping but receives no learning signal.
5. **Single-vector bottleneck.** The decoder reconstructs text from one 256-d thought vector.
6. **Train/inference mismatch.** Training runs one iteration; inference runs several and injects the thought as a token the model never saw in training.
7. **No baseline.** There is no memory-off comparison, so it is impossible to say whether memory helps.

## 7. Next steps

1. Baseline: the same stack with memory off; measure loss and perplexity.
2. Event schema in Neo4j: node types and typed links from section 2, valence and intensity on events.
3. Seed memory with meaning: commonsense events from ATOMIC 2020, plus episodes extracted from text by an LLM (participants, action, place, consequence, valence).
4. Change the training task from "predict the next token" to **"predict what happens next"** in a situation; this is where memory should matter.
5. Key-value memory: search by the meaning of a situation, return the event and its consequences, so query and memory live in one space.
6. Value-weighted, prediction-error-driven link updates; train confidence (as predicted loss or a learned halting signal).
7. Ablations: memory off / generic memory / event memory / event memory plus valence. Report results either way.
