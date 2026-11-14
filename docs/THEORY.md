# CRANE theory: memory, thought and value

This document describes the idea behind CRANE: what the system is supposed to be, why I think it can work, and how far the current code is from it. The code in this repository is an early scaffold; this is the target.

## 1. The core claim

Today's language models all share roughly one architecture: text is tokenized, turned into latent vectors by an embedding layer, and then pushed through a very large stack of attention and feed-forward blocks. Those blocks do two jobs at once. They store what the model knows, and they do whatever thinking the model does. Memory is baked into thinking.

The consequence is that a sentence like "I went to the lake" is pushed through the entire network, through everything it knows about Michael Jackson, Shakespeare and tax law, and only a small part of that computation is actually about lakes. However small a model is, it still carries facts in its weights and answers from them.

I don't think that is how cognition works. Thinking and memory are different processes:

- **Thinking** is the process: it holds the current situation, compares, infers, builds a hypothesis, decides.
- **Memory** is what surfaces during thinking. The meaning of the current thought acts as a trigger, activates a region of memory, and activation spreads along associations: lake, then fish, then the morning I went fishing and it was cold, then what usually happens next.

CRANE separates the two. **The network is trained to think, not to store facts.** A graph memory holds what was experienced, what was concluded, and what follows from what.

The argument is not that the brain holds less data than a large model; with around 100 trillion synapses it does not. The argument is about efficiency and structure: a brain runs on about 20 watts, learns from a tiny fraction of the data a language model sees, and does not route every thought through everything it has ever learned.

### What is wrong with thinking in text

- **Text is not the mental process.** Language is an interface to thought, not the thought itself. Reasoning models "think" by writing text and then reading it back, which is closer to retrieval over their own output than to thinking.
- **Insights are not kept.** Whatever a model "realised" while answering disappears when the answer ends. A person who realises something holds on to it, strengthens it when it is confirmed, and builds new abstractions on top of it that make later thinking easier.
- Moving from words to images as a medium of reasoning does not fix this. The step that is missing is reasoning in a representation of **thought** itself, not in any surface form.

When a person learns, they:

1. add to memory,
2. change the weights of the parts of memory that were involved,
3. change how they think, but do not rewrite all of memory.

## 2. The pipeline

```
text -> tokenization -> embeddings -> salient (sparse) selection
     -> memory grabbing -> hypothesis construction
     -> transformer blocks (thought cleaning) -> thought
     -> thought-to-language converter -> tokens -> text
```

The transformer blocks are no longer storage; they are processing units that turn retrieved memory and the current situation into a thought. The final converter is one of several possible outputs: text today, but the same thought could drive speech or actions through other converters.

## 3. What a memory is

A memory is not a word and not a definition. It is an **event-like structure**: objects, the situation they were in, what happened, and what followed. Memory is the set of links between objects, events and consequences. Thinking uses it to find, for the current situation, the nearest link to a consequence, and then to try to repeat it or prevent it.

Word embeddings alone are not enough for this. They place "dog" near "cat" and far from "synchrophasotron", but it is unclear where an event like "a murder" should sit relative to them. Events need their own representation, built from their participants, context and outcome, not the coordinates of a word.

The graph holds three kinds of units:

| Unit | Example | Role |
|---|---|---|
| Object | lake, fish, cold water | things that take part in events |
| Event | "fishing at the lake at dawn, nothing was biting, I got cold" | one concrete experience |
| Situation / abstraction | "fishing" | a generalised script or concept built from many similar events |

And typed, weighted links between them:

| Link | Meaning |
|---|---|
| `INVOLVES` | event → objects that took part |
| `LOCATED_AT`, `BEFORE` | context: where, in what order |
| `CAUSES` / `LEADS_TO` | event → what followed from it |
| `INSTANCE_OF` | event → the situation or abstraction it is an example of |
| `ASSOCIATED_WITH` | weak learned association, the current catch-all |

Because memory stores consequences, recalling it gives the thinking process **predictions**: in situations like this, this is what usually happens next.

### Several memory spaces, not one

"Reading a book" and "casting iron" have nothing in common, and forcing them into one flat space makes every search noisy. Memory is split into several graphs or regions by domain of experience. The meaning of the current thought first selects the relevant region, then searches inside it. This is also what keeps the system sparse: most of memory is never touched by a given thought.

## 4. Memory learns

Memory is not a fixed knowledge base filled in advance. This is the difference from graph RAG, where the structure is built once by an algorithm (for example Leiden community detection) and then only read.

In CRANE memory changes as the system works and trains:

- **New units appear.** A thought that was useful is written as a new event or concept and linked to the memories that produced it.
- **Link strengths change.** Links used together in a good outcome grow; unused ones fade and are pruned.
- **Units move.** The vectors of memory units are themselves updated, so related experiences drift closer together.
- **Abstractions form.** When a hypothesis is confirmed, it is reinforced and becomes something new thinking can build on. Periodically, similar events are merged into a generalised situation or concept, the way sleep consolidates memory in people. This happens as a separate consolidation phase, not on every step.

## 5. Value: good, bad, painful, pleasant

Every situation is evaluated: good or bad, painful or pleasant, important or not. That evaluation is what fixes things in memory. In the brain this is done chemically (dopamine, noradrenaline, stress hormones); in CRANE it is a signal attached to events and used when updating links.

- Each event carries **valence** (from bad to good, -1..+1) and **intensity** (how much it mattered, 0..1).
- Links are strengthened by co-activation **multiplied by how much the outcome mattered and how surprising it was**. Things that fired together in an unexpectedly good or bad situation are linked most strongly.
- **Emotion follows from memory.** When a situation activates memories, their expected valence is a prediction of whether what comes next will be good or bad. That prediction is the system's emotional state; its change over time gives analogues of surprise, relief or disappointment. The state is fed back into reasoning as context, so thinking depends on it.

I use "emotion" in a functional sense: a value prediction built from experience that changes how the system reasons and what it remembers. Whether such a system feels anything is a separate philosophical question that this project does not try to answer.

## 6. The loop

**Inference**

1. Input text is encoded; the salient parts form a query: the meaning of the current thought.
2. The query selects a memory region, activates similar events and situations, and activation spreads along links (1–2 hops), weighted by link strength.
3. A significance filter keeps the few memories that matter most, including expected consequences reached through `CAUSES` links and their valence.
4. The retrieved memories are fused into the current representation; the reasoner builds a hypothesis, a confidence, and an expected outcome.
5. If not confident, the hypothesis becomes part of the next query and the loop repeats, as reasoning models do, but in thought space rather than text.
6. The result is written back to memory as a new abstraction linked to what produced it.
7. The final thought is converted into language (or another output).

**Training** is the same loop, plus a weight update of the thinking network after the produced tokens are compared with the target, and a memory update driven by prediction error: confirmed links grow, wrong ones weaken, genuinely new experience becomes a new event node. The training data is the same as for a language model, and next-token prediction is the starting objective.

## 7. The main trap: facts leak into weights

Gradient descent does not know that facts are supposed to live in the graph. If storing a fact in the weights lowers the loss, it will be stored there, and the "thinking-only" network quietly becomes a small language model with a memory attached.

Two ways to prevent it:

- **Make memory the cheaper path.** Keep the thinking network small and make retrieval good enough that remembering through the graph beats memorising. RETRO showed that retrieval can replace a large share of parameters.
- **Train on worlds where memorising is useless.** Synthetic episodes where every episode has its own objects, rules and facts. Nothing carries over between episodes except the skill of using memory, so the only way to lower the loss is to write experience to memory and reason over it. This is the cleanest test of the whole idea: a thinker that is forced to use memory.

## 8. Why I think this is plausible

None of the pieces are new on their own; the combination is.

- **Thought without language.** People with severe aphasia still solve logic and arithmetic problems, and the brain's language network is separate from the network used for reasoning (Fedorenko and colleagues). In ML, reasoning in continuous latent space instead of tokens is an active direction (e.g. Coconut, Meta, 2024).
- **Two memory systems.** Complementary Learning Systems theory (McClelland, McNaughton, O'Reilly, 1995): the hippocampus stores episodes quickly, the cortex learns general structure slowly. In CRANE the graph is the fast episodic store and the network weights are the slow learner.
- **Episodic memory as events.** Tulving's episodic memory (what, where, when, with whom); event segmentation theory (Zacks): experience is cut into events with boundaries; scripts (Schank and Abelson): generalised situations like "going to a restaurant".
- **Spreading activation.** Collins and Loftus (1975): concepts form a network and activation spreads to neighbours, confirmed by priming experiments.
- **Commonsense as event → consequence.** ATOMIC (Sap et al., 2019; ATOMIC 2020) and COMET encode knowledge as if-then links between events, causes and effects.
- **Value shapes memory.** Emotional arousal strengthens hippocampal encoding via the amygdala; dopamine signals reward prediction error (Schultz); Damasio's somatic markers: past valuations attached to situations guide decisions; Barrett's theory of constructed emotion: emotions are predictions built from past experience.
- **Prediction error as the learning signal.** Predictive processing (Friston, Clark): the brain predicts and learns from the mismatch.
- **Knowledge does not have to live in weights.** RETRO (DeepMind, 2021): a model with retrieval matched models roughly 25 times larger. Mixture of Experts and interpretability work (sparse features, sparse autoencoders) show that only a small part of a network is relevant for a given input, even when the whole network is computed.
- **Sparse trainable memory.** Product Key Memory (Lample et al., 2019): a memory of millions of trainable slots where each step reads only the top few, and only those are updated. Titans (Google, 2025): a memory that keeps learning at inference time, driven by surprise.

Honest counterpoints:

- Transformers already behave as associative memories (attention is closely related to modern Hopfield networks; feed-forward layers act as key-value stores).
- Explicit-memory architectures (Neural Turing Machine, DNC) were elegant and still lost to plain transformers plus scale.

So CRANE is unlikely to beat a same-size transformer on next-token perplexity of ordinary text. Its claim is narrower and testable: **on tasks where experience and consequences matter, an explicit event memory with value-weighted links should help.** Examples: learning new facts after training without retraining, predicting what happens next in a familiar situation, not forgetting old knowledge when new is added, and the synthetic worlds from section 7.

## 9. Where the current code stands

The current scaffold implements the loop with a generic vector memory. Known gaps against this document:

1. **Memory units have no meaning.** Training starts from an empty graph and writes one "thought" vector per step from an untrained network; there are no objects, events or consequences. This is why the database fills with noise.
2. **One graph, one link type, no valence.** All links are `ASSOCIATED_WITH` with a single weight; there are no memory regions; events carry no value.
3. **Retrieval and memory are not aligned or trained.** The query (mean of salient slots) and the written thought vectors live in different spaces, and retrieval is not differentiable, so nothing teaches the network what to store or how to search. Memory unit vectors never move.
4. **Confidence is not trained.** It gates both writing and stopping but receives no learning signal.
5. **Single-vector bottleneck.** The decoder reconstructs text from one 256-d thought vector.
6. **Train/inference mismatch.** Training runs one iteration; inference runs several and injects the thought as a token the model never saw in training.
7. **No consolidation.** Abstractions are never formed from repeated events.
8. **No baseline.** There is no memory-off comparison, so it is impossible to say whether memory helps.

## 10. Next steps

1. Baseline: the same stack with memory off; measure loss and perplexity.
2. Event schema in Neo4j: node types and typed links from section 3, memory regions, valence and intensity on events.
3. Seed memory with meaning: commonsense events from ATOMIC 2020, plus episodes extracted from text by an LLM (participants, action, place, consequence, valence).
4. Key-value memory with trainable unit vectors: search by the meaning of a situation, return the event and its consequences, so query and memory live in one space and units can move.
5. Synthetic worlds (section 7) and a "predict what happens next" task, where memory has to matter.
6. Value-weighted, prediction-error-driven link updates; train confidence (as predicted loss or a learned halting signal).
7. A consolidation phase that merges repeated events into situations and abstractions.
8. Ablations: memory off / generic memory / event memory / event memory plus valence / plus consolidation. Report results either way.
