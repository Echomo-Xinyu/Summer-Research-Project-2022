# Week1 Report

Week 1: *8 May - 13 May 2022*

## Learning Progress

This week is to self-learn the NLP course [CS224N - 2019 Winter](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)

- [x] Lecture1-6
- [ ] Lecture7-8: to be finished by the start of weekly meeting on Sunday 3pm
- [x] assignment1+2 on course website

## Learning Outcome

As this week is mostly self-learning, the outcome is essentially a summary of what I have learned.

### How to represent word as vectors

*Locallist* (one-hot vector) vs *Distributed representation*: the latter encodes the word's meaning -- allowing finding synonyms or even vector decomposition.

#### How to derive the word vector

1. **SVD**: find co-occurence counts in a context window as a matrix and then apply *Singular Value Decomposition* to reduce the dimension -- sufficient for encoding similarities
    - high computing cost (large matrix size + quadratic cost for SVD)
    - drastic imbalance in word frequency: "the", "he", "has", etc -- addressed by ignoring such words
2. **word2Vec**: train a one-layer neural network that is capable to compute probability of a word given its context. The hidden layer will be the word encoding by the end of the training.
    - *Skip-gram*, to predict the distribution of context words from a center word, as opposed to *Continuous Bag of Words* (CBOW).
    - *negative sampling*: update only a selective number of rows during one epoch, as computing for the whole matrix is costy
3. **GloVe**: a weighted least squares model that trains on global word-word co-occurrence counts -- sota performance in word analogy task. It utilizes the global co-occurence stats which SVD and word2Vec fail to.

### Neural Network

- Self catch-up on *differential calculus* and *Jacobian*.
- Recap of concepts relevant to NN: *Backpropagation*, *learning rate*, *activation function*, *preprocessing*.

### Dependency Parsing

Due to inherent confusion in some languages, a *dependency graph* is built to show which words depend on (modify or are arguments of) which other words. Three methods of building such a graph is mentioned:

1. Dynamic Programming based parsing -- no longer used due to high time complexity
2. Transition-based shift-reduced parsing --  Formulate current situation (stack + buffer + actions) and let NN classifier decide which action (left arc, right arc, shift) to take. Repeat till the end.
   - greedy and deterministic, linear time complexity
   - sparse feature representation -- undesirable and addressed by neural dependency parser
3. Neural dependency parser: carefully formulate the situation and use NN to directly obtain results, hence faster than method 2.

### Intro to RNN

Problem of language model: to compute the distribution given t-1 previous words.

- take in variable-length input -- as opposed on traditional n-grams language
- reusing the same weight matrix to avoid inefficient learning while saving memory -- as opposed to fix window size NN
- a few training details such as backpropagation gradient, evaluation with perplexity

## Goal for Next Week

- [ ] Lecture 9-17 videos
- [ ] Lecture 9-17 Complementary Notes (if have spare time)
