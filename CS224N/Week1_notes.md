# Week1 notes

## Lecture 1: Introduction and Word Vectors

Random thoughts:

- Language can mean different for different people.
- languages serve as the glue to connect people as community and promote connection and collaboration.

in linguistics, denotational semnatics to represent meaning.

### WordNet

- greart as a resource but missing nuance
- fixed meanings
- surjective
- require human labor to generate
- no metric to measure word similarity

Traditional NLP: words are treated as discrete symbols like a localist representation like an one-hot vector. *one-hot vector*: one 1 the rest 0s. (classified output result) -- vector dimension = number of words in vocabulary (~500,000)

Problems: different words have orthogonal vectors -> no natural notion of *similarity*.
Solution: to incorporate the similarity into vector representation.

### Distributional Semantics

meaning: a word's meaning is given by the words that frequently appear close-by.

When a word $w$ appears in a text, its *context* is the set of words that appear nearby (within a fixed-size window).

Build a dense vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts. Such *word vectors* are called *word embeddings* or *word representations* -- *distributed* representation.

We can build a vector space from the word embeddings (with projections on 2D plane) for *visualization*.

### Application of Distributional Semantics: Word2Vec

a framework for learning word vectors:

- with a large corpus of text
- every word in a fixed vocabulary is represented by a *vector*
- go through each position $t$ in the text, which has a center word $c$ and context ("outside") word $o$
- use the *similarity of the word vectors* for $c$ and $o$ to *calculate the probability* of $o$ given $c$ (or vice versa)
- *keep adjusting the word vectors* to maximize this probaility

For each position $t = 1, \dots, T$, predict context words within a window of fixed size $m$, given center word $w_j$.

$$ \text{Likelihood} = L(\theta) = \prod^T_{t = 1} \prod_{\substack{-m \le j \le m \\ j \neq 0}} P(w_{t+j} | w_t; \theta) $$

where $\theta$ is all variables to be optimized.

The *objective function $J(\theta)$ is the (average) negative log likelihood:

$$ J(\theta) = -\frac{1}{T}logL(\theta) = -\frac{1}{T} \sum^T_{t=1}\sum_{\substack{-m \le j \le m \\ j \neq 0}}logP(w_{t+j} | w_t; \theta) $$

*Minimizing objective function $\Leftrightarrow$ Maximizing predictive accuracy*

In order to calculate $P(w_{t+j} | w_t; \theta)$, one way is to use two vectors per word $w$: $v_w$ when $w$ is a center word and $u_w$ when $w$ is a context word.

For a center word $c$ and a context word $o$:

$$ P(o|c) = \frac{\exp(u^T_ov_c)}{\sum_{w \in V} \exp(u^T_wv_c)} $$

the dot product compares similarity of $o$ and $c$. Larger dot product implies larger probability. (think about the angle between two vectors)

Above is an example of the *softmax function*, where "max" amplifies probability of largest value and "soft" still assigns some probability for smaller value.

To start, we assign random values to each vector and go down gradient to find convergence.

$$ \frac{\partial}{\partial v_c}log\frac{\exp(u^T_ov_c)}{\sum_{w \in V} \exp(u^T_wv_c)} = \frac{\partial}{\partial v_c} (u^T_ov_c) - \frac{\partial}{\partial v_c}log\sum_{w \in V} \exp(u^T_wv_c)$$

numerator can be simplified (by multivariable calculus) to be $u_o$ and denominator can be simplied (by chain rule) to be $\frac{\sum^V_{x=1}\exp(u^T_xv_c)u_x}{\sum^V_{w=1}\exp(u^T_wv_c)}$.

Hence $\frac{\partial}{\partial v_c}logP(o|c) = u_o - \sum^V_{x=1}\frac{\exp(u^T_xv_c)}{\sum^V_{w=1}\exp(u^T_wv_c)}u_x$, which consists of $\frac{\exp(u^T_xv_c)}{\sum^V_{w=1}\exp(u^T_wv_c)}$ -- the definition of $P(x|c)$

Vector composition can be applied similarly (in ideal context) (eg king - man + woman == queen)

### Personal Random Note

the formula on the blackboard for $J(\theta)$ has no $\theta$ inside.
