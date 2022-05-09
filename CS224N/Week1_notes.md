# Week1 notes

## Lecture1: Introduction and Word Vectors

Random thoughts:

- Language can mean different for different people.
- languages serve as the glue to connect people as community and promote connection and collaboration.

in linguistics, denotational semantics to represent meaning.

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

## Lecture2: Word Vectors and Word Senses

### Continue fromn Lecture1

The vector space from the word embeddings may have more meanings in terms of the vector direction. For example, `analogy("man", "king", "woman")` should yield `"queen"`.

The issue of labelling individual points in a scatter plot can be addressed with this [stackoverflow answer](https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point).

### Word2vec notes

We are generating a model that gives a reasonably high probability estimate to all words that occur in the context -- same predictions at each position.

### Word2vec Optimization

We use *gradient descent* to minimize the cost function $J(\theta)$ in previous class. During each iteration, we udpate $\theta$ value in the direction of negative gradient.

$$ \theta^\text{new} = \theta^\text{old} - \alpha \nabla_\theta J(\theta)$$

with $\alpha$ as the *step size* or *learning rate*.

**Problem**: $\nabla_\theta J(\theta)$ can be very expensive to compute as $J(\theta)$ can potentially mean billions of windows in the corpus.

**Solution**: *Stochastic gradient descent* (SGD) top repeatedly sample windows and update after each one. (updating $\theta$ value right after one single sampled window) [Tradeoff between SGD and GD](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent).

**Potential problems**:

- The vectors in SGD can be very sparse (with at most 2m+1) words. We hence need to be careful about the update we make.
- we can have only one vector instead of the two vector mentioned (which can be easier to optimize). We have gone through *skip-grams* during lecture and Continuous Bag of Words (CBOW) which predict center word from (bag of) context words.
- Negative sampling can add additional efficiency in training. (which can be more complex but cheaper in training compared to softmax method) -- practice of skip-gram methods with negative sampling in a2.

Hyperparameters such as choosing window size and 3/4 in Unigram distribution are useful practical tricks.

### A Traditiaonal Stats perspective

We may consider counting the frequency of all the words within the window centered on a certain word $w$ in the whole corpus. -- name it as a matrix of co-occurance accounts.

Problems:

- increase in size with vocabulary
- very high dimensional: storage consuming
- subsequent clasification models have sparsity issues
- classification models are less robust

One solution: reduce the matrix to ow dimensionality (practice in a1) -- Latent Semantic Analysis(LSA) and Singular value decomposition([SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)).

Some tricks:

- scaling down the counts of words that appear too frequently (eg taking a ceiling $min(X, t)$ with $t \approx 100$)
- ramped windwos that count more of closer words
- use Pearson correlations instead of counts, then set negative values to 0.

| Count based                                       | direct prediction                                   |
|---------------------------------------------------|-----------------------------------------------------|
| LSA etc                                           | Skip-gram / CBOW                                    |
| Fast training                                     | Scales with corpus size                             |
| Efficient usage of statistics                     | Inefficient usage of statistiscs                    |
| Primarily used to capture word similarity         | Generate improved performance on other tasks        |
| Disproportionate improtance given to large counts | Can capture complex patterns beyond word similarity |

hence, can we combine them together? Ratio of co-occurrence probabilities.

### GloVe model

Log-bilinear model: $w_i \cdot w_j = \log P(i|j)$ to capture ratios of co-occurrence probabilities as linear meaning components in a word vector space.

$$ J = \sum^V_{i, j = 1}f(X_{ij})(w^T_i\tilde{w_j} + b_1 + \tilde{b_j} - \log X_{ij})^2$$

where f is a function that increases around linearly and stop increasing after a certain threshold.

### Evaluate word vectors

Intrinsic:

- evaluation on a spefici / intermediate subtask
- fast to compute
- helps to understand that system
- not clear if really helpful unless correlation to real task is established

Extrinsic:

- evaluation on a real task
- can be time consuming to compute accuracy
- unclear if the subsyem is the problem or its interaction or other subsytems
- if replacing exactly one subsystem with another improves accuracy

In human languages, words are inherently carrying multiple meanings and hence can be ambiguous.

One way is to decompose a word into a few culster, each with different sense meanings, and combine the vectors with a weighted average.

## Word2Vec Tutorial

### Part1: Skip-gram model

Link to tutorial is here: [Link](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). This section only aims to record some personal thoughts while reading.

Neural Network structure:

- input: a 1 \* 10000 one-hot vector (representing the center word);
- hidden layer: 300 linear neurons which then constitue of 10000 \* 300 matrix as a lookup table such that the input vector times this matrix will yield the 1 \* 300 features corresponding to this word;
- output player: 10000 neurons with softmax classifier to compute the probability of the word at corresponding position appears near the given center word.

### Part2: Negative Sampling

Link to tutorial: [link](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

Problem: huge size of hidden layer

Solutions:

1. subsampling frequent words to decrease the number of training examples
2. modifying the optimization objective with "negative sampling", which causes each training sample to updaye only a small percentage of the model's weights.

**Subsampling** computes a probability function with two parameters: the fraction of the total words in the corpus that are that word, and the threshold value to control how much subsampling occurs. Formula in the tutorial sheet.

**Negative sampling** has each training sample only modify a small percentage of the weights, rather than all of them. We randomlly select a small number ($n$) of "negative" words to update the weights such that the output becomes 0 and also still update the weights for our "positive" word. In this way, we only updates the $n+1$ rows in the hidden layers.

We should also consider to treat phrases as a single word under specific contexts.
