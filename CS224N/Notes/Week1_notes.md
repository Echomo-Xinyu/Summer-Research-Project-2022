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

## Lectuere3: Neural Networks

### Classification setup

$$\{x_i, y_i\}^N_{i=1}$$

We then need to train softmax / logistic regression weights $W$ in order to make prediction. After this, we apply softmax function to get normalized probabilty of $y$ given input $x_i$.

**Cross-entropy loss** (derived from information theory):

$$ H(p, q) = -\sum^C_{c=1}p(c)\log q(c)$$

As we are having one-hot vector as input, the only term left is the negative log probability of the true class.

Softmax alone is merely a linear decision boundaries and hence can be quite limiting. -- high bias. That is why we introduce neural network which has strong abilities to create non-linear approximation.

**Representational learning**: representation of words are updated together with the weights in order to improve the performance of the model.

### Neural Network

$$ f(\sum_iw_ix_i + b) $$

to output the result of this neuron and $f$ is the activitatioin function like logistis regression model $\frac{1}{1+\exp^{-z}}$.

A neural netwrok has several logistic regresion ongoing. Bias is an additional row in the parameter matrix.

non-linearity is important to approximate high-order functions.

### Named Entity Recognition (NER)

Task: find and classify names in a given text.

Subproblems: classify a word in its context window of neighboring words -- concatentate word vectors to be a bigger vector and perform classification on this.

**True window**: the center word is a named entity location; **Corrupt window**: any window whose center word isn't specifically labeled as NER location in our corpus.

max-margin loss: to ensure window's score larger and corrupt window's score lower

### Matrix Calculus

Jacobian matrix: handy to use in vectorized gradients.

### Back Propogation

Chain rule.

The intermediate terms are local error signal and can be reused for $\frac{\partial z}{\partial W}$ after  $\frac{\partial z}{\partial b}$

Shape of the gradient follows the shape of parameters.

## Lecture3 Complementary Notes

[A tutorial on Back Propogation](https://cs231n.github.io/optimization-2/) provides an interesting perspective to consider the computation as circuits and hence to analyze the operation's effect on the gradient, as well as to show how temporary value storage may ease the computation process.

[Matrix calculus notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/gradient-notes.pdf) includes the proof of some basic gradient calculation examples. (the example 5 and 7 mean a too big jump for me personally) but this document may be useful to refer to.

- [ ] read differential calculus [notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/review-differential-calculus.pdf)

## Lecture4: Backpropagation

### Followup from Lecture3

$W_{ij}$ only contributes to $z_i$, which could then enable us to transform the matrix.

$$ \frac{\partial s}{\partial W_{ij}}  = \delta_i x_j$$

Hence, the overall answer is:

$$ \frac{\partial s}{\partial W} = \delta^T x^T $$

Tip:

- we may consider bundling certain synonyms together to ensure the training effect of one word is equally applied on its synonyms;
- making use of pre-trained word vectors can make training easy. But this depends on the context of training, eg when the corpus size is huge;
- depending on the size of training data set, we may consider to update (fine tune) our word vectors.

### Backpropagation

*Forward propagation* is just to evaluate the expressions in programs.

*Downstream gradient* can be computed as the produce of local gradient and upstream gradient by chain rule.

The effect of node (in an expression tree) is similar to what has been covered in Lecture3 Complementary Notes.

We should always try to make use of existing results to improve the computation efficiency. Compute in the order of reverse topological ordering.

A numeric gradient calculation can be handy in checking the whether gradient has been calculated correctly. But this would be costy in real computation.

### Additional Tips

**Regularization**: to avoid overfit, eg L2 regularization by adding $\lambda \sum_k \theta^2_k$ at the end of cost function $J$.

**Tensorisation** or *matrixization* or *vectorization* can improve the calculation speed significantly compared with using for loops.

**Non-linearities**:

- logistic(sigmoid) vs $\tanh$ vs hard $\tanh$: $\tanh$ is a rescaled and shifted sigmoid from $-1$ to $1$. Hard tanh flatters the slops and makes the computing cheaper;
- *ReLU* (rectified linear unit): $\text{rect}(z) = \max(z, 0)$ which trains quickly and performs well due to good gradient backflow. A mutant is *leaky ReLU* to have $y=0.01x$ over the negative side.

**parameter initialization**: we need to avoid symmetries that prevent learning

**Optimizer**: plain SGD works but learning rate needs to be tuned. A family of "adaptive" optimizers that scale the parameter adjustment by an accumulated gradient such as Adam, Adagrad, RMSprop, etc.

**learning rate**: init around 0.001. The order of magnitude must be right. Better results can generally be obtained by allowing learning rates to decrease as you train.

## Lecture4 Complementary Notes

### CS231N Notes on netwrok architecture

**Sigmoid** activation function's drawbacks:

1. close to zero local gradient at extreme values which would make the backpropagation close to zero;
2. output not zero-centered will introduce a zig-zag dynamics in the gradient updates. Although this issue has little effect on the final update on the weights.

Discussion on **ReLU**:

1. greatly accelerate the convergence of SGD due to its linear, non-saturating form (not definitive)
2. simple and cheap computing process compared with exponentials in tanh and sigmoid
3. a poorly set learning rate can effectively make a signficant portion of the units zero, as the negative side of ReLU function has zero gradient

To address the issue of point 3, **leaky ReLU** is used. However, the consistency of teh benefit across tasks is presently unclear.

**Maxout**: a generalized version of ReLU by computing $\max(w^T_1x+b_1, w^T_2x+b_2)$. (ReLU is to have $w^T_1$ and $b_1$ be zero). This would introduce a doubling of parameters however.

The Neural Neywork with at least one hidden layer are *universal approximators*. This means that neural networks with a single hidden layer can be used to approximate any continuous function to any desired precision.

**Regularization** is preferred to address the issue of overfitting.

### Derivatives, Backpropagation, and Vectorization

This [file](http://cs231n.stanford.edu/handouts/derivatives.pdf) is a very clear introduction from basic single-variable derivative to Jacobian and then derivatives related to tensors.

**Tensor** in machine learning is a D-dimentional grid of numbers. Suppose $f: R^{N_1 \times \dots \times N_{D_x}} \rightarrow R^{M_1 \times \dots \times M_{D_y}}$, the derivative $\frac{\partial y}{\partial x}$ is a generalized Jacobian and is of shape $(M_1 \times \dots \times M_{D_u}) \times (N_1 \times \dots \times N_{D_x})$. We can think of the generalized Jacobian as generalization of a matrix, where each "row" has the same shape as $y$ and each "column" has the same shape as $x$.

$$ x \rightarrow x + \Delta x \Rightarrow y \rightarrow \approx y+\frac{\partial y}{\partial x} \Delta x$$

Explicitly constructing Jacobian can be very costy in memory. Given $y = f(x, w) = wx$ and assuming we are given $\frac{\partial L}{\partial y}$, we can then play around the Math and derive $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}x^T$. By my personal derivation, $\frac{\partial L}{\partial w} = x^T\frac{\partial L}{\partial y}$.

## Lecture5

### Syntactic Structure

Constituency = phrase structure grammar = context-free grammars: words - phrases - bigger phrases -- dominated in linguistics

Dependency sturcture: shows which words depend on (modify or are arguments of) which other words

A few inherent issue with English grammar

- prepositional phrase attachment ambiguity -- Potential exponential number of possible combinations. (Catalan numbers)
- Coordination scope ambiguity
- verb phrase attachment ambiguity

### Dependency Grammar and Treebanks

Dependency graph is eventually a connected, single-rooted directed acyclic graph.

Treebank: annotated universal dependency data.

A history of phrasing:

1. Dynamic Programming: before 2000s, cubic or even more complexity
2. Greedy deterministic transition-based shift-reduced parsing: linear time complexity: put current situation as input to neural network and decide the next actions. repeat this process.
3. Neural dependency parser: to address issue of 2: sparse + incomplete labels + expensive computation, why not just put the whole stack and buffers into neural network?

## Lecture6: Language Models and RNNs

Language modeling is the task of predicting what word comes next: $P(x^{(t+1)} | x^{(t)}, \dots, x^{(1)})$ where $x^{(t+1)}$ can be any word in the vocabulary $V  = \{w_1, \dots, w_{|V|}\}$.

The probability of a given text is essentially the product of all the word given the predecessor words.

### N-gram Language Model

*n-gram* is a chunk of $n$ consecutive words.

Idea: collect statistcis about how frequent different n-grams are and then make predictions.

Simplifying Assumptions: $x^{t+1}$ depends only on the preceding $n-1$ words.

Data is generated by counting them in some large corpus.

Limitation:

- the full context may be beyond the $n-1$ words.
- highly dependent on the corpus -- only make predictions based on existing data (sparsity problem: correct result never appear)

**smoothing**: adding a small probability for all zero-probability result to address sparasity problem

*back-off*: if n-gram cannot be found, fall back to n-1 gram model.

choice of n must be small to avoid sparsity problm.

The text generated is incoherent. We need to consider more than three words at a time if we want to model language well. But increasing n worsens sparsity problem and increases model size.

### Machine Learning approach

A fixed-window neual language model: similar to n-gram model (discard words outside window), concatenate word embeddings and compute distribution.

Improvements:

- no sparsity problem
- memory saving

Problems:

- window size insufficient
- large window size enlarges weights in NN
- learning process is repeated for four columns in the weight matrix of NN because of dot product.

Hence, we need a neural network structure that accept variable-length input.

### Recurrent Neural Networks (RNN)

We have as many hidden states as the length of the input. Each hidden state is computed based on previous hidden state and corresponding input.

Core idea: apply the same weights W repeatedly.

Initial hidden state can be something we learned or simply zeros.

advantages:

- any length input
- computation for step t can (in theory) use information from many steps back
- model size doesn't increase for longer input
- same weights applied on every timestep, so there is symmetry in how inputs are processed.

disadvantages:

- recurrent computation is slow -- cannot be parallelised
- difficult to access information from many steps back in practice.

#### How to train a RNN language model?

- get a big corpus of text
- compute output distribution for every step t
- loss function on step t is cross-entropy between predicted probability and the true next word
- average this to get overall loss for entire training set

computing loss and graidents across entire corpus is too expensive, hence in practice, we consider the vectors of words as a sentence

Bakpropagation: gradient is the sum of the gradient at each step. (by multivariable chain rule). *Backpropagation through time*: simply summing up along the way back.

Generating text with a RNN language model: put in one word and continuously put existing words to get the next.

perplexity: the standard evaluation metric for language models., which is equal to the exponential of the cross-entropy loss. (lower is better)

##### Why do we care about language modeling

1. Language modeling is a benchmark task that helps us measure our progress on understanding language
2, it is a subcomponent of many NLP tasks -- predictive typing, speech recognition, grammar correction

RNN can be used in a wide array of way. One impressive one is an encoder module.

Vanilla RNN: RNN covered here. There may be other flavors like GRU, LSTM, multi-layer RNNs

## Lecture7: Vanishing Gradients + Fancy RNNs

### Problems with RNNs and how to fix them

**Vanishing gradient problem**: when intermediate gradients are small, the gradient signal gets smaller and smaller as it backpropagates further.

In formal definition, when $W_h$ is small (related to largest exponential), its exponenial index $i-j$ will affect its value significantly -- exponentially small.

This problem implies that the *gradient signal from faraway* is lost because it's much smaller than *gradient signal from close-by*. So model weights are only updated only with respect to *near effects*, not *long-term effects*.

**Question**: why are we interested in $\frac{\partial J}{\partial h}$ while we are treating $h$ as an activation instead of a weight we are updating?

one temporary answer is that when we treat $h_0$ as trainable vectors.

Gradient can be viewed as a measure of the effect of the past on the future. If the gradient becomes vanishingly small over longer distances, then we cannot tell whether there is no dependency between them or we are having wrong parameters to capture the true dependency over this distance.

example of a such problem: The writer of the books _, choose between is and are. Sequential recency is preferred over syntactic recency.

There is a similar problem of *exploding gradients* when largest eigenvalue is more than 1. -- during gradient descent, the update may be too huge and fail to obtain good results, resulting in Inf or NaN in network.

To address exploding fgradient, gradient clipping: if the norm of the gradient is greater than some threshold, scale it down before applying SGD update. (take a step in the same direction, but a smaller step)

To address the vanishing gradient, what about some extra memory? -- motive of Long Short-Term Memory.

On step $t$, there is a hidden state and a cell state. We introduce forget gate, input gate and output gate in order to determine what information is retained and passed on from the hidden state and cell state from step $t-1$. This is clearly illustrated by the LSTM diagram on slides.

**Question**: why is the forget function not include the cell content from the previous step in its computation? -- one possible answer is that the hidden state from the previous step contains some parts of the cell content.

This atchitecture makes it easier to preserve information from many steps earlier. -- on extreme case, forget gate can choose to never forget anything.

**Gated Recurrent Units** (GRU): to reduce complexity of LSTM while achieving same functionality.

Gate control for information flow. Update gate (fortget + input) + reset gate (select) to have only a new hidden state.

Difference between LSTM and GRU:

GRU is quicker to compute and has fewer parameters;

no conclusive evidence that one consistently performs better than the other. LSTM is a good default choice, switch if efficiency is valued.

vanishing/exploding gradient is not just a RNN problem, especially deep ones (feed-forward and convolutional) -> lower layers are learned very slowly.

solution: add moredirect connections (thus allowing the gradient to flow). eg residual connections (skip-connections) - skip layers to preserve information by default.

dense connections ie DenseNet, directly connect everything together.

highway connections ie HighwayNet: similar to residual connections, but identity connection vs the transformation layer is controlled by a dynamic gate.

RNNs are particularly unstable due to the repeated multiplication by the *same* weight matrix.

### RNN variants

Bidirectional RNNs

motivation: considers the context of both sides.

go from left and right and concatenate the two hidden states and use it for training.

This is not applicable for Language Modeling as we only have left context in LM task. Bidirectional RNNs assume we have access to entire input sequence.

Multi-layer RNNs (stacked RNNs)

allow the network to compute more complex representations. Higher RNNs should compute higher-level features.

High-performing RNNs are often multi-layer (but are not as deep as convolutional or feed-forward networks)

## Lecture8: Translation, Seq2Seq, Attention

### Machine Translation

**Machine Translation**: the task of translating a sentence $x$ from one language to a sentence $y$ in another language.

Early Machine Translation: *rule-based*, using a bilingual dictionary to map source language to the target language.

Statistical Machine Translation: learn a probabilistic model from data. find best target language phrase, given the source language phrase. This can be converted into a combination of Language Model and Translation Model.

Parallel data (pairs of human-translated target / source sentences)

alignment: one-to-one, one-to-none, one-to-many, many-to-many, etc. *fertile word*: the one word that corresponds with many words from the other language.

to learn $\argmax_yP(x|y)P(y)$

enumerate every possible y and calculate the probability is too expensive -- use a heuristic search algorithm to search for the best translation, discarding hypotheses that are too low-probability (this process is called *decoding*).

the best systems were extremely complex and had many separately-designed subcomponents. -- significant human effort.

### Neural Machine Translation

do all the work with a single neural network. sequence-to-sequence, which involves two RNN.

*Encoder RNN* produces an encoding of the source sentence, which eventually generates initial hidden state for Decoder RNN. *Decoder RNN* is a Language Model that generates target senetnce, *conditioned* on *encoding*.

Training end-to-end -- pre-training might be considered.

**Greedy decoding**: local optimal is not necessarily global optimal. -- this may be addressed with a search algorithm

**beam search decoding**: keep track of the *k most probable* partial translations (hypotheses). k - *beam size*. This is not guaranteed to find optimal solution but much more efficient.

**stopping criterion**: in greedy, decode until END token; in beam search decoding, when a hypothesis produces END, place it aside and continue exploring others. we may continue until reach certain threshold timestep or reach a threshold number of complete hypothesis.

Problem: longer hypotheses are likely to have lower scores. -- fix by normalizing by length.

advantages:

- better performance - more fluent, better use of context, better use of phrase similarities
- a single NN to be optimized end-to-end rather than work with many subcomponents
- less human engineering effort - no feature engineering + same method for all language pairs

disadvantages:

- NMT less interpretable - hard to debug
- difficult to control - hard to impose certain rules + safety concerns

metrics to evaluate progress: **BLEU** (Bilingual Evaluation Understudy): compute a similarity score based on *n-gram precision* + *short sentence penalty* between machine-written translation to human-written translation.

Problem with BLEU: useful but imperfect -- many possible valid ways to translation.

difficulties remain:

- out-of-vocab words
- domain mismatch between train and test data
- maintain context over longer text
- low-resource language pairs
- common sense
- prejudice
- uninterpretable systems do strange things -- nonsensible input generate essentially noise and hence the decoder is not conditioned
- the list goes on

NLP tasks including summarization, dialogue, parsing, code generation can be phrased as seq2seq.

### Attention

motivation: informational bottleneck as we encode the source sentence to a single vector that includes all information of the sentence.

On each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence by computing a attention distribution which would then produce attention output.

A soft alignment is achieved by the attention distribution.

Q: implication of a change in vector?

Advantages:

- improves NMT performance
- solve the bottleneck problem
- helps with vanishing gradient problem
- provide some interpretability (softa alignment for free)

Attention is a general DL technique: attention is a technique to compute a weighted sum of the values ( a set of vector) dependent on the query (a vector) - query attends to the values.
