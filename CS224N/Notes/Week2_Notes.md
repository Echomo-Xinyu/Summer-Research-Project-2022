# Week2 Notes

## Lecture9: Review of Lecture8 Content

The Practical Part is neglected.

Video 33:20 for linguistic data source.

adaptive shortcut connections.

The simple yet significant difference between normal RNN and LSTM+GRU is that instead of simply applying the multiplication using the same matrix, there is an "addition" step which ensures the direct linear relation between cell content at $t$ and cell content at $t-1$.

Computing all possible words' distribution is expensive. Possible approaches:

- hierarchical softmax;
- noise-contrastive estimation -- negative sample;
- train on a subset of the vocabulary at a time, test on a smart on the set of possible translations;
- attention to work out what you are translating;
- more: word pieces, char.models

Evaluation metric for MT:

- manual: adeqacy and fluency + error categorization + comparative ranking of translations;
- testing in an application that makes use of MT;
- automatic metric: BLEU, TER, METEOR, ...

BLEU4 formula: counts n-grams up to length4 and take weighted average, removing brevity penalty

overfit is not necessarily evil. Split dataset into train + dev + test.

Build up a complex model slowly. Start with a small subset and increase when the model looks good.
