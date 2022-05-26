# Week3 Notes

## Lecture15: Natural Language Generation (NLG)

**NLG**: any setting in which we generate *new* text.

**Teacher Forcing**: using the designated result for training regardless of the current training output.

effect of changing beam size $k$:

- small k leads to greedy decoding
- increase k reduces "ungrammatical, unnatural, nonsensical incorrect" cases but more computationally expensive;
- for NMT, increasing k too much may decrease BLEU score -- no strong link between the optimal result of high prob and the BLEU score
- open-ended tasks, large k can make output more generic -- safe, "coorect" response

**sampling-based decoding**: *pure sampling* (randomly sample from probability distribution) vs *top-n sampling* (randomly choose from top-n most probable words)

**Softmax temperature**: divide probability by the *temperature hyperparameter* $\tau$ before applying the exponential function. -- a larger $\tau$ will make distribution becomes more uniform and hence more diverse output. -- a technique that to control safety vs risky

**summarization**: given input text $x$, write a summary $y$ which is shorter and contains the main information of $x$.

single-file vs multiple-file

datasets can be of different lengths and styles: headlines, sentence summary, brief, etc.

*sentence simplification* is a different but related task: rewrite the source text in a simpler way

*extractive summarization* (select parts) vs *abstractive summarization* (generate new text)

**ROUGE**: Recall-Oriented Understudy for Gisting Evaluation --  based on n-gram overlap without brevity penalty an dis based on recall instead of precision.

*bottom-up summarization*: overall content selection and less copying of long sequences

## Lecture16: co-reference resolution

Coreference remains unclear in many NLP's tasks.

*Coreference* is when two mentions refer to the same entity in the world. *Anaphora*: a term (anaphor) refers to another term (antecedent). *Obama* said *he* ...; *cataphora*: similar to anaphora but the pronouns comes before he antecedent.

rule-based:

- Hobb's naive algorithm (baseline). find nearby noun phrases and search from low to high promixity.
- knowledge-based pronominal coreference: *Winograd Schema*
  - She poured water from the pitcher into the cup unitl *it* was full;
  - She poured water from the pitcher into the cup until *it* was empty

mention pair:

- train the model to predict only one antecedent for each mention instead of all of them.

mention ranking:

add a dummy NA mention at the beginning to allow the model to decline linking the current mention to anything

clustering:

progressive clustering -- cluster-pair decision is easier than mention-pair decision

evaluation metric: b-cubed and more, consider both precision and recall (to avoid single clustering and generalized clustering).

## Lecture17: Multi-tasking

a unified paradigm to unite different tasks into one task (or do pre-train for the whole model)

*curriculumn learning*: learn tasks from simple to difficult.

## Lecture18: TRNN + Constitutency Parsing

building on vector space -- the meaning of a phrase should be determined by its components and such that the phrase can be fit into the vector space.

recursive neural nets vs RNN

greedy approach to merge the components.

in order to introduce composition (very is a function for the word after it eg), we  take each word as two representations (a vector and a matrix) -- recursive matrix-vector model.
