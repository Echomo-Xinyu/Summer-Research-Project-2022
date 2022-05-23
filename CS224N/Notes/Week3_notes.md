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
