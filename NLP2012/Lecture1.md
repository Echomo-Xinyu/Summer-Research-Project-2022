# Lecture 1

Analysed the sentiment and aggregate the features, machine translation
summarization and answer questions

Crash blossoms - two meanings

```math
[A-Z] all capital letters
[^S] negation S
| disjunction a|b|c == [abc]
? pervious character optional
* 0 or more of previous char
+ 1 or mre previous char
. any character
^ begining of line
$ end of line
\ escape character
```

Word tokenisation - text normalisation
N = number of teens
V = set of words
Standardisation of word break down
Specific issues in different languages
Max-match: given a wordlist of Chinese and find the longest word that match the dictionary - greedy algorithm - doesn’t work well in English

Normalisation:

- make indexed text have same form
- implicit define equivalent classes
- asymmetric expansion
- reduce all letters to lower case (depends)
- reduce inflections or variant forms to base form — lemmatisation
- morphemes: the small meaningful units that make up words
  - stems: the core meaning-bearing units — stemming
  - affixes: bits ad pieces that adhere to stems

Porter’s algorithm: replace certain parts into simple forms

Sentence segmentation - build a binary classifier (decision tree eg) to decide the functionalities of an ambiguous symbol like “.” (whether end of sentence eg)
