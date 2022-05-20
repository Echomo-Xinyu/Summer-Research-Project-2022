# Week2 Report

Week2: *16 May - 22 May*.

## Learning Progress

- [x] Lecture9-13 Videoes and Complementary Notes

## Learning Outcome

This is a summary of Lecture 6-13 in continuation of Week1 Report summary.

- the **Language Model** (LM) problem
  - *n-gram* model and its limitation and *fixed-window-size Neural Network*
  - how **Recurrent Neural Network**(RNN) is structured *(hidden states and recurrently applying same weight matrix)* and how it is trained and applied in LM context;
  - the limitation of RNN -- *exploding*/**vanishing gradient** *(due to the existence of matrix term in gradient)*; and how we modify existing structures to have **Long Short-Term Memory** (LSTM) (additional cell content for long-term information storage) and **Gated Recurrent Unit** (GRU)
  - other mutations such as *ResNet*, *Deep Bi-Directional RNNs* and so on.
- learn about the **Machine Translation** (MT) problem
  - traditional *Statistical Machine Translation* model
  - how two RNNs are put together as **seq2seq** (one as encoder and the other as decoder) and *beam search* is applied to find desired output
  - **attention** is introduced to better contextualized output (and implicitly form soft alignment between phrases)
  - performance measured by *manual review*, *test as subcomponent for another task*, *BLEU*, etc.
- learn about the (Textual) **Question Answering** (QA) problem
  - how the components learned before (*bidirectional context*, *attention*, etc) can be combined and applied in this context
  - modifications like *attention flow layer* and *coattention layer* are made to address the QA problem
- learn about how *Computer Vision* can be linked to NLP fields
  - how *Convolutional NN* (CNN) is applied here
  - review of ideas including *kernel*, *pooling*, and *channel*
  - 1-conv against fully connected layer
  - attempt for *quasi-recurrent NN* to combine the best of LSTM and CNN
- learn about how words can be modelled at *sub-word* level -- **Byte Pair Encoding** and **FastText** embeddings
- learn about making use of contextual word representation *(instead of word2vec)* and pre-trained models: **ELMo**, **ULMfit**, and **BERT**

## Current Problems

Many of the models mentioned in summary are merely understood at the level of its motivation and core idea, without going into their technical details. Implementation of these models is needed to strengthen understanding of the topic.

## Goal for Next Week

- [ ] Lecture14-16: by the end of Week2
- [ ] Lecture17-22
- [ ] implement some of the most important models covered so far, depending on the time remaining
