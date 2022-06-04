# Week4 Report

Week4: *30 May - 5 June*

## Learning Progress

- [ ] read surveys on Neural Machine Translation -- not completed, change to Video Summarization after weekly meeting
- [x] make a preliminary plan for direction to work on -- Question Answering
- [ ] ~~Implement BERT basic model during spare time (overestimate the spare time)~~
- [x] read survey on [Video Summarization](https://arxiv.org/abs/2101.06072)
- [x] read survey on [Question Answering](https://arxiv.org/abs/2007.13069) (ongoing, complete by Sunday)

## Learning Outcome

In the survey on Video Summarization, I have

- understood the problem Video Summarization: cause, application domain, output form (storyboard or video skim), and refined problem statement
- understood the current progress of research in this area:
  - *supervised* approaches which take original videos and human-generated ground-truth importance score for each frame/shot and output the predicted importance score for each frame/shot via modeling *temporal dependency* or *spatiotemporal structure* or using *Generative Adversarial Network*
  - *unsupervised* approaches which in general only take in videos as their input and either reconstructs summarized videos from selected frames or concatenate video fragments by making use of *Generative Adversarial Network*, *reinforcement learning* by targeting certain properties including *diversity, representativeness, and uniformity* or *modeling key-motionof important visual objects*
  - *semi-supervised* appraoches which make additional use of textual data as additional input besides the two in supervised learning case
  - performance measuring *metrics*, mainly based on *F1 score* as well as some extends of human evaluation -- main issue: hard to quantify and no perfect answer
  - commonly used *datasets* and how are they useful and limited -- main issue: limited size
  - learned about *performance* of above-mentioned approaches -- okay but far from good

In the survey on Question Answering, I have:

- understood the problem on Question Answering and how the SOTA models are still restricted and underperforming on certain *complex* tasks
- have general knowledge about different approaches to the problem. [TBC by Sunday when the reading is fully completed]

## Goals for Next Week

- [ ] Find relevant papers on a more refined direction and start to reproduce its work
