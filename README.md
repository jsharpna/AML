# Advanced Machine Learning

### Prof. James Sharpnack
### Winter 2019

## Introduction

Machine learning has undergone dramatic changes in the past decade.  Empirical risk minimization, enabled by batch and online convex optimization, is well understood, and has been used in classification, regression, and unsupervised learning.  Recently, stochastic optimization and a suite of tools has been used for non-convex optimization, which has enabled learning functions that are the composition of other simpler functions.  This has enabled the learning of representations in deep neural networks, and has transformed the field of computer vision and natural language processing.

Decision making in Markovian environments is a separate field of research in machine learning.  Passive learning in Markov models include the hidden markov models and recurrent neural networks.  For active decision making, settings and methods range from multiarmed bandits to reinforcement learning.  All of these developments have culminated to deep Q-networks and other deep reinforcement learning techniques.  It is our ultimate goal to track these ideas from basic machine learning to deep reinforcement learning.

## Syllabus

You should read this syllabus as a list of topics that will take on average 30 minutes to go over, and some topics are dependent on other topics.  The only rule for going through this material is that you must complete the dependencies before you go over a given topic.  In this way the course forms a directed acyclic graph (DAG) and we will visit each vertex in an ordering that is consistent with the DAG structure.  The format of the syllabus is the following:

topic: dependency 1, dependency 2, ...
- description
- subtopics
- references

