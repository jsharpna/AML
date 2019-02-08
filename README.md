# Advanced Machine Learning

### Prof. James Sharpnack
### Winter 2019
### Office hours: 10am-12pm MSB 4107 or by appointment

## Introduction

Machine learning has undergone dramatic changes in the past decade.  Empirical risk minimization, enabled by batch and online convex optimization, is well understood, and has been used in classification, regression, and unsupervised learning.  Recently, stochastic optimization and a suite of tools has been used for non-convex optimization, which has enabled learning functions that are the composition of other simpler functions.  This has enabled the learning of representations in deep neural networks, and has transformed the field of computer vision and natural language processing.

Decision making in Markovian environments is a separate field of research in machine learning.  Passive learning in Markov models include the hidden markov models and recurrent neural networks.  For active decision making, settings and methods range from multiarmed bandits to reinforcement learning.  All of these developments have culminated to deep Q-networks and other deep reinforcement learning techniques.  It is our ultimate goal to track these ideas from basic machine learning to deep reinforcement learning.

## How to read this syllabus

You should read this syllabus as a list of topics that will take on average 30 minutes to go over, and some topics are dependent on other topics.  The only rule for going through this material is that you must complete the dependencies before you go over a given topic.  In this way the course forms a directed acyclic graph (DAG) and we will visit each vertex in an ordering that is consistent with the DAG structure.  The format of the content section is the following:

### topic (ABV) : dependency 1, dependency 2, ...
- description
- subtopics
- references

The refences are some of my source material for a topic.  You should in no way interpret this as required reading, and the course content are the lectures that I give and the content in this repository.  

### Prerequisites:

- Linear Algebra (projections, matrix operations, singular value decomposition and spectrum)
- Calculus and probability (differentiation, Newton's method, LLN, CLT, expectation, conditioning, Bayes theorem, etc.)
- Statistical decision making (MLE, hypothesis testing, confidence intervals, linear and logistic regression)
- Computer programming (Basic data structures, sorting, big-O notation, sequential programming language such as C, Python, R, Java)

## Contents

### Intro to machine learning (IML) : 
- Look into the basic terminology of machine learning, and a preview of what is to come
- Definition of learning machine, supervised learning, loss, risk, cross-validation, basic APIs, computational complexity, OLS, KNN
- "Machine Learning", Tom Mitchell, Chapter 1

### Classification and Surrogate Loss (Class) : IML
- Classification is an important subclass of learning, and it is the first example of a computational-statistical tradeoff with surrogate losses.
- Hinge loss, logistic loss, surrogate losses, ROC and PR curves
- "A Probabilistic Theory of Pattern Recognition", Devroye et al., Chapters 1-2
- ["Surrogate losses and F-divergences"](https://arxiv.org/pdf/math/0510521.pdf), Nguyen, Wainwright, Jordan, 2009.

### Information Theory (IT) : Class
- This is a continuation of Class with a look at F-divergences and other topics in information theory
- F-divergences, differential entropy, and hypothesis testing
- "Introduction to Non-parametric Estimation", Tsybakov, Chapter 2.
- ["Information Theory, Inference, and Learning Algorithms"](http://www.inference.org.uk/itprnn/book.pdf), Mackay, Chapter 1. 
- "A Probabilistic Theory of Pattern Recognition", Devroye et al., Chapter 3

### Bias-Variance Tradeoff (BV) : IML
- We will see a bias variance tradeoff in supervised learning with a toy model, which is the sparse normal means model.  Also some strategies for model selection.
- sparse normal means, soft-thresholding, lasso, greedy selection
- ["On Minimax Estimation of a Sparse Normal Mean Vector"](https://projecteuclid.org/download/pdf_1/euclid.aos/1176325368), Iain Johnstone, 1994
- ["Elements of Statistical Learning"](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, Friedman, Chapter 3

### Directed Graphical Models and Hidden Markov Models (HMM) : IML
- We will look at directed graphical models and how to learn in Markovian environments.  In particular, we will see hidden Markov models (HMMs) and how to learn parameters in this context.
- Forward backward algorithm, EM algorithm, DAGS
- ["An Introduction to Hidden Markov Models and Bayesian Networks"](http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf), Zoubin Ghahramani.

### Convex Optimization (Conv) : IML
- We will see convex functions, subgradients, and why convexity is nice.
- convexity, subgradient, subdifferential, first order conditions
- ["Convex Optimization"](http://web.stanford.edu/~boyd/cvxbook/), Boyd and Vandenberghe, Unit 1
- ["Subgradients"](https://web.stanford.edu/class/ee364b/lectures/subgradients_notes.pdf), Boyd, Duchi, Vandenberghe.

### Online Learning (Online) : Conv
- In online learning, we read the data sequentially and need to make predictions in sequence.  We will introduce the basic idea of online convex optimization and regret and see how this fits into the ideas we have learned in IML.  We will see some basics of learning from experts.
- ["Introduction to Online Convex Optimization"](http://ocobook.cs.princeton.edu/OCObook.pdf), Hazan, Chapter 1.
- ["Prediction, learning, and games"](https://pdfs.semanticscholar.org/b791/9ff2179bea8dbe9241332fbb4137e2661825.pdf), Cesa-Bianchi, Lugosi, Introduction. 

### Online convex optimization (OCO) : Online, Class
- We will see subgradient descent algorithms, and convergence guarantees.  We will see that the perceptron is just SGD applied to the hinge loss.  
- Online optimization, subgradient descent, SGD, preceptron, regret
- ["Introduction to Online Convex Optimization"](http://ocobook.cs.princeton.edu/OCObook.pdf), Hazan, Chapters 1-3.
- ["Online convex programming and generalized infinitesimal gradient ascent"](https://dl.acm.org/citation.cfm?id=3041955), Zinkevich, 2003.

### Unsupervised Learning (UL): Conv, Class
- PCA is the first solvable non-convex programs that we will encounter.  PCA can be used for learning latent factors and dimension reduction.  We will cast it as a contrained loss minimization problem, and show it's connections to K-means clustering.
- PCA, Frobenius norm, SVD, K-means, Lloyds algorithm
- ["Elements of Statistical Learning"](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, Friedman, Chapter 14

### Kernel Machines (Kern) : Class, UL
- The kernel trick is a simple way to go from linear methods to non-linear methods, while ensuring computational feasibility.  We will see the general kernel trick and apply it to SVMs and PCA.
- Kernel trick, Kernel SVMs, Kernel PCA
- ["Kernel Methods in Machine Learning"](http://www.kernel-machines.org/publications/pdfs/0701907.pdf), Hoffman, Scholkopf, Smola, 2008.

### Decision Trees and Random Forests (Tree) : Class
- We will look at algorithms like CART and other decision trees.  We will recall the bootstrap and bagging, then go over random forests.  We will talk about random forests in the context of the bias-variance tradeoff.
- CART, bagging, random forests
- ["Elements of Statistical Learning"](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, Friedman, Chapter 15

### Boosting (Boost) : OCO
- We will go over boosting stumps and how smart ensembles of weak learners can make strong learners.  Specific attention will be paid to Adaboost and we will prove a bound on training error.  Time permitting, we will go over gradient boosting.
- Adaboost, weak learners, exponential weighting, gradient boosting
- ["Rapid Object Detection using a Boosted Cascade of Simple Features"](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf), Viola, Jones, 2001.
- ["Experiments with a New Boosting Algorithm"](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf), Freund, Schapire, 1996.
- ["Boosting notes"](https://www.cs.princeton.edu/courses/archive/fall08/cos402/readings/boosting.pdf), Schapire.
- ["Gradient tree boosting paper"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf), Friedman

### Neural Nets (NNets) : Class, Conv
- Neural networks construct non-linear functions by composing simple convex functions, which produces non-convex functions.  We will see that neural networks can learn non-linear separators like XOR, and look at how to optimize them.
- Fully connected layers, backpropagation, computation graphs
- ["Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0), Rumelhart, Hinton, Williams, 1986.
- ["Elements of Statistical Learning"](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, Friedman, Chapter 11
- ["Overfitting in Neural Nets: Backpropagation, Conjugate Gradient, and Early Stopping"](https://papers.nips.cc/paper/1895-overfitting-in-neural-nets-backpropagation-conjugate-gradient-and-early-stopping.pdf), Caruana, Lawrence, Giles, 2001.

### Non-convex Optimization (NonConv) : UL, NNets, Conv
- We will look at some examples of non-convex optimization problems in machine learning, such as neural networks and PCA.  We will see that perturbation and SGD can help non-convex optimization escape local minima.
- Local minima, Saddle points, convergence guarantees, random starts
- ["Nonconvex optimization lecture notes"](http://www.cs.cornell.edu/courses/cs6787/2017fa/Lecture7.pdf), De Sa 
- ["Escaping from Saddle Points"](https://www.offconvex.org/2016/03/22/saddlepoints/), Ge
- ["Saddles again"](https://www.offconvex.org/2016/03/24/saddles-again/), Recht.

### Deep learning and optimization (Deep) : NonConv, NNets
- We will see how deep neural nets are universal function approximators, but optimizing them can be challenging.  We will look at optimization methods, such as Nesterov acceleration.  We will also look at tricks like gradient clipping, dropouts, and variance reduction.
- ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf), Srivastava et al., 2014.
- ["On the importance of initialization and momentum in deep learning"](http://proceedings.mlr.press/v28/sutskever13.pdf), Sutskever et al., 2013
- ["Universal Approximation Bounds for Superpositions of a Sigmoidal Function "](http://www.stat.yale.edu/~arb4/publications_files/UniversalApproximationBoundsForSuperpositionsOfASigmoidalFunction.pdf), Barron, 1993.
- ["Optimization for Training Deep Models"](https://www.deeplearningbook.org/contents/optimization.html), Deep Learning, Goodfellow, Bengio, Courville.

### Recurrent Neural Nets (RNN) : Deep, HMM
- We will see that recurrent neural nets provide an alternative formulation to the HMM for prediction in Markov models.  We will look at unravelling the computation graph and the DAG implied by RNNs.
- RNN, Recurrent gradient calculation, RNN DAG, LSTM
- ["Sequence Modeling: Recurrent and Recursive Nets"](https://www.deeplearningbook.org/contents/rnn.html), Deep Learning, Goodfellow, Bengio, Courville.

### Convolutional Neural Nets (ConvNets) : Deep
- We will see how convolution can enforce parameter sharing, and look at this in the context of computer vision.  We will see how convolution can be used with fixed low level features such as SIFT features and Gabor filters.  This will lead to deep convolutional NNs.  
- Convolution, FFT, ConvNets
- ["Convolutional Networks"](https://www.deeplearningbook.org/contents/convnets.html), Deep Learning, Goodfellow, Bengio, Courville.

### Batch normalization (BatchNorm): Deep
- In deep learning, at an intermediate layer, the input distribution will shift during training, causing learning rates to needs be small.  Batch normalization is one way to get around this problem by normalizing the neuron in mini-batches.
- batch normalization
- ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167), Ioffe, Szegedy, 2015.
- ["How Does Batch Normalization Help Optimization?"](https://arxiv.org/abs/1805.11604), Santurkar et al., 2018

### Deep Autoencoders (Auto) : Deep, UL
- Much like adding layers to linear classifiers can form non-linear classifiers, autoencoders add layers to PCA to perform non-linear dimension reduction.  Time permitting we will look at variants such as sparse, and convolutional.
- autoencoders, convolutional autoencoders
- ["Autoencoders"](https://www.deeplearningbook.org/contents/autoencoders.html), Deep Learning, Goodfellow, Bengio, Courville.

### Generalized Adversarial Networks (GAN) : Auto
- GANs is another deep unsupervised learner that seeks to make it hard to distinguish between fake and training images.  This is accomplished by simultaneously learning and adversary and a generative model.
- GANs
- ["Generalized Adversarial Networks"](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al., 2014

### Bandits (Bandit) : OCO
- Our first sequential decision making setting is the multi-armed and stochastic bandit setting.  We will look at exponential weighting and UCB.
- Multi-armed bandit, EXP3, UCB
- ["Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems"](https://arxiv.org/pdf/1204.5721.pdf), Bubeck, Cesa-Bianchi, 2012, Chapters 2-3.

### Contextual Bandits (Context) : Bandit
- A more realistic setting for recommendation systems is bandits with context features.  We will look at contextual bandits in the stochastic framework and the linUCB algorithm.
- Contextual Bandits, linUCB
- ["Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems"](https://arxiv.org/pdf/1204.5721.pdf), Bubeck, Cesa-Bianchi, 2012, Chapter 4.

### Reinforcement learning (RL) : Bandit
- The bandit setting is a specific Markov decision process.  Finding a policy that can perform well in the MDP setting is the focus of reinforcement learning.  We will see the basic definitions of RL, Bellman iteration, and Monto Carlo methods.
- Bellman iteration, RL, MDP, on/off-policy
- ["Reinforcement Learning"](http://incompleteideas.net/book/the-book-2nd.html), Sutton, Barto, Chapters 3-4

### Policy gradients and REINFORCE (PGrad) : RL

### Proximal policy optimization (PPO) : PGrad, IT

### Temporal difference learning (TD) : RL

### Q-learning and SARSA (Qlearn) : TD

### Deep Q-learning (DQN) : Qlearn

- ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mihn et al.
- ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/pdf/1710.02298.pdf)


## Instruction Plan and Grading

- the scribes have been assigned, you can find [your lesson here](https://docs.google.com/spreadsheets/d/1AOxBEi1xyqoQX1aaQ1fPBTYGm2-nGLjtng7GZHPOR7Q/edit?usp=sharing)
- an example notebook can be found in [the classification notebook](example/classification.ipynb)

We may not be able to get to every topic.  My job is to present these methods and summarize the material in lecture.  Each of you will act as scribe for one of these topics, and will implement and test one of these methods.  You will create your own branch of this repo, with code, and jupyter notebooks.  You will make a pull request, I will edit it, and then we will merge to the master branch.  Once this is completed to my satisfaction then you will pass the class with an A.  If you are missing some component, or I am not satisfied with your implementations, such as it not being a serious attempt at implementing it, then you may get a B.



