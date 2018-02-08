In this project, we use collapsed Gibbs sampling to implement the Latent Diritchlet Allocation (LDA) for document topic modeling. The collapsed Gibbs sampling for LDA was introduced by Griffiths et al. in the paper [Finding scientific topics](http://www.pnas.org/content/101/suppl_1/5228.full.pdf). 

The major challenge in implementing the model was in trying to rederive the conditional distribution for Gibbs sampling as given in equation 5 of the [Griffiths paper](http://www.pnas.org/content/101/suppl_1/5228.full.pdf). Reading the tutorials by [Carpenter et al.](https://lingpipe.files.wordpress.com/2010/07/lda3.pdf), [Resnik et al.](https://www.cs.umd.edu/~hardisty/papers/gsfu.pdf) and [Darling](http://u.cs.biu.ac.il/~89-680/darling-lda.pdf) helped me in this process. For my own reference, the derivation can be found in (need to include scan).

For the purpose of illustration, we consider a toy sample with the following documents:
```
Documents
d1 = [v1, v1, v1, v1, v1, v1, v2]
d2 = [v2, v1, v1, v1, v1, v1, v3]
d3 = [v3, v3, v4, v3, v3, v3, v4, v3, v2, v2, v2, v2]
d4 = [v4, v4, v4, v3, v3, v4, v4, v2, v2, v2, v2]
D = [d1, d2, d3, d4]
```
We notice that documents d1 and d2 are mde up predominantly of a single vocab: v1. On the other hand, d3 and d4 are mostly made up of v2, v3 and v4. If we consider the case of having only two topics, then for the above docs, we can argue d1 and d2 are made up of words originating mostly from a single topic whose probability is mostly on v1. On the other hand, words in d2 and d3 originate from the other topic for which the probability over vocabs are distributed evenly over v2, v3 and v4. This intuition is confirmed by looking at the following figure.

Each figure below consists of four squares, one square representing the probability of a vocab. Each column corresponds to one topic. Each panel corresponds to a given number of iterations in the Gibbs sampling algorightm. As expected, one can see that with increasing number of iterations, one topic's probability is mostly on v1 and the other topic's probability is distributed over v2, v3 and v4.  

![Convergence image](https://github.com/spookyQubit/lda_from_scratch/blob/master/images/topics_v_dist.jpg)

