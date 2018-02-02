I am still working on this project. 

In this project, we implement the Latent Diritchlet Allocation (LDA ) for document topic modeling, using collapsed Gibbs sampling, from scratch. The collapsed Gibbs sampling for LDA was introduced by Griffiths et al. in the paper [Finding scientific topics](http://www.pnas.org/content/101/suppl_1/5228.full.pdf). 

The major challenge in implementing the model was in trying to rederive the conditional distribution for Gibbs sampling as given in equation 5 of the [paper](http://www.pnas.org/content/101/suppl_1/5228.full.pdf). Reading the tutorials by [Carpenter et al.](https://lingpipe.files.wordpress.com/2010/07/lda3.pdf), [Resnik et al.](https://www.cs.umd.edu/~hardisty/papers/gsfu.pdf) and [Darling](http://u.cs.biu.ac.il/~89-680/darling-lda.pdf) helped me in this process. For my own reference, the derivation can be found in (need to include scan).

![alt text](https://github.com/spookyQubit/lda_from_scratch/blob/master/images/topics_v_dist.jpg "Logo Title Text 2")

