# Code for IJCNN 2018 submission

### About the paper

- Title: A Simple Reservoir Model of Working Memory with Real Values
- Authors: Anthony Strock, Nicolas P. Rougier, Xavier Hinaut
- Abstract: The prefrontal cortex is known to be involved in many high-level cognitive functions, in particular, working memory. Here, we study to what extent a group of randomly connected units (namely an Echo State Network, ESN) can store and maintain (as output) an arbitrary real value from a streamed input, i.e. can act as a sustained working memory unit. Furthermore, we explore to what extent such an architecture can take advantage of the stored value in order to produce non-linear computations. Comparison between different architectures (with and without feedback, with and without a working memory unit) shows that an explicit memory improves the performances.

### Instructions to use the code

To make all the figure you should first use make in the root of the repository.
It will produce all the figure but the 4 and the 6.
For figure 4 and 6, the figures are produced in the jupyter notebooks figure4.ipynb and figure6.ipynb

To create a specific figure, for instance the figure 1, you can use make figure1 to produce it.

