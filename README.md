# Linear-Support-Vector-Machine-SVM-using-Pegasos-Algorithm

# Dataset

We will use mnist subset (images of handwritten digits from 0 to 9). As before, the dataset is stored in a JSON-formated
file mnist subset.json. You can access its training, validation, and test splits using the keys
‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load mnist subset.json to the
variable x. Then, x\['train'\] refers to the training set of mnist subset. This set is a list with two
elements: x\['train'\] \[0\] containing the features of size N (samples) ×D (dimension of features), and
x\['train'\]\[1\] containing the corresponding labels of size N.

# Cautions

Please do not import packages that are not listed in the provided code. Follow the instructions in each section strictly to code up your solutions. Do not change the output format. Do not modify the code unless we instruct you to do so.

# Pegasos: a stochastic gradient based solver for linear SVM

Instead of turning linear SVM into dual formulation, we are going to solve the primal formulation directly
with a gradient-based algorithm. Note that here we include the bias term b into parameter w by
appending x with 1.

In (batch) gradient descent, at each iteration of parameter update, we compute the gradients
for all data points and take the average (or sum). When the training set is large (i.e., N is a large
number), this is often too computationally expensive (for example, a too large chunk of data cannot
be held in memory). Stochastic gradient descent with mini-batch alleviates this issue by computing
the gradient on a subset of the data at each iteration.
One key issue of using (stochastic) gradient descent is that max{0, z} is not
differentiable at z = 0. Here, you are going to learn and implement Pegasos, that applies stochastic gradient descent with mini-batch and explicitly takes care of the non-differentiable issue.


Running pegasos.sh will run the Pegasos algorithm for 500 iterations with 6 settings (mini-batch size K = 100 with different λ ∈ {0.01, 0.1, 1} and λ = 0.1 with different K ∈ {1, 10, 1000}), and output a pegasos.json that records the test accuracy and the value of objective function at each iteration during the training process.
