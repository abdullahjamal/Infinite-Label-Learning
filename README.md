# Infinite-Label-Learning

This is the partial Implementation of Infinite Label learning using synthetic data.

First, we generate some synthetic data and randomly sample 500 training data points and 1000 testing data points from a five-component Gaussian mixture model. We also sample 10 seen labels L  and additional 2990 unseen labels U = {λ11, · · · ,λ3000} from a Gaussian distribution. We actually use Dirichlet distribution to sample mixture weights.

A groundtruth matrix V [2x3] is generated from a standard normal distribution. The label assignments are thus given by yml = sgn< Vxm,λl > for both training and testing data and both seen and unseen labels. Given the training set,the model parameters are learned by minimizing a hinge loss, and then try to assign both seen and unseen labels to 1000 test data points.

Hamming loss is used for evaluation metric.
