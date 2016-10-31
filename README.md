# Infinite-Label-Learning

This is the torch Implementation of Infinite Label learning https://arxiv.org/abs/1608.06608

I generate some synthetic data. I randomly sample 500 training data points and 1000 testing data points from a five-component Gaussian mixture model. I also sample 10 seen labels L  and additional 2990 unseen labels U = {λ11, · · · ,λ3000} from a Gaussian distribution.I actually use Dirichlet distribution to sample mixture weights.
