import torch


def beta_binom_sample(n_sample, n, a=4.0, b=3.0):
    beta_samples = torch.distributions.Beta(a, b).sample(torch.Size([n_sample]))
    binomial_samples = torch.distributions.Binomial(n, beta_samples).sample()
    return binomial_samples.int()


def random_bool_vector(vector_size, n_true):
    """
    create vector_size boolean vector randomly
    with n_true True elements
    """
    if n_true > vector_size:
        raise ValueError("n_true cannot be greater than tensor_size")

    tensor = torch.zeros(vector_size, dtype=torch.bool)

    if n_true == 0:
        return tensor

    indices = torch.randperm(vector_size)[:n_true]
    tensor[indices] = True

    return tensor
