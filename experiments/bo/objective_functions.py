import torch
import numpy as np


def unnorm(val, val_lb, val_ub, new_lb, new_ub):
    """
    function to unnormalize inputs from [val_lb, val_ub] ([[-1], [1]] for our standard setting)
    to [new_lb, new_ub] (domain of the true function)

    val [N, D]
    val_lb, val_ub, new_lb, new_ub [D]
    """
    unnormalized = ((val - val_lb) / (val_ub - val_lb)) * (new_ub - new_lb) + new_lb
    return unnormalized

# 1D function ----------------------------------------------------------------------------
gramacy_lee_1d = {
    "name": "gramacy lee",
    "func": lambda x, unnormalize: np.sin(10 * np.pi * unnormalize(x))
    / (2 * unnormalize(x))
    + (unnormalize(x) - 1) ** 4,
    "bounds": torch.tensor([[0.5], [2.5]], dtype=torch.float32),
    "optimum": -0.8690111349894923,
    "optimizer": torch.Tensor([0.5485634404856344]),
}

multimodal_f1 = {
    # for illustrative purposes
    "name": "multimodal f",
    "func": lambda x, unnormalize: -(1.4 - 3.0 * unnormalize(x))
    * torch.sin(18.0 * unnormalize(x)),
    "bounds": torch.tensor([[0.1], [1.2]], dtype=torch.float32),
}


ackley_1d = {
    "name": "ackley1d",
    "func": lambda x, unnormalize: -20
    * np.exp(-0.2 * np.sqrt(0.5 * unnormalize(x) ** 2))
    - np.exp(0.5 * np.cos(2 * np.pi * unnormalize(x)))
    + 20
    + np.e,
    "bounds": torch.tensor([[-5.0], [5.0]], dtype=torch.float32),
    "optimum": 1,
    "optimizer": torch.Tensor([0]),
}

neg_easom_1d = {
    "name": "neg easom",
    "func": lambda x, unnormalize: np.cos(unnormalize(x))
    * np.exp(-((unnormalize(x) - np.pi) ** 2) / 2),
    "bounds": torch.tensor([[-10.0], [10.0]], dtype=torch.float32),
    "optimum": -2,
    "optimizer": torch.Tensor([3.141592731415928]),
}


# 2D function ----------------------------------------------------------------------------
ackley_2d = {
    "name": "ackley2d",
    "func": lambda x, unnormalize: -20
    * np.exp(
        -0.2 * np.sqrt(0.5 * (unnormalize(x)[:, 0] ** 2 + unnormalize(x)[:, 1] ** 2))
    )
    - np.exp(
        0.5
        * (
            np.cos(2 * np.pi * unnormalize(x)[:, 0])
            + np.cos(2 * np.pi * unnormalize(x)[:, 1])
        )
    )
    + np.e
    + 20,
    "optimum": 0,
    "bounds": torch.tensor([[-5, -5], [5, 5]], dtype=torch.float32),
    "optimizer": torch.Tensor([0.0, 0.0]),
}

branin_scaled_2d = {
    "name": "branin_scaled",
    "func": lambda x, unnormalize: (1 / 51.95)
    * (
        (
            (unnormalize(x)[:, 1] * 15.0)
            - (5.1 / (4 * np.pi**2)) * (unnormalize(x)[:, 0] * 15.0 - 5.0) ** 2
            + (5 / np.pi) * (unnormalize(x)[:, 0] * 15.0 - 5.0)
            - 6
        )
        ** 2
        + 10 * (1 - (1 / (8 * np.pi))) * np.cos(unnormalize(x)[:, 0] * 15.0 - 5.0)
        + -44.81
    ),
    "bounds": torch.tensor([[0, 0], [1, 1]], dtype=torch.float32),
    "optimum": -1.047393,
    "optimizer": torch.Tensor([0.9617, 0.1650]),
}

michalewicz_2d = {
    "name": "michalewicz",
    "func": lambda x, unnormalize: -(
        np.sin(unnormalize(x)[:, 0]) * np.sin((unnormalize(x)[:, 0] ** 2) / np.pi) ** 20
        + np.sin(unnormalize(x)[:, 1])
        * np.sin(2 * (unnormalize(x)[:, 1] ** 2) / np.pi) ** 20
    ),
    "optimum": -1.8013034,
    "bounds": torch.tensor([[0, 0], [np.pi, np.pi]], dtype=torch.float32),
    "optimizer": torch.Tensor([2.20, 1.57]),
}


# 3D function ----------------------------------------------------------------------------


def hartmann_3(x, unnormalize):
    """
    The Hartmann 3 test function over [0, 1]^3. This function has 3 local
    and one global minima. See https://www.sfu.ca/~ssurjano/hart3.html for details.

    :param x: The points at which to evaluate the function, with shape [N, 3].
    :param unnormalize: A function to unnormalize the input `x`.
    :return: The function values at `x`, with shape [N, 1].
    :raise ValueError: If `x` has an invalid shape.
    """
    x_unnormalized = unnormalize(x)
    a = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float32)
    A = torch.tensor(
        [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]],
        dtype=torch.float32,
    )
    P = torch.tensor(
        [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0381, 0.5743, 0.8828],
        ],
        dtype=torch.float32,
    )

    inner_sum = -torch.sum(A * (x_unnormalized.unsqueeze(1) - P) ** 2, dim=-1)
    res = -torch.sum(a * torch.exp(inner_sum), dim=-1)

    return res


hartmann3 = {
    "name": "hartmann3d",
    "func": hartmann_3,
    "bounds": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32),
    "optimum": -3.86278,
    "optimizer": torch.Tensor([0.114614, 0.555649, 0.852547]),
}


levy3d = {
    "name": "levy3d",
    "func": lambda x, unnormalize: (
        np.sin(np.pi * (1 + (unnormalize(x)[:, 0] * 10 - 5 - 1) / 4)) ** 2
        + (
            (1 + (unnormalize(x)[:, 0] * 10 - 5 - 1) / 4 - 1) ** 2
            * (
                1
                + 10
                * np.sin(np.pi * (1 + (unnormalize(x)[:, 0] * 10 - 5 - 1) / 4) + 1) ** 2
            )
            + (1 + (unnormalize(x)[:, 1] * 10 - 5 - 1) / 4 - 1) ** 2
            * (
                1
                + 10
                * np.sin(np.pi * (1 + (unnormalize(x)[:, 1] * 10 - 5 - 1) / 4) + 1) ** 2
            )
        )
        + (1 + (unnormalize(x)[:, 2] * 10 - 5 - 1) / 4 - 1) ** 2
        * (1 + np.sin(2 * np.pi * (1 + (unnormalize(x)[:, 2] * 10 - 5 - 1) / 4)) ** 2)
    ),
    "optimum": 0,
    "bounds": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32),
    "optimizer": torch.Tensor([11./20, 11./20, 11./20]),
}

ackley_3d = {
    "name": "ackley",
    "func": lambda x, unnormalize: ((-20 * torch.exp(
        -0.2 * torch.sqrt(0.333 * (unnormalize(x)[:, 0]**2 + 
                                   unnormalize(x)[:, 1]**2 + 
                                   unnormalize(x)[:, 2]**2))
    ) - torch.exp(
        0.333 * (torch.cos(2 * torch.pi * unnormalize(x)[:, 0]) + 
                 torch.cos(2 * torch.pi * unnormalize(x)[:, 1]) + 
                 torch.cos(2 * torch.pi * unnormalize(x)[:, 2]))
    ) + 20 + torch.exp(torch.tensor(1.0)))),  # torch.e equivalent
    "bounds": torch.tensor([[-32.768, -32.768, -32.768], [32.768, 32.768, 32.768]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0, 0.0, 0.0]),
}
def hartmann_4(x, unnormalize):
    """
    https://www.sfu.ca/~ssurjano/Code/hart4r.html
    """
    x = unnormalize(x)
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float32)
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ], dtype=torch.float32)
    P = 1e-4 * torch.tensor([
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]
    ], dtype=torch.float32)
    # Add a batch dimension for broadcasting [N, 4] -> [N, 4, 1]
    x = x.unsqueeze(1)  # [N, 1, 4]
    # Compute inner term
    inner = torch.sum(A[:, :4] * (x - P[:, :4].unsqueeze(0)) ** 2, dim=2)  # [N, 4]
    # Compute outer term
    outer = torch.sum(alpha * torch.exp(-inner), dim=1)  # [N]
    # Final function value
    y = (1.1 - outer) / 0.839
    return y

hartmann_4d = {
    "name": "hartmann_4",
    "func": hartmann_4,
    "bounds": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    "optimum": -3.1189,
    "optimizer": torch.tensor([0.1749, 0.2138, 0.5415, 0.2634]),
}

rosenbrock_4d = {
    "name": "rosenbrock",
    "func": lambda x, unnormalize: torch.sum(
        100 * (unnormalize(x)[:, 1:] - unnormalize(x)[:, :-1]**2)**2 + (1 - unnormalize(x)[:, :-1])**2, dim=-1
    ),
    "bounds": torch.tensor([[-5.0, -5.0, -5.0, -5.0], [5.0, 5.0, 5.0, 5.0]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([1.0, 1.0, 1.0, 1.0]),
}

xt = lambda x: 15*x - 5
rosenbrock_4d_picheny = {
    "name": "rosenbrock",
    "func": lambda x, unnormalize: ((torch.sum(
        (100 * (xt(unnormalize(x))[:, 1:] - xt(unnormalize(x))[:, :-1]**2)**2 + (1 - xt(unnormalize(x))[:, :-1])**2), dim=-1
    ))- 3.827 * 1e5) / (3.755 * 1e5),
    "bounds": torch.tensor([[0., 0., 0., 0.], [1.0, 1.0, 1.0, 1.0]]),
    "optimum": -1.01917,
    "optimizer": torch.tensor([0.4, 0.4, 0.4, 0.4]),
}

ackley_4d = {
    "name": "ackley",
    "func": lambda x, unnormalize: (
        -20 * torch.exp(-0.2 * torch.sqrt(0.25 * torch.sum(unnormalize(x)**2, dim=-1)))
        - torch.exp(0.25 * torch.sum(torch.cos(2 * torch.pi * unnormalize(x)), dim=-1))
        + 20 + torch.exp(torch.tensor(1.0))
    ),  
    "bounds": torch.tensor([[-32.768, -32.768, -32.768, -32.768], [32.768, 32.768, 32.768, 32.768]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0, 0.0, 0.0, 0.0]),
}


def shekel_4(x, unnormalize):
    x_unnormalized = unnormalize(x)
    C = torch.tensor([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
    ], dtype=torch.float32).T
    beta = torch.tensor([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=torch.float32)

    
    inner_sum = torch.sum((x_unnormalized.unsqueeze(1) * 10.0 - C) ** 2, dim=-1)
    res = -torch.sum(1 / (inner_sum + beta), dim=-1)
    return res

shekel_4d = {
    "name": "shekel_4",
    "func": shekel_4,
    "bounds": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    "optimum": -10.5363,
    "optimizer": torch.tensor([0.4, 0.4, 0.4, 0.4]),
}


rosenbrock_5d = {
    "name": "rosenbrock_5",
    "func": lambda x, unnormalize: (
        (
            torch.sum(
                100 * (15 * unnormalize(x)[:,1:] - 5 - (15 * unnormalize(x)[:,:-1] - 5) ** 2) ** 2 +
                (1 - (15 * unnormalize(x)[:,:-1] - 5)) ** 2
            ,dim=-1) - 3.827 * 10 ** 5
        ) / (3.755 * 10 ** 5)
    ),
    "bounds": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
    "optimum": -1.0195,  #1M evaluations
    "optimizer": torch.tensor([0.3782, 0.3626, 0.3654, 0.3720, 0.3635]),  #1M evaluations
}

levy_5d = {
    "name": "levy_5",
    "func": lambda x, unnormalize: (
        torch.sin(torch.pi * (1 + (unnormalize(x) - 1) / 4))**2 +
        torch.sum(((1 + (unnormalize(x) - 1) / 4) - 1)**2 * (1 + 10 * torch.sin(torch.pi * (1 + (unnormalize(x) - 1) / 4))**2), dim=-1) +
        (((1 + (unnormalize(x)[:, -1] - 1) / 4) - 1)**2) * (1 + torch.sin(2 * torch.pi * (1 + (unnormalize(x)[:, -1] - 1) / 4))**2)
    ),
    "bounds": torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0], [10.0, 10.0, 10.0, 10.0, 10.0]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
}

def levy_5(x, unnormalize):
    z = 1 + ((unnormalize(x) - 1) / 4)
    term1 = torch.sin(torch.pi * z[:, 0]) ** 2
    term2 = torch.sum((z[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * z[:, :-1] + 1) ** 2), dim=-1)
    term3 = (z[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * z[:, -1]) ** 2)
    return term1 + term2 + term3

levy_5d = {
    "name": "levy_5",
    "func": levy_5,
    "bounds": torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0], [10.0, 10.0, 10.0, 10.0, 10.0]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
}
griewank_5d = {
    "name": "griewank_5",
    "func": lambda x, unnormalize: (
        torch.sum(unnormalize(x) ** 2, dim=-1) / 4000.0 - 
        torch.prod(torch.cos(unnormalize(x) / torch.sqrt(torch.arange(1, 6, device=x.device).float())), dim=-1) + 1
    ),
    "bounds": torch.tensor([[-600, -600, -600, -600, -600], [600, 600, 600, 600, 600]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
}

rastrigin_5d = {
    "name": "rastrigin_5",
    "func": lambda x, unnormalize: (
        10 * 5 + torch.sum(unnormalize(x) ** 2 - 10 * torch.cos(2 * torch.pi * unnormalize(x)), dim=-1)
    ),
    "bounds": torch.tensor([[-5.12, -5.12, -5.12, -5.12, -5.12], [5.12, 5.12, 5.12, 5.12, 5.12]]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
}


def hartmann_6(x, unnormalize):
    """
    https://www.sfu.ca/~ssurjano/Code/hart6scr.html
    """
    x = unnormalize(x)
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float32)
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ], dtype=torch.float32)
    P = 1e-4 * torch.tensor([
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]
    ], dtype=torch.float32)
    # Add a batch dimension for broadcasting [N, 6] -> [N, 4, 1]
    x = x.unsqueeze(1)  # [N, 1, 6]
    # Compute inner term
    inner = torch.sum(A[:, :6] * (x - P[:, :6].unsqueeze(0)) ** 2, dim=2)  # [N, 4]
    # Compute outer term
    outer = torch.sum(alpha * torch.exp(-inner), dim=1)  # [N]
    # Final function value
    y = -outer
    return y

hartmann_6d = {
    "name": "hartmann_6",
    "func": hartmann_6,
    "bounds": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
    "optimum": -3.32237,
    "optimizer": torch.tensor([0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.6573]),
}

def ackley_6(x, unnormalize, d=6):
    """
    Implements the 6-dimensional Ackley function.
    """
    x = unnormalize(x)
    a = 20
    b = 0.2
    c = 2 * torch.pi

    term1 = -a * torch.exp(-b * torch.sqrt(torch.sum(x**2, dim=1) / d))
    term2 = -torch.exp(torch.sum(torch.cos(c * x), dim=1) / d)
    y = term1 + term2 + a + torch.exp(torch.tensor(1.0))
    return y

ackley_6d = {
    "name": "ackley_6",
    "func": ackley_6,
    "bounds": torch.tensor([[-32.768] * 6, [32.768] * 6]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0] * 6),
}

def levy_6(x, unnormalize):
    """
    Implements the 6-dimensional Levy function.
    """
    x = unnormalize(x)
    w = 1 + (x - 1) / 4
    term1 = torch.sin(torch.pi * w[:, 0]) ** 2
    term2 = torch.sum((w[:, :-1] - 1)**2 * (1 + 10 * torch.sin(torch.pi * w[:, :-1] + 1)**2), dim=1)
    term3 = (w[:, -1] - 1)**2 * (1 + torch.sin(2 * torch.pi * w[:, -1])**2)
    y = term1 + term2 + term3
    return y

levy_6d = {
    "name": "levy_6",
    "func": levy_6,
    "bounds": torch.tensor([[-10.0] * 6, [10.0] * 6]),
    "optimum": 0.0,
    "optimizer": torch.tensor([1.0] * 6),
}

def griewank_6(x, unnormalize):
    """
    Implements the 6-dimensional Griewank function.
    """
    x = unnormalize(x)
    d = x.size(1)  # Dimensionality
    term1 = torch.sum(x**2, dim=1) / 4000
    term2 = torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, d + 1, dtype=torch.float32, device=x.device))), dim=1)
    y = term1 - term2 + 1
    return y

griewank_6d = {
    "name": "griewank_6",
    "func": griewank_6,
    "bounds": torch.tensor([[-600.0] * 6, [600.0] * 6]),
    "optimum": 0.0,
    "optimizer": torch.tensor([0.0] * 6),
}

benchmark_dict = {
    "ackley1d": ackley_1d,
    "gramacylee1d": gramacy_lee_1d,
    "negeasom1d": neg_easom_1d,
    "ackley2d": ackley_2d,
    "braninscaled2d": branin_scaled_2d,
    "michalewicz2d": michalewicz_2d,
    "hartmann3d": hartmann3,
    "levy3d": levy3d,
    "ackley3d": ackley_3d,
    "ackley4d": ackley_4d,
    "hartmann4d": hartmann_4d,
    "rosenbrock4d": rosenbrock_4d_picheny,
    "shekel4d": shekel_4d,
    "rosenbrock5d": rosenbrock_5d,
    "levy5d": levy_5d,
    "griewank5d": griewank_5d,
    "rastrigin5d": rastrigin_5d,
    "hartmann6d": hartmann_6d,
    "ackley6d": ackley_6d,
    "levy6d": levy_6d,
    "griewank6d": griewank_6d,
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def visualize_1d_function(func_dict):
        x = torch.linspace(func_dict["bounds"][0].item(), func_dict["bounds"][1].item(), 1000).unsqueeze(1)
        y = func_dict["func"](x, lambda x: unnorm(x, func_dict["bounds"][0], func_dict["bounds"][1], func_dict["bounds"][0], func_dict["bounds"][1]))
        plt.plot(x.numpy(), y.numpy(), label=func_dict["name"])
        plt.scatter(func_dict["optimizer"].numpy(), func_dict["optimum"], color='red', label='Optimum')
        plt.legend()
        plt.title(f"1D Function: {func_dict['name']}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def visualize_2d_function(func_dict):
        x0 = torch.linspace(func_dict["bounds"][0, 0].item(), func_dict["bounds"][1, 0].item(), 100)
        x1 = torch.linspace(func_dict["bounds"][0, 1].item(), func_dict["bounds"][1, 1].item(), 100)
        X0, X1 = torch.meshgrid(x0, x1)
        X = torch.cat([X0.reshape(-1, 1), X1.reshape(-1, 1)], dim=1)
        y = func_dict["func"](X, lambda x: unnorm(x, func_dict["bounds"][0], func_dict["bounds"][1], func_dict["bounds"][0], func_dict["bounds"][1]))
        Y = y.reshape(100, 100).detach().numpy()
        plt.contourf(X0.numpy(), X1.numpy(), Y, levels=50, cmap='viridis')
        plt.colorbar()
        plt.scatter(func_dict["optimizer"][0].item(), func_dict["optimizer"][1].item(), color='red', label='Optimum')
        plt.legend()
        plt.title(f"2D Function: {func_dict['name']}")
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.show()

    def visualize_histogram(func_dict, num_samples=100000000):
        dim = func_dict["bounds"].shape[1]
        x = torch.rand(num_samples, dim) * (func_dict["bounds"][1] - func_dict["bounds"][0]) + func_dict["bounds"][0]
        y = func_dict["func"](x, lambda x: unnorm(x, func_dict["bounds"][0], func_dict["bounds"][1], func_dict["bounds"][0], func_dict["bounds"][1]))
        print(f"function: {func_dict['name']}, minimum: {y.min().item()}, minimizer: {x[torch.argmin(y)]}")
        plt.hist(y.numpy(), bins=50, alpha=0.75)
        plt.title(f"Histogram of y values for {func_dict['name']}")
        plt.xlabel("y")
        plt.ylabel("Frequency")
        plt.show()

    # Example usage:
    #visualize_1d_function(gramacy_lee_1d)
    #visualize_2d_function(ackley_2d)
    #visualize_histogram(ackley_3d)

    visualize_histogram(griewank_6d)