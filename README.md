# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference

This repository will provide the implementation and code used in the preprint article *Amortized Probabilistic Conditioning for Optimization, Simulation and Inference* (Chang et al., 2024).
The full paper can be found on arXiv at: [https://arxiv.org/abs/2410.15320](https://arxiv.org/abs/2410.15320).

## Installation with Anaconda

To install the required dependencies, run:

```bash
conda install python=3.9.19 pytorch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 -c pytorch
pip install -e .
```

## Demos

At the moment, we release three demo notebooks with examples of our method, the Amortized Conditioning Engine (ACE).

- [`1.MNIST_demo.ipynb`](1.MNIST_demo.ipynb): Image completion demo with MNIST.
- [`2.BO_demo.ipynb`](2.BO_demo.ipynb): Bayesian optimization demo.
- [`3.SBI_demo.ipynb`](3.SBI_demo.ipynb): Simulation-based inference demo.

Each notebook demonstrates a specific application of ACE. Simply open the notebooks in Jupyter or in GitHub to visualize the demos.

Code to run the demos will be added soon, and full code for this project will be made available later.

### License
This code is released under the Apache 2.0 License.
