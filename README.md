# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference

This repository will provide the implementation and code used in the AISTATS 2025 article *Amortized Probabilistic Conditioning for Optimization, Simulation and Inference* (Chang et al., 2025).
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

Full code for this project will be made available later.

## Citation
If you find this work valuable for your research, please consider citing our paper:

```
@article{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  journal={28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)},
  year={2025}
}
```

## License
This code is released under the Apache 2.0 License.
