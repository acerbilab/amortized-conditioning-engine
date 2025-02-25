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

---

# Training and Running experiments

## Bayesian Optimization

### Training

Before training the model, we first need to generate offline datasets using the following command:

```bash
python -m src.dataset.optimization.offline_bo_data_generator_prior -m dataset=offline_bo_prior_1d
```
This will generate offline datasets for both the prior and non-prior cases. In the non-prior case, the prior information is omitted. The offline data will be saved in offline_data/bonprior. Once the offline data is generated, we can proceed with training the models.

Training can be performed using the following commands:
```bash
python train.py -m dataset=offline_bo_1d #for non prior case
```
or 

```bash
python train.py -m dataset=offline_bo_prior_1d #for prior case
```
The resulting model checkpoint (`.ckpt`) and Hydra configuration will be saved in `multirun/<date>/<time>/0/`.

Below is the full script to reproduce the models used in the paper:

```bash 
# Data generation for 1-6 dimensions case
python -m src.dataset.optimization.offline_bo_data_generator_prior -m dataset=offline_bo_prior_1d,offline_bo_prior_2d,offline_bo_prior_3d,offline_bo_prior_4d,offline_bo_prior_5d,offline_bo_prior_6d

# Training for 1-3 dimension
python train.py -m dataset=offline_bo_1d,offline_bo_2d,offline_bo_3d encoder=tnpd_dm256df128l6h16 num_steps=500000 batch_size=64

# Training for 1-3 dimension with prior
python train.py -m dataset=offline_bo_prior_1d,offline_bo_prior_2d,offline_bo_prior_3d encoder=tnpd_dm256df128l6h16 num_steps=500000 batch_size=64

# Training for 4-6 dimension
python train.py -m dataset=offline_bo_4d,offline_bo_5d,offline_bo_6d encoder=tnpd_dm128df512l6h8 num_steps=350000 batch_size=128

```

### Running Experiments

<TODO>


