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

## Regression 

Training GP:
```bash
python train.py -m dataset=gp_sampler_kernel
```
Training MNIST:
```bash
python train.py -m dataset=image_sampler
```

## Bayesian Optimization

### Training BO

Before training the model, we first need to generate offline datasets using the following command:

```bash
python -m src.dataset.optimization.offline_bo_data_generator_prior -m dataset=offline_bo_prior_1d
```
This will generate offline datasets for both the prior and non-prior cases. In the non-prior case, the prior information is omitted. The offline data will be saved in `offline_data/bonprior`. Once the offline data is generated, we can proceed with training the models.

Training can be performed using the following commands:
```bash
python train.py -m dataset=offline_bo_1d #for non prior case
```
or 

```bash
python train.py -m dataset=offline_bo_prior_1d #for prior case
```
The resulting model checkpoint (`.ckpt`) and Hydra configuration will be saved in `multirun/` folder.

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

### Running BO Experiments

After training the models, the next step is to run the BO experiments on the benchmark functions. This can be done as follows:

1. Navigate to the `experiments/bo` folder.

2. Run the following script, specifying the benchmark functions:  
   
```bash
sh run_bo.sh 1d_ackley 10 results/bo_run/  
```
This runs the Ackley function with **10 repetitions** and saves the results in the specified folder.

3. Once the experiment is complete, plot the results using:  
```bash
python bo_plot.py result_path=results/bo_run/ plot_path=results/bo_plot/
```

**Reproducing the Experiments**

To fully reproduce the experiments, first ensure that the trained models are saved in the appropriate location before running the scripts. 
In the default settings, the trained models are saved in `models_ckpt/` folder, also see individual `.yml` files in tge `cfgs/benchmark` folder to see the full path of the models. 

Full script to reproduce the experiments is as folows:

```bash
# no prior experiments

# main paper experiments
sh run_bo.sh 1d_gramacy_lee 10 results/bo_run/ 
sh run_bo.sh 2d_branin_scaled 10 results/bo_run/ 
sh run_bo.sh 3d_hartmann 10 results/bo_run/ 
sh run_bo.sh 4d_rosenbrock 10 results/bo_run/
sh run_bo.sh 5d_rosenbrock 10 results/bo_run/
sh run_bo.sh 6d_hartmann 10 results/bo_run/
sh run_bo.sh 6d_levy 10 results/bo_run/    

# extended experiments in appendix
sh run_bo.sh 1d_ackley 10 results/bo_run/ 
sh run_bo.sh 1d_neg_easom 10 results/bo_run/
sh run_bo.sh 2d_michalewicz 10 results/bo_run/
sh run_bo.sh 2d_ackley 10 results/bo_run/
sh run_bo.sh 3d_levy 10 results/bo_run/
sh run_bo.sh 4d_hartmann 10 results/bo_run/
sh run_bo.sh 5D_griewank 10 results/bo_run/
sh run_bo.sh 6D_griewank 10 results/bo_run/      

# plotting results
python bo_plot.py result_path=results/bo_run/ plot_path=results/bo_plot/
```
For with-prior experiments we also need to specify the standard deviation of the gaussian prior (see the manuscript for more detail).

Full script to reproduce the prior experiments is as follows:

```bash
# with prior experiments

# main paper with prior experiments
sh run_bo_prior.sh 2d_michalewicz_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 2d_michalewicz_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 3d_levy_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 3d_levy_prior 0.2 10 results/bo_run_strongprior/

# extended experiments with prior in appendix
sh run_bo_prior.sh 1d_ackley_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 1d_ackley_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 1d_gramacy_lee_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 1d_gramacy_lee_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 1d_neg_easom_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 1d_neg_easom_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 2d_branin_scaled_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 2d_branin_scaled_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 2d_ackley_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 2d_ackley_prior 0.2 10 results/bo_run_strongprior/

sh run_bo_prior.sh 3d_hartmann_prior 0.5 10 results/bo_run_weakprior/
sh run_bo_prior.sh 3d_hartmann_prior 0.2 10 results/bo_run_strongprior/


# plotting results
python bo_plot.py result_path=results/bo_run_strongprior/ plot_path=results/bo_plot/ prefix_file_name="strongprior_"
python bo_plot.py result_path=results/bo_run_weakprior/ plot_path=results/bo_plot/ prefix_file_name="weakprior_"
```

## Simulation-based Inference

### Training SBI tasks
Before training the model, we first need to generate offline datasets using the following command:

```bash
python -m src.dataset.sbi.oup 
python -m src.dataset.sbi.sir
python -m src.dataset.sbi.turin
```
This will generate offline datasets for both the prior and non-prior cases. In the non-prior case, the prior information is omitted. 
The offline data will be saved in `data/`. Once the offline data is generated, we can proceed with training the models.

Training can be performed using the following commands:
```bash
# Non-prior case
python train_sbi.py dataset=oup embedder=embedder_marker_skipcon
python train_sbi.py dataset=sir embedder=embedder_marker_skipcon
python train_sbi.py dataset=turin embedder=embedder_marker_skipcon

# prior-injection case
python train_sbi.py dataset=oup_prior embedder=embedder_marker_prior_sbi
python train_sbi.py dataset=sir_prior embedder=embedder_marker_prior_sbi
python train_sbi.py dataset=turin_prior embedder=embedder_marker_prior_sbi
```

### Evaluating SBI tasks
After training the models, you can put your trained ckpt models and hydra config folders under `results/SIMULATOR`. E.g., for standard OUP task, you can put your ACE models and configs under `results/oup` and ACEP under `results/oup_pi`.

You can then use the notebooks in `experiments/sbi` to evaluate the models on the SBI tasks. Each notebook contains the NPE and NRE baselines, also including the evaluation code used to create the Table 1 in our paper, and all the visualization code. We use latex to create the plot, if you don't have latex in your local machine, you can set `"text.usetex": False` in `update_plot_style()` function under `sbi_demo_utils.py`.