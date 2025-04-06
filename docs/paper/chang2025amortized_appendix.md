# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference - Appendix

---

#### Page 14

# Supplementary Material

## Contents

A TABLE OF ACRONYMS ..... 15
B METHODS ..... 15
B. 1 Details and experiments with prior injection ..... 16
B. 2 Architecture ..... 20
B. 3 Training batch construction ..... 21
B. 4 Autoregressive predictions ..... 21
C EXPERIMENTAL DETAILS ..... 21
C. 1 Gaussian process (GP) experiments ..... 21
C. 2 Image completion and classification ..... 22
C. 3 Bayesian optimization ..... 24
C. 4 Simulation-based inference ..... 33
C. 5 Computational resources and software ..... 39

---

#### Page 15

# A TABLE OF ACRONYMS

For ease of reference, Table S1 reports a list of key acronyms and abbreviations used in the paper.

|                Acronym                 |                  Full Name                  |                                               Description                                               |
| :------------------------------------: | :-----------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
|             Architectures              |                                             |                                                                                                         |
|                 TPM-D                  |    Transformer Prediction Map - Diagonal    |   Family of transformer architectures for diagonal prediction maps, including all architectures below   |
|                  ACE                   |        Amortized Conditioning Engine        |    Our transformer-based meta-learning model for probabilistic tasks with explicit latent variables     |
|                  ACEP                  | Amortized Conditioning Engine (with Priors) |                 ACE variant allowing runtime injection of priors over latent variables                  |
|                  CNP                   |         Conditional Neural Process          |                          Context-to-target mapping with permutation invariance                          |
|                  PFN                   |            Prior-Fitted Network             | Meta-learning approach using transformers for inference and introducing Riemannian output distributions |
|                 TNP-D                  |    Transformer Neural Process - Diagonal    |                 Transformer neural process variant with independent target predictions                  |
|    Bayesian Optimization (BO) Terms    |                                             |                                                                                                         |
|                   BO                   |            Bayesian Optimization            |                         Black-box function optimization using surrogate models                          |
|                  MES                   |          Max-Value Entropy Search           |                      Acquisition function based on uncertainty over optimum value                       |
|                   TS                   |              Thompson Sampling              |                   Optimization via sampling from the posterior over optimum location                    |
|                $\pi$ BO                |            Prior-information BO             |                          BO incorporating prior knowledge on optimum location                           |
|              AR-TNP-D-TS               |   Autoregressive TNP-D Thompson Sampling    |                            TNP extension with autoregressive sampling for BO                            |
| Simulation-Based Inference (SBI) Terms |                                             |                                                                                                         |
|                  SBI                   |         Simulation-Based Inference          |                           Parameter posterior inference using synthetic data                            |
|                  NPE                   |         Neural Posterior Estimation         |                             Direct posterior modeling with neural networks                              |
|                  NRE                   |           Neural Ratio Estimation           |                               Likelihood-ratio-based posterior inference                                |
|                  OUP                   |         Ornstein-Uhlenbeck Process          |                                    Mean-reverting stochastic process                                    |
|                  SIR                   |      Susceptible-Infectious-Recovered       |                                  Epidemiological disease spread model                                   |

Table S1: Key acronyms used in the paper, grouped by category.

## B METHODS

This section details several technical aspects of our paper, such as the prior amortization techniques, neural network architecture and general training and inference details.

---

#### Page 16

# B. 1 Details and experiments with prior injection

Prior generative process. To expose ACE to a wide array of distinct priors during training, we generate priors following a hierarchical approach that generates smooth priors over a bounded range. The process is as follows, separately for each latent variable $\theta_{l}$, for $1 \leq l \leq L$ :

- We first sample the type of priors for the latent variable. Specifically, with $80 \%$ probability, we sample from a mixture of Gaussians to generate a smooth prior, otherwise, we create a flat prior with uniform distribution.
- If we sample from a mixture of Gaussians:
- We first sample the number of Gaussian components $K$ from a geometric distribution with $q=0.5$ :

$$
K \sim \text { Geometric }(0.5)
$$

- If $K>1$, we randomly choose among three configurations with equal probability:

1. Same means and different standard deviations.
2. Different means and same standard deviations.
3. All different means and standard deviations.

- Given the predefined global priors for mean and standard deviation (uniform distributions whose ranges are determined by the range of the corresponding latent variable), we sample the means and standard deviations for each component from the predefined uniform distributions.
- The weights of the mixture components are sampled from a Dirichlet distribution:

$$
\mathbf{w} \sim \operatorname{Dirichlet}\left(\alpha_{0}=1\right)
$$

- Finally, we convert the mixture of Gaussians into a normalized histogram over a grid $\mathcal{G}$ with $N_{\text {bins }}$ uniformly-spaced bins. For each bin $b$, we compute the probability mass $\mathbf{p}_{l}^{(b)}$ by calculating the difference between the cumulative distribution function values at the bin edges. This is done for each Gaussian component and then summed up, weighted by the mixture weights.
- We normalize the bin probabilities to ensure a valid probability distribution:

$$
\mathbf{p}_{l}=\frac{\mathbf{p}_{l}}{\sum_{b=1}^{N_{\text {bins }}} \mathbf{p}_{l}^{(b)}}
$$

- If we sample from a uniform distribution:
- We assign equal probability to each bin over the grid:

$$
\mathbf{p}_{l}=\frac{1}{N_{\text {bins }}} \mathbf{1}_{N_{\text {bins }}}
$$

where $\mathbf{1}_{N_{\text {bins }}}$ is a vector of ones of length $N_{\text {bins }}$.

For all experiments, we select $N_{\text {bins }}=100$ as the number of bins for the prior grid. See Fig. S1 for some examples of sampled priors.

Investigation of prior injection with a Gaussian toy model. To investigate the effect of the injected prior, we test our method with a simple 1D Gaussian model with two latent variables: mean $\mu$ and standard deviation $\sigma$. The data is the samples drawn from this distribution, $\mathcal{D}_{N}=\left\{y_{n}\right\}_{n=1}^{N} \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$. We can numerically compute the exact Bayesian posterior on the predefined grid given the data and any prior, and subsequently compare the ground-truth posterior with the ACE's predicted posterior after injecting the same prior.

We first sample random priors using the generative process described above. Then we sample $\mu$ and $\sigma$ from the priors and generate the corresponding data $\mathcal{D}_{N}$. We pass the data along with the prior to ACE to get the predictive distributions $p\left(\mu \mid \mathcal{D}_{N}\right)$ and $p\left(\sigma \mid \mathcal{D}_{N}\right)$ as well as the autoregressive predictions $p\left(\mu \mid \sigma, \mathcal{D}_{N}\right)$ and $p\left(\sigma \mid \mu, \mathcal{D}_{N}\right)$. With these equations, we can autoregressively compute our model's prediction for $p\left(\mu, \sigma \mid \mathcal{D}_{N}\right)$ on the grid. The true posterior is calculated numerically via Bayes rule on the grid. See Fig. S2 for several examples.

---

#### Page 17

> **Image description.** The image is a grid of 25 plots, each displaying a different probability distribution. Each plot is labeled "Sample [number]" where the number ranges from 1 to 25, and is positioned above its corresponding plot.
>
> Each plot has the following characteristics:
>
> - **Axes:** Each plot has an x-axis ranging from 0 to 2, and a y-axis that varies in scale depending on the distribution, but generally ranges from 0 to a maximum value between 0.01 and 0.15. The axes are labeled with numerical values at regular intervals.
> - **Curve:** Each plot displays a blue curve representing a probability distribution. The shapes of these curves vary considerably, including:
>   - Bell-shaped curves resembling Gaussian distributions (e.g., Samples 2, 3, 7, 8, 12, 13, 16, 22, 25).
>   - Skewed curves (e.g., Samples 4, 5, 6, 10, 11, 14, 15, 18, 19, 21).
>   - Uniform distributions (horizontal lines) (e.g., Samples 17, 20).
>   - Bimodal distributions (e.g., Samples 1, 23).
>   - A sharp peak distribution (e.g., Sample 9, 24).
> - **Data Points:** The curves are formed by connected blue dots.
>
> The plots are arranged in a 5x5 grid, with each plot visually separated from its neighbors.

Figure S1: Examples of randomly sampled priors over the range $[0,2]$. Samples include mixtures of Gaussians and Uniform distributions.

To quantitatively assess the quality of our model's predicted posteriors, we compare the posterior mean and standard deviation (i.e., the first two moments ${ }^{3}$ ) for $\mu$ and $\sigma$ of predicted vs. true posteriors, visualized in Fig. S3. The scatter points are aligned along the diagonal line, indicating that the moments of the predicted posterior closely match the moments of true posterior. These results show that ACE is effectively incorporating the information provided by the prior and adjusts the final posterior accordingly. In Appendix C.4.4 we perform a more extensive analysis of posterior calibration in ACE with a complex simulator model.

[^0]
[^0]: ${ }^{3}$ We prefer standard deviation to variance as it has the same units as the quantity of interest, as opposed to squared units which are less interpretable.

---

#### Page 18

> **Image description.** The image is a figure containing a 5x4 grid of plots, visually comparing different probability distributions. Each row represents a different scenario, and each column represents a different type of distribution: (a) Prior Distribution, (b) Likelihood, (c) True Posterior, and (d) ACE Posterior.
>
> The plots are all 2D contour plots with the x-axis labeled "μ" (mu) and the y-axis labeled "σ" (sigma). The background color of each plot is a dark purple, with contours represented by lines of varying colors, ranging from dark blue/purple to yellow/green, indicating probability density. Higher density regions are indicated by warmer colors (yellow/green), while lower density regions are indicated by cooler colors (blue/purple).
>
> The first column, labeled "(a) Prior Distribution," shows different prior distributions across the rows. The first row shows a vertical band of high probability. The second row shows a bimodal distribution. The third row shows a horizontal band of high probability. The fourth row shows a horizontal ellipse of high probability. The fifth row shows a vertical line of high probability.
>
> The second column, labeled "(b) Likelihood," shows the likelihood function for each scenario. The likelihoods are generally concentrated around a single point, indicated by concentric contours.
>
> The third column, labeled "(c) True Posterior," shows the ground-truth Bayesian posterior distribution for each scenario. These plots show distributions that are generally concentrated around a single point, but with varying shapes and orientations.
>
> The fourth column, labeled "(d) ACE Posterior," shows the predicted posterior distribution from ACE (presumably an algorithm). These plots are visually similar to the "True Posterior" plots, suggesting that ACE is effectively approximating the true posterior.
>
> The rows are separated by white lines.

Figure S2: Examples of the true and predicted posterior distributions in the toy 1D Gaussian case. (a) Prior distribution over $\boldsymbol{\theta}=(\mu, \sigma)$ set at runtime. (b) Likelihood for the observed data (the data themselves are not shown). (c) Ground-truth Bayesian posterior. (d) ACE's predicted posterior, based on the user-set prior and observed data, approximates well the true posterior.

---

#### Page 19

> **Image description.** This image contains four scatter plots arranged in a 2x2 grid. Each plot compares predicted vs. true posterior values for either the mean (μ) or standard deviation (σ).
>
> Here's a breakdown of each plot:
>
> - **Top Left:** "Predicted vs True Posterior Mean (μ)".
>
>   - The x-axis is labeled "Predicted Posterior Mean (μ)" and ranges from approximately -1 to 1.
>   - The y-axis is labeled "True Posterior Mean (μ)" and ranges from approximately -1 to 1.
>   - A cluster of blue data points forms a linear pattern.
>   - A red dashed line runs diagonally through the data points.
>   - A box in the upper left corner displays "R2: 1.00".
>
> - **Top Right:** "Predicted vs True Posterior Std (μ)".
>
>   - The x-axis is labeled "Predicted Posterior Std (μ)" and ranges from 0.00 to 0.30.
>   - The y-axis is labeled "True Posterior Std (μ)" and ranges from 0.00 to 0.25.
>   - A cluster of blue data points forms a linear pattern.
>   - A red dashed line runs diagonally through the data points.
>   - A box in the upper left corner displays "R2: 0.94".
>
> - **Bottom Left:** "Predicted vs True Posterior Mean (σ)".
>
>   - The x-axis is labeled "Predicted Posterior Mean (σ)" and ranges from 0.0 to 1.0.
>   - The y-axis is labeled "True Posterior Mean (σ)" and ranges from approximately 0.0 to 0.8.
>   - A cluster of blue data points forms a linear pattern.
>   - A red dashed line runs diagonally through the data points.
>   - A box in the upper left corner displays "R2: 1.00".
>
> - **Bottom Right:** "Predicted vs True Posterior Std (σ)".
>   - The x-axis is labeled "Predicted Posterior Std (σ)" and ranges from 0.00 to 0.18.
>   - The y-axis is labeled "True Posterior Std (σ)" and ranges from 0.00 to 0.175.
>   - A cluster of blue data points forms a linear pattern.
>   - A red dashed line runs diagonally through the data points.
>   - A box in the upper left corner displays "R2: 0.89".
>
> In all four plots, the data points are clustered closely around the red dashed line, suggesting a strong correlation between the predicted and true posterior values. The R-squared values, displayed in the upper left corner of each plot, quantify the strength of this correlation.

Figure S3: The scatter plots compare the predicted and true posterior mean and standard deviation values for both $\mu$ and $\sigma$ across 100 examples. We can see that the points lie closely along the diagonal red dashed line, indicating that the moments (mean and standard deviation) of the predicted posteriors closely match the true posteriors.

---

#### Page 20

# B. 2 Architecture

Here we give an overview of two key architectures used in our paper. First, we show the TNP-D (Nguyen and Grover, 2022) architecture in Fig. S4, which our method extends. Fig. S5 shows the ACE architecture introduced in this paper.

> **Image description.** This is a diagram illustrating an architecture, likely a neural network or similar computational model.
>
> The diagram is structured from left to right, showing the flow of data through different layers or components.
>
> - **Input:** On the left, there are six inputs represented as gray rounded rectangles. The top three contain coordinate pairs (x1, y1), (x3, y3), and (x5, y5). The bottom three contain only x values: (x2), (x4), and (x6). The x values are in red font. Arrows point from each input to a central block labeled "Embedder" in light gray.
>
> - **Embedder:** The "Embedder" is a large, light-gray rectangle.
>
> - **MHSA and CA:** The output of the "Embedder" splits into two paths. The top path leads to a block labeled "MHSA" (Multi-Head Self-Attention). This block contains three gray rounded rectangles labeled (z1), (z3), and (z5). The bottom path leads to a block labeled "CA" (Cross-Attention). This block also receives input from the "MHSA" block. The "MHSA" and "CA" blocks are enclosed within a larger rounded rectangle labeled "k-blocks" in blue. The labels "MHSA" and "CA" are in red font. The inputs to CA are gray rounded rectangles labeled (z2), (z4), and (z6). The z values are in red font.
>
> - **Head:** The output of the "k-blocks" feeds into another light-gray rectangle labeled "Head".
>
> - **Output/Loss:** The output of the "Head" leads to a final block labeled "Loss" in gold. This block contains six rounded rectangles arranged in two columns. The left column contains white rectangles with hat{y} values: (hat{y}2), (hat{y}4), and (hat{y}6). The right column contains gray rectangles with y values: (y2), (y4), and (y6). The y values are in red font.
>
> Arrows connect the components, indicating the flow of data. The color scheme uses light-gray for blocks, black for arrows, red for certain labels and values, blue for the "k-blocks" enclosure, and gold for the "Loss" enclosure.

Figure S4: A conceptual figure of TNP-D architecture. The TNP-D architecture can be summarized in the embedding layer, attention layers and output head. The $x$ denotes locations where the output is unknown (target inputs). The $z$ is the embedded data, while MHSA stands for multi head cross attention and CA for cross attention. The head for TNP-D is Gaussian, so it outputs a mean and variance for each target point.

> **Image description.** This is a diagram illustrating the ACE architecture, a neural network architecture. The diagram shows the flow of data through different layers and components of the network.
>
> Here's a breakdown of the visible elements:
>
> - **Input:** On the left side, there are several inputs represented by rectangles.
>   - Two green rectangles labeled "(θ₁)" and "(?θ₂)".
>   - Three grey rectangles labeled "(x₃, y₃)", "(x₅, y₅)", "(x₄)", and "(x₆)". The label "(z₂)" is written in red.
> - **Embedder:** A large, light-pink rectangle labeled "Embedder" in grey is positioned in the center of the left side. Arrows connect the input rectangles to the Embedder.
> - **MHSA:** A blue rounded rectangle labeled "MHSA" in black. Inside the rectangle are three grey rectangles labeled "(z₁)", "(z₃)", and "(z₅)". Arrows connect the Embedder to these rectangles.
> - **CA:** Below the MHSA rectangle, there is a light-pink rectangle labeled "CA" in dark red. An arrow connects the MHSA rectangle to the CA rectangle. The label "k-blocks" is written below the CA rectangle.
> - **Head:** To the right of the CA rectangle, there is a light-pink rectangle labeled "Head (GMM or Cat)" in grey. An arrow connects the CA rectangle to the Head rectangle.
> - **Output/Loss:** On the right side, there is an orange rounded rectangle labeled "Loss" in orange. Inside the rectangle are six rectangles:
>   - Two white rectangles labeled "(^θ₂)" and "(^y₄)" and "(^y₆)".
>   - Four grey rectangles labeled "(y₄)" and "(y₆)".
>   - One green rectangle labeled "(θ₂)".
> - **Arrows:** Arrows connect the different components, indicating the flow of data.
>
> The diagram uses different colors to highlight different types of data or components. The labels provide information about the variables and operations involved in each step of the architecture. The overall layout suggests a sequential processing of data from the input to the output/loss calculation.

Figure S5: A conceptual figure of ACE architecture. The diagram shows key differences between ACE and TNP-D. The differences boil down to the embedder layer that now incorporates latents $\theta_{l}$ (and possibly priors over these) and the output head that is now a Gaussian mixture model (GMM, for continuous variables) or categorical (Cat, for discrete variables). Both latent and data can be of either type.

---

#### Page 21

# B. 3 Training batch construction

ACE can condition on and predict data, latent variables, and combinations of both. Here, we outline the sampling process used to construct the training batch.

- First, we generate our dataset by following the steps outlined for the respective cases (GP, Image Completion, BO, SBI); see Appendix C. For example, in the GP emulation case, we draw $n_{\text {data }}$ points from a function sampled from a GP along with its respective latent variables.
- Next, we sample the number of context points, $n_{\text {context }}$, uniformly between the minimum and maximum context points, min*ctx and max_ctx. We then split our data based on this $n*{\text {context }}$ value; the remaining data points that are not in the context set are allocated to the target set.
- We then determine whether the context includes any latent variables at all with a $50 \%$ probability. If latent variables are to be included in the context set, we sample the number of latents residing in the context set, uniformly from 1 to $n_{\text {latents }}$. All latent variables not in the context set are assigned to the target set.

The above steps are applied for each element (dataset) in the training batch. In the implementation, we also ensure that, within each batch, the number of context points remains consistent across all elements, as does the number of target points, to facilitate batch training for our model. However, the number of latents in the context set may vary for each element, introducing variability that improves the model's training process.

## B. 4 Autoregressive predictions

While ACE predicts conditional marginals independently, we can still obtain joint predictions over both data and latents autoregressively (Nguyen and Grover, 2022; Bruinsma et al., 2023). Suppose we want to predict the joint target distribution $p\left(\mathbf{z}_{1: M}^{\star} \mid \boldsymbol{\xi}_{1: M}^{\star}, \mathfrak{D}_{N}\right)$, where we use compact indexing notation. We can write:

$$
p\left(\mathbf{z}_{1: M}^{\star} \mid \boldsymbol{\xi}_{1: M}^{\star}, \mathfrak{D}_{N}\right)=\prod_{m=1}^{M} p\left(z_{m}^{\star} \mid \mathbf{z}_{1: m-1}^{\star}, \boldsymbol{\xi}_{1: m}^{\star}, \mathfrak{D}_{N}\right)=\mathbb{E}_{\boldsymbol{\pi}}\left[\prod_{m=1}^{M} p\left(z_{\pi_{m}}^{\star} \mid \mathbf{z}_{\pi_{1}: \pi_{m-1}}^{\star}, \boldsymbol{\xi}_{\pi_{1}: \pi_{m}}^{\star}, \mathfrak{D}_{N}\right)\right]
$$

where $\boldsymbol{\pi}$ is a permutation of $(1, \ldots, M)$, i.e. an element of the symmetric group $\mathcal{S}_{M}$. The first passage follows from the standard rules of probability and the second passage follows from permutation invariance of the joint distribution with respect to the ordering of the variables $\boldsymbol{\xi}_{1: M}$. The last expression can be used to enforce permutation invariance and validity of our joint predictions even if sequential predictions of the model are not natively invariant (Murphy et al., 2019). In practice, for moderate to large $M(M \gtrsim 4)$ we approximate the expectation over permutations via Monte Carlo sampling.

## C EXPERIMENTAL DETAILS

In this section, we show additional experiments to validate our method and provide additional details about sampling, training, and model architecture.

## C. 1 Gaussian process (GP) experiments

We now demonstrate the use of ACE for performing amortized inference tasks in the Gaussian processes (GP) model class. GPs are a Bayesian non-parametric method used as priors over functions (see Rasmussen and Williams, 2006). To perform inference with a GP, one must first define a kernel function $\kappa_{\boldsymbol{\theta}}$ parameterized by hyperparameters $\boldsymbol{\theta}$ such as lengthscales and output scale. As a flexible model of distributions over functions used for regression and classification, GPs are a go-to generative model for meta-learning and feature heavily in the (conditional) neural process literature (CNP; Garnelo et al., 2018b). ACE can handle many of the challenges faced when applying GPs. Firstly, it can accurately amortize the GP predictive distribution as is usually shown in the CNP literature. In addition, ACE can perform other crucial tasks in the GP workflow, such as amortized learning of $\boldsymbol{\theta}$, usually found through optimization of the marginal likelihood (Gaussian likelihood) or via approximate inference for non-Gaussian likelihoods (e.g., Hensman et al. 2015). Furthermore, we can also do kernel selection by treating the kernel as a latent discrete variable, and incorporate prior knowledge about $\boldsymbol{\theta}$.

---

#### Page 22

> **Image description.** The image contains three line graphs, labeled (a), (b), and (c) respectively, arranged horizontally. Each graph plots data against "Size of $\mathcal{D}_N$" on the x-axis, which ranges from approximately 0 to 20. All graphs have gridlines.
>
> - **Graph (a):** This graph displays "Log predictive density $p(y|\cdot)$" on the y-axis, ranging from -1 to 2. Three lines are plotted:
>
>   - A dashed orange line representing "$p(y|\mathcal{D}_N)$".
>   - A solid green line representing "$p(y|\theta, \mathcal{D}_N)$".
>   - A solid blue line representing "GP predictive".
>     Each line has circular markers at data points and is surrounded by a shaded area of the same color, indicating a confidence interval.
>
> - **Graph (b):** This graph displays "Kernel identification accuracy" on the y-axis, ranging from 0.4 to 1. A single solid orange line with circular markers is plotted, also surrounded by a shaded area of the same color.
>
> - **Graph (c):** This graph displays "Log predictive density $p(\theta|\mathcal{D}_N)$" on the y-axis, ranging from 0 to 0.6. A single solid orange line with circular markers is plotted, surrounded by a shaded area of the same color.

Figure S6: (a) Conditioning on the latent variable $\boldsymbol{\theta}$ (kernel hyperparameters and type) improves predictive performance, approaching the GP upper bound for the log predictive density. (b) ACE can identify the kernel $\kappa$. (c) ACE can learn kernel hyperparameters.

Results. The main results from our GP regression experiments are displayed in Fig. S6. We trained ACE on samples from four kernels, the RBF and Matérn- $(1 / 2,3 / 2,5 / 2)$, using the architecture described in Section 3; see below for details. In Fig. S6a, we show ACE's ability to condition on provided information: data only, or data and $\boldsymbol{\theta}$ (kernel hyperparameters and type). As expected, there is an improvement when conditioning on more information, specifically when the context set $\mathcal{D}_{N}$ is small. As an upper bound, we show the ground-truth GP predictive performance. The method can accurately predict the kernel, i.e. model selection (Fig. S6b), while at the same time learn the hyperparameters (Fig. S6c), both improving as a function of the context set size.

Sampling from a GP. Both the GP experiments and the Bayesian optimization experiments reported in Section 4.2 and further detailed in Appendix C. 3 use a similar sampling process to generate data.

- We first sample the latents. These are kernel hyperparameters, the output scale $\sigma_{f}$ and lengthscale $\ell$. Each input dimension of $\mathbf{x}$ is assigned its own lengthscale $\ell=\left(\ell^{(1)}, \ell^{(2)}, \ldots\right)$ and a corresponding kernel. For all GP examples the RBF and three Matérn- $(1 / 2,3 / 2,5 / 2)$ kernels were used with equal weights. The kernel output scale $\sigma_{f} \sim U(0.1,1)$ and each $k$-th lengthscale is $\ell^{(k)} \sim \mathcal{N}(1 / 3,0.75)$ truncated between $[0.05,2]$.
- Once all latent information is defined, we draw from a GP prior from a range $[-1,1]$ for each input dimension. The realizations from the prior form our context data $\mathcal{D}_{N}$ where the size of the context set $N$ is drawn from a discrete uniform distribution $3,4, \ldots .50$. The target data $\left(\mathbf{X}^{*}, \mathbf{y}^{*}\right)$ of size $200-N$ is then drawn from the predictive posterior of the GP conditioned on $\mathcal{D}_{N}$.

Architecture. The ACE model used in the GP experiments had embedding dimension 256 and 6 transformer layers. The attention blocks had 6 heads and the MLP block had hidden dimension 128. The output head had $K=20$ MLP components with hidden dimension 256. The model was trained for $5 \times 10^{4}$ steps with batch size 32 , using learning rate $1 \times 10^{-4}$ with cosine annealing. Following Nguyen and Grover (2022), and unlike the original transformer implementation (Vaswani et al., 2017), we do not use dropout in any of our experiments.

# C. 2 Image completion and classification

In this section, we detail the image experiments in Section 4.1 as well as report additional experiments. Image completion experiments have long been a benchmark in the neural process literature treating them as regression problems (Garnelo et al., 2018a; Kim et al., 2019). We use two standard datasets, MNIST (Deng, 2012) and CelebA (Liu et al., 2015). The MNIST results presented are with the full image size $28 \times 28$, while CelebA results were downsized to $32 \times 32$. However, as shown in Fig. S11, ACE can also handle the full image size. All image datasets were normalised based on the complete dataset average and standard deviation. The data input $\mathbf{x}$ for image experiments is the 2-D image pixel-coordinates and the data value $y$ for MNIST is one output dimension, while CelebA uses all three RGB channels and thus is a multi-output $\mathbf{y}$.

The experiments on images demonstrate the versatility of the ACE method and its advantages over conventional

---

#### Page 23

> **Image description.** The image presents a series of image completion results for handwritten digits, arranged in a 2x5 grid. Each image is a square. The top row shows the digit '9', and the bottom row shows the digit '7'.
>
> - **Column 1 (a) Image:** The first column displays the original, complete images of the digits. The digit '9' is in the top row, and the digit '7' is in the bottom row. Digits are white on a black background.
> - **Column 2 (b) $\mathcal{D}_{N}$:** The second column shows the context provided to the models. The background is blue, and a sparse scattering of small white and dark blue squares represents the observed pixels (10% of the pixels are observed).
> - **Column 3 (c) TNPD:** The third column shows the image completion results from the TNPD model. The digits are blurry and faint, with a gray scale appearance. The background is black, and there is a scattering of small blue squares.
> - **Column 4 (d) ACE:** The fourth column shows the image completion results from the ACE model. The digits are clearer than in the TNPD column. The background is black, and there is a scattering of small blue squares.
> - **Column 5 (e) ACE- $\boldsymbol{\theta}$:** The fifth column shows the image completion results from the ACE model conditioned on the class label. The digits are the clearest and most similar to the original images. The background is black, and there is a scattering of small blue squares.
>
> Below each column, there is a label: (a) Image, (b) $\mathcal{D}_{N}$, (c) TNPD, (d) ACE, and (e) ACE- $\boldsymbol{\theta}$.

(a) Image
(b) $\mathcal{D}_{N}$
(c) TNPD
(d) ACE
(e) ACE- $\boldsymbol{\theta}$

> **Image description.** A line graph compares the performance of three models: ACE, ACE-θ, and TNPD.
>
> The graph has the following elements:
>
> - **X-axis:** Ranges from 0 to 20, with ticks at intervals of 10.
> - **Y-axis:** Ranges from -1.2 to -0.6, with ticks at intervals of 0.2.
> - **Three lines:**
>   - **ACE:** A blue line with circular markers.
>   - **ACE-θ:** An orange line with circular markers.
>   - **TNPD:** A green line with circular markers.
> - **Shaded regions:** Each line has a corresponding shaded region around it, representing a confidence interval. The colors of the shaded regions match the colors of the lines.
> - **Legend:** A box in the upper right corner identifies each line by its color and label.

(f) NLPD v Context(\% of image)

Figure S7: Image regression (MNIST). Image (a) serves as the reference for the problem, while (b) is the context where $10 \%$ of the pixels are observed. Figures (c) - (e) are the respective model predictions, while (f) shows performance over varying context (mean and $95 \%$ confidence interval). In (e) the model is also conditioned on the class label, showing a clear improvement in performance.

CNPs. We outperform the current state-of-the-art TNP-D on the standard image completion task (Fig. 3). Given a random sample from the image space as context $\mathcal{D}_{N}$, the model predicts the remaining $M$ image pixel values at $\mathbf{x}^{*}$. The total number of points $N+M$ for MNIST is thus 784 points and 1024 for CelebA where the split is randomly sampled (see below for details). The model is then trained as detailed in Section 3.3. Thus, the final trained model can perform image completion, also sometimes known as in-painting.

In addition to image completion, our method can condition on and predict latent variables $\boldsymbol{\theta}$. For MNIST, we use the class labels as latents, so 0 , $1,2, \ldots$, which were encoded into a single discrete variable. Meanwhile, for CelebA we use as latents the 40 binary features that accompany the dataset, e.g. BrownHair, Man, Smiling, trained with the sampling procedure discussed below. We recall that in ACE the latents $\boldsymbol{\theta}$ can be both conditioned on and predicted. Thus, we can do conditional generation based on the class or features or, given a sample of an image, predict its class or features, as initially promised in Fig. 1a.

> **Image description.** A line graph depicts the relationship between "Context Size %" on the x-axis and "Classification Accuracy" on the y-axis.
>
> The x-axis ranges from 0 to 100, with labels at 0, 20, 40, 60, 80, and 100. The y-axis ranges from 0 to 1, with labels at 0, 0.2, 0.4, 0.6, 0.8, and 1.
>
> A blue line with circular markers represents the data points. The line starts at approximately (0, 0.2) and increases sharply to around (20, 0.85). It then gradually rises, reaching nearly 1.0 around (60, 1.0). The line remains close to 1.0 for the rest of the graph, with a slight dip near the end.
>
> A shaded light blue area surrounds the line, indicating the confidence interval or variability of the data. A grid of light gray lines is visible in the background, providing a visual aid for reading values on the graph.

Figure S8: Classification accuracy for MNIST for varying context size.

Results. The main image completion results for the CelebA dataset are shown in Fig. 3, with the same experiment performed on MNIST and displayed in Fig. S7. In both figures, we display some example images and predictions and negative log-probability density for different context sizes (shaded area is $95 \%$ confidence interval). Our method demonstrates a clear improvement over the TNP-D method across all context sizes on both datasets (Fig. 3 and Fig. S7). Moreover, incorporating latent information for conditional generation further enhances the performance of our base method. A variation of the image completion experiment is shown in Fig. S9, where the context is no longer randomly sampled from within the image but instead selected according

> **Image description.** The image shows a close-up of a person's head and shoulders with a bright green rectangle obscuring the top half of their face. The person's skin tone appears light, and the visible portion of their face shows features like a nose and mouth. The shoulders and neck are outlined in black. The background is white. The green rectangle covers the forehead and eyes.

(a) Context

> **Image description.** The image is a close-up photograph of a person's face. The person appears to be a man with light skin, a white beard, and brown hair that is balding on top. The image quality is somewhat pixelated and blurry, especially around the edges. The man is wearing a dark-colored shirt or jacket. The background is plain white.

(b) $\mathrm{BALD}=$ True

> **Image description.** The image consists of two panels.
>
> The left panel shows a blurry image of a man's face. The face is light-skinned with dark hair. He is wearing a dark-colored shirt or jacket with a high collar. The image is somewhat pixelated and lacks fine detail.
>
> The right panel is divided into two vertical rectangles. The left rectangle is bright green, and the right rectangle is black.

(c) $\mathrm{BALD}=\mathrm{False}$

> **Image description.** The image shows two panels side-by-side, each containing a partial image.
>
> Panel 1: The top half of the image is a solid bright green color. The bottom half shows a blurry image with hints of blue, white, and a brownish-orange color. The shapes are indistinct.
>
> Panel 2: This panel shows a blurred image with a color palette of blue, black, and red. The shapes are indistinct, but there is a suggestion of a rounded form in the center of the image.

(d) Context

> **Image description.** The image contains two blurry images of faces.
>
> The left image shows a person's face with a brown complexion, set against a blurred blue background. The person appears to be bald or have very short hair. The image quality is low, making it difficult to discern fine details.
>
> The right image also shows a person's face against a blurred blue background. However, in this image, the person's hair appears dark and covers most of their forehead. The image is similarly blurry, obscuring facial features. A black vertical bar is visible on the left side of the image.

(e) $\mathrm{BALD}=\mathrm{True}$

> **Image description.** The image contains two blurry headshot-style photographs.
>
> The left photograph shows a person with dark skin, a bald head, and a blurred expression. The background is a gradient of blue.
>
> The right photograph shows a person with dark skin and dark hair. A vertical black bar obscures part of the left side of the image. The background is also a gradient of blue. The image is blurry, making it difficult to discern specific facial features.

(f) $\mathrm{BALD}=\mathrm{False}$

Figure S9: Example of ACE conditioning on the value of the BALD feature when the image is masked for the first 22 rows. (a) and (d) show the context points used for prediction, where (b) and (e) show predictions where the Bald feature is conditioned on True. Meanwhile, c and f are conditioned on False.

---

#### Page 24

to a top 22-row image mask. For this example, the latent information BALD is either conditioned on True or False. The results show that the model adjusts its generated output based on the provided latent information, highlighting the potential of conditional generation. Furthermore, in Fig. S10, we show examples of ACE's ability to perform image classification showing a subset of the 40 features in CelebA dataset. Despite only having $10 \%$ of the image available, ACE can predict most features successfully. Finally, in Fig. S8 the accuracy for predicting the correct class label for MNIST is reported.

> **Image description.** The image presents a figure with three panels labeled (a), (b), and (c). The figure appears to be related to image classification based on partial information.
>
> Panel (a), labeled "Context," shows a pixelated representation of an image. The background is predominantly green, with scattered pixels of other colors (white, black, orange, and gray). This likely represents the available context or input to a classification model.
>
> Panel (b), labeled "Full image," displays a pixelated image of a face. The top image shows a light-skinned person, potentially male, with light-colored hair. The bottom image shows a light-skinned person, potentially female, with dark hair. The pixelation obscures fine details, but the basic facial features are discernible.
>
> Panel (c), labeled "Classification probability for some features," shows two horizontal bar charts, one for each image in (b). Each chart displays the classification probabilities for a subset of features. The features listed vertically on the left side of each chart are: "Bald," "Gray_Hair," "Smiling," "Black_Hair," "Big_Lips," "Wearing_Necktie," "Male," "Bangs," "Young," and "No_Beard." The x-axis ranges from 0 to 1, representing the probability. Each feature has a blue horizontal bar indicating the probability, a black dot representing the average probability, and a symbol indicating the ground truth label: a red asterisk (\*) for label = 1 and a black cross (x) for label = 0. A vertical dashed red line is present at x = 0.5. The bars extend to the right or left of the average depending on the probability.

Figure S10: An example showing the classification ability of ACE. (a) is the context available of the full image displayed in the panel (b). The probabilities for a subset of features are in (c).

Sampling for Image experiments. For sampling, we use the full available dataset for both MNIST and CelebA, detailed in Appendix B.3. In the MNIST dataset there is one latent class label and for CelebA all 40 features were used. In Fig. S9, the sampling procedure was adjusted to represent features that would influence the top 22 rows of the images. Therefore, we selected a subset of seven features, which were BALD, BANGS, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair and Eyeglasses. The same sampling procedure was performed again but, now on a smaller set of features.

Architecture and training. For the image experiments, we used the same embedder layer as in the other experiments. Through grid search, we found that a transformer architecture with 8 heads, 6 layers, and a hidden dimension of 128 performed best. For the MLP layer, we used a dimension of 256 . Finally, we reduced the number of components for the output head to $K=3$. We trained the model for 80,000 iterations using a cosine scheduler with Adam (Kingma and Ba, 2015), with a learning rate 0.0005 and a batch size of 64 .

# C. 3 Bayesian optimization

This section presents ACE's Bayesian Optimization (BO) experiments (Section 4.2 in the main paper) in more detail, including the training data generation, algorithms, benchmark functions, and baselines used in this paper.

## C.3.1 Bayesian Optimization dataset, architecture and training details

Dataset. The BO datasets are generated by sampling from a GP, following the approach described in Appendix C.1. The sampling procedure is adjusted to include the known optimum location and value of the function within the generative process. The detailed dataset generation procedure is outlined as follows:

1. Sampling GP hyper-parameters, to determine the base function shape:

---

#### Page 25

> **Image description.** This image shows a comparison of image reconstruction using different methods. It consists of two rows, each with four panels, and each panel displays a 64x64 pixel image.
>
> - **Row 1:** The first row shows a reconstruction of a woman's face.
>
>   - Panel (a), labeled "Context", shows a green background with scattered colored pixels (red, black, white). This appears to be a sparse or noisy representation of the image.
>   - Panel (b), labeled "Image", displays the original image of a woman's face in profile. She has dark hair and fair skin.
>   - Panel (c), labeled "ACE", shows a reconstruction of the woman's face, slightly blurred compared to the original.
>   - Panel (d), labeled "ACE-θ", shows another reconstruction of the woman's face, also slightly blurred, and appears similar to panel (c).
>
> - **Row 2:** The second row shows a reconstruction of a man's face.
>   - Panel (a), labeled "Context", shows a green background with scattered colored pixels (red, black, white). This appears to be a sparse or noisy representation of the image.
>   - Panel (b), labeled "Image", displays the original image of a man's face wearing a red baseball cap and sunglasses. He has fair skin. The background is blurred and appears to be blue.
>   - Panel (c), labeled "ACE", shows a reconstruction of the man's face, slightly blurred compared to the original.
>   - Panel (d), labeled "ACE-θ", shows another reconstruction of the man's face, also slightly blurred, and appears similar to panel (c).
>
> The reconstructions in panels (c) and (d) of both rows appear to be attempts to recreate the original images (panel b) from the sparse context provided in panel (a).

Figure S11: Examples of ACE on 64x64 image size.

- First, we randomly select a kernel from a set comprising the RBF kernel and three Matérn kernels (Matérn$1 / 2$, Matérn-3/2, and Matérn-5/2) based on predefined weights $[0.35,0.1,0.2,0.35]$, corresponding respectively to the RBF kernel and the Matérn kernels in the specified order.
- Then, we sample whether the kernel is isotropic or not with $p=0.5$.
- The output scale $\sigma_{f}$ and lengthscales $l^{(k)}$ are sampled following the procedure outlined in Appendix C.1.
- We assume the GP (constant) mean to be 0 for now.

2. Sampling the latent values, $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$ :

- We sample the optimum location $\mathbf{x}_{\text {opt }}$ uniformly inside $[-1,1]^{D}$.
- We sample the value of the global minimum $y_{\text {opt }}$ from a minimum-value distribution of a zero-mean Gaussian with variance equal to the output variance of the GP. The number of samples for the minimumvalue distribution, $N$, is approximated as the number of uncorrelated samples from the GP in the hypercube, determined based on the GP's length scale. This approach ensures that $y_{\text {opt }}$ roughly respects the statistics of optima for the GP hyperparameters.
- With $p=0.1$ probability, we add $\Delta y \sim \exp (1)$ to the mean function to represent an "unexpectedly low" optimum.

3. Sampling from GP posterior to get the context and target sets:

- We build a posterior GP with the above specification and a single observation at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$.
- We sample a total of $100 \cdot D$ (context + target) locations where the number of context points is sampled similarly to the GP dataset generation. The maximum number of context points is 50 for the 1D case and 100 for both 2 D and 3 D cases.
- Then, the values of this context set are jointly sampled from a GP posterior conditioned on one observation at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$.
- Instead, the target points are sampled independently from a GP posterior conditioned on $\mathcal{D}_{N}$ (the previously sampled context points) and $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$. Independent sampling of the targets speeds up GP data generation and is valid since during training we only predict 1D marginal distributions at the target points.

4. Further adjustment of $y$, and consequently $y_{\text {opt }}$ :

- To ensure that the global optimum is at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$ we add a convex envelope (a quadratic component). Specifically, we transform the $y$ values of the datasets as $y_{i}^{\prime}=\left|y_{i}\right|+\frac{1}{5}\left\|\mathbf{x}_{\text {opt }}-\mathbf{x}_{i}\right\|^{2}$ where $\mathbf{x}_{i}$ and $y_{i}$ are the input and output values of all sampled points.

---

#### Page 26

> **Image description.** A figure containing 16 subplots, each displaying a one-dimensional graph. Each subplot has the same x-axis range from -1.0 to 1.0, and a different y-axis range depending on the function plotted. The x-axis is labeled "x" at the bottom right of the figure, and the y-axis is labeled "y" at the left of the figure. Each graph shows a blue line representing a function, and a red dot marking the global optimum of that function within the plotted range. The title of each subplot indicates the sample number, from "Sample 1" to "Sample 16". The functions plotted vary in complexity, with some being smooth and others exhibiting more erratic behavior.

Figure S12: One-dimensional Bayesian optimization dataset samples, with their global optimum (red dot).

- Lastly, we add an offset to the $y^{\prime}$ values of sampled points uniformly drawn from $[-5,5]$, meaning that $y_{\text {opt }} \in[-5,5]$.

One and two-dimensional examples of the sampled functions are illustrated in Fig. S12 and Fig. S13, respectively.
Architecture and training details. In the Bayesian Optimization (BO) experiments, the ACE model was configured differently depending on the dimensionality of the problem. For the 1-3D cases, the model used an embedding dimension of $D_{\text {emb }}=256$ with six transformer layers. Each attention block had 16 heads, while the MLP block had a hidden dimension of 128 . The output head consisted of $K=20 \mathrm{MLP}$ components, each with a hidden dimension of 128 . For the 4-6D cases, the model was configured with embedding dimension of $D_{\text {emb }}=128$ while still using six transformer layers. Each attention block had 8 heads, and the MLP block had a hidden dimension of 512 . The output head structure remained unchanged, consisting of $K=20 \mathrm{MLP}$ components, each with a hidden dimension of 128 . The model configuration varied with problem dimensionality to balance capacity and efficiency.

The model was trained for $5 \times 10^{5}$ steps with a batch size of 64 for 1-3D cases and $3.5 \times 10^{5}$ steps and 128 batch size for 4-6D cases, using learning rate $5 \times 10^{-4}$ with cosine annealing. We apply loss weighing to give more

---

#### Page 27

> **Image description.** This image presents a set of nine contour plots arranged in a 3x3 grid. Each plot visualizes a two-dimensional function, likely representing Bayesian optimization dataset samples.
>
> Each subplot is labeled "Sample [number]" where the number ranges from 1 to 9. The x-axis, labeled "X1" at the bottom of the 8th subplot, and y-axis, labeled "X2" to the left of the 4th subplot, both range from -1.0 to 1.0. Each subplot displays a contour plot with varying color gradients, likely indicating the value of the function at different points in the 2D space. The color gradient ranges from dark purple (representing low values) to bright yellow (representing high values). Each subplot also contains a single red dot, which is identified in the caption as the optimum. Each subplot has a colorbar to its right indicating the range of values represented by the color gradient. The range of values varies from subplot to subplot.

Figure S13: Two-dimensional Bayesian optimization dataset samples, with their optimum (red dot).
importance to the latent variables during training. This adjustment accounts for the fact that the number of latent variables, $n_{\text {latent }}$, is generally much smaller than the number of data points, $\left(n_{\text {total }}-n_{\text {latent }}\right)$. The weight assigned to the latent loss is calculated as $w_{\text {latent }}=\left(n_{\text {total }}-1 / 2\left(\max \_c t x+\min \_c t x\right) / n_{\text {latent }}\right)^{T}$ where $T$ is a tunable parameter, max_ctx and max_ctx are the maximum and minimum number of context points during the dataset generation. We conducted a grid search over $T=1,2 / 3,1 / 3,0$ to identify the best-performing model. In our experiments, the optimal $T$ values are $T=1$ for $1 \mathrm{D}, T=2 / 3$ for 2 D and 3 D , and $T=0$ for $4 \mathrm{D}-6 \mathrm{D}$. Note that ACE has different models trained with different datasets for each input dimensionality.

# C.3.2 ACE-BO Algorithm

Bayesian optimization with Thompson sampling (ACE-TS). For Thompson sampling, we sample the query point at each step from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$ where $\tau$ is a threshold lower than the minimum point seen so far. This encourages exploration to sample a new point that is lower than the current optimum. We set $\tau=y_{\min }-\alpha \max \left(1, y_{\max }-y_{\min }\right)$, where $y_{\max }$ and $y_{\min }$ are the maximum and minimum values currently observed so far, and $\alpha$ a parameter controlling the minimum improvement. We set $\alpha=0.01$ throughout all experiments. First, we sample $y_{\text {opt }}$ from a truncated mixture of Gaussian obtained from ACE's predictive distribution $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$, truncated for $y_{\text {opt }}<\tau$. After that, we sample $\mathbf{x}_{\text {opt }}$ conditioned on that sampled $y_{\text {opt }}$ (i.e., sample from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$ ). For higher dimension $(D>1)$ we sample $\mathbf{x}_{\text {opt }}$ in an autoregressive manner, one dimension at a time. The order of the dimensions is randomly permuted to mitigate order bias among the dimensions. The detailed pseudocode for ACE-TS ( $\mathrm{D}>1$ ) is presented in Algorithm Algorithm 1. An example evolution of ACE-TS is reported in Fig. S14.

---

#### Page 28

# Algorithm 1 ACE-Thompson Sample ( $\mathrm{D}>1$ )

Input: observed data points $\mathcal{D}_{N}=\left\{\mathbf{x}_{1: N}, y_{1: N}\right\}$, improvement parameter $\alpha$, input dimensionality $D \in \mathbb{N}^{+}$, whether to condition on $y_{\text {opt }}$ or not flag $c \in\{$ True, False $\}$.
Initialization $y_{\min } \leftarrow \min y_{1: N}, y_{\max } \leftarrow \max y_{1: N}$.
if $c$ is True then
set threshold value: $\tau \leftarrow y_{\min }-\alpha \max \left(1, y_{\max }-y_{\min }\right)$.
sample $y_{\text {opt }}$ from mixture truncated at $\tau: y_{\text {opt }} \sim p\left(y_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$.
end if
randomly permute dimension indices: $(1, \ldots, D) \rightarrow\left(\pi_{1}, \ldots, \pi_{D}\right) . \quad \triangleright \pi$ is permutation of $(1, \ldots, D)$
for $i \leftarrow \pi_{1}, \ldots, \pi_{D}$ do
if $c$ is True then
sample $x_{\text {opt }}^{i}$ conditioned on $y_{\text {opt }}, \mathcal{D}_{N}$, and already sampled $\mathbf{x}_{\text {opt }}$ dimensions if any:
$x_{\text {opt }}^{i} \sim p\left(x_{\text {opt }}^{i} \mid \mathcal{D}_{N}, y_{\text {opt }}, x_{\text {opt }}^{l(i-1)}\right)$.
else
sample $x_{\text {opt }}^{i}$ conditioned on $\mathcal{D}_{N}$ and already sampled $\mathbf{x}_{\text {opt }}$ dimensions if any:
$x_{\text {opt }}^{i} \sim p\left(x_{\text {opt }}^{i} \mid \mathcal{D}_{N}, x_{\text {opt }}^{l(i-1)}\right)$.
end if
end for
get full value of $\mathbf{x}_{\text {opt }}$ using the true indices: $\mathbf{x}_{\text {opt }} \leftarrow\left(x_{\text {opt }}^{1}, \ldots, x_{\text {opt }}^{D}\right)$.
return $\mathbf{x}_{\text {opt }}$

## Algorithm 2 ACE-MES

Input: observed data points $\mathcal{D}_{N}=\left\{x_{1: N}, y_{1: N}\right\}$, number of candidate points $N_{\text {cand }}$, Thompson sampling ratio for candidate point $T S_{\text {ratio }}$.

1: Initialization $N_{T S 1} \leftarrow N_{\text {cand }} \times T S_{\text {ratio }}, N_{T S 2} \leftarrow N_{\text {cand }} \times\left(1-T S_{\text {ratio }}\right)$.
2: propose $N_{\text {cand }}$ candidate points $X_{1: N_{\text {cand }}}^{*}$ according to $T S_{\text {ratio }}$ :
3: sample $X_{1: N_{T S 1}}^{*}$ using ACE-TS with conditioning on $y_{\text {opt }}(c=$ True $)$.
4: sample $X_{N_{T S 1}+1: N_{T S 1}+N_{T S 2}}^{*}$ using ACE-TS without conditioning on $y_{\text {opt }}(c=$ True $)$.
5: for $i \leftarrow 1$ to $N_{\text {cand }}$ do:
6: $\quad$ sample $y_{\text {opt }}$ for conditioning: $y_{\text {opt }} \sim p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$.
7: $\quad \alpha_{(i)}\left(\mathbf{x}_{(i)}^{*}\right)=H\left[p\left(y_{(i)}^{*} \mid \mathbf{x}_{(i)}^{*}, \mathcal{D}_{N}\right)\right]-\mathbb{E}\left(H\left[p\left(y_{(i)}^{*} \mid \mathbf{x}_{(i)}^{*}, \mathcal{D}_{N}, y_{\text {opt }}\right)\right]\right) \quad \triangleright$ see Appendix C.3.2 for more detail
8: end for
9: $\mathbf{x}_{\text {opt }}=\arg \max \boldsymbol{\alpha}$.
10: return $\mathbf{x}_{\text {opt }}$

---

#### Page 29

> **Image description.** The image shows three panels, each representing an iteration of a Bayesian optimization process on a 1D function. Each panel contains a plot with an x-axis labeled "x" ranging from -1.0 to 1.0 and a y-axis labeled "y".
>
> - **Main Plot:** A dashed gray line represents the underlying function. Black dots indicate observed points on the function. A dotted line connects the observed points, and a shaded purple area around the dotted line represents the uncertainty or confidence interval. A red asterisk marks the queried point at each iteration. In the third panel, two blue dots are also present.
>
> - **Left PDF:** An orange probability density function (PDF) is displayed on the left side of each panel, oriented vertically. This PDF likely represents the probability distribution of the optimal y-value.
>
> - **Bottom PDF:** A red PDF is shown at the bottom of each panel, oriented horizontally. This PDF likely represents the probability distribution of the optimal x-value given a sampled optimal y-value.
>
> - **Horizontal Line:** A dashed-dot orange line runs horizontally across the main plot in each panel. This line represents a sampled optimal y-value.
>
> - **Vertical Line:** A dotted gray vertical line runs from the x axis to the top of the plot, intersecting with the red asterisk.
>
> - **Panel Titles:** Each panel is labeled with "Iteration 1", "Iteration 2", and "Iteration 3" respectively.

Figure S14: Bayesian optimization example. We show here an example evolution of ACE-TS on a 1D function. The orange pdf on the left of each panel is $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$, the red pdf at the bottom of each panel is $p\left(x_{\text {opt }} \mid y_{\text {opt }}, \mathcal{D}_{N}\right)$, for a sampled $y_{\text {opt }}$ (orange dashed-dot line). The queried point at each iteration is marked with a red asterisk, while black and blue dots represent the observed points. Note how ACE is able to learn complex conditional predictive distributions for $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$.

Bayesian optimization with Minimum-value Entropy Search (ACE-MES). For Minimum-value Entropy Search (MES; Wang and Jegelka, 2017), the procedure is as follows:

1. First, we propose $N_{\text {candidate }}$ points. We generate these candidate points by sampling $80 \%$ of them using the conditional Thompson sampling approach described earlier, i.e., $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$, and the remaining $20 \%$ directly from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}\right)$. In our experiments we use $N_{\text {candidate }}=20$.
2. For each candidate point $\mathbf{x}^{*}$, we evaluate the acquisition function, which in our case is the gain in mutual information between the maximum $y_{\text {opt }}$ and the candidate point $\mathbf{x}^{*}$ (Eq. (5)).
3. To compute the first term of the right-hand side of Eq. (5), for a candidate point $\mathbf{x}^{*}$, we calculate the predictive distribution $p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}\right)$ represented in our model by a mixture of Gaussians. We compute its entropy via numerical integration over a grid.
4. For the second term of the right-hand side of Eq. (5), we perform Monte Carlo sampling to evaluate the expected entropy. For each candidate point $\mathbf{x}^{*}$, we draw $N_{\mathrm{mc}}$ samples of $y_{\mathrm{opt}}$ from the predictive distribution $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$. We set $N_{\mathrm{mc}}=20$ to ensure the procedure remains efficient while maintaining accuracy.
5. For each sampled $y_{\text {opt }}$, we determine the predictive distribution $p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}, y_{\text {opt }}\right)$. Then, for each mixture, we compute the entropy as in step 2 . We then average over samples to compute the expectation.
6. To compute the estimated MES value of candidate point $\mathbf{x}^{*}$ we subtract the computed first term to the second term of the equation Eq. (5).
7. We repeat this procedure for all candidate points $\mathbf{x}^{*}$ and select the point with the highest information gain. This point is expected to yield the lowest uncertainty about the value of the minimum, thus guiding our next query in the Bayesian optimization process.

To illustrate the implementation details of ACE-MES, we present its pseudocode in Algorithm 2.

# C.3.3 Bayesian optimization with prior over $\mathbf{x}_{\text {opt }}$

ACE is capable of injecting a prior over latents when predicting values and latents. In the context of BO, this prior could incorporate information about the location of the optimum, $\mathbf{x}_{\text {opt }}$. Several works, such as (Souza et al., 2021; Hvarfner et al., 2022; Müller et al., 2023), have explored the use of priors in BO to improve predictive performance. In our experiments, we evaluate two types of priors: strong and weak, to assess the robustness of the model under varying levels of prior knowledge. As a baseline, we utilize a $\pi$ BO-like procedure (Hvarfner et al., 2022), as described below, to perform Thompson sampling across all experiments.

---

#### Page 30

Training. For training, we generate a prior distribution similar to Appendix B.1, but with slight adjustments: when sampling the mixture distribution, we include a $50 \%$ chance of adding a uniform component. If present, the uniform distribution weight $w_{\text {unif }}$ is sampled uniformly from 0.0 to 0.2 (otherwise $w_{\text {unif }}=0$ ). The uniform component is then added as follows:

$$
\mathbf{p}=\left(w_{\text {unif }} \cdot \mathbf{p}_{\text {unif }}\right)+\left(1-w_{\text {unif }}\right) \cdot \mathbf{p}_{\text {mixture }}
$$

where $\mathbf{p}_{\text {unif }}$ represents the uniform component, and $\mathbf{p}_{\text {mixture }}$ is the sampled mixture. The inclusion of a uniform component during training means that the prior can be a mixture of an informative and a non-informative (flat) component, which will be useful later. Using this binned distribution, we then sample $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$, and use these two latent samples to construct our function, as outlined in Appendix C.3.1.

Testing. During the BO testing phase, we consider two scenarios:

1. Strong prior: We first sample a mean for the $\mathbf{x}_{\text {opt }}$ prior by drawing from a Gaussian distribution centered on the true $\mathbf{x}_{\text {opt }}$ with a standard deviation set to $10 \%$ of the domain (in our case $[-1,1]$ ), resulting in a standard deviation of 0.2 . We use this sampled prior mean and standard deviation to construct the binned prior.
2. Weak prior: The same steps are applied to generate the prior, but with a standard deviation of $25 \%$, which translates to 0.5 for our domain.

In both scenarios, we add a uniform prior component with $w_{\text {uniform }}=0.1$. The uniform component helps with model and prior misspecification, by allowing the model to explore outside the region indicated by the prior.

We compare ACE with Prior Thompson Sampling (ACEP-TS) to the no-prior ACE-TS and a baseline GP-TS. We also consider a state-of-the-art heuristic for prior injection in BO, $\pi$ BO (Hvarfner et al., 2022), with the TS acquisition procedure described below ( $\pi$ BO-TS). The procedure is repeated 10 times for each case, with different initial points sampled at random.
$\pi$ BO-TS. The main technique in $\pi \mathrm{BO}$ for injecting a prior in the BO procedure consists of rescaling the chosen acquisition function $\alpha(\mathbf{x})$ by the user-provided prior over the optimum location $\pi(\mathbf{x})$ (Eq. 6 in Hvarfner et al., 2022),

$$
\alpha_{\pi \mathrm{BO}}(\mathbf{x} ; \alpha) \propto \alpha(\mathbf{x}) \pi(\mathbf{x})^{\gamma_{n}}
$$

where $n$ is the BO iteration and $\gamma_{n}$ governs the relative influence of the prior with respect to the acquisition function, which is heuristically made to decay over iterations to reflect the increased role of the observed data. As in Hvarfner et al. (2022), we set $\gamma_{n}=\frac{\beta}{n}$ where $\beta$ is a hyperparameter reflecting the user confidence on the prior.

To implement Thompson sampling (TS) with $\pi \mathrm{BO}$, we first note that the TS acquisition function $\alpha_{\mathrm{TS}}(\mathbf{x})$ corresponds to the current posterior probability over the optimum location, and the TS procedure consists of drawing one sample from this acquisition function (as opposed to optimizing it). Thus, the $\pi \mathrm{BO}$ variant of TS ( $\pi$ BO-TS) corresponds to sampling from Eq. (S7), where the current posterior over the optimum takes the role of $\alpha(\mathbf{x})$. We sample from Eq. (S7) using a self-normalized importance sampling-resampling approach (Robert and Casella, 2004). Namely, we sample $N_{\mathrm{TS}}=100$ points from $\alpha_{\mathrm{TS}}$ using batch GP-TS, then resample one point from this batch using importance sampling weight $w \propto \frac{\alpha_{\mathrm{TS}}(\mathbf{x}) \pi(\mathbf{x})^{\beta / n}}{\alpha_{\mathrm{TS}}(\mathbf{x})}=\pi(\mathbf{x})^{\beta / n}$, where all weights are then normalized to sum to 1 . Following (Hvarfner et al., 2022), we set $\beta=10$, i.e., equal to their setting when running BO experiments with 100 iterations, as in our case.

# C.3.4 Benchmark functions and baselines

BO benchmarks. We use a diverse set of benchmark functions with input dimensions ranging from 1D to 6D to thoroughly evaluate ACE's performance on the BO task. These include (1) the non-convex Ackley function in both 1D and 2D, (2) the 1D Gramacy-Lee function, known for its multiple local minima, (3) the 1D Negative Easom function, characterized by a sharp, narrow global minimum and deceptive flat regions, (4) the non-convex 2D Branin Scaled function with multiple global minima, (5) the 2D Michalewicz function, which features a complex landscape with multiple local minima, (6) the 3D, 5D, and 6D Levy function, with numerous local minima due to its sinusoidal component, (7) the 5D and 6D Griewank function, which is highly multimodal and regularly spaced local minima, but a single smooth global minimum, (8) the 4D and 5D Rosenbrock function,

---

#### Page 31

> **Image description.** The image consists of eight line graphs arranged in a 2x4 grid. Each graph depicts the "Regret" on the y-axis versus "Iteration" on the x-axis for different test functions.
>
> - **Overall Layout:** The graphs are arranged in two rows and four columns. Each graph has the same general structure: x-axis labeled "Iteration," y-axis labeled "Regret," and multiple lines representing different algorithms.
>
> - **Axes and Labels:**
>
>   - The x-axis (Iteration) ranges from approximately 0 to 25, 50, 75, or 90, depending on the specific graph.
>   - The y-axis (Regret) ranges from 0 to varying maximum values, such as 3.8, 0.9, 1.2, 5.8, and 1.8, depending on the specific graph.
>   - Each graph has a title indicating the test function used, such as "Ackley 1D," "Easom 1D," "Michalewicz 2D," "Ackley 2D," "Levy 3D," "Hartmann 4D," "Griewank 5D," and "Griewank 6D."
>
> - **Lines and Algorithms:** Each graph contains multiple lines, each representing a different Bayesian optimization algorithm. The algorithms are:
>
>   - ACE-TS (solid blue line)
>   - ACE-MES (dashed blue line)
>   - AR-TNPD-TS (solid green line)
>   - GP-TS (solid orange line)
>   - GP-MES (dashed orange line)
>   - Random (dotted pink line)
>
> - **Shaded Regions:** Each line has a shaded region around it, representing the standard error (mean $\pm$ standard error) of the algorithm's performance. The color of the shaded region corresponds to the color of the line representing the algorithm.
>
> - **Visual Patterns:** The graphs show how the regret decreases as the number of iterations increases for each algorithm on different test functions. The performance of different algorithms varies across the different test functions. The "Random" algorithm generally performs worse than the other algorithms, as indicated by its higher regret values.

Figure S15: Bayesian optimization additional results. Regret comparison (mean $\pm$ standard error) on extended BO benchmark results on distinct test functions.
which has a narrow, curved valley containing the global minimum, and (9) the 3D, 4D, and 6D Hartmann function, a widely used standard benchmark. These functions present a range of challenges, allowing us to effectively test the robustness and accuracy of ACE across different scenarios.

BO baselines. For our baselines, we employ three methods: autoregressive Thompson Sampling with TNP-D (AR-TNPD-TS) (Nguyen and Grover, 2022), Gaussian Process-based Bayesian Optimization with the MES acquisition function (GP-MES) (Wang and Jegelka, 2017), and Gaussian Process-based Thompson Sampling (GP-TS) with 5000 candidate points. In addition, we use $\pi$ BO-TS for the prior injection case as the baseline Hvarfner et al. (2022) (using the same number of candidate points used in GP-TS). We optimize the acquisition function in GP-MES using the 'shotgun' procedure detailed later in this section (with 1000 candidate points for minimum value approximation via Gumbel sampling). Both GP-MES and GP-TS implementations are written using the BOTorch library (Balandat et al., 2020). For AR-TNPD-TS, we use the same network architecture configurations as ACE, but with a non-linear embedder and a single Gaussian head (Nguyen and Grover, 2022). Additionally, AR-TNPD-TS uses autoregressive sampling, as described in (Bruinsma et al., 2023).
We conducted our experiments with 100 BO iterations across all benchmark functions. The number of initial points was set to 3 for 1D experiments and 10 for 2D-6D experiments. These initial points were drawn uniformly randomly within the input domain. We evaluated the runtime performance of our methods and baseline algorithms on a local machine equipped with a 13th Gen Intel(R) Core(TM) i5-1335U processor and 15GB of RAM. On average, the runtime for 100 BO iterations was as follows: ACE-TS and ACEP-TS completed in approximately 5 seconds; ACE-MES required about 1.3 minutes; GP-TS and $\pi$ BO-TS took roughly 2 minutes; GP-MES took about 1.4 minutes; and AR-TNPD-TS was the slowest, requiring approximately 10 minutes, largely due to the computational cost of its autoregressive steps.

Shotgun optimizer. To perform fast optimization in parallel, we first sample 10000 points from a quasirandom grid using the Sobol sequence. Then we pick the point with the highest acquisition function value, referred to as $\mathbf{x}_{0}$. Subsequently, we sample 1000 points around $\mathbf{x}_{0}$ using a multivariate normal distribution with diagonal covariance $\sigma^{2} \mathbf{I}$, where the initial $\sigma$ is set to the median distance among the points. We re-evaluate the acquisition function over this neighborhood, including $\mathbf{x}_{0}$, and select the best point. After that, we reduce $\sigma$ by a factor of five and repeat the process, iterating from the current best point. This 'shotgun' approach allows us to zoom into a high-valued region of the acquisition function while exploiting large parallel evaluations.

---

#### Page 32

> **Image description.** This image contains six line graphs arranged in a 2x3 grid, each showing the performance of different Bayesian optimization algorithms on different test functions. Each graph plots "Regret" on the y-axis against "Iteration" on the x-axis. The algorithms compared are ACE-TS (solid blue line), ACEP-TS (dashed blue line), GP-TS (solid orange line), and πBO-TS (dashed orange line). Shaded regions around the lines indicate the standard error.
>
> Here's a breakdown of each subplot:
>
> - **Top Left:** "Ackley 1D (weak)" is the title. The y-axis ranges from 0 to 4.2. The x-axis ranges from 0 to 25.
> - **Top Middle:** "Gramacy Lee 1D (weak)" is the title. The y-axis ranges from 0 to 0.7. The x-axis ranges from 0 to 75.
> - **Top Right:** "Negeasom 1D (weak)" is the title. The y-axis ranges from 0 to 0.8. The x-axis ranges from 0 to 50.
> - **Bottom Left:** "Branin 2D (weak)" is the title. The y-axis ranges from 0 to 0.13. The x-axis ranges from 0 to 75.
> - **Bottom Middle:** "Ackley 2D (weak)" is the title. The y-axis ranges from 0 to 4.5. The x-axis ranges from 0 to 90.
> - **Bottom Right:** "Hartmann 3D (weak)" is the title. The y-axis ranges from 0 to 1.3. The x-axis ranges from 0 to 50.
>
> A legend at the top of the figure identifies the line styles and colors for each algorithm: ACE-TS (solid blue), ACEP-TS (dashed blue), GP-TS (solid orange), and πBO-TS (dashed orange).
>
> All subplots show a general trend of decreasing regret as the number of iterations increases, indicating that the algorithms are improving their performance over time. The "weak" designation in the titles likely refers to the use of a weak prior in the Bayesian optimization process.

Figure S16: Bayesian optimization with weak prior. Simple regret (mean $\pm$ standard error). Prior injection can improve the performance of ACE, making it perform competitively compared to $\pi$ BO-TS.

> **Image description.** The image consists of six line graphs arranged in a 2x3 grid. Each graph plots "Regret" on the y-axis against "Iteration" on the x-axis. All graphs share a similar style, with step-like lines representing different algorithms.
>
> - **Overall Layout:** The six graphs are arranged in two rows and three columns. Each graph has a title indicating the function being optimized and the strength of the prior.
>
> - **Axes and Labels:**
>
>   - The y-axis is labeled "Regret" on the left side of the graphs. The y-axis scales vary between the graphs, ranging from 0 to 4.2, 0 to 0.7, 0 to 0.8, 0 to 0.13, 0 to 4.5, and 0 to 1.3.
>   - The x-axis is labeled "Iteration" at the bottom of the graphs. The x-axis scales vary between the graphs, ranging from 0 to 25, 0 to 75, 0 to 50, 0 to 75, 0 to 90, and 0 to 50.
>
> - **Lines and Shaded Regions:**
>
>   - Each graph contains multiple lines representing different algorithms. The lines are colored blue, dark blue, orange, and yellow.
>   - A solid blue line represents "ACE-TS".
>   - A dashed dark blue line represents "ACEP-TS".
>   - A solid orange line represents "GP-TS".
>   - A dotted yellow line represents "πBO-TS".
>   - Each line is surrounded by a shaded region of the same color, representing the standard error.
>
> - **Titles:**
>
>   - Top row: "Ackley 1D (strong)", "Gramacy Lee 1D (strong)", "Negeasom 1D (strong)"
>   - Bottom row: "Branin 2D (strong)", "Ackley 2D (strong)", "Hartmann 3D (strong)"
>
> - **Legend:** Located at the top of the image, the legend identifies the lines: "ACE-TS" (solid blue), "ACEP-TS" (dashed dark blue), "GP-TS" (solid orange), and "πBO-TS" (dotted yellow).

Figure S17: Bayesian optimization with strong prior. Simple regret (mean $\pm$ standard error). When strong priors are used, the gap between ACE-TS and ACEP-TS is more evident compared to weak priors.

# C.3.5 Additional Bayesian optimization results.

Standard BO setting additional results. Additional results in Fig. S15 complement those in Fig. 5. While our method performs generally well across different benchmark functions, we find that it struggles on the Michalewicz function, likely because its sharp, narrow optima and highly irregular landscape differ significantly from the function classes used during training. Conversely, ACE performs competitively on Griewank, where the structured landscape aligns well with our approach. On the 2D Ackley function, the challenge may stem from its highly non-stationary nature, while our method was trained only on draws from stationary kernels. Addressing functions like Michalewicz and Ackley may require extending our relatively simple function generation process and incorporating specialized techniques like input and output warping (Müller et al., 2023) to better handle non-stationarity.

---

#### Page 33

BO with prior over $\mathbf{x}_{\text {opt }}$ additional results. Additional results on the weak prior scenario are presented in Fig. S16 and with strong prior in Fig. S17. The results indicate that ACEP-TS consistently outperforms ACE-TS, particularly when using a strong prior. In this case, the model benefits from the prior information, leading to a notable improvement in performance. Specifically, the strong prior allows the model to converge more rapidly toward the optimum.

# C. 4 Simulation-based inference

## C.4.1 Simulators

The experiments reported in Section 4.3 used three time-series models to simulate the training and test data. This section describes the simulators in more details.

Ornstein-Uhlenbeck Process (OUP) is widely used in financial mathematics and evolutionary biology due to its ability to model mean-reverting stochastic processes (Uhlenbeck and Ornstein, 1930). The model is defined as:

$$
y_{t+1}=y_{t}+\Delta y_{t}, \quad \Delta y_{t}=\theta_{1}\left[\exp \left(\theta_{2}\right)-y_{t}\right] \Delta t+0.5 w, \quad \text { for } t=1, \ldots, T
$$

where $T=25, \Delta t=0.2, x_{0}=10$, and $w \sim \mathcal{N}(0, \Delta t)$. We use a uniform prior $U([0,2] \times[-2,2])$ for the latent variables $\boldsymbol{\theta}=\left(\theta_{1}, \theta_{2}\right)$ to generate the simulated data.

Susceptible-Infectious-Recovered (SIR) is a simple compartmental model used to describe infectious disease outbreaks (Kermack and McKendrick, 1927). The model divides a population into susceptible (S), infectious (I), and recovered (R) individuals. Assuming population size $N$ and using $S_{t}, I_{t}$, and $R_{t}$ to denote the number of individuals in each compartment at time $t, t=1, \ldots, T$, the disease outbreak dynamics can be expressed as

$$
\Delta S_{t}=-\beta \frac{I_{t} S_{t}}{N}, \quad \Delta I_{t}=\beta \frac{I_{t} S_{t}}{N}-\gamma I_{t}, \quad \Delta R_{t}=\gamma I_{t}
$$

where the parameters $\beta$ and $\gamma$ denote the contact rate and the mean recovery rate. An observation model with parameters $\phi$ is used to convert the SIR model predictions to observations $\left(t, y_{t}\right)$. The experiments carried out in this work consider two observation models and simulator setups.

The setups considered in this work are as follows. First, we consider a SIR model with fixed initial condition and 10 observations $y_{t} \sim \operatorname{Bin}\left(1000, I_{t} / N\right)$ collected from $T=160$ time points at even interval, as proposed in (Lueckmann et al., 2021). Here the population size $N=10^{6}$ and the initial condition is fixed as $S_{0}=N-1$, $I_{0}=1, R_{0}=0$. We use uniform priors $\beta \sim U(0.01,1.5)$ and $\gamma \sim U(0.02,0.25)$. We used this model version in the main experiments presented in Section 4.3 and Appendix C.4.2.

In addition we consider a setup where $N$ and $I_{0}$ are unknown and we collect 25 observations $y_{t} \sim \operatorname{Poi}\left(\phi I_{t} / N\right)$ from $T=250$ time points at even interval. We use $\beta \sim U(0.5,3.5), \gamma \sim U(0.0001,1.5), \phi \sim U(50,5000)$, and $I_{0} / N \sim U(0.0001,0.01)$ with $S_{0} / N=1-I_{0} / N$ and $R_{0} / N=0$ to generate simulated samples. We used this model version in an additional experiment to test ACE on real world data, presented in Appendix C.4.5.

Turin model is a time-series model used to simulate radio propagation phenomena, making it useful for testing and designing wireless communication systems (Turin et al., 1972; Pedersen, 2019; Bharti et al., 2019). The model generates high-dimensional complex-valued time-series data and is characterized by four key parameters that control different aspects of the radio signal: $G_{0}$ controls the reverberation gain, $T$ determines the reverberation time, $\nu$ specifies the arrival rate of the point process, and $\sigma_{W}^{2}$ represents the noise variance.
The model starts with a frequency bandwidth $B=0.5 \mathrm{GHz}$ and simulates the transfer function $H_{k}$ over $N_{s}=101$ equidistant frequency points. The measured transfer function at the $k$-th point, $Y_{k}$, is given by:

$$
Y_{k}=H_{k}+W_{k}, \quad k=0,1, \ldots, N_{s}-1
$$

where $W_{k}$ denotes additive zero-mean complex circular symmetric Gaussian noise with variance $\sigma_{W}^{2}$. The transfer function $H_{k}$ is defined as:

$$
H_{k}=\sum_{l=1}^{N_{\text {points }}} \alpha_{l} \exp \left(-j 2 \pi \Delta f k \tau_{l}\right)
$$

---

#### Page 34

where $\tau_{l}$ are the time delays sampled from a one-dimensional homogeneous Poisson point process with rate $\nu$, and $\alpha_{l}$ are complex gains. The gains $\alpha_{l}$ are modeled as i.i.d. zero-mean complex Gaussian random variables conditioned on the delays, with a conditional variance:

$$
\mathbb{E}\left[\left|\alpha_{l}\right|^{2} \mid \tau_{l}\right]=\frac{G_{0} \exp \left(-\tau_{l} / T\right)}{\nu}
$$

The time-domain signal $\tilde{y}(t)$ can be obtained by taking the inverse Fourier transform:

$$
\tilde{y}(t)=\frac{1}{N_{s}} \sum_{k=0}^{N_{s}-1} Y_{k} \exp (j 2 \pi k \Delta f t)
$$

with $\Delta f=B /\left(N_{s}-1\right)$ being the frequency separation. Our final real-valued output is calculated by taking the absolute square of the complex-valued data and applying a logarithmic transformation $y(t)=10 \log _{10}\left(|\tilde{y}(t)|^{2}\right)$.
The four parameters of the model are sampled from the following uniform priors: $G_{0} \sim \mathcal{U}\left(10^{-9}, 10^{-8}\right), T \sim$ $\mathcal{U}\left(10^{-9}, 10^{-8}\right), \nu \sim \mathcal{U}\left(10^{7}, 5 \times 10^{9}\right), \sigma_{W}^{2} \sim \mathcal{U}\left(10^{-10}, 10^{-9}\right)$.

# C.4.2 Main experiments

ACE was trained on examples that included simulated time series data and model parameters divided between target and context. In these experiments, the time series data were divided into context and target data by sampling $N_{d}$ data points into the context set and including the rest in the target set. The context size $N_{d} \sim U(10,25)$ in the OUP experiments, $N_{d} \sim U(5,10)$ in the SIR experiments, and $N_{d} \sim U(50,101)$ in the Turin experiments. In addition, the model parameters were randomly assigned to either the context or target set. NPE and NRE cannot handle partial observations and was trained with the full time series data in both cases.

The ACE model used in these experiments had embedding dimension 64 and 6 transformer layers. The attention blocks had 4 heads and the MLP block had hidden dimension 128. The output head had $K=20$ MLP components with hidden dimension 128. The model was trained for $5 \times 10^{4}$ steps with batch size 32 , using learning rate $5 \times 10^{-4}$ with cosine annealing.

We used the sbi package (Tejero-Cantero et al., 2020) (https://sbi-dev.github.io/sbi/, Version: 0.22.0, License: Apache 2.0) to implement NPE and NRE. Specifically, we chose the NPE-C (Greenberg et al., 2019) and NRE-C (Miller et al., 2022) with Masked Autoregressive Flow (MAF) (Papamakarios et al., 2017) as the inference network. We used the default configuration with 50 hidden units and 5 transforms for MAF, and training with a fixed learning rate $5 \times 10^{-4}$. For Simformer (Gloeckler et al., 2024), we used their official package (https://github.com/mackelab/simformer, Version: 2, License: MIT). We used the same configuration as in our setup for the transformer, while we used their default configuration for the diffusion part. For a fair comparison, we pre-generated $10^{4}$ parameter-simulation pairs for all methods. We also normalized the parameters of the Turin model when feeding into the networks. For evaluation, we randomly generated 100 observations and assessed each method across 5 runs. For the RMSE evaluation, given $N_{\text {obs }}$ observations, with $N_{\text {post }}$ posterior samples generated for each observation, and $L$ latent parameters, our RMSE metric is calculated as:

$$
\operatorname{RMSE}=\frac{1}{N_{\mathrm{obs}}} \sum_{i=1}^{N_{\mathrm{obs}}} \sqrt{\frac{1}{L \cdot N_{\mathrm{post}}} \sum_{l=1}^{L} \sum_{j=1}^{N_{\mathrm{post}}}\left(\theta_{i, l}-\hat{\theta}_{i, l, j}\right)^{2}}
$$

where $\theta_{i, l}$ represents the true value of the $l$-th latent parameter for the $i$-th observation, and $\hat{\theta}_{i, l, j}$ represents the $j$-th posterior sample of the $l$-th latent parameter for the $i$-th observation. This approach first calculates the RMSE for each observation (averaging across all latent dimensions and posterior samples for that observation), and then averages these observation-specific RMSE values to obtain the final metric. For MMD, we use an exponentiated quadratic kernel with a lengthscale of 1 .

Statistical comparisons. We evaluate models based on their average results across multiple runs and perform pairwise comparisons to identify models with comparable performance. The results from pairwise comparisons are used in Table 1 to highlight in bold the models that are considered best in each experiment. The following procedure is used to determine the best models:

---

#### Page 35

- First, we identify the model (A) with the highest empirical mean and highlight it in bold.
- For each alternative model (B), we perform $10^{5}$ bootstrap iterations to resample the mean performance for both model A and model B.
- We then calculate the proportion of bootstrap iterations where model B outperforms model A.
- If this proportion is larger than the significance level $(\alpha=0.05)$, model B is considered statistically indistinguishable from model A.
- All models that are not statistically significantly different from the best model are highlighted in bold.

# C.4.3 Ablation study: Gaussian vs. mixture-of-Gaussians output heads

To assess the impact of using a Gaussian versus a mixture-of-Gaussians output head in ACE, we conduct an ablation study on the SBI tasks. In theory, a mixture-of-Gaussians head should improve performance when the predictive or posterior data distributions are non-Gaussian. Table S2 shows the results. As expected, we observe improvements in OUP and Turin when using a mixture-of-Gaussians head. This suggests that more flexible distributional families better capture complex distributions. However, for the SIR task, the performance difference is negligible as the posteriors are largely Gaussian. These findings align with our expectations.

|       |                                            | Gaussian (ablation) | Mixture-of-Gaussians (ACE) |
| :---: | :----------------------------------------: | :-----------------: | :------------------------: |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $0.90(0.01)$     |  $\mathbf{1 . 0 3}(0.02)$  |
|  OUP  |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.48(0.01)$     |        $0.48(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.52(0.00)$     |  $\mathbf{0 . 5 1}(0.00)$  |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $6.80(0.02)$     |        $6.78(0.02)$        |
|  SIR  |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.02(0.00)$     |        $0.02(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.02(0.00)$     |        $0.02(0.00)$        |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $2.73(0.02)$     |  $\mathbf{3 . 1 4}(0.02)$  |
| Turin |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.24(0.00)$     |        $0.24(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.36(0.00)$     |  $\mathbf{0 . 3 5}(0.00)$  |

Table S2: Ablation study comparing single Gaussian versus mixture-of-Gaussians output heads across SBI tasks. Mean and standard deviation from 5 runs are reported. mixture-of-Gaussians heads benefit complex distributions (OUP and Turin), while maintaining similar performance on simpler tasks (SIR).

## C.4.4 Simulation-based calibration

To evaluate the calibration of the approximate posteriors obtained by ACE, we apply simulation-based calibration (SBC; Talts et al. 2018) on the Turin model to evaluate whether the approximate posteriors produced by ACE are calibrated. We recall that SBC checks if a Bayesian inference process is well-calibrated by repeatedly simulating data from parameters drawn from the prior and inferring posteriors under those priors and simulated datasets. If the inference is calibrated, the average posterior should match the prior. Equivalently, when ranking the true parameters within each posterior, the ranks should follow a uniform distribution (Talts et al., 2018).

We use the following procedure for SBC: for a given prior, we first sample 1000 samples from the prior distribution and generate corresponding simulated data. Then we use ACE to approximate the posteriors and subsequently compare the true parameter values with samples drawn from the inferred posterior distribution. To visualize the calibration, we plot the density of the posterior samples against the prior samples. If the model is well-calibrated, the posterior distribution should recover the true posterior, which results in a close match between the density of the posterior samples and the prior. We also present the fractional rank statistic against the ECDF difference (Săilynoja et al., 2022). Ideally, the ECDF difference between the rank statistics and the theoretical uniform distribution should remain close to zero, indicating well-calibrated posteriors.

Fig. S18 shows that our ACE is well-calibrated with pre-defined uniform priors across all four latents. Since ACEP allows conditioning on different priors at runtime, we also test the calibration of ACEP using randomly generated priors (following Appendix B.1). For comparison, we show what happens if we forego prior-injection, using vanilla ACE instead of ACEP. The visualization on one set of priors is shown in Fig. S19. As expected,

---

#### Page 36

> **Image description.** This image presents a figure composed of eight subplots arranged in two rows and four columns, displaying simulation-based calibration results. The top row shows density plots, while the bottom row shows fractional rank statistics against ECDF (Empirical Cumulative Distribution Function) differences.
>
> - **Top Row (Density Plots):** Each of the four plots in the top row displays the density of posterior samples from ACE (presumably an algorithm) compared with prior samples.
>
>   - The y-axis is labeled "Density."
>   - Each plot contains two curves: one in gray representing "prior samples" and one in purple representing "ACE."
>   - The x-axis is scaled differently in each plot, with labels like "x10^8", "x10^-10", and "x10^9", indicating different scales for the x-axis values. The x-axis values range from 0.0 to 1.0 in the first two plots, from 0 to 6 in the third plot, and from 0.0 to 1.0 in the fourth plot.
>   - The shapes of the curves vary slightly between the plots, but generally, the purple ACE curve is more peaked than the gray prior samples curve.
>
> - **Bottom Row (Fractional Rank Statistics):** Each of the four plots in the bottom row shows the fractional rank statistic against the ECDF difference.
>   - The y-axis is labeled "Δ ECDF".
>   - The x-axis is labeled "Fractional Rank" and ranges from 0.00 to 1.00.
>   - Each plot contains a purple line representing the ACE data.
>   - A gray shaded oval region is present in each plot, presumably representing a 95% confidence band. The purple line fluctuates within or near the boundaries of this gray region.
>   - Labels are present below the x-axis of each plot: "G0", "T", "V", and "σw^2".
>
> Overall, the figure appears to be comparing the performance of an algorithm (ACE) against prior samples, with the bottom row indicating the calibration of the algorithm within a certain confidence interval.

Figure S18: Simulation-based calibration of ACE on the Turin model. The top row shows the density of the posterior samples from ACE compared with the prior samples. The bottom row shows the fractional rank statistic against the ECDF difference with $95 \%$ confidence bands. ACE is well-calibrated.
vanilla ACE (without prior-injection) does not include the correct prior information and shows suboptimal calibration performance, whereas ACEP correctly leverages the provided prior information and shows closer alignment with the prior and lower ECDF deviations. We also calculate the average absolute deviation over 100 randomly sampled priors. In the prior-injection setting, ACEP demonstrates better calibration, with an average deviation of $0.03 \pm 0.01$ compared to $0.10 \pm 0.05$ for ACE without the correct prior.

# C.4.5 Extended SIR model on real-world data

We present here results obtained by considering an extended four-parameter version of the SIR model then applied to real-world data. We further include details on the training data and model configurations used in the real-data experiment as well as additional evaluation results from experiments carried out with simulated data. As our real-world data, we used a dataset that describes an influenza outbreak in a boarding school. The dataset is available in the R package outbreaks (https://cran.r-project.org/package=outbreaks, Version: 1.9.0, License: MIT).

Methods. The four-parameter SIR model we used is detailed in Appendix C.4.1 (last paragraph). The ACE models were trained with samples constructed based on simulated data as follows. The observations were divided into context and target points by sampling $N_{d} \sim U(2,20)$ data points into the context set and 2 data points into the target set. The examples included $50 \%$ interpolation tasks where the context and target points were sampled at random (without overlap) and $50 \%$ forecast tasks where the points were sampled in order. The model parameters were divided between the context and target set by sampling the number to include $N_{l} \sim U(0,4)$ and sampling the $N_{l}$ parameters from the parameter set at random. The parameters were normalized to range $[-1,1]$ and the observations were square-root compressed and scaled to the approximate range $[-1,1]$.

The ACE models had the same architecture as the models used in the main experiment, but the models were trained for $10^{5}$ steps with batch size 32 . In this experiment, we generated the data online during the training, which means that the models were trained with $3.2 \times 10^{6}$ samples. The NPE models used in this experiment had the same configuration as the model used in the main experiment, for fair comparison, the models were now trained with $3.2 \times 10^{6}$ samples. Each sample corresponded to a unique simulation and the full time series was used as the observation data.

To validate model predictions, we note that ground-truth parameter values are not available for real data. Instead, we examined whether running the simulator with parameters sampled from the posterior can replicate the observed data. For reference, we also included MCMC results. The MCMC posterior was sampled with Pyro (Bingham et al., 2018) (https://pyro.ai/, Version: 1.9.0, License: Apache 2.0) using the random walk kernel

---

#### Page 37

> **Image description.** The image presents a figure composed of eight subplots arranged in two rows and four columns. The top row displays probability density functions (PDFs), while the bottom row shows the difference in empirical cumulative distribution functions (ΔECDF). Each column corresponds to a different parameter: G0, T, v, and σw^2.
>
> - **Top Row (PDFs):** Each subplot in the top row displays three overlapping density curves. The y-axis is labeled "Density". The x-axis label varies depending on the column, with units of x10^8, x10^8, x10^-10, and x10^-9 respectively. A legend in the first subplot identifies the curves as "prior samples" (gray), "ACE" (purple), and "ACEP" (green). The curves show the distribution of the parameters.
>
> - **Bottom Row (ΔECDF):** Each subplot in the bottom row shows the difference in the empirical cumulative distribution function (ΔECDF) on the y-axis, ranging from -0.10 to 0.10. The x-axis is labeled "Fractional Rank" and ranges from 0.00 to 1.00. The curves represent "ACE" (purple) and "ACEP" (green). A gray, horizontally-oriented oval is present in the background of each subplot, visually indicating a region of acceptable deviation. The x-axis labels for each column are G0, T, v, and σw^2.
>
> In summary, the figure compares the performance of ACE and ACEP against prior samples in terms of parameter distributions (PDFs) and ECDF differences.

Figure S19: Simulation-based calibration of ACE and ACEP on the Turin model with an example custom prior. ACEP demonstrates improved calibration by closely following the prior distribution and showing lower deviations in the ECDF difference, highlighting its ability to condition on user-specified priors effectively.
and sampling 4 chains with $5 \times 10^{4}$ warm-up steps and $5 \times 10^{4}$ samples.

> **Image description.** The image contains three line graphs arranged side-by-side, comparing different methods: ACE, NPE, and MCMC. Each graph plots "Count" on the y-axis versus "Time" on the x-axis.
>
> - **General Layout:** The three graphs are labeled "ACE," "NPE," and "MCMC" respectively above each plot. Each graph displays a light blue line representing the "PPD mean," a shaded light blue area representing the "PPD 95% CI" (Credible Interval), and black dots representing "observed" data points. The y-axis ranges from 0 to 400, and the x-axis ranges from 0 to 10 in all three graphs.
>
> - **ACE Graph:** The graph labeled "ACE" shows the PPD mean line rising to a peak around Time=5 and then decreasing. The black dots are scattered around the PPD mean line, generally within the shaded 95% CI. The text "log-prob ↑ -64.4" is displayed above the graph.
>
> - **NPE Graph:** The graph labeled "NPE" is visually similar to the ACE graph, with the PPD mean line peaking around Time=5 and decreasing. The black dots are also scattered around the PPD mean line and mostly within the 95% CI. The text "log-prob ↑ -64.6" is displayed above the graph.
>
> - **MCMC Graph:** The graph labeled "MCMC" follows the same general pattern as the ACE and NPE graphs. The PPD mean line peaks around Time=5 and decreases. The black dots are scattered around the PPD mean line, largely within the 95% CI. The text "log-prob ↑ -62.9" is displayed above the graph.
>
> - **Legend:** A legend is present to the right of the graphs, indicating that the blue line represents the "PPD mean," the shaded blue area represents the "PPD 95% CI," and the black dots represent "observed" data.

Figure S20: SIR model on a real dataset. Posterior predictive distributions based on the ACE, NPE, and MCMC posteriors. The dataset is mildly misspecified, in that even MCMC does not fully match the data.

Results. The posterior predictive distributions and log-probabilities for observed data calculated based on ACE, NPE, and MCMC results are shown in Fig. S20. For this visualization, ACE and NPE models were trained once, and simulations were carried out with 5000 parameters sampled from each posterior distribution. The log-probabilities observed in this experiment are -64.4 with ACE, -64.6 with NPE. Repeating ACE and NPE training and posterior estimation 10 times, the average log-probabilities across the 10 runs were -65.1 (standard deviation 0.4 ) with ACE and -65.5 (standard deviation 0.7 ) with NPE, showing a similar performance. The ACE predictions used in this experiment are sampled autoregressively (see Appendix B.4). These results show that ACE can handle inference with real data.

Validation on simulated data. For completeness, we performed a more extensive validation of ACE and other methods with the extended SIR model using simulated data. Specifically, we assessed the ACE and NPE models on simulated data and evaluated the same ACE models in a data completion task with the TNP-D baseline. All the training details remain the same as in the real-world experiment for ACE and NPE. The TNP-D models had the same overall architecture as ACE but used a different embedder and output head. The MLP block in the TNP-D embedder had hidden dimension 64 and the MLP block in the single-component output head hidden dimension 128. The TNP-D models were trained for $10^{5}$ steps with batch size 32 . The evaluation set used in these experiments included 1000 simulations sampled from the training distribution and the evaluation metrics included log-probabilities and coverage probabilities calculated based on $95 \%$ quantile intervals that were

---

#### Page 38

Table S3: Comparison between ACE and NPE in posterior estimation task in the extended SIR model. The ACE predictions were generated autoregressively so both methods target the joint posterior. The estimated posteriors are compared based on log-probabilities and $95 \%$ marginal coverage probabilities. The evaluation set includes 1000 examples and we report the mean and (standard deviation) from 10 runs. ACE log-probabilities are on average better than NPE log-probabilities and the coverage probabilities are close to the nominal level 0.95 .

|          | $\log$-probs $(\uparrow)$ | cover $\beta$ | cover $\gamma$ | cover $\phi$ | cover $I_{0}$ |  cover ave   |
| :------: | :-----------------------: | :-----------: | :------------: | :----------: | :-----------: | :----------: |
|   NPE    |       $6.63(0.16)$        | $0.92(0.01)$  |  $0.94(0.01)$  | $0.94(0.01)$ | $0.92(0.01)$  | $0.93(0.01)$ |
| ACE (AR) |       $7.38(0.04)$        | $0.96(0.00)$  |  $0.97(0.00)$  | $0.97(0.00)$ | $0.96(0.00)$  | $0.97(0.00)$ |

Table S4: ACE posterior estimation based on incomplete data with $M$ observation points using either independent or autoregressive predictions. The estimated posteriors are evaluated using (a) log-probabilities and (b) average $95 \%$ marginal coverage probabilities. We report the mean and (standard deviation) from 10 runs. The logprobabilities improve when the context size $M$ increases and when autoregressive predictions are used.

|     |  $M=25$  |    $M=20$    |    $M=15$    |    $M=10$    |    $M=5$     |
| :-: | :------: | :----------: | :----------: | :----------: | :----------: | ------------ |
| (a) |   ACE    | $4.94(0.04)$ | $4.55(0.03)$ | $3.87(0.02)$ | $2.82(0.03)$ | $0.88(0.03)$ |
|     | ACE (AR) | $7.38(0.04)$ | $6.93(0.04)$ | $6.21(0.04)$ | $5.11(0.04)$ | $2.91(0.05)$ |
|     |   ACE    | $0.97(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ | $0.95(0.00)$ | $0.96(0.00)$ |
|     | ACE (AR) | $0.97(0.00)$ | $0.97(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.97(0.00)$ |

estimated based on 5000 samples.
We start with the posterior estimation task where we used ACE and NPE to predict simulator parameters based on the simulated observations with 25 observation points. The results are reported in Table S3. We observe that the ACE log-probabilities are on average better than NPE log-probabilities and that both methods have marginal coverage probabilities close to the nominal level 0.95 .

The simulated observations used in the previous experiment were complete with 25 observation points. Next, we evaluate ACE posteriors estimated based on incomplete data with $5-20$ observation points. NPE is not included in this experiment since it cannot handle incomplete observations. Instead, we use this experiment to compare independent and autoregressive ACE predictions. The results are reported in Table S4. The log-probabilities indicate that both independent and autoregressive predictions improve when more observation points are available while the coverage probabilities are close to the nominal level in all conditions. That autoregressive predictions result in better log-probabilities than independent predictions indicates that ACE is able to use dependencies between simulator parameters.

Table S5: Comparison between ACE and TNP-D in data completion task in the extended SIR model. The estimated predictive distributions are compared based on (a) log-probabilities and (a) $95 \%$ coverage probabilities. We report the mean and (standard deviation) from 10 runs. ACE log-probabilities are on average better than TNP-D log-probabilities and improve both when the context size $M$ increases or when predictions are conditioned on the simulator parameters $\theta$.

|     |        $M=20$         |    $M=15$    |    $M=10$    |    $M=5$     |
| :-: | :-------------------: | :----------: | :----------: | :----------: | ------------ |
| (a) |         TNP-D         | $10.1(0.11)$ | $9.99(0.09)$ | $9.44(0.10)$ | $8.02(0.07)$ |
|     |          ACE          | $14.2(0.31)$ | $13.8(0.31)$ | $13.2(0.31)$ | $11.4(0.28)$ |
|     | $\mathrm{ACE}+\theta$ | $14.7(0.31)$ | $14.6(0.31)$ | $14.6(0.30)$ | $14.3(0.30)$ |
| (b) |         TNP-D         | $0.96(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ | $0.95(0.00)$ |
|     |          ACE          | $0.97(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ |
|     | $\mathrm{ACE}+\theta$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ |

The same ACE models that have been evaluated in the posterior estimation (latent prediction) task can also make predictions about the unobserved values in incomplete data. To evaluate ACE in the data completion task,

---

#### Page 39

we selected 5 target observations from each evaluation sample and used 5-20 remaining observations as context. We used ACE to make target predictions either based on the context data alone or based on both context data and the simulator parameters $\theta$. For comparison, we also evaluated data completion with TNP-D. The results are reported in Table S5. We observe that ACE log-probabilities are on average better than TNP-D log-probabilities and improve when simulator parameters are available as context. In these experiments, both ACE and TNP-D were used to make independent predictions.

# C. 5 Computational resources and software

For the experiments and baselines, we used a GPU cluster containing AMD MI250X GPUs. All experiments can be run using a single GPU with a VRAM of 50 GB . Most of the experiments took under 6 hours, with the exception of a few BO experiments that took around 10 hours. The core code base was built using Pytorch (Paszke et al., 2019) (https://pytorch.org/ Version: 2.2.0, License: modified BSD license) and based on the Pytorch implementation for TNP (Nguyen and Grover, 2022) (https://github.com/tung-nd/TNP-pytorch, License: MIT). Botorch (Balandat et al., 2020) (https://github.com/pytorch/botorch Version: 0.10.0, License: MIT) was used for the implementation of GP-MES, GP-TS, and $\pi$ BO-TS.
