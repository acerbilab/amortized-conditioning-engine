```
@article{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  journal={28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)},
  year={2025}
}
```

---

#### Page 1

# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference

Paul E. Chang ${ }^{* 1}$ Nasrulloh Loka*1 Daolang Huang ${ }^{* 2}$ Ulpu Remes ${ }^{3}$ Samuel Kaski ${ }^{2,4}$ Luigi Acerbi ${ }^{1}$<br>${ }^{1}$ Department of Computer Science, University of Helsinki, Helsinki, Finland<br>${ }^{2}$ Department of Computer Science, Aalto University, Espoo, Finland<br>${ }^{3}$ Department of Mathematics and Statistics, University of Helsinki, Helsinki, Finland<br>${ }^{4}$ Department of Computer Science, University of Manchester, Manchester, United Kingdom

#### Abstract

Amortized meta-learning methods based on pre-training have propelled fields like natural language processing and vision. Transformerbased neural processes and their variants are leading models for probabilistic meta-learning with a tractable objective. Often trained on synthetic data, these models implicitly capture essential latent information in the data-generation process. However, existing methods do not allow users to flexibly inject (condition on) and extract (predict) this probabilistic latent information at runtime, which is key to many tasks. We introduce the Amortized Conditioning Engine (ACE), a new transformer-based meta-learning model that explicitly represents latent variables of interest. ACE affords conditioning on both observed data and interpretable latent variables, the inclusion of priors at runtime, and outputs predictive distributions for discrete and continuous data and latents. We show ACE's practical utility across diverse tasks such as image completion and classification, Bayesian optimization, and simulation-based inference, demonstrating how a general conditioning framework can replace task-specific solutions.

## 1 INTRODUCTION

Amortization, or pre-training, is a crucial technique for improving computational efficiency and generalization across many machine learning tasks, from regression (Garnelo et al., 2018a) to optimization (Amos, 2022)

> **Image description.** This image presents a diagram illustrating probabilistic conditioning and prediction across three different scenarios, labeled (a), (b), and (c). Each scenario is depicted in a row, showing the relationship between "Data" and "Latent" variables before and after a process represented by an arrow.
>
> - **Row (a):** This row deals with image completion and classification.
>
>   - The first panel shows a blurred grayscale image of the digit "3".
>   - The second panel shows a bar chart representing the latent variable. The y-axis is labeled with values 2, 3, and 7. The bars are horizontal and orange. A "Label" is written above the bar chart.
>   - An arrow points to the third panel, which shows the completed image of the digit "3" in purple and black.
>   - The fourth panel shows another bar chart, similar to the second panel, representing the latent variable after the process.
>
> - **Row (b):** This row demonstrates Bayesian optimization.
>
>   - The first panel shows a scatter plot with several black dots.
>   - The second panel shows another scatter plot. In the bottom and left of the scatter plot are a horizontal orange bar labeled "xopt" and a vertical green bar labeled "yopt".
>   - An arrow points to the third panel, which shows a plot of a function with data points marked as black dots with error bars. A purple shaded region surrounds the function.
>   - The fourth panel shows two plots, one above the other. The top plot shows a function with a grey shaded region surrounding it. The bottom plot shows a distribution represented by an orange filled curve.
>
> - **Row (c):** This row shows simulation-based inference.
>   - The first panel shows a scatter plot with several black dots.
>   - The second panel shows a distribution represented by an orange filled curve. A "θ" symbol is written above the distribution.
>   - An arrow points to the third panel, which shows a plot of a function with data points marked as black dots with error bars.
>   - The fourth panel shows a distribution represented by an orange filled curve.
>
> The rows are separated by dashed lines. The labels "Data" and "Latent" are written above the columns.

Figure 1: Probabilistic conditioning and prediction. Many tasks reduce to probabilistic conditioning on data and key latent variables (left) and then predicting data and latents (right). (a) Image completion and classification (data: pixels; latents: classes). Top: Class prediction. Bottom: Conditional generation. (b) Bayesian optimization (data: function values; latents: optimum location $x_{\text {opt }}$ and value $y_{\text {opt }}$ ). We predict both the function values and $x_{\text {opt }}, y_{\text {opt }}$ given function observations and a prior over $x_{\text {opt }}, y_{\text {opt }}$ (here flat). (c) Simulator-based inference (data: observations; latents: model parameter $\theta$ ). Given data and a prior over $\theta$, we can compute both the posterior over $\theta$ and predictive distribution over unseen data. Our method fully amortizes probabilistic conditioning and prediction.

[^0]
[^0]: \*Equal contribution.

---

#### Page 2

and simulation-based inference (Cranmer et al., 2020). By training a deep neural network on a large dataset of related problems and solutions, amortization can achieve both fast inference time, solving problems with a single forward pass, and meta-learning, better adapting to new problems by capturing high-level statistical relations (Brown et al., 2020). Probabilistic metalearning models based on the transformer architecture (Vaswani et al., 2017) are the state-of-the-art for amortizing complex predictive data distributions (Nguyen and Grover, 2022; Müller et al., 2022).

This paper capitalizes on the fact that many machine learning problems reduce to predicting data and task-relevant latent variables after conditioning on other data and latents (Ghahramani, 2015); see Fig. 1. Moreover, in many scenarios the user has exact or probabilistic information (priors) about task-relevant variables that they would like to leverage, but incorporating such prior knowledge is challenging, requiring dedicated, expensive solutions.

For instance, in Bayesian optimization (Garnett, 2023), the goal is to find the location $\mathbf{x}_{\text {opt }}$ and value $y_{\text {opt }}$ of the global minimum of a function (Fig. 1b). These are latent variables, distinct from the data $\mathcal{D}_{N}=$ $\left\{\left(\mathbf{x}_{1}, y_{1}\right), \ldots,\left(\mathbf{x}_{N}, y_{N}\right)\right\}$ consisting of observed function location and values. Following information-theoretical principles, we should query points that would reduce uncertainty (entropy) about the latent optimum's location or value. However, predictive distributions over $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$ are intractable, leading to complex approximation techniques (Hennig and Schuler, 2012; HernándezLobato et al., 2014; Wang and Jegelka, 2017). Another case of interest is when prior information is available, such as knowing $y_{\text {opt }}$ due to the problem formulation (e.g., $y_{\text {opt }}=0$ for some theoretical reason), or expert knowledge of more likely locations for $\mathbf{x}_{\text {opt }}$ - but injecting such information in current methods is highly nontrivial (Nguyen and Osborne, 2020; Souza et al., 2021; Hvarfner et al., 2022). Crucially, if we had access to $p\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }} \mid \mathcal{D}_{N}\right)$, and we could likewise condition on $\mathbf{x}_{\text {opt }}$ or $y_{\text {opt }}$ or set priors over them, many challenging tasks that so far have required dedicated heuristics or computationally expensive solutions would become straightforward. Similar challenges extend to many machine learning tasks, including regression and classification (Fig. 1a), and simulation-based inference (Fig. 1c), all involving predicting, sampling, and probabilistic conditioning on either exact values or distributions (priors) at runtime.

In this work, we address the desiderata above by introducing the Amortized Conditioning Engine (ACE),
a general amortization framework which extends transformer-based meta-learning architectures (Nguyen and Grover, 2022; Müller et al., 2022) with explicit and flexible probabilistic modeling of task-relevant latent variables. Our main goal with ACE is to develop a method capable of addressing a variety of tasks that would otherwise require bespoke solutions and approximations. Through the lens of amortized probabilistic conditioning and prediction, we provide a unifying methodological bridge across multiple fields.

Contributions. Our contributions include:

- We propose ACE, a transformer-based architecture that simultaneously affords, at inference time, conditioning and autoregressive probabilistic prediction for arbitrary combinations of data and latent variables, both continuous and discrete.
- We introduce a new technique for allowing the user to provide probabilistic information (priors) over each latent variable at inference time.
- We substantiate the generality of our framework through a series of tasks from different fields, including image completion and classification, Bayesian optimization, and simulation-based inference, on both synthetic and real data.

ACE requires availability of predefined, interpretable latent variables during training (e.g., $\mathbf{x}_{\text {opt }}, y_{\text {opt }}$ ). For many tasks, this can be achieved through explicit construction of the generative model, as shown in Section 4.

## 2 PRELIMINARIES

In this section, we review previous work on transformerbased probabilistic meta-learning models within the framework of prediction maps (Foong et al., 2020; Markou et al., 2022) and Conditional Neural Processes (CNPs; Garnelo et al., 2018b). We denote with $\mathbf{x} \in \mathcal{X} \subseteq \mathbb{R}^{D}$ input vectors (covariates) and $y \in \mathcal{Y} \subseteq \mathbb{R}$ scalar output vectors (values). Table S1 in Appendix A summarizes key acronyms used in the paper.

Prediction maps. A prediction map $\pi$ is a function that maps (1) a context set of input/output pairs $\mathcal{D}_{N}=$ $\left\{\left(\mathbf{x}_{1}, y_{1}\right), \ldots,\left(\mathbf{x}_{N}, y_{N}\right)\right\}$ and (2) a collection of target inputs $\mathbf{x}_{1: M}^{\star} \equiv\left(\mathbf{x}_{1}^{\star}, \ldots, \mathbf{x}_{M}^{\star}\right)$ to a distribution over the corresponding target outputs $y_{1: M}^{\star} \equiv\left(y_{1}^{\star}, \ldots, y_{M}^{\star}\right)$ :

$$
\pi\left(y_{1: M}^{\star} \mid \mathbf{x}_{1: M}^{\star} ; \mathcal{D}_{N}\right)=p\left(y_{1: M}^{\star} \mid \mathbf{r}\left(\mathbf{x}_{1: M}^{\star}, \mathcal{D}_{N}\right)\right)
$$

where $\mathbf{r}$ is a representation vector of the context and target sets that parameterizes the predictive distribution. Such map should be invariant with respect to permutations of the context set and, separately, of the

---

#### Page 3

targets (Foong et al., 2020). The Bayesian posterior is a prediction map, with the Gaussian Process (GP; Rasmussen and Williams, 2006) posterior a special case.

Diagonal prediction maps. We call a prediction map diagonal if it represents and predicts each target independently:

$$
\pi\left(y_{1: M}^{*}\left|\mathbf{x}_{1: M}^{*} ; \mathcal{D}_{N}\right)=\prod_{m=1}^{M} p\left(y_{m}^{*} \mid \mathbf{r}\left(\mathbf{x}_{m}^{*}, \mathbf{r}_{\mathcal{D}}\left(\mathcal{D}_{N}\right)\right)\right)\right.
$$

where $\mathbf{r}_{\mathcal{D}}$ denotes a representation of the context alone. Diagonal outputs ensure that a prediction map is permutation and marginalization consistent with respect to the targets for a fixed context, necessary conditions for a valid stochastic process (Markou et al., 2022). Importantly, while diagonal prediction maps directly model conditional 1D marginals, they can represent any conditional joint distribution autoregressively (Bruinsma et al., 2023). CNPs are diagonal prediction maps parameterized by deep neural networks (Garnelo et al., 2018b). CNPs encode the context set to a fixeddimension vector $\mathbf{r}_{\mathcal{D}}$ via a DeepSet (Zaheer et al., 2017) to ensure permutation invariance of the context, and each target predictive distribution is a Gaussian whose mean and variance are decoded by a multi-layer perceptron (MLP). Given the likelihood in Eq. (2), CNPs are easily trainable via maximum-likelihood optimization of parameters of encoder and decoder networks by sampling batches of context and target sets.

Transformers. Transformers (Vaswani et al., 2017) are deep neural networks based on the attention mechanism, which computes a weighted combination of hidden states of dimension $D_{\text {emb }}$ through three learnable linear projections: query $(Q)$, key $(K)$, and value $(V)$. The attention operation, $\operatorname{softmax}\left(Q K^{T} / \sqrt{D_{\text {emb }}}\right) V$, captures complex relationships between inputs. Selfattention computes $Q, K$, and $V$ from the same input set, whereas cross-attention uses two sets, one for computing the queries and another for keys and values. A standard transformer architecture consists of multiple stacked self-attention and MLP layers with residual connections and layer normalization (Vaswani et al., 2017). Without specifically injecting positional information, transformers process inputs in a permutationequivariant manner.

Transformer diagonal prediction maps. We define here a general transformer prediction map model family, focusing on its diagonal variant (TPM-D), which includes the TNP-D model from Nguyen and Grover (2022) and prior-fitted networks (PFNs; Müller et al., 2022). TPM-Ds are not strictly CNPs because the context set is encoded by a variable-size representation, but they otherwise share many similarities. In

> **Image description.** This image shows a set of eight heatmaps arranged in a 2x4 grid, each representing a probability distribution. The heatmaps are colored with a gradient from dark purple to bright yellow, indicating increasing probability density.
>
> - **Axes and Labels:** Each heatmap has implicit x and y axes. The x-axis is labeled with "μ" (mu) below each column of heatmaps, and the y-axis is labeled with "σ" (sigma) to the left of each row of heatmaps.
>
> - **Marginal Distributions:** The first heatmap in each row also includes marginal distributions plotted along the axes.
>
>   - The top-left heatmap has a histogram-like plot along the top edge and a similar plot along the left edge.
>   - The bottom-left heatmap also has histogram-like plots along the top and left edges.
>
> - **Heatmap Patterns:** The heatmaps display different patterns of probability density:
>
>   - **(a) Top Row:** A vertical stripe of high probability density. Bottom Row: An elongated blob of high probability density.
>   - **(b) Top Row:** An elliptical blob with the major axis aligned vertically. Bottom Row: Similar to the top row, but the blob is more compact.
>   - **(c) Top Row:** A vertically oriented, teardrop-shaped blob of high probability density. Bottom Row: A small, horizontally oriented blob of high probability density.
>   - **(d) Top Row:** A vertically oriented, teardrop-shaped blob of high probability density, but more compact than (c). Bottom Row: A small, horizontally oriented blob of high probability density, but more compact than (c).
>
> - **Panel Labels:** Each column of heatmaps is labeled with a lowercase letter in parentheses below the bottom heatmap: (a), (b), (c), and (d).

Figure 2: Prior amortization. Two example posterior distributions for the mean $\mu$ and standard deviation $\sigma$ of a 1D Gaussian. (a) Prior distribution over $\boldsymbol{\theta}=(\mu, \sigma)$ set at runtime. (b) Likelihood for the observed data. (c) Ground-truth Bayesian posterior. (d) ACE's predicted posterior approximates well the true posterior.
a TPM-D, context data $\left(\mathbf{x}_{n}, y_{n}\right)_{n=1}^{N}$ and target inputs $\left(\mathbf{x}_{m}^{*}\right)_{m=1}^{M}$ are first individually mapped to vector embeddings of size $D_{\text {emb }}$ via an embedder $f_{\text {emb }}$, often a linear map or an MLP. The embedded context points are processed together via a series of $B-1$ transformer layers implementing self-attention within the context set. We denote by $\mathbf{E}^{(b)}=\left(\mathbf{e}_{1}^{(b)}, \ldots, \mathbf{e}_{N}^{(b)}\right)$ the matrix of output embeddings of the $b$-th transformer layer, with $b=0$ the embedding layer. The encoded context representation is the stacked output of all layers, i.e. $\mathbf{r}_{\mathcal{D}}=\left(\mathbf{E}^{(0)}, \ldots, \mathbf{E}^{(B-1)}\right)$, whose size is linear in the context size $N$. The decoder is represented by a series of $B$ transformer layers that apply cross-attention from the embedded target points to the context set layer-wise, with the $b$-th target transformer layer attending the output $\mathbf{E}^{(b-1)}$ of the previous context transformer layer. The decoder transformer layers operate in parallel on each target point. The $M$ outputs of the $B$-th decoder block are fed in parallel to an output head yielding the predictive distribution, Eq. (2). This shows that indeed TPM-Ds are diagonal prediction maps. The predictive distribution is a single Gaussian in TNP-D (Nguyen and Grover, 2022) and a 'Riemannian distribution' (a mixture of uniform distributions with fixed bin edges and half-Gaussian tails on the sides) in PFNs (Müller et al., 2022). While in TPM-Ds encoding is mathematically decoupled from decoding, in practice encoding and decoding are commonly implemented in parallel within a single transformer layer via masking (Nguyen and Grover, 2022; Müller et al., 2022).

## 3 AMORTIZED CONDITIONING ENGINE

We describe now our proposed Amortized Conditioning Engine (ACE) architecture, which affords arbitrary probabilistic conditioning and predictions. We assume

---

#### Page 4

the problem has $L$ task-relevant latent variables of interest $\boldsymbol{\theta}=\left(\theta_{1}, \ldots, \theta_{L}\right)$. ACE amortizes arbitrary conditioning over latents (in context) and data to predict arbitrary combinations of latents (in target) and data. ACE also amortizes conditioning on probabilistic information about unobserved latents, represented by an approximate prior distribution $p\left(\theta_{l}\right)$ for $l \in\{1, \ldots, L\}$; see Fig. 2 for an example (details in Appendix B.1).

### 3.1 ACE encodes latents and priors

We demonstrate here that ACE is a new member of the TPM-D family, by extending the prediction map formalism to explicitly accommodate latent variables. In ACE, we aim to seamlessly manipulate variables that could be either data points $(\mathbf{x}, y)$ or latent variables $\theta_{l}$, for a finite set of continuous or discrete-valued latents $1 \leq l \leq L$. We redefine inputs as $\boldsymbol{\xi} \in \mathcal{X} \cup\left\{\ell_{1}, \ldots, \ell_{L}\right\}$ where $\mathcal{X} \subseteq \mathbb{R}^{D}$ denotes the data input space (covariates) and $\ell_{l}$ is a marker for the $l$-th latent. We also redefine the values as $z \in \mathcal{Z} \subseteq \mathbb{R}$ where $\mathcal{Z}$ can be continuous or a finite set of integers for discrete-valued output. Thus, $(\boldsymbol{\xi}, z)$ could denote either a (input, output) data pair or a (index, value) latent pair with either continuous or discrete values. With these new flexible definitions, ACE is indeed a transformer diagonal prediction map (TPM-D). In particular, we can predict any combination of target variables (data or latents) conditioning on any other combination of context data and latents, $\mathfrak{D}_{N}=\left\{\left(\boldsymbol{\xi}_{1}, z_{1}\right), \ldots,\left(\boldsymbol{\xi}_{N}, z_{N}\right)\right\}$ :

$$
\pi\left(z_{1: M}^{*}\left|\boldsymbol{\xi}_{1: M}^{*} ; \mathfrak{D}_{N}\right)=\prod_{m=1}^{M} p\left(z_{m}^{*} \mid \mathbf{r}\left(\boldsymbol{\xi}_{m}^{*}, \mathbf{r}_{\mathcal{D}}\left(\mathfrak{D}_{N}\right)\right)\right)\right.
$$

Prior encoding. ACE also allows the user to express probabilistic information over latent variables as prior probability distributions at runtime. Our method affords prior specification separately for each latent, corresponding to a factorized prior $p(\boldsymbol{\theta})=\prod_{l=1}^{L} p\left(\theta_{l}\right)$. To flexibly approximate a broad class of distributions, we convert each one-dimensional probability density function $p\left(\theta_{l}\right)$ to a normalized histogram of probabilities $\mathbf{p}_{l} \in[0,1]^{N_{\text {grid }}}$ over a predefined grid $\mathcal{G}$ of $N_{\text {bins }}$ bins uniformly covering the range of values. We can represent this probabilistic conditioning information within the prediction map formalism by extending the context output representation to $z \in\left\{\mathcal{Z} \cup[0,1]^{N_{\text {bins }}}\right\}$, meaning that a context point either takes a specific value or a prior defined on $\mathcal{G}$ (see Appendix B.1).

### 3.2 ACE architecture

We detail below how ACE extends the general TPMD architecture presented in Section 2 to implement latent and prior encoding, enabling flexible probabilistic conditioning and prediction. We introduce a novel
embedding layer for latents and priors, adopt an efficient transformer layer implementation, and provide an output represented by a flexible Gaussian mixture or categorical distribution. See Appendix B. 2 for an illustration and comparison to the TNP-D architecture.

Embedding layer. In ACE, the embedders map context and target data points and latents to the same embedding space of dimension $D_{\text {emb }}$. The ACE embedders handle discrete and continuous inputs without the need of tokenization. For the context set, we embed an observed data point $\left(\mathbf{x}_{n}, y_{n}\right)$ as $f_{\mathbf{x}}\left(\mathbf{x}_{n}\right)+f_{\text {val }}\left(y_{n}\right)+\mathbf{e}_{\text {data }}$, while we embed an observed latent variable $\theta_{l}$ as $f_{\text {val }}\left(\theta_{l}\right)+\mathbf{e}_{l}$, where $\mathbf{e}_{\text {data }}$ and $\mathbf{e}_{l}$ for $1 \leq l \leq L$ are learnt vector embeddings, and $f_{\mathbf{x}}$ and $f_{\text {val }}$ are learnt nonlinear embedders (MLPs) for the covariates and values, respectively. For discrete-valued variables (data or latents), $f_{\text {val }}$ is replaced by a vector embedding matrix $\mathbf{E}_{\text {val }}$ with a separate row for each discrete value. Latent variables with a prior $\mathbf{p}_{l}$ are mapped to the context set as $f_{\text {prob }}\left(\mathbf{p}_{l}\right)+\mathbf{e}_{l}$, where $f_{\text {prob }}$ is a learnt MLP. In the target set, the value of a variable is unknown and needs to be predicted, so we replace the value embedders above with a learnt 'unknown' embedding $\mathbf{e}_{?}$, i.e. $f_{\mathbf{x}}\left(\mathbf{x}_{n}\right)+\mathbf{e}_{?}+\mathbf{e}_{\text {data }}$ for data and $\mathbf{e}_{?}+\mathbf{e}_{l}$ for latents.

Transformer layers. The embedding layer is followed by $B$ stacked transformer layers, each with a multi-head attention block followed by a MLP (Vaswani et al., 2017). Both the attention and MLP blocks are followed by a normalization layer and include skip connections. The attention block combines encoder and decoder in the same step, with self-attention on the context points (encoding) and cross-attention from the target points to the context (decoding). Computing a single masked context + target attention matrix would incur $O\left((N+M)^{2}\right)$ cost (Nguyen and Grover, 2022; Müller et al., 2022). Instead, by separating the context self-attention and target cross-attention matrices we incur a $O\left(N^{2}+N M\right)$ cost (Feng et al., 2023).

Output heads. A prediction output head is applied in parallel to all target points after the last transformer layer. For a continuous-valued variable, the output head is a Gaussian mixture, consisting of $K$ MLPs that separately output the parameters of $K 1$ D Gaussians, i.e., 'raw' weight, mean, standard deviation for each mixture component (Uria et al., 2016). A learnt global raw bias term ( $3 \times K$ parameters) is added to each raw output, helping the network learn deviations from the global distribution of values. Then weights, means and standard deviations for each Gaussian are obtained through appropriate transformations (softmax, identity, and softplus, respectively). For discrete-valued variables, the output head is a MLP that outputs a softmax categorical distribution over the discrete values.

---

#### Page 5

### 3.3 Training and prediction

ACE is trained via maximum-likelihood on synthetic data consisting of batches of context and target sets, using the Adam optimizer (details in Appendix B.3).

Training. We generate each problem instance hierarchically by first sampling the latent variables $\boldsymbol{\theta}$, and then data points $(\mathbf{X}, \mathbf{y})$ according to the generative model of the task. For example, $\boldsymbol{\theta}$ could be length scale and output scale of a 1D Gaussian process with a given kernel, and $(\mathbf{X}, \mathbf{y})$ input locations and function values. Data and latents are randomly split between context and target. For training with probabilistic information $\mathbf{p}_{t}$, we first sample the priors for each latent variable from a hierarchical model $\mathcal{P}$ which includes mixtures of Gaussians and Uniform distributions (see Appendix B.1) and then sample the value of the latent from the chosen prior. During training, we minimize the expected negative log-likelihood of the target set conditioned on the context, $\mathcal{L}(\mathbf{w})$ :

$$
\mathbb{E}_{\mathbf{p} \sim \mathcal{P}}\left[\mathbb{E}_{\mathcal{D}_{N}, \boldsymbol{\xi}_{1: M}, \boldsymbol{x}_{1: M} \sim \mathbf{p}}\left[-\sum_{m=1}^{M} \log q\left(z_{m}^{*} \mid \mathbf{r}_{\mathbf{w}}\left(\boldsymbol{\xi}_{m}^{*}, \mathfrak{D}_{N}\right)\right)\right]\right]
$$

where $q$ is our model's prediction (a mixture of Gaussians or categorical), and $\mathbf{w}$ are the model parameters. Minimizing Eq. (4) is equivalent to minimizing the Kullback-Leibler (KL) divergence between the data sampled from the generative process and the model. Since the generative process is consistent with the provided contextual prior information, training will aim to converge (KL-wise) as close as possible, for the model capacity, to the correct Bayesian posteriors and predictive distributions for the specified generative model and priors (Müller et al., 2022; Elsemüller et al., 2024).

Prediction. ACE is trained via independent predictions of target data and latents, Eq. (4). Given the closed-form likelihood (mixture of Gaussians or categorical), we can easily evaluate or sample from the predictive distribution at any desired target point (data or latent) in parallel, conditioned on the context. Moreover, we can predict non-independent joint distributions autoregressively (Nguyen and Grover, 2022; Bruinsma et al., 2023); see Appendix B. 4 for details.

Task-specific contributions. The availability of predictive distributions in closed form allows ACE to simplify tasks or perform new ones. We give an example in Section 4.2, where ACE facilitates the computation of acquisition functions in Bayesian optimization.

> **Image description.** The image is a composite figure containing two rows of images and a line graph.
>
> The top two rows display images. The first image in the first row (labeled "(a) Image") shows a low-resolution image of a woman's face. The second image (labeled "(b) D_N") is a pixelated image with mostly green pixels and some other colored pixels scattered around. The third image (labeled "(c) TNP-D") shows a blurry image of a woman's face. The fourth image (labeled "(d) ACE") shows a slightly clearer image of a woman's face compared to (c). The fifth image (labeled "(e) ACE-θ") shows a slightly clearer image of a woman's face compared to (d). The second row mirrors the first row, but with a man's face instead of a woman's.
>
> The bottom part of the image is a line graph. The x-axis is labeled "Context %" and ranges from 0 to 30. The y-axis ranges from -1 to 1. There are three lines plotted on the graph, each with markers.
>
> - A blue line with circular markers represents "ACE".
> - An orange line with circular markers represents "ACE-θ".
> - A green line with circular markers represents "TNP-D".
>   The lines show a decreasing trend as "Context %" increases. The green line ("TNP-D") is consistently higher than the blue ("ACE") and orange ("ACE-θ") lines. The blue and orange lines are close to each other. Shaded regions around the lines indicate uncertainty or variance.

(f) Negative log-probability density vs. Context

Figure 3: Image completion. Image (a) serves as the reference for the problem, where $10 \%$ of the pixels are observed (b). Figures (c) through (e) display different models' prediction conditioned on the observed pixels '(b). In addition, (e) incorporates latent variable $\boldsymbol{\theta}$ information for the ACE model. Figure (f) illustrates the different models' performance across varying levels of context.

## 4 EXPERIMENTS

The following section showcases ACE's capabilities as a general framework applicable to diverse machine learning and modeling tasks. ${ }^{1}$

Firstly, Section 4.1 demonstrates how ACE complements transformer-based meta-learning in image completion and classification. In Section 4.2, we show how ACE can be applied to Bayesian optimization (BO) by treating the location and value of the global optimum as latent variables. We then move to simulation-based inference (SBI) in Section 4.3, where ACE unifies the SBI problem into a single framework, treating parameters as latent variables and affording both forward and inverse modelling. Notably, SBI and BO users may have information about the simulator or target function. ACE affords incorporation of informative priors about latent variables at runtime, as detailed in Section 3.1, a variant we call ACEP in these experiments. Finally, in Appendix C. 1 we provide extra experimental results on Gaussian Processes (GPs) where ACE can accurately predict the kernel, i.e. model selection (Fig. S6), while at the same time learn the hyperparameters, in addition to the common data prediction task.

[^0]
[^0]: ${ }^{1}$ The code implementation of ACE is available at github.com/acerbilab/amortized-conditioning-engine/.

---

#### Page 6

> **Image description.** The image shows two graphs, labeled (a) and (b), illustrating Bayesian Optimization. Both graphs share a similar structure, plotting a function with respect to x and y axes.
>
> Here's a breakdown of the elements in each graph:
>
> - **Axes:** Both graphs have x and y axes, labeled accordingly.
> - **True Function:** A dashed gray line represents the "True function". This line shows the actual function being optimized.
> - **Observations:** Black dots mark the "Observations," which are data points sampled from the true function.
> - **Prediction:** A dotted line represents the prediction of the function based on the observations.
> - **Uncertainty:** A shaded purple area around the prediction line indicates the uncertainty in the prediction. The width of the shaded area reflects the degree of uncertainty.
> - **Additional Curves:** Each graph contains additional curves:
>   - In graph (a), there's an orange curve that starts from the origin and increases rapidly. A blue curve is present at the bottom of the graph, showing a peak corresponding to the location of the optimum.
>   - In graph (b), a horizontal dashed orange line represents y_opt (true minimum value). A blue curve is present at the bottom of the graph, showing a peak corresponding to the location of the optimum.
>
> The key difference between the two graphs is the conditioning on y_opt in graph (b). The prediction and uncertainty in graph (b) are updated based on knowing the true minimum value.

Figure 4: Bayesian Optimization example. (a) ACE predicts function values ( $\cdots \cdots p\left(y \mid x, \mathcal{D}_{N}\right)$ ) as well as latents: optimum location ( $\left.\left.\left.\right|\left.\right.\right\rangle p\left(x_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)$ ) and optimum value ( $\left.\left.\left.\right|\left.\right\rangle p\left(y_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)\right)$. (b) Further conditioning on $\cdots-y_{\mathrm{opt}}$ (here the true minimum value) leads to updated predictions.

### 4.1 Image completion and classification

We treat image completion as a regression task (Garnelo et al., 2018a), where the goal is given some limited $\mathcal{D}_{N}$ of image coordinates and corresponding pixel values to predict the complete image. For the MNIST (Deng, 2012) task, we downsize the images to $16 \times 16$ and likewise for CelebA to $32 \times 32$ (Liu et al., 2015). We turn the class label into a single discrete latent for MNIST while for CelebA, we feed the full 40 corresponding binary features (e.g., BrownHair, Man, Smiling). The latents are sampled using the procedure outlined in Appendix B. 3 and more experimental details of the image completion task can be found in Appendix C.2. Notably, ACE affords conditional image generation, i.e., predictions of pixels based on latent variables $(\boldsymbol{\theta})$ such as class labels in MNIST and appearance features in CelebA - as well as image classification, i.e. the prediction of these latent variables themselves from pixel values.

Results. In Fig. 3 we present a snapshot of the results for CelebA and in Appendix C. 2 we present the same for MNIST. The more complex output distribution allows ACE to outperform other Transformer NPs convincingly, and the integration of latent information shows a clear improvement. In Appendix C.2, we present our full results, including predicting $\boldsymbol{\theta}$.

### 4.2 Bayesian optimization (BO)

BO aims to find the global minimum $y_{\text {opt }}=f\left(\mathbf{x}_{\text {opt }}\right)$ of a black-box function. This is typically achieved iteratively by building a surrogate model that approximates the target and optimizing an acquisition function $\alpha(\mathbf{x})$
to determine the next query point. ACE provides additional modeling options for the BO loop by affording direct conditioning on, and prediction of, key latents $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$, yielding closed-form predictive distributions and samples for $p\left(\mathbf{x}_{\mathrm{opt}} \mid \mathcal{D}_{N}\right), p\left(y_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)$, and $p\left(\mathbf{x}_{\mathrm{opt}} \mid \mathcal{D}_{N}, y_{\mathrm{opt}}\right)$; see Fig. 4.
For the BO task, we trained ACE on synthetic functions generated from GP samples with RBF and Matérn- $(1 / 2,3 / 2,5 / 2)$ kernels and a random global optimum ( $\mathbf{x}_{\text {opt }}, y_{\text {opt }}$ ) within the function domain; see Appendix C. 3 for details. We leverage ACE's explicit modeling of latents in multiple ways.

Acquisition functions with ACE. ACE affords a straightforward implementation of a variant of Thompson Sampling (TS) (Dutordoir et al., 2023; Liu et al., 2024). First, we sample a candidate optimum value, $y_{\text {opt }}^{*}$, conditioning on it being below a threshold $\tau$, from the truncated predictive distribution $y_{\text {opt }}^{*} \sim p\left(y_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$; see Fig. 4a. Given $y_{\text {opt }}^{*}$, we then sample the query point $\mathbf{x}^{*} \sim p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}^{*}\right)$; Fig. 4b. ${ }^{2}$ This process is repeated iteratively within the BO loop (see Fig. S14). For higher input dimensions $(D>1)$, we sample $\mathbf{x}^{*}$ autoregressively; see Appendix C.3.2. Crucially, ACE's capability of directly modeling $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$ bypasses the need for surrogate optimization or grid searches typical of standard TS implementations (e.g., GP-TS or TNP-D based TS).
ACE also easily supports advanced acquisition functions used in BO, such as Max-Value Entropy Search (MES; Wang and Jegelka, 2017). For a candidate point $\mathbf{x}^{*}$, MES evaluates the expected gain in mutual information between $y_{\text {opt }}$ and $\mathbf{x}^{*}$ :

$$
\begin{aligned}
\alpha_{\mathrm{MES}}\left(\mathbf{x}^{*}\right) & =H\left[p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}\right)\right] \\
& -\mathbb{E}_{p\left(y_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)}\left[H\left[p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}, y_{\mathrm{opt}}\right)\right]\right]
\end{aligned}
$$

With all predictive distributions available in closed form, ACE can readily calculate the expectation and entropies in Eq. (5) via Monte Carlo sampling and fast 1D numerical integration, unlike other methods that require more laborious approximations. For maximizing $\alpha\left(\mathbf{x}^{*}\right)$, we obtain a good set of candidate points via Thompson Sampling (see Appendix C.3.2 for details).

Results. We compare ACE with gold-standard Gaussian processes (GPs) and the state-of-the-art TNP-D model (Fig. 5). Additionally, we test a setup where prior information is provided about the location of the optimum (Souza et al., 2021; Hvarfner et al., 2022;

[^0]
[^0]: ${ }^{2}$ Why not sampling directly from $p\left(\mathbf{x}_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)$ ? The issue is that $p\left(\mathbf{x}_{\mathrm{opt}} \mid \mathcal{D}_{N}\right)$ may reasonably include substantial probability mass at the current optimum, which would curb exploration. The constraint $y_{\text {opt }}<\tau$, with $\tau$ (just) below the current optimum value, ensures continual exploration.

---

#### Page 7

> **Image description.** The image is a figure containing eight line graphs arranged in a 2x4 grid. Each graph displays the performance of different optimization methods on benchmark tasks, plotting "Regret" on the y-axis against "Iteration" on the x-axis.
>
> - **Overall Structure:** The figure consists of eight individual plots, each representing a different benchmark function. The plots are arranged in two rows and four columns.
>
> - **Axes and Labels:**
>
>   - Each plot has an x-axis labeled "Iteration" and a y-axis labeled "Regret".
>   - The x-axis ranges vary slightly between plots, typically from 0 to 75 or 90.
>   - The y-axis ranges also vary, with maximum values ranging from 0.1 to 28.
>   - Each plot has a title indicating the benchmark function being evaluated, such as "Gramacy Lee 1D", "Branin Scaled 2D", "Hartmann 3D", "Rosenbrock 4D", "Rosenbrock 5D", "Levy 5D", "Hartmann 6D", and "Levy 6D".
>
> - **Data Representation:**
>
>   - Each plot contains multiple lines, each representing a different optimization method.
>   - The methods are identified in a legend at the top of the figure: "ACE-TS" (solid blue line), "ACE-MES" (dashed blue line), "AR-TNPD-TS" (solid green line), "GP-TS" (solid orange line), "GP-MES" (dashed orange line), and "Random" (dotted pink line).
>   - Each line is accompanied by a shaded region of the same color, representing the standard error.
>
> - **Visual Patterns:**
>
>   - In most plots, the "Regret" values decrease as the "Iteration" number increases, indicating that the optimization methods are converging towards a solution.
>   - The "Random" method generally performs worse than the other methods, as indicated by its higher "Regret" values.
>   - The relative performance of the other methods varies depending on the benchmark function.
>
> - **Text:** The following text is present in the image:
>   - "ACE-TS"
>   - "ACE-MES"
>   - "AR-TNPD-TS"
>   - "GP-TS"
>   - "GP-MES"
>   - "Random"
>   - "Regret" (y-axis label)
>   - "Iteration" (x-axis label)
>   - "Gramacy Lee 1D"
>   - "Branin Scaled 2D"
>   - "Hartmann 3D"
>   - "Rosenbrock 4D"
>   - "Rosenbrock 5D"
>   - "Levy 5D"
>   - "Hartmann 6D"
>   - "Levy 6D"

Figure 5: Bayesian optimization results. Regret comparison (mean $\pm$ standard error) for different methods across benchmark tasks.

Müller et al., 2023); Fig. 6. Unlike other methods that employ heuristics or complex approximations, ACE's architecture affords seamless incorporation of a prior $p\left(\mathbf{x}_{\text {opt }}\right)$. Here, we consider two types of priors: strong and weak, represented by Gaussians with a standard deviation equal to, respectively, $10 \%$ and $25 \%$ of the optimization box width in each dimension, and mean drawn from a Gaussian centered on the true optimum and same standard deviation (see Appendix C.3.3).

In Fig. 5, we show the performance of ACE Thompson sampling (ACE-TS) and MES (ACE-MES) with GP-based MES (GP-MES; Wang and Jegelka, 2017), GP-based Thompson Sampling (GP-TS; Balandat et al., 2020), and Autoregressive TNP-D based Thompson Sampling (AR-TNPD-TS; Bruinsma et al., 2023; Nguyen and Grover, 2022) on several benchmark functions (see Appendix C.3.4 for details). ACE-MES frequently outperforms ACE-TS and often matches the gold-standard GP-MES. In the prior setting, we compare ACE without (ACE-TS) and with prior (ACEPTS) against $\pi$ BO-TS, a state-of-the-art heuristic for prior injection in BO (Hvarfner et al., 2022), as well as the GP-TS baseline. ACEP-TS shows significant improvement over its no-prior variant ACE-TS while showing competitive performance compared to $\pi$ BO-TS in both weak and strong prior case (Fig. 6).

### 4.3 Simulation-based inference (SBI)

We now apply ACE to simulation-based inference (SBI; Cranmer et al., 2020). With ACE, we can predict the posterior distribution of (latent) model parameters, simulate data based on parameters, predict missing data given partial observations, and set priors at run-

> **Image description.** The image contains four line graphs arranged in a 2x2 grid. Each graph displays the "Regret" on the y-axis versus "Iteration" on the x-axis. The graphs compare the performance of four different methods: ACE-TS (solid blue line), ACEP-TS (dashed blue line), GP-TS (solid orange line), and πBO-TS (dotted orange line). Shaded regions around the lines represent the standard error.
>
> - **Top Left:** The graph is titled "Michalewicz 2D (weak)". The x-axis ranges from 10 to 90, and the y-axis ranges from 0 to 1.2.
> - **Top Right:** The graph is titled "Michalewicz 2D (strong)". The x-axis ranges from 10 to 90, and the y-axis ranges from 0 to 1.2.
> - **Bottom Left:** The graph is titled "Levy 3D (weak)". The x-axis ranges from 10 to 50, and the y-axis ranges from 0 to 1.3.
> - **Bottom Right:** The graph is titled "Levy 3D (strong)". The x-axis ranges from 10 to 50, and the y-axis ranges from 0 to 1.3.
>
> The legend at the top of the image identifies the line styles and colors corresponding to each method.

Figure 6: Bayesian optimization with prior over $\mathbf{x}_{\text {opt }}$. Regret comparison (mean $\pm$ standard error) on 2D and 3D optimization benchmarks. Left: Weak Gaussian prior ( $25 \%$ ), Right: Strong prior ( $10 \%$ ). ACEP-TS performs competitively compared to $\pi$ BO-TS.
time. We consider two benchmark time-series models, each with two latents: the Ornstein-Uhlenbeck Process (OUP; Uhlenbeck and Ornstein 1930) and the Susceptible-Infectious-Recovered model (SIR; Kermack and McKendrick 1927); and a third more complex engineering model from the field of radio propagation (Turin; Turin et al., 1972), which has four parameters and produces 101-dimensional data representing a radio signal. See Appendix C. 4 for all model descriptions.

---

#### Page 8

|       |                                                 |    NPE     |    NRE     | Simformer  |    ACE     | $\mathrm{ACEP}_{\text {weak prior }}$ | $\mathrm{ACEP}_{\text {strong prior }}$ |
| :---: | :---------------------------------------------: | :--------: | :--------: | :--------: | :--------: | :-----------------------------------: | :-------------------------------------: |
|  OUP  | $\log -\operatorname{probs}_{\theta}(\uparrow)$ | 1.09(0.10) | 1.07(0.13) | 1.03(0.04) | 1.03(0.02) |              1.05(0.02)               |               1.44(0.03)                |
|       |   $\operatorname{RMSE}_{\theta}(\downarrow)$    | 0.48(0.01) | 0.49(0.00) | 0.50(0.02) | 0.48(0.00) |              0.43(0.01)               |               0.27(0.00)                |
|       |      $\operatorname{MMD}_{y}(\downarrow)$       |     -      |     -      | 0.43(0.02) | 0.51(0.00) |              0.37(0.00)               |               0.35(0.00)                |
|  SIR  | $\log -\operatorname{probs}_{\theta}(\uparrow)$ | 6.53(0.11) | 6.24(0.16) | 6.89(0.09) | 6.78(0.02) |              6.62(0.10)               |               6.69(0.10)                |
|       |   $\operatorname{RMSE}_{\theta}(\downarrow)$    | 0.02(0.00) | 0.03(0.00) | 0.02(0.00) | 0.02(0.00) |              0.02(0.00)               |               0.02(0.00)                |
|       |      $\operatorname{MMD}_{y}(\downarrow)$       |     -      |     -      | 0.02(0.00) | 0.02(0.00) |              0.02(0.00)               |               0.00(0.00)                |
| Turin | $\log -\operatorname{probs}_{\theta}(\uparrow)$ | 1.99(0.05) | 2.33(0.07) | 3.16(0.03) | 3.14(0.02) |              3.58(0.04)               |               4.87(0.08)                |
|       |   $\operatorname{RMSE}_{\theta}(\downarrow)$    | 0.26(0.00) | 0.28(0.00) | 0.25(0.00) | 0.24(0.00) |              0.21(0.00)               |               0.13(0.00)                |
|       |      $\operatorname{MMD}_{y}(\downarrow)$       |     -      |     -      | 0.35(0.00) | 0.35(0.00) |              0.35(0.00)               |               0.34(0.00)                |

Table 1: Comparison metrics for SBI models on parameters $(\boldsymbol{\theta})$ and data $(y)$ prediction; mean and (standard deviation) from 5 runs. Left: Statistically significantly (see Appendix C.4.2) best results are bolded. ACE shows performance comparable to the other dedicated methods on latents prediction. In the data prediction task, ACE performs similarly to Simformer with much lower sampling cost at runtime (see text). Right: ACE can leverage probabilistic information provided at runtime by informative priors (ACEP), yielding improved performance.

We compare ACE with Neural Posterior Estimation (NPE; Greenberg et al. 2019), Neural Ratio Estimation (NRE; Miller et al. 2022), and Simformer (Gloeckler et al., 2024), from established to state-of-the-art methods in amortized SBI. We evaluate ACE in three different scenarios. For the first one, we use only the observed data as context. For the other two scenarios, we inform ACE with priors over the parameters (ACEP), to assess their impact on posterior prediction. These priors are Gaussians with standard deviation equal to $25 \%$ (weak) or $10 \%$ (strong) of the parameter range, and mean drawn from a Gaussian centered on the true parameter value and the same standard deviation.

We evaluate the performance of posterior estimation using the log probabilities of the ground-truth parameters and the root mean squared error (RMSE) between the true parameters and posterior samples. Since both ACE and Simformer can predict missing data from partial observations - an ability that previous SBI methods lack - we also test them on a data prediction task. For each observed dataset, we randomly designate half of the data points as missing and use the remaining half as context for predictions. We then measure performance via the maximum mean discrepancy (MMD) between the true data and the predicted distributions.

Results. Results are reported in Table 1; see Appendix C. 4 for details. For ACE, we see that joint training to predict data and latents does not compromise its posterior estimation performance compared to NPE and NRE, even achieving better performance on the Turin model. ACE and Simformer obtain similar results. However, as Simformer uses diffusion, data sampling is substantially slower. For example, we measured the time required to generate 1,000 posterior samples for 100 sets of observations on the OUP model using a CPU (GPU) across 5 runs: the average time for

Simformer is $\sim 130$ minutes ( 14 minutes), whereas ACE takes 0.05 seconds ( 0.02 s ). When we provide ACE with informative priors (ACEP; Table 1 right), performance improves in proportion to the provided information. Importantly, simulation-based calibration checks (Talts et al., 2018) show that both ACE and ACEP output good approximate posteriors (Appendix C.4.4).
Finally, we applied ACE to a real-world outbreak dataset (Avilov et al., 2023) using an extended, fourparameter version of the SIR model. We show that ACE can handle real data, providing reasonable results under a likely model mismatch (see Appendix C.4.5).

## 5 RELATED WORK

Our work combines insights from different fields such as neural processes, meta-learning, and simulation-based inference, providing a new unified and versatile framework for amortized inference.

Neural processes. ACE relates to the broader work on neural processes (Garnelo et al., 2018a,b; Kim et al., 2019; Gordon et al., 2020; Markou et al., 2022; Nguyen and Grover, 2022; Huang et al., 2023a) and shares similarities with autoregressive diffusion models (Hoogeboom et al., 2022), whose permutation-invariant conditioning set mimics the context set in neural processes. Unlike previous methods that focused on predictive data distributions conditioned on observed data, ACE also explicitly models and conditions on latent variables of interest. While Latent Neural Processes (LNPs) capture correlations via non-interpretable latent vectors that yield an intractable learning objective (Garnelo et al., 2018a), ACE uses autoregressive sampling to model correlations (Bruinsma et al., 2023), and 'latents' in ACE denote explicitly modelled task variables.

---

#### Page 9

Meta-learning. Our approach falls within the fields of amortized inference, meta-learning, and pre-trained models, which overlap significantly with neural process literature (Finn et al., 2017, 2018). Prior-Fitted Networks (PFNs) demonstrated the use of transformers for Bayesian inference (Müller et al., 2022) and optimization (Müller et al., 2023), focusing on predictive posterior distributions over data using fixed bins ('Riemannian' distributions). In particular, Müller et al. (2023) allow users to indicate a 'high-probability' interval for the optimum location at runtime within a set of specified ranges. ACE differs from these methods by allowing the specification of more flexible priors at runtime, using learnable mixtures of Gaussians for continuous values, and, most importantly, for yielding explicit predictive distributions over both data and task-relevant, interpretable latents. While other works target specific optimization tasks (Liu et al., 2020; Simpson et al., 2021; Amos, 2022), we provide a generative model for random functions with known optima, directly amortizing predictive distributions useful for optimization (see Appendix C.3). More broadly, ACE relates to efforts in amortized inference over explicit task-relevant latents (Mittal et al., 2023), though our focus extends beyond posterior estimation to flexible conditioning and prediction of both data and latents. Additionally, Mittal et al. (2024) show that enforcing a bottleneck in transformers captures task-relevant latents but does not necessarily improve generalization. Unlike their learned latents, ACE conditions on predefined, interpretable task variables, enabling direct control and adaptability. By providing methods to condition on interpretable latents as well as injecting probabilistic knowledge (priors), our work aligns with the principles of informed meta-learning (Kobalczyk and van der Schaar, 2024). Subsequent to our work, distribution transformers were recently proposed as amortized meta-learning models for inference with flexible priors and posteriors (Whittle et al., 2025).

Simulation-based inference. ACE is related to the extensive literature on amortized or neural simulationbased inference (SBI) (Cranmer et al., 2020), where deep networks are used to (a) recover the posterior distribution over model parameters (Lueckmann et al., 2017), and (b) emulate the simulator or likelihood (Papamakarios et al., 2019). While these two steps are often implemented separately, recent work has started to combine them (Radev et al., 2023). In particular, the recently proposed Simformer architecture allows users to freely condition on observed variables and model parameters, in a manner similar to ACE (Gloeckler et al., 2024). Simformer uses a combination of a transformer architecture and a denoising diffusion model (Song and Ermon, 2019), which makes it suitable for continuous predictions but does not immediately apply
to discrete outputs. Notably, Simformer allows the user to specify intervals for variables at runtime, but not (yet) general priors. To our knowledge, before our paper the only work in amortized SBI that afforded some meaningful flexibility in prior specification at runtime was Elsemüller et al. (2024). Even so, the choice there is between a limited number of fixed priors or a global prior scale parameter, with the main focus being sensitivity analysis (Elsemüller et al., 2024). Importantly, all these works focus entirely on SBI settings, while our paper showcases the applicability of ACE to a wider variety of machine learning tasks.

## 6 DISCUSSION

In this paper, we introduced the Amortized Conditioning Engine (ACE), a unified transformer-based architecture that affords arbitrary probabilistic conditioning and prediction over data and task-relevant variables for a wide variety of tasks. In each tested domain ACE performed on par with state-of-the-art, bespoke solutions, and with greater flexibility. As a key feature, ACE allows users to specify probabilistic information at runtime (priors) without the need for retraining as required instead by the majority of previous amortized inference methods (Cranmer et al., 2020).

Limitations and future work. As all amortized and machine learning methods, ACE's predictions become unreliable if applied to data unseen during training due to mismatch between simulated and real data. This is an active area of research in terms of developing both more robust training objectives (Huang et al., 2023b) and diagnostics (Schmitt et al., 2023).

The method's quadratic complexity in context size could benefit from sub-quadratic attention variants (Feng et al., 2023), while incorporating equivariances (Huang et al., 2023a) and variable covariate dimensionality (Liu et al., 2020; Dutordoir et al., 2023; Müller et al., 2023) could further improve performance.

While ACE can learn full joint distributions through 1D marginals (Bruinsma et al., 2023), scaling to many data points and latents remains challenging. Similar scaling challenges affect our prior-setting approach, currently limited to few latents under factorized, smooth priors. Future work should also explore discovering interpretable latents (Mittal et al., 2024) and handling multiple tasks simultaneously (Kim et al., 2022; Ashman et al., 2024), extending beyond our current single-task supervised learning approach.

Conclusions. ACE shows strong promise as a new unified and versatile method for amortized probabilistic conditioning and prediction, able to perform various probabilistic machine learning tasks.

---

# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference - Backmatter

---

#### Page 10

## Acknowledgments

PC, DH, UR, SK, and LA were supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI). NL was funded by Business Finland (project 3576/31/2023). LA was also supported by Research Council of Finland grants 358980 and 356498 . SK was also supported by the UKRI Turing AI World-Leading Researcher Fellowship, [EP/W002973/1]. The authors wish to thank the Finnish Computing Competence Infrastructure (FCCI), Aalto Science-IT project, and CSC-IT Center for Science, Finland, for the computational and data storage resources provided, including access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium (LUMI project 462000551 ).

## References

Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Ali Eslami, and Yee Whye Teh. Neural processes. In ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models, 2018a.
Brandon Amos. Tutorial on amortized optimization for learning to optimize over continuous domains. arXiv e-prints, pages arXiv-2202, 2022.
Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48): 30055-30062, 2020.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877-1901, 2020.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.
Tung Nguyen and Aditya Grover. Transformer Neural Processes: Uncertainty-aware meta learning via sequence modeling. In Proceedings of the International Conference on Machine Learning (ICML), pages 123134. PMLR, 2022.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022.
Zoubin Ghahramani. Probabilistic machine learning
and artificial intelligence. Nature, 521(7553):452-459, 2015.

Roman Garnett. Bayesian optimization. Cambridge University Press, 2023.
Philipp Hennig and Christian J Schuler. Entropy search for information-efficient global optimization. Journal of Machine Learning Research, 13(6), 2012.
José Miguel Hernández-Lobato, Matthew W Hoffman, and Zoubin Ghahramani. Predictive entropy search for efficient global optimization of black-box functions. Advances in Neural Information Processing Systems, 27, 2014.
Zi Wang and Stefanie Jegelka. Max-value entropy search for efficient Bayesian optimization. In International Conference on Machine Learning, pages 3627-3635. PMLR, 2017.
Vu Nguyen and Michael A Osborne. Knowing the what but not the where in Bayesian optimization. In International Conference on Machine Learning, pages 7317-7326. PMLR, 2020.
Artur Souza, Luigi Nardi, Leonardo B Oliveira, Kunle Olukotun, Marius Lindauer, and Frank Hutter. Bayesian optimization with a prior for the optimum. In Machine Learning and Knowledge Discovery in Databases. Research Truck: European Conference, ECML PKDD 2021, Bilbao, Spain, September 1317, 2021, Proceedings, Part III 21, pages 265-296. Springer, 2021.
Carl Hvarfner, Danny Stoll, Artur Souza, Luigi Nardi, Marius Lindauer, and Frank Hutter. $\pi$ BO: Augmenting acquisition functions with user beliefs for bayesian optimization. In International Conference on Learning Representations, 2022.
Andrew YK Foong, Wessel P Bruinsma, Jonathan Gordon, Yann Dubois, James Requeima, and Richard E Turner. Meta-learning stationary stochastic process prediction with convolutional neural processes. In Advances in Neural Information Processing Systems, volume 33, pages 8284-8295, 2020.
Stratis Markou, James Requeima, Wessel P Bruinsma, Anna Vaughan, and Richard E Turner. Practical conditional neural processes via tractable dependent predictions. In International Conference on Learning Representations, 2022.
Marta Garnelo, Dan Rosenbaum, Chris J Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J Rezende, and SM Ali Eslami. Conditional neural processes. In International Conference on Machine Learning, pages 1704-1713, 2018b.
Carl Edward Rasmussen and Christopher KI Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

---

#### Page 11

Wessel P Bruinsma, Stratis Markou, James Requeima, Andrew YK Foong, Tom R Andersson, Anna Vaughan, Anthony Buonomo, J Scott Hosking, and Richard E Turner. Autoregressive conditional neural processes. In International Conference on Learning Representations, 2023.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J Smola. Deep sets. In Advances in Neural Information Processing Systems, volume 30, pages 3391-3401, 2017.

Leo Feng, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Latent bottlenecked attentive neural processes. In The Eleventh International Conference on Learning Representations, ICLR 2023. PMLR (Proceedings of Machine Learning Research), 2023.

Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, and Hugo Larochelle. Neural autoregressive distribution estimation. Journal of Machine Learning Research, 17(205):1-37, 2016.

Lasse Elsemüller, Hans Olischläger, Marvin Schmitt, Paul-Christian Bürkner, Ullrich Koethe, and Stefan T. Radev. Sensitivity-aware amortized bayesian inference. Transactions on Machine Learning Research, 2024.

Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141-142, 2012.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, and Fergus Simpson. Neural diffusion processes. In International Conference on Machine Learning, pages 8990-9012. PMLR, 2023.

Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar. Large language models to enhance Bayesian optimization. International Conference on Learning Representations, 2024.

Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. Pfns4bo: In-context learning for bayesian optimization. In International Conference on Machine Learning, pages 25444-25470. PMLR, 2023.

Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. Advances in Neural Information Processing Systems, 33:2152421538, 2020.

George E Uhlenbeck and Leonard S Ornstein. On the theory of the Brownian motion. Physical Review, 36 (5):823, 1930.

William O Kermack and Anderson G McKendrick. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London. Series A, Containing papers of a mathematical and physical character, 115(772):700-721, 1927.

George L Turin, Fred D Clapp, Tom L Johnston, Stephen B Fine, and Dan Lavry. A statistical model of urban multipath propagation. IEEE Transactions on Vehicular Technology, 21(1):1-9, 1972.

David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood-free inference. In International Conference on Machine Learning, pages 2404-2414. PMLR, 2019.

Benjamin K Miller, Christoph Weniger, and Patrick Forré. Contrastive neural ratio estimation. Advances in Neural Information Processing Systems, 35:32623278, 2022.

Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H Macke. All-in-one simulation-based inference. In International Conference on Machine Learning. PMLR, 2024.

Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating bayesian inference algorithms with simulation-based calibration. arXiv preprint arXiv:1804.06788, 2018.

Konstantin Avilov, Qiong Li, Lewi Stone, and Daihai He. The 1978 english boarding school influenza outbreak: Where the classic seir model fails. SSRN 4586177, 2023.

Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In International Conference on Learning Representations, 2019.

Jonathan Gordon, Wessel P Bruinsma, Andrew YK Foong, James Requeima, Yann Dubois, and Richard E Turner. Convolutional conditional neural processes. In International Conference on Learning Representations, 2020.

Daolang Huang, Manuel Haussmann, Ulpu Remes, ST John, Grégoire Clarté, Kevin Luck, Samuel Kaski, and Luigi Acerbi. Practical equivariances via Relational Conditional Neural Processes. Advances in Neural Information Processing Systems, 36:2920129238, 2023a.

Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim

---

#### Page 12

Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pages 1126-1135. PMLR, 2017.
Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. Advances in Neural Information Processing Systems, 31, 2018.
Sulin Liu, Xingyuan Sun, Peter J Ramadge, and Ryan P Adams. Task-agnostic amortized inference of Gaussian process hyperparameters. In Advances in Neural Information Processing Systems, volume 33, pages 21440-21452, 2020.
Fergus Simpson, Ian Davies, Vidhi Lalchand, Alessandro Vullo, Nicolas Durrande, and Carl Edward Rasmussen. Kernel identification through transformers. In Advances in Neural Information Processing Systems, volume 34, pages 10483-10495, 2021.
Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, and Marcus A Brubaker. Exploring exchangeable dataset amortization for bayesian posterior inference. In ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling, 2023.
Sarthak Mittal, Eric Elmoznino, Leo Gagnon, Sangnie Bhardwaj, Dhanya Sridhar, and Guillaume Lajoie. Does learning the right latent variables necessarily improve in-context learning? arXiv preprint arXiv:2405.19162, 2024.
Katarzyna Kobalczyk and Mihaela van der Schaar. Informed meta-learning. arXiv preprint arXiv:2402.16105, 2024.
George Whittle, Juliusz Ziomek, Jacob Rawling, and Michael A Osborne. Distribution transformers: Fast approximate Bayesian inference with on-the-fly prior adaptation. arXiv preprint arXiv:2502.02463, 2025.
Jan-Matthis Lueckmann, Pedro J Goncalves, Giacomo Bassetto, Kaan Öcal, Marcel Nonnenmacher, and Jakob H Macke. Flexible statistical inference for mechanistic models of neural dynamics. Advances in Neural Information Processing Systems, 30, 2017.
George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 837-848. PMLR, 2019.
Stefan T Radev, Marvin Schmitt, Valentin Pratz, Umberto Picchini, Ullrich Köthe, and Paul-Christian Bürkner. JANA: Jointly amortized neural approximation of complex Bayesian models. In Uncertainty
in Artificial Intelligence, pages 1695-1706. PMLR, 2023.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.
Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, and Samuel Kaski. Learning robust statistics for simulation-based inference under model misspecification. Advances in Neural Information Processing Systems, 36, 2023b.
Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, and Stefan T Radev. Detecting model misspecification in amortized Bayesian inference with neural networks. In DAGM German Conference on Pattern Recognition, pages 541-557. Springer, 2023.
Donggyun Kim, Seongwoong Cho, Wonkwang Lee, and Seunghoon Hong. Multi-task neural processes. In International Conference on Learning Representations, 2022.

Matthew Ashman, Cristiana Diaconu, Adrian Weller, and Richard E Turner. In-context in-context learning with transformer neural processes. In Proceedings of the 6th Symposium on Advances in Approximate Bayesian Inference, 2024.
Ryan L Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Janossy pooling: Learning deep permutation-invariant functions for variablesize inputs. In International Conference on Learning Representations, 2019.
James Hensman, Alexander Matthews, and Zoubin Ghahramani. Scalable variational Gaussian process classification. In Artificial Intelligence and Statistics, pages 351-360. PMLR, 2015.
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015. PMLR (Proceedings of Machine Learning Research), 2015.

Christian P. Robert and George Casella. Monte Carlo Statistical Methods. Springer Texts in Statistics. Springer, 2nd edition, 2004.
Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Benchmarking simulation-based inference. In Proc. AISTATS, pages 343-351. PMLR, 2021.
Troels Pedersen. Stochastic multipath model for the inroom radio channel based on room electromagnetics. IEEE Transactions on Antennas and Propagation, 67(4):2591-2603, 2019.
Ayush Bharti, Ramoni Adeogun, and Troels Pedersen. Estimator for stochastic channel model without

---

#### Page 13

multipath extraction using temporal moments. In 2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), pages 1-5. IEEE, 2019.
Alvaro Tejero-Cantero, Jan Boelts, Michael Deistler, Jan-Matthis Lueckmann, Conor Durkan, Pedro J. Gonçalves, David S. Greenberg, and Jakob H. Macke. sbi: A toolkit for simulation-based inference. Journal of Open Source Software, 5(52):2505, 2020.
George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. Advances in Neural Information Processing Systems, 30, 2017.
Teemu Sälynoja, Paul-Christian Bürkner, and Aki Vehtari. Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing, $32(2): 32,2022$.
Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D. Goodman. Pyro: Deep Universal Probabilistic Programming. Journal of Machine Learning Research, 2018.
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. Advances in Neural Information Processing Systems, 32, 2019.

## Checklist

1. For all models and algorithms presented, check if you include:
   (a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes] See description in Section 3 and further details in Appendix B and Appendix C.
   (b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Not Applicable] Our architecture is based on the standard attention mechanism (Vaswani et al., 2017) which has well-known properties.
   (c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes] Our source code is available at https://github.com/acerbilab/ amortized-conditioning-engine/.
2. For any theoretical claim, check if you include:
   (a) Statements of the full set of assumptions of all theoretical results. [Not Applicable]
   (b) Complete proofs of all theoretical results. [Not Applicable]
   (c) Clear explanations of any assumptions. [Not Applicable]
3. For all figures and tables that present empirical results, check if you include:
   (a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL).
   [Yes] The linked GitHub repo contains working examples demonstrating our main results, in addition to the full codebase.
   (b) All the training details (e.g., data splits, hyperparameters, how they were chosen).
   [Yes] In Appendix C we provide all experimental details.
   (c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times).
   [Yes] See Section 4.1, Section 4.2, Section 4.3 and Appendix C.
   (d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes] See Appendix C.5.
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:
   (a) Citations of the creator If your work uses existing assets. [Yes] See Appendix C.5.
   (b) The license information of the assets, if applicable. [Yes] See Appendix C.5.
   (c) New assets either in the supplemental material or as a URL, if applicable. [Not Applicable]
   (d) Information about consent from data providers/curators. [Not Applicable]
   (e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable]
5. If you used crowdsourcing or conducted research with human subjects, check if you include:
   (a) The full text of instructions given to participants and screenshots. [Not Applicable]
   (b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]
   (c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable]

---

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