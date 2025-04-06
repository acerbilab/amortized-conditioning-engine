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
