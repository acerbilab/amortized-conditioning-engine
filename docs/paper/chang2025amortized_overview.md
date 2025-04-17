# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference

Paul E. Chang[^1]\*, Nasrulloh Loka[^1]\*, Daolang Huang[^2]\*, Ulpu Remes[^3], Samuel Kaski[^2],[^4], Luigi Acerbi[^1]

[^1]: Department of Computer Science, University of Helsinki, Helsinki, Finland
[^2]: Department of Computer Science, Aalto University, Espoo, Finland
[^3]: Department of Mathematics and Statistics, University of Helsinki, Helsinki, Finland
[^4]: Department of Computer Science, University of Manchester, Manchester, United Kingdom

\*Equal contribution

Accepted to the 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025)

[Code](https://github.com/acerbilab/amortized-conditioning-engine/) | [Paper](https://arxiv.org/abs/2410.15320) | [Social](https://bsky.app/profile/lacerbi.bsky.social/post/3ljpc4zkyl22k) | [Markdown](https://github.com/acerbilab/amortized-conditioning-engine/tree/main/docs/paper)

### TL;DR

We introduce the **Amortized Conditioning Engine (ACE)**, a transformer-based meta-learning model that enables flexible probabilistic conditioning and prediction for machine learning tasks. ACE can condition on both observed data and latent variables, include priors at runtime, and output predictive distributions for both data and latents. This general framework unifies and simplifies diverse ML tasks like image completion, Bayesian optimization, and simulation-based inference, and has the potential to be applied to many others.

```
@article{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  journal={28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)},
  year={2025}
}
```

## Introduction

Amortization, or pre-training, is a crucial technique for improving computational efficiency and generalization across many machine learning tasks. This paper capitalizes on the observation that many machine learning problems reduce to predicting data and task-relevant latent variables after conditioning on other data and latents. Moreover, in many scenarios, the user has exact or probabilistic information (priors) about task-relevant variables that they would like to leverage, but incorporating such prior knowledge is challenging and often requires dedicated, expensive solutions.

As an example, consider **Bayesian optimization (BO)**, where the goal is to find the location $\\mathbf{x}*{\\text{opt}}$ and value $y*{\\text{opt}}$ of the global minimum of a function. These are latent variables, distinct from the observed data $\\mathcal{D}*{N}$ consisting of function values at queried locations. Following information-theoretical principles, we should query points that would reduce uncertainty about the latent optimum. This task would be easier if we had direct access to predictive distribution over the latents of interest, $p(\\mathbf{x}*{\\text{opt}} | \\mathcal{D}*{N})$ and $p(y*{\\text{opt}} | \\mathcal{D}\_{N})$, among others, but predictive distributions over these variables are intractable, leading to many complex techniques and a variety of papers just to approximate these distributions.

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
>
>   - The first panel shows a scatter plot with several black dots.
>   - The second panel shows a distribution represented by an orange filled curve. A "θ" symbol is written above the distribution.
>   - An arrow points to the third panel, which shows a plot of a function with data points marked as black dots with error bars.
>   - The fourth panel shows a distribution represented by an orange filled curve.
>
> The rows are separated by dashed lines. The labels "Data" and "Latent" are written above the columns.

**Probabilistic conditioning and prediction.** Many tasks
reduce to probabilistic conditioning on data and key latent variables
(left) and then predicting data and latents (right). (a) Image completion
and classification. (b) Bayesian optimization. (c) Simulator-based
inference.

We address these challenges by introducing the **Amortized Conditioning Engine (ACE)**, a general amortization framework that extends transformer-based meta-learning architectures with explicit and flexible probabilistic modeling of task-relevant latent variables. With ACE, we can seamlessly obtain predictive distribution over variables of interest, replacing bespoke techniques across different fields with a unifying framework for amortized probabilistic conditioning and prediction.

## Probabilistic Conditioning and Prediction

In the framework of prediction maps and Conditional Neural Processes (CNPs), a prediction map $\\pi$ is a function that takes a context set of input/output pairs $\\mathcal{D}*{N}$ and target inputs $\\mathbf{x}*{1:M}^\*$ to predict a distribution over the corresponding target outputs:

$$\pi(y_{1:M}^* | \mathbf{x}_{1:M}^* ; \mathcal{D}_{N}) = p(y_{1:M}^* | \mathbf{r}(\mathbf{x}_{1:M}^*, \mathcal{D}_{N}))$$

where $\\mathbf{r}$ is a representation vector of the context and target sets. Diagonal prediction maps model each target independently:

$$\pi(y_{1:M}^* | \mathbf{x}_{1:M}^* ; \mathcal{D}_{N}) = \prod_{m=1}^{M} p(y_{m}^* | \mathbf{r}(\mathbf{x}_{m}^*, \mathbf{r}_{\mathcal{D}}(\mathcal{D}_{N})))$$

While diagonal maps directly model conditional 1D marginals, they can represent any conditional joint distribution autoregressively.

## The Amortized Conditioning Engine (ACE)

### Key Innovation: Encoding Latents and Priors

ACE extends the prediction map formalism to explicitly accommodate latent variables. We redefine inputs as $\\boldsymbol{\\xi} \\in \\mathcal{X} \\cup {\\ell\_1, \\ldots, \\ell\_L}$ where $\\mathcal{X}$ is the data input space and $\\ell\_l$ is a marker for the $l$-th latent. Values are redefined as $z \\in \\mathcal{Z}$ where $\\mathcal{Z}$ can be continuous or discrete. This allows ACE to predict any combination of target variables conditioning on any other combination of context data and latents:

$$\pi(z_{1:M}^* | \boldsymbol{\xi}_{1:M}^* ; \mathfrak{D}_{N}) = \prod_{m=1}^{M} p(z_{m}^* | \mathbf{r}(\boldsymbol{\xi}_{m}^*, \mathbf{r}_{\mathcal{D}}(\mathfrak{D}_{N})))$$

> **Key Innovation:** ACE also allows the user to express probabilistic information over latent variables as prior probability distributions at runtime. To flexibly approximate a broad class of distributions, we convert each one-dimensional probability density function to a normalized histogram of probabilities over a predefined grid.

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

**Prior amortization.** Two example posterior distributions for the mean $\\mu$ and standard deviation $\\sigma$ of a 1D Gaussian. (a) Prior distribution over $\\boldsymbol{\\theta}=(\\mu, \\sigma)$ set at runtime. (b) Likelihood for the observed data. (c) Ground-truth Bayesian posterior. (d) ACE's predicted posterior approximates well the true posterior.

### Architecture

ACE consists of three main components:

1.  **Embedding Layer:** Maps context and target data points and latents to the same embedding space. For context data points $(\\mathbf{x}*n, y\_n)$, we use $f*{\\mathbf{x}}(\\mathbf{x}*n) + f*{\\text{val}}(y\_n) + \\mathbf{e}*{\\text{data}}$, while latent variables $\\theta\_l$ are embedded as $f*{\\text{val}}(\\theta\_l) + \\mathbf{e}\_l$. For latents with a prior $\\mathbf{p}*l$, we use $f*{\\text{prob}}(\\mathbf{p}\_l) + \\mathbf{e}\_l$.
2.  **Transformer Layers:** ACE employs multi-head self-attention for context points and cross-attention from target points to context, implemented efficiently to reduce computational complexity.
3.  **Output Heads:** For continuous-valued variables, ACE uses a Gaussian mixture output consisting of $K$ components. For discrete-valued variables, it employs a categorical distribution.

> **Image description.** This is a diagram illustrating the ACE architecture, a neural network architecture. The diagram shows the flow of data through different layers and components of the network.
>
> Here's a breakdown of the visible elements:
>
> - **Input:** On the left side, there are several inputs represented by rectangles.
> - Two green rectangles labeled "(θ₁)" and "(?θ₂)".
> - Three grey rectangles labeled "(x₃, y₃)", "(x₅, y₅)", "(x₄)", and "(x₆)". The label "(z₂)" is written in red.
> - **Embedder:** A large, light-pink rectangle labeled "Embedder" in grey is positioned in the center of the left side. Arrows connect the input rectangles to the Embedder.
> - **MHSA:** A blue rounded rectangle labeled "MHSA" in black. Inside the rectangle are three grey rectangles labeled "(z₁)", "(z₃)", and "(z₅)". Arrows connect the Embedder to these rectangles.
> - **CA:** Below the MHSA rectangle, there is a light-pink rectangle labeled "CA" in dark red. An arrow connects the MHSA rectangle to the CA rectangle. The label "k-blocks" is written below the CA rectangle.
> - **Head:** To the right of the CA rectangle, there is a light-pink rectangle labeled "Head (GMM or Cat)" in grey. An arrow connects the CA rectangle to the Head rectangle.
> - **Output/Loss:** On the right side, there is an orange rounded rectangle labeled "Loss" in orange. Inside the rectangle are six rectangles:
> - Two white rectangles labeled "(^θ₂)" and "(^y₄)" and "(^y₆)".
> - Four grey rectangles labeled "(y₄)" and "(y₆)".
> - One green rectangle labeled "(θ₂)".
> - **Arrows:** Arrows connect the different components, indicating the flow of data.
>
> The diagram uses different colors to highlight different types of data or components. The labels provide information about the variables and operations involved in each step of the architecture. The overall layout suggests a sequential processing of data from the input to the output/loss calculation.

**A conceptual figure of ACE architecture.**
ACE's architecture can be summarized in the embedding layer,
attention layers and output head. The $(\\mathbf{x}*n, y\_n)$ pairs
denote known data (context). The red $\\mathbf{x}*{j}$ denotes
locations where the output is unknown (target inputs). The main
innovation in ACE is that the embedder layer can incorporate known
or unknown latents $\\theta\_{l}$ and possibly priors over these.
The $z$ is the embedded data, while MHSA stands for multi head
cross attention and CA for cross-attention. The output head is a
Gaussian mixture model (GMM, for continuous variables) or
categorical (Cat, for discrete variables). Both latent and data
can be of either type.

The diagram illustrates ACE's key architectural enhancements, including:

- The ability to incorporate latent variables ($\\theta$) and their priors in the embedder layer
- More expressive output heads using Gaussian mixture models (GMM) or categorical distributions
- Flexible representation of both continuous and discrete data and latent variables

These modifications allow ACE to amortize distributions over both data and latent variables while maintaining permutation invariance for the context set.

### Training and Prediction

ACE is trained via maximum-likelihood on synthetic data. During
training, we generate each problem instance hierarchically by first
sampling the latent variables $\\boldsymbol{\\theta}$, and then data
points $(\\mathbf{X}, \\mathbf{y})$ according to the generative model of
the task. Data and latents are randomly split between context and
target. ACE requires access to latent variables during training, which
can be easily achieved in many generative models for synthetic data.

ACE minimizes the expected negative log-likelihood of the target set
conditioned on the context:

$$\mathcal{L}(\mathbf{w}) = \mathbb{E}_{\mathbf{p} \sim \mathcal{P}}\left[\mathbb{E}_{\mathfrak{D}_{N}, \boldsymbol{\xi}^*_{1:M}, \mathbf{z}^*_{1:M} \sim \mathbf{p}}\left[-\sum_{m=1}^{M} \log q(z_{m}^* | \mathbf{r}_{\mathbf{w}}(\boldsymbol{\xi}_{m}^*, \mathfrak{D}_{N}))\right]\right]$$

In this equation, $\\mathbf{w}$ represents the model parameters, $q$ is
the model's predictive distribution (a mixture of Gaussians for
continuous variables or categorical for discrete variables),
$\\mathcal{P}$ is the hierarchical model for sampling priors, and
$\\mathbf{r}*{\\mathbf{w}}$ is the transformer network that encodes the
context $\\mathfrak{D}*{N}$ and relates it to the target inputs
$\\boldsymbol{\\xi}\_{m}^\*$ (which can be data, latents, or a mix of both).

## Applications and Experimental Results

We demonstrate ACE's capabilities across diverse machine learning tasks:

### 1\. Image Completion and Classification

ACE treats image completion as a regression task, where given limited pixel values (context), it predicts the complete image. For MNIST and CelebA datasets, ACE outperforms other Transformer Neural Processes, with notable improvement when integrating latent information.

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

**Image completion.** (a) Reference image. (b) Observed
pixels (10%). (c-e) Predictions from different models. (f) Performance
across varying levels of context.

ACE also performs well at conditional image generation and image classification, as we can condition and predict latent variables such as CelebA features.

> **Image description.** The image shows a close-up of a person's head and shoulders with a bright green rectangle obscuring the top half of their face. The person's skin tone appears light, and the visible portion of their face shows features like a nose and mouth. The shoulders and neck are outlined in black. The background is white. The green rectangle covers the forehead and eyes.
>
> (a) Context
>
> **Image description.** The image is a close-up photograph of a person's face. The person appears to be a man with light skin, a white beard, and brown hair that is balding on top. The image quality is somewhat pixelated and blurry, especially around the edges. The man is wearing a dark-colored shirt or jacket. The background is plain white.
>
> (b) $\\mathrm{BALD}=$ True
>
> **Image description.** The image consists of two panels.
>
> The left panel shows a blurry image of a man's face. The face is light-skinned with dark hair. He is wearing a dark-colored shirt or jacket with a high collar. The image is somewhat pixelated and lacks fine detail.
>
> The right panel is divided into two vertical rectangles. The left rectangle is bright green, and the right rectangle is black.
>
> (c) $\\mathrm{BALD}=\\mathrm{False}$
>
> **Image description.** The image shows two panels side-by-side, each containing a partial image.
>
> Panel 1: The top half of the image is a solid bright green color. The bottom half shows a blurry image with hints of blue, white, and a brownish-orange color. The shapes are indistinct.
>
> Panel 2: This panel shows a blurred image with a color palette of blue, black, and red. The shapes are indistinct, but there is a suggestion of a rounded form in the center of the image.
>
> (d) Context
>
> **Image description.** The image contains two blurry images of faces.
>
> The left image shows a person's face with a brown complexion, set against a blurred blue background. The person appears to be bald or have very short hair. The image quality is low, making it difficult to discern fine details.
>
> The right image also shows a person's face against a blurred blue background. However, in this image, the person's hair appears dark and covers most of their forehead. The image is similarly blurry, obscuring facial features. A black vertical bar is visible on the left side of the image.
>
> (e) $\\mathrm{BALD}=\\mathrm{True}$
>
> **Image description.** The image contains two blurry headshot-style photographs.
>
> The left photograph shows a person with dark skin, a bald head, and a blurred expression. The background is a gradient of blue.
>
> The right photograph shows a person with dark skin and dark hair. A vertical black bar obscures part of the left side of the image. The background is also a gradient of blue. The image is blurry, making it difficult to discern specific facial features.
>
> (f) $\\mathrm{BALD}=\\mathrm{False}$

**Conditional image completion.** Example of ACE conditioning
on the value of the BALD feature when the top part of the image is masked.

### 2\. Bayesian Optimization (BO)

In Bayesian optimization, ACE explicitly models the global optimum location $\\mathbf{x}*{\\text{opt}}$ and value $y*{\\text{opt}}$ as latent variables. This enables:

- Direct sampling from the predictive distribution $p(\\mathbf{x}\_{\\text{opt}} | \\mathcal{D}*N, y*{\\text{opt}} \< \\tau)$ for Thompson Sampling (ACE-TS)
- Straightforward implementation of Max-Value Entropy Search (MES) acquisition function
- Seamless incorporation of prior information about the optimum location

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

**Bayesian Optimization.** Example evolution of ACE-TS on a
1D function. The orange pdf on the left of each panel is $p(y\_{\\text{opt}}
| \\mathcal{D}*N)$, the red pdf at the bottom of each panel is
$p(\\mathbf{x}*{\\text{opt}} | y\_{\\text{opt}}, \\mathcal{D}*N)$, for a
sampled $y*{\\text{opt}}$ (orange dashed-dot line). The queried point at
each iteration is marked with a red asterisk, while black and blue dots
represent the observed points. Note how ACE is able to learn complex
conditional predictive distributions for $\\mathbf{x}*{\\text{opt}}$ and
$y*{\\text{opt}}$.

Results show that ACE-MES frequently outperforms ACE-TS and often matches the gold-standard GP-MES. When prior information about the optimum location is available, ACE-TS with prior (ACEP-TS) shows significant improvement over its no-prior variant and competitive performance compared to state-of-the-art methods (see paper).

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
>
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

**Bayesian optimization results.** Regret comparison for
different methods across benchmark tasks.

### 3\. Simulation-Based Inference (SBI)

For simulation-based inference, ACE can predict posterior distributions of model parameters, simulate data, predict missing data, and incorporate priors at runtime. We evaluated ACE on three simulation models:

- Ornstein-Uhlenbeck Process (OUP)
- Susceptible-Infectious-Recovered model (SIR)
- Turin model (a complex radio propagation simulator)

ACE shows performance comparable to dedicated SBI methods on posterior estimation. When injecting informative priors (ACEP), performance improves proportionally to the provided information. Notably, while Simformer achieves similar results, ACE is significantly faster at sampling (0.05 seconds vs. 130 minutes for 1,000 posterior samples).

|       |                       Metric                        |    NPE     |    NRE     | Simformer  |    ACE     | $\\mathrm{ACEP}*{\\text {weak prior }}$ | $\\mathrm{ACEP}*{\\text {strong prior }}$ |
| :---: | :-------------------------------------------------: | :--------: | :--------: | :--------: | :--------: | :-------------------------------------: | :---------------------------------------: |
|  OUP  | $\\log -\\operatorname{probs}*{\\theta}(\\uparrow)$ | 1.09(0.10) | 1.07(0.13) | 1.03(0.04) | 1.03(0.02) |               1.05(0.02)                |                1.44(0.03)                 |
|       |    $\\operatorname{RMSE}*{\\theta}(\\downarrow)$    | 0.48(0.01) | 0.49(0.00) | 0.50(0.02) | 0.48(0.00) |               0.43(0.01)                |                0.27(0.00)                 |
|       |       $\\operatorname{MMD}*{y}(\\downarrow)$        |     -      |     -      | 0.43(0.02) | 0.51(0.00) |               0.37(0.00)                |                0.35(0.00)                 |
|  SIR  | $\\log -\\operatorname{probs}*{\\theta}(\\uparrow)$ | 6.53(0.11) | 6.24(0.16) | 6.89(0.09) | 6.78(0.02) |               6.62(0.10)                |                6.69(0.10)                 |
|       |    $\\operatorname{RMSE}*{\\theta}(\\downarrow)$    | 0.02(0.00) | 0.03(0.00) | 0.02(0.00) | 0.02(0.00) |               0.02(0.00)                |                0.02(0.00)                 |
|       |       $\\operatorname{MMD}*{y}(\\downarrow)$        |     -      |     -      | 0.02(0.00) | 0.02(0.00) |               0.02(0.00)                |                0.00(0.00)                 |
| Turin | $\\log -\\operatorname{probs}*{\\theta}(\\uparrow)$ | 1.99(0.05) | 2.33(0.07) | 3.16(0.03) | 3.14(0.02) |               3.58(0.04)                |                4.87(0.08)                 |
|       |    $\\operatorname{RMSE}*{\\theta}(\\downarrow)$    | 0.26(0.00) | 0.28(0.00) | 0.25(0.00) | 0.24(0.00) |               0.21(0.00)                |                0.13(0.00)                 |
|       |       $\\operatorname{MMD}\_{y}(\\downarrow)$       |     -      |     -      | 0.35(0.00) | 0.35(0.00) |               0.35(0.00)                |                0.34(0.00)                 |

**Comparison metrics for simulator-based inference models.**
ACE shows performance comparable to dedicated methods while offering
additional flexibility. _Table adapted from Table 1 in the main paper._

## Conclusions

1.  ACE provides a unified framework for probabilistic conditioning and prediction across diverse machine learning tasks.
2.  The ability to condition on and predict both data and latent variables enables ACE to handle tasks that would otherwise require bespoke solutions.
3.  Runtime incorporation of priors over latent variables offers additional flexibility.
4.  Experiments show competitive performance compared to task-specific methods across image completion, Bayesian optimization, and simulation-based inference.

ACE shows strong promise as a new unified and versatile method for amortized **probabilistic conditioning and prediction**. While the current implementation has limitations, such as quadratic complexity in context size and scaling challenges with many data points and latents, these provide clear directions for future work, with the goal of unlocking the power of amortized probabilistic inference for every task.

> **Image description.** This is a four-panel comic strip titled "BAYES EXPLAINS EVERYTHING!".
>
> - **Panel 1:** A sign advertises "ASK BAYES! ALL LIFE'S QUESTIONS ANSWERED PROBABILISTICALLY!". A young person approaches a historical figure labeled "Mr. Bayes" at a desk, stating they have a problem.
> - **Panel 2:** The young person elaborates on their problem, mentioning struggles with "image completion, Bayesian optimization, AND inference from simulations...". Another person looks on.
> - **Panel 3:** Mr. Bayes dismisses these as variations of the same core concept: "Ah! But they're all just probabilistic conditioning and prediction." He gestures towards a computer screen displaying the acronym "ACE" and emphasizes, "Everything."
> - **Panel 4:** The young person looks skeptical and asks, "Really?". Mr. Bayes confirms, "Everything!", pointing to a chart below that lists examples like "Image pixels," "Latent variables," "Best pizza toppings," and "Weather tomorrow," each illustrated with a probability distribution curve.

**Everything is probabilistic conditioning and prediction.**
(Comic written by GPT-4.5 and graphics by gpt-4o.)

> **Acknowledgments:** This work was supported by the Research Council of Finland (grants 358980 and 356498 and Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/)); Business Finland (project 3576/31/2023); the UKRI Turing AI World-Leading Researcher Fellowship [EP/W002973/1]. The authors thank Finnish Computing Competence Infrastructure (FCCI), Aalto Science-IT project, and CSC–IT Center for Science, Finland, for the computational and data storage resources provided, including access to the LUMI supercomputer.

## References

1.  Marta Garnelo, Dan Rosenbaum, Chris J Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J Rezende, and SM Ali Eslami. Conditional neural processes. In International Conference on Machine Learning, pages 1704-1713, 2018.
2.  Tung Nguyen and Aditya Grover. Transformer Neural Processes: Uncertainty-aware meta learning via sequence modeling. In Proceedings of the International Conference on Machine Learning (ICML), pages 123-134. PMLR, 2022.
3.  Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022.
4.  Wessel P Bruinsma, Stratis Markou, James Requeima, Andrew YK Foong, Tom R Andersson, Anna Vaughan, Anthony Buonomo, J Scott Hosking, and Richard E Turner. Autoregressive conditional neural processes. In International Conference on Learning Representations, 2023.
5.  Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48): 30055-30062, 2020.
6.  Roman Garnett. Bayesian optimization. Cambridge University Press, 2023.
7.  Zi Wang and Stefanie Jegelka. Max-value entropy search for efficient Bayesian optimization. In International Conference on Machine Learning, pages 3627-3635. PMLR, 2017.
8.  Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H Macke. All-in-one simulation-based inference. In International Conference on Machine Learning. PMLR, 2024.

---

© 2025 Paul E. Chang, Nasrulloh Loka, Daolang Huang, Ulpu Remes, Samuel Kaski, Luigi Acerbi

Webpage created with the help of [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet/). Code available at: [https://github.com/acerbilab/amortized-conditioning-engine/](https://github.com/acerbilab/amortized-conditioning-engine/)
