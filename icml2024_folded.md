# ICML 2024

## 0. A Bayesian Approach to Online Planning

<details>

<summary>Abstract</summary>

The combination of Monte Carlo tree search and neural networks has revolutionized online planning. As neural network approximations are often imperfect, we ask whether uncertainty estimates about the network outputs could be used to improve planning.   We develop a Bayesian planning approach that facilitates such uncertainty quantification, inspired by classical ideas from the meta-reasoning literature.   We propose a Thompson sampling based algorithm for searching the tree of possible actions, for which we prove the first (to our knowledge) finite time Bayesian regret bound, and propose an efficient implementation for a restricted family of posterior distributions. In addition we propose a variant of the Bayes-UCB method applied to trees. Empirically, we demonstrate that on the ProcGen Maze environment, when the uncertainty estimates are accurate but the neural network output is inaccurate, our Bayesian approach searches the tree much more effectively. In addition, we investigate whether popular uncertainty estimation methods are accurate enough to yield significant gains in planning.Two Heads Are Better Than One: Boosting Graph Sparse Training via Semantic and Topological Awareness

</details>

## 1. Accelerating Parallel Sampling of Diffusion Models

<details>

<summary>Abstract</summary>

Diffusion models have emerged as state-of-the-artgenerative models for image generation. However, sampling from diffusion models is usuallytime-consuming due to the inherent autoregressive nature of their sampling process. In thiswork, we propose a novel approach that accelerates the sampling of diffusion models by parallelizing the autoregressive process. Specifically,we reformulate the sampling process as solving asystem of triangular nonlinear equations throughfixed-point iteration. With this innovative formulation, we explore several systematic techniquesto further reduce the iteration steps required bythe solving process. Applying these techniques,we introduce ParaTAA, a universal and training-free parallel sampling algorithm that can leverage extra computational and memory resourcesto increase the sampling speed. Our experimentsdemonstrate that ParaTAA can decrease the inference steps required by common sequential sampling algorithms such as DDIM and DDPM bya factor of 4$\sim$14 times. Notably, when applyingParaTAA with 100 steps DDIM for Stable Diffusion, a widely-used text-to-image diffusion model,it can produce the same images as the sequentialsampling in only 7 inference steps.

</details>

## 2. Active Statistical Inference 🌟

<details>

<summary>Abstract</summary>

Inspired by the concept of active learning, we propose active inference---a methodology for statistical inference with machine-learning-assisted data collection. Assuming a budget on the number of labels that can be collected, the methodology uses a machine learning model to identify which data points would be most beneficial to label, thus effectively utilizing the budget. It operates on a simple yet powerful intuition: prioritize the collection of labels for data points where the model exhibits uncertainty, and rely on the model's predictions where it is confident. Active inference constructs provably valid confidence intervals and hypothesis tests while leveraging any black-box machine learning model and handling any data distribution. Moreover, it achieves far smaller errors than existing baselines relying on i.i.d. data, enabling smaller confidence intervals and more powerful p-values. We evaluate active inference on datasets from survey research and proteomics.

</details>

## 3. Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models

<details>

<summary>Abstract</summary>

Inverse problems arise in a multitude of applications, where the goal is to recover a clean signal from noisy and possibly (non)linear observations. The difficulty of a reconstruction problem  depends on multiple factors, such as the structure of the ground truth signal, the severity of the degradation and the complex interactions between the above. This results in natural sample-by-sample variation in the difficulty of a reconstruction task, which is often overlooked by contemporary techniques. Our key observation is that most existing inverse problem solvers lack the ability to adapt their compute power to the difficulty of the reconstruction task, resulting in subpar performance and wasteful resource allocation. We propose a novel method that we call severity encoding,  to estimate the degradation severity of noisy, degraded signals in the latent space of an autoencoder. We show that the estimated severity has strong correlation with the true corruption level and can give useful hints at the difficulty of reconstruction problems on a sample-by-sample basis. Furthermore, we propose a reconstruction method based on latent diffusion models that leverages the predicted degradation severities to fine-tune the reverse diffusion sampling trajectory and thus achieve sample-adaptive inference times. Our framework acts as a wrapper that can be combined with any latent diffusion-based baseline solver, imbuing it with sample-adaptivity and acceleration. We perform numerical experiments on both linear and nonlinear inverse problems and demonstrate that our technique greatly improves the performance of the baseline solver and achieves up to $10\times$ acceleration in mean sampling speed.

</details>

## 4. Adaptive Hierarchical Certification for Segmentation using Randomized Smoothing 🌟

<details>

<summary>Abstract</summary>

Common certification methods operate on a flat pre-defined set of fine-grained classes. In this paper, however, we propose a novel, more general, and practical setting, namely adaptive hierarchical certification for image semantic segmentation. In this setting, the certification can be within a multi-level hierarchical label space composed of fine to coarse levels. Unlike classic methods where the certification would abstain for unstable components, our approach adaptively relaxes the certification to a coarser level within the hierarchy. This relaxation lowers the abstain rate whilst providing more certified semantically meaningful information. We mathematically formulate the problem setup and introduce, for the first time, an adaptive hierarchical certification algorithm for image semantic segmentation, that certifies image pixels within a hierarchy and prove the correctness of its guarantees. Since certified accuracy does not take the loss of information into account when traversing into a coarser hierarchy level, we introduce a novel evaluation paradigm for adaptive hierarchical certification, namely the certified information gain metric, which is proportional to the class granularity level. Our evaluation experiments on real-world challenging datasets such as Cityscapes and ACDC demonstrate that our adaptive algorithm achieves a higher certified information gain and a lower abstain rate compared to the current state-of-the-art certification method, as well as other non-adaptive versions of it.

</details>

## 5. Adaptive Robust Learning using Latent Bernoulli Variables 🌟

<details>

<summary>Abstract</summary>

We present an adaptive approach for robust learning from corrupted training sets. We identify corrupted and non-corrupted samples with latent Bernoulli variables and thus formulate the learning problem as maximization of the likelihood where latent variables are marginalized. The resulting problem is solved via variational inference, using an efficient Expectation-Maximization based method. The proposed approach improves over the state-of-the-art by automatically inferring the corruption level, while adding minimal computational overhead. We demonstrate our robust learning method and its parameter-free nature on a wide variety of machine learning tasks including online learning and deep learning where it adapts to different levels of noise and maintains high prediction accuracy.

</details>

## 6. A Geometric Explanation of the Likelihood OOD Detection Paradox

<details>

<summary>Abstract</summary>

Likelihood-based deep generative models (DGMs) commonly exhibit a puzzling behaviour: when trained on a relatively complex dataset, they assign higher likelihood values to out-of-distribution (OOD) data from simpler sources. Adding to the mystery, OOD samples are never generated by these DGMs despite having higher likelihoods. This two-pronged paradox has yet to be conclusively explained, making likelihood-based OOD detection unreliable. Our primary observation is that high-likelihood regions will not be generated if they contain minimal probability mass. We demonstrate how this seeming contradiction of large densities yet low probability mass can occur around data confined to low-dimensional manifolds. We also show that this scenario can be identified through local intrinsic dimension (LID) estimation, and propose a method for OOD detection which pairs the likelihoods and LID estimates obtained from a pre-trained DGM. Our method can be applied to normalizing flows and score-based diffusion models - which we show are also afflicted by the paradox - and often obtains results which surpass state-of-the-art OOD detection benchmarks using the same DGM backbones.

</details>

## 7. A Graph is Worth $K$ Words: Euclideanizing Graph using Pure Transformer 🌟

<details>

<summary>Abstract</summary>

Can we model non-Euclidean graphs as pure language or even Euclidean vectors while retaining their inherent information? The non-Euclidean property have posed a long term challenge in graph modeling. Despite recent GNN and Graphformer efforts encoding graphs as Euclidean vectors, recovering original graph from the vectors remains a challenge. We introduce GraphsGPT, featuring a Graph2Seq encoder that transforms non-Euclidean graphs into learnable graph words in a Euclidean space, along with a GraphGPT decoder that reconstructs the original graph from graph words to ensure information equivalence. We pretrain GraphsGPT on 100M molecules and yield some interesting findings: (1) Pretrained Graph2Seq excels in graph representation learning, achieving state-of-the-art results on 8/9 graph classification and regression tasks. (2) Pretrained GraphGPT serves as a strong graph generator, demonstrated by its ability to perform both unconditional and conditional graph generation. (3)  Graph2Seq+GraphGPT enables effective graph mixup in the Euclidean space, overcoming previously known challenges about non-Euclidean data mixup. (4) Our proposed novel edge-centric GPT pretraining task is effective in graph fields, underscoring its success in both representation and generation.

</details>

## 8. Aligned Objective for Soft-Pseudo-Label Generation in Supervised Learning 🌟

<details>

<summary>Abstract</summary>

Soft pseudo-labels, generated by the softmax predictions of the trained networks, offer a probabilistic rather than binary form, and have been shown to improve the performance of deep neural networks in supervised learning.   Most previous methods adopt classification loss to train a classifier as the soft-pseudo-label generator and fail to fully exploit their potential due to the misalignment with the target of soft-pseudo-label generation, aimed at capturing the knowledge in the data rather than making definitive classifications. Nevertheless, manually designing an effective objective function for a soft-pseudo-label generator is challenging, primarily because datasets typically lack ground-truth soft labels, complicating the evaluation of the soft pseudo-label accuracy.  To deal with this problem, we propose a novel framework that alternately trains the predictive model and the soft-pseudo-label generator guided by a meta-network-parameterized objective function. The parameters of the objective function are optimized based on the feedback from both the performance of the predictive model and the soft-pseudo-label generator in the learning task.  Additionally, the framework offers versatility across different learning tasks by allowing direct modifications to the task loss. Experiments on the benchmark datasets validate the effectiveness of the proposed framework.

</details>

## 9. Ambiguity-Aware Abductive Learning

<details>

<summary>Abstract</summary>

Abductive Learning (ABL) is a promising framework for integrating sub-symbolic perception and logical reasoning through abduction. In this case, the abduction process provides supervision for the perception model from the background knowledge. Nevertheless, this process naturally contains uncertainty, since the knowledge base may be satisfied by numerous potential candidates. This implies that the result of the abduction process, i.e., a set of candidates, is ambiguous; both correct and incorrect candidates are mixed in this set. The prior art of abductive learning selects the candidate that has the minimal inconsistency of the knowledge base. However, this method overlooks the ambiguity in the abduction process and is prone to error when it fails to identify the correct candidates. To address this, we propose Ambiguity-Aware Abductive Learning ($\textrm{A}^3\textrm{BL}$), which evaluates all potential candidates and their probabilities, thus preventing the model from falling into sub-optimal solutions. Both experimental results and theoretical analyses prove that $\textrm{A}^3\textrm{BL}$ markedly enhances ABL by efficiently exploiting the ambiguous abduced supervision.

</details>

## 10. Ameliorate Spurious Correlations in Dataset Condensation

<details>

<summary>Abstract</summary>

Dataset Condensation has emerged as a technique for compressing large datasets into smaller synthetic counterparts,  facilitating downstream training tasks. In this paper, we study the impact of bias inside the original dataset on the performance of dataset condensation. With a comprehensive empirical evaluation on canonical datasets with color, corruption and background biases, we found that color and background biases in the original dataset will be amplified through the condensation process, resulting in a notable decline in the performance of models trained on the condensed dataset, while corruption bias is suppressed through the condensation process.   To reduce bias amplification in dataset condensation, we introduce a simple yet highly effective approach based on a sample reweighting scheme utilizing kernel density estimation.Empirical results on multiple real-world and synthetic datasets demonstrate the effectiveness of the proposed method.   Notably, on CMNIST with 5\% bias-conflict ratio and IPC 50, our method achieves 91.5\% test accuracy compared to 23.8\% from vanilla DM, boosting the performance by 67.7\%, whereas applying state-of-the-art debiasing method on the same dataset only achieves 53.7\% accuracy.   Our findings highlight the importance of addressing biases in dataset condensation and provide a promising avenue to address bias amplification in the process.

</details>

## 11. Amortized Variational Deep Kernel Learning

<details>

<summary>Abstract</summary>

Deep kernel learning (DKL) marries the uncertainty quantification of Gaussian processes (GPs) and the representational power of deep neural networks. However, training DKL is challenging and often leads to overfitting. Most notably, DKL often learns "non-local" kernels --- incurring spurious correlations. To remedy this pathology, we propose using amortized inducing points and a parameter-sharing scheme, which ties together the amortization and DKL networks. This design imposes an explicit dependency between the ELBO's model fit and capacity terms. In turn, this prevents the former from dominating the optimization procedure and incurring the aforementioned spurious correlations. Extensive experiments show that our resulting method, amortized varitional DKL (AVDKL), i) consistently outperforms DKL and standard GPs for tabular data; ii) achieves significantly higher accuracy than DKL in node classification tasks; and iii) leads to substantially better accuracy and negative log-likelihood than DKL on CIFAR100.

</details>

## 12. An Empirical Study of Realized GNN Expressiveness

<details>

<summary>Abstract</summary>

Research on the theoretical expressiveness of Graph Neural Networks~(GNNs) has developed rapidly, and many methods have been proposed to enhance the expressiveness. However, most methods do not have a uniform expressiveness measure except for a few that strictly follow the $k$-dimensional Weisfeiler-Lehman ($k$-WL) test hierarchy. Their theoretical analyses are often limited to distinguishing certain families of non-isomorphic graphs, leading to difficulties in quantitatively comparing their expressiveness. In contrast to theoretical analysis, another way to measure expressiveness is by evaluating model performance empirically with 1-WL-indistinguishable graphs. Previous datasets face problems with difficulty (any model surpassing 1-WL has nearly 100\% accuracy), granularity (models tend to be either 100\% correct or near random guess), and scale (only several essentially different graphs involved). To address these limitations, we restudied the realized expressive power of different expressive GNN models on a new expressiveness dataset, BREC, which poses greater difficulty (with up to 4-WL-indistinguishable graphs), finer granularity (can compare models between 1-WL and 3-WL), a larger scale (800 1-WL-indistinguishable graphs non-isomorphic to each other).%, and a more reliable evaluation result (with controllable error rate). We synthetically test 23 models with higher-than-1-WL expressiveness on BREC. Our experiment gives the first thorough measurement of the realized expressiveness of those state-of-the-art beyond-1-WL GNN models and reveals the gap between theoretical and realized expressiveness. Dataset and evaluation codes are released at: https://github.com/brec-icml2024/brec-icml2024.

</details>

## 13. A Theory of Fault-Tolerant Learning 🌟

<details>

<summary>Abstract</summary>

Developing machine learning models that account for potential faults encountered in real-world environments presents a fundamental challenge for mission-critical applications. In this paper, we introduce a novel theoretical framework grounded in learning theory for dealing with faults. In particular, we propose a framework called *fault-tolerant PAC learning*, aimed at identifying the most fault-tolerant models from a given hypothesis class (such as neural networks). We show that if faults occur randomly, fault-tolerant learning is equivalent to regular PAC learning. However, for *adversarial* faults, we show that the sample complexity of fault-tolerant PAC learning can grow linearly w.r.t. the number of perturbing functions induced by the faults, even for a hypothesis class with VC-dimension 1. We then provide a matching upper bound by restricting the number of perturbing functions. Finally, we show that the linear dependency on the number of perturbing functions can be substantially improved for *deletion faults* in neural networks. Our work provides a powerful formal framework and avenues for a number of future investigations on the precise characterization of fault-tolerant learning.

</details>

## 14. Balanced Data, Imbalanced Spectra: Unveiling Class Disparities with Spectral Imbalance

<details>

<summary>Abstract</summary>

Classification models are expected to perform equally well for different classes, yet in practice, there are often large gaps in their performance. This issue of class bias is widely studied in cases of datasets with sample imbalance, but is relatively overlooked in balanced datasets. In this work, we introduce the concept of spectral imbalance in features as a potential source for class disparities and study the connections between spectral imbalance and class bias in both theory and practice.  To build the connection between spectral imbalance and class gap, we develop a theoretical framework for studying class disparities and derive exact expressions for the per-class error in a high-dimensional mixture model setting. We then study this phenomenon in 11 different state-of-the-art pre-trained encoders, and show how our proposed framework can be used to compare the quality of encoders, as well as evaluate and combine data augmentation strategies to mitigate the issue. Our work sheds light on the class-dependent effects of learning, and provides new insights into how state-of-the-art pre-trained features may have unknown biases that can be diagnosed through their spectra.

</details>

## 15. Bayesian Knowledge Distillation: A Bayesian Perspective of Distillation with Uncertainty Quantification 🌟

<details>

<summary>Abstract</summary>

Knowledge distillation (KD) has been widely used for model compression and deployment acceleration.  Nonetheless, the statistical insight of the remarkable performance of KD remains elusive, and methods for evaluating the uncertainty of the distilled model/student model are lacking. To address these issues, we establish a close connection between KD and a Bayesian model. In particular, we develop an innovative method named Bayesian Knowledge Distillation (BKD) to provide a transparent interpretation of the working mechanism of KD, and a suite of Bayesian inference tools for the uncertainty quantification of the student model. In BKD, the regularization imposed by the teacher model in KD is formulated as a  teacher-informed prior for the student model's parameters. Consequently, we establish the equivalence between minimizing the KD loss and estimating the posterior mode in BKD. Efficient Bayesian inference algorithms are developed based on the stochastic gradient Langevin Monte Carlo and examined with extensive experiments on uncertainty ranking and credible intervals construction for predicted class probabilities.

</details>

## 16. Benign Overfitting in Adversarially Trained Neural Networks

<details>

<summary>Abstract</summary>

Benign overfitting is the phenomenon wherein none of the predictors in the hypothesis class can achieve perfect accuracy (i.e., the non-realizable or noisy setting), but a model that interpolates the training data sill achieves good generalization. A series of recent works aim to understand this phenomenon, for regression and classification tasks using linear predictors as well as two-layer neural networks. In this paper, we study such a benign overfitting phenomenon in an adversarial setting. We show that under a distributional assumption, interpolating neural networks found using adversarial training generalize well despite additive inference-time attacks. Specifically, we provide convergence and generalization guarantees for adversarial training of two-layer networks (both, with smooth and non-smooth activation functions), showing that under moderate $\ell_2$ norm perturbation budget, the trained model has near-zero robust training loss and near-optimal robust generalization error. We support our theoretical findings with an empirical study on synthetic and real-world data.

</details>

## 17. Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling

<details>

<summary>Abstract</summary>

Monte Carlo methods, Variational Inference, and their combinations play a pivotal role in sampling from intractable probability distributions. However, current studies lack a unified evaluation framework, relying on disparate performance measures and limited method comparisons across diverse tasks, complicating the assessment of progress and hindering the decision-making of practitioners. In response to these challenges, our work introduces a benchmark that evaluates sampling methods using a standardized task suite and a broad range of performance criteria. Moreover, we study existing metrics for quantifying mode collapse and introduce novel metrics for this purpose. Our findings provide insights into strengths and weaknesses of existing sampling methods, serving as a valuable reference for future developments.

</details>

## 18. Biharmonic Distance of Graphs and its Higher-Order Variants: Theoretical Properties with Applications to Centrality and Clustering 🌟

<details>

<summary>Abstract</summary>

Effective resistance is a distance between nodes of a graph that is both theoretically interesting and useful in applications. We study a variant of effective resistance called the biharmonic distance. While the effective resistance measures how well-connected two nodes are, we prove several theoretical results supporting the idea that the biharmonic distance measures how important an edge is to the global topology of the graph. Our theoretical results connect the biharmonic distance to well-known measures of connectivity of a graph like its total resistance and sparsity. Based on these result, we introduce two clustering algorithms using the biharmonic distance. Finally, we introduce a further generalization of the biharmonic distance that we call the $k$-harmonic distance. We empirically study the utility of biharmonic and $k$-harmonic distance for graph clustering.

</details>

## 19. Binning as a Pretext Task: Improving Self-Supervised Learning in Tabular Domains 🌟

<details>

<summary>Abstract</summary>

The ability of deep networks to learn superior representations hinges on leveraging the proper inductive biases, considering the inherent properties of datasets. In tabular domains, it is critical to effectively handle heterogeneous features (both categorical and numerical) in a unified manner and to grasp irregular functions like piecewise constant functions. To address the challenges in the self-supervised learning framework, we propose a novel pretext task based on the classical binning method. The idea is straightforward: reconstructing the bin indices (either orders or classes) rather than the original values. This pretext task provides the encoder with an inductive bias to capture the irregular dependencies, mapping from continuous inputs to discretized bins, and mitigates the feature heterogeneity by setting all features to have category-type targets. Our empirical investigations ascertain several advantages of binning: capturing the irregular function, compatibility with encoder architecture and additional modifications, standardizing all features into equal sets, grouping similar values within a feature, and providing ordering information. Comprehensive evaluations across diverse tabular datasets corroborate that our method consistently improves tabular representation learning performance for a wide range of downstream tasks. The codes are available in the supplementary material.

</details>

## 20. Bridging the gap between mini-batch and asymptotic analysis in contrastive learning: From InfoNCE to Kernel-based losses 🌟

<details>

<summary>Abstract</summary>

What do different contrastive learning (CL) losses actually optimize for? Although multiple CL methods have demonstrated remarkable representation learning capabilities, the differences in their inner workings remain largely opaque. In this work, we analyse several CL families and prove that, under certain conditions, they admit the same minimisers when optimizing either their batch-level objectives or their expectations asymptotically. In both cases, an intimate connection with the hyperspherical energy minimisation (HEM) problem resurfaces. Drawing inspiration from this, we introduce a novel CL objective, coined Decoupled Hyperspherical Energy Loss (DHEL). DHEL simplifies the problem by decoupling the target hyperspherical energy from the alignment of positive examples while preserving the same theoretical guarantees. Going one step further, we show the same results hold for another relevant CL family, namely kernel contrastive learning (KCL), with the additional advantage of the expected loss being independent of batch size, thus identifying the minimisers in the non-asymptotic regime. Empirical results demonstrate improved downstream performance and robustness across combinations of different batch sizes and hyperparameters and reduced dimensionality collapse, on several computer vision datasets.

</details>

## 21. Calibration Bottleneck: Over-compressed Representations are Less Calibratable

<details>

<summary>Abstract</summary>

Although deep neural networks have achieved remarkable success, they often exhibit a significant deficiency in reliable uncertainty calibration. This paper focus on model calibratability, which assesses how amenable a model is to be well recalibrated post-hoc. We find that the widely used weight decay regularizer detrimentally affects model calibratability, subsequently leading to a decline in final calibration performance after post-hoc calibration. To identify the underlying causes leading to poor calibratability, we delve into the calibratability of intermediate features across the hidden layers. We observe a U-shaped trend in the calibratability of intermediate features from the bottom to the top layers, which indicates that over-compression of the top representation layers significantly hinders model calibratability. Based on the observations, this paper introduces a weak classifier hypothesis, i.e., given a weak classification head that has not been over-trained, the representation module can be better learned to produce more calibratable features. Consequently, we propose a progressively layer-peeled training (PLP) method to exploit this hypothesis, thereby enhancing model calibratability. Our comparative experiments show the effectiveness of our method, which improves model calibration and also yields competitive predictive performance.

</details>

## 22. CKGConv: General Graph Convolution with Continuous Kernels 🌟

<details>

<summary>Abstract</summary>

The existing definitions of graph convolution, either from spatial or spectral perspectives, are inflexible and not unified.Defining a general convolution operator in the graph domain is challenging due to the lack of canonical coordinates, the presence of irregular structures, and the properties of graph symmetries.In this work, we propose a general graph convolution framework by parameterizing the kernels as continuous functions of pseudo-coordinates derived via graph positional encoding.  We name this Continuous Kernel Graph Convolution (CKGConv).Theoretically, we demonstrate that CKGConv is flexible and expressive.CKGConv encompasses many existing graph convolutions, and exhibits the same expressiveness as graph transformers in terms of distinguishing non-isomorphic graphs.Empirically, we show that CKGConv-based Networks outperform existing graph convolutional networks and perform comparably to the best graph transformers across a variety of graph datasets.

</details>

## 23. Class-Imbalanced Graph Learning without Class Rebalancing 🌟

<details>

<summary>Abstract</summary>

Class imbalance is prevalent in real-world node classification tasks and poses great challenges for graph machine-learning models. Most existing studies are rooted in a class-rebalancing (CR) perspective and aim to address class imbalance with class-wise reweighting or resampling. In this work, we approach the root cause of class-imbalance bias from an orthogonal topological paradigm. Specifically, we theoretically reveal and empirically observe two fundamental phenomena in the underlying graph topology that can greatly exacerbate the predictive bias stemming from class imbalance. In light of these findings, we devise a lightweight topological augmentation framework called TOBE to mitigate the class-imbalance bias without class rebalancing. Being orthogonal to CR, the proposed TOBE is a model-agnostic and efficient solution that can be seamlessly combined with and further boost existing CR techniques. Systematic experiments on real-world imbalanced graph learning tasks show that TOBE can deliver up to 46.27% performance gain and up to 72.74% bias reduction over existing techniques. Code is available at https://anonymous.4open.science/r/ToBE/.

</details>

## 24. Community-Invariant Graph Contrastive Learning 🌟

<details>

<summary>Abstract</summary>

Graph augmentation has received great attention in recent years for graph contrastive learning (GCL) to learn well-generalized node/graph representations. However, mainstream GCL methods often favor randomly disrupting graphs for augmentation, which shows limited generalization and inevitably leads to the corruption of high-level graph information, i.e., the graph community. Moreover, current knowledge-based graph augmentation methods can only focus on either topology or node features, causing the model to lack robustness against various types of noise. To address these limitations, this research investigated the role of the graph community in graph augmentation and figured out its crucial advantage for learnable graph augmentation. Based on our observations, we propose a community-invariant GCL framework to maintain graph community structure during learnable graph augmentation. By maximizing the spectral changes, this framework unifies the constraints of both topology and feature augmentation, enhancing the model's robustness. Empirical evidence on 21 benchmark datasets demonstrates the exclusive merits of our framework. Code is released on Github (https://anonymous.4open.science/r/CI-GCL-E718).

</details>

## 25. Comparing Graph Transformers via Positional Encodings 🌟

<details>

<summary>Abstract</summary>

The distinguishing power of graph transformers is closely tied to the choice of *positional encoding*: features used to augment the base transformer with information about the graph. There are two primary types of positional encoding: *absolute positional encodings (APEs)}*and *relative positional encodings (RPEs)*. APEs assign features to each node and are given as input to the transformer. RPEs instead assign a feature to each *pair of nodes*, e.g., graph distance, and are used to augment the attention block. A priori, it is unclear which method is better for maximizing the power of the resulting graph transformer. In this paper, we aim to understand the relationship between these different types of positional encodings. Interestingly, we show that graph transformers using APEs and RPEs are equivalent in terms of distinguishing power. In particular, we demonstrate how to interchange APEs and RPEs while maintaining their distinguishing power in terms of graph transformers. Based on our theoretical results, we provide a study on several APEs and RPEs (including the recently introduced stable and expressive positional encoding (SPE) as well as the resistance distance) and compare their distinguishing power in terms of transformers. We believe our work will help navigate the huge number of positional encoding choices and will provide guidance on the future design of positional encodings for graph transformers.

</details>

## 26. Complexity Matters: Feature Learning in the Presence of Spurious Correlations

<details>

<summary>Abstract</summary>

Existing research often posits spurious features as *easier* to learn than core features in neural network optimization, but the impact of their relative simplicity remains under-explored. In this paper, we propose a theoretical framework and associated synthetic dataset grounded in boolean function analysis which allows for fine-grained control on the relative complexity (compared to core features) and correlation strength (with respect to the label) of spurious features. Our setup uncovers several interesting phenomenon: (1) stronger spurious correlations or simpler spurious features slow down the rate of learning for the core features, (2) learning phases of spurious features and core features are not always separable, (3) spurious features are not forgotten even after core features are fully learned. We show that our findings justify the success of retraining the last layer to remove spurious correlation and also identifies limitations of popular debiasing algorithms that exploit early learning of spurious features. We support our empirical findings with theoretical analyses for the case of learning XOR features with a one-hidden-layer ReLU network.

</details>

## 27. Compress Clean Signal from Noisy Raw Image: A Self-Supervised Approach

<details>

<summary>Abstract</summary>

Raw images offer unique advantages in many low-level visual tasks due to their unprocessed nature. However, this unprocessed state accentuates noise, making raw images challenging to compress effectively. Current compression methods often overlook the ubiquitous noise in raw space, leading to increased bitrates and reduced quality. In this paper, we propose a novel raw image compression scheme that selectively compresses the noise-free component of the input, while discarding its real noise using a self-supervised approach. By excluding noise from the bitstream, both the coding efficiency and reconstruction quality are significantly enhanced. We curate an full-daydataset of raw images with calibrated noise parameters and reference images to evaluate the performance of models under a wide range of input signal-noise ratios. Experimental results demonstrate that our method surpasses existing compression techniques, achieving a more advantageous rate-distortion balance with improvements ranging from +2 to +10dB and yielding a bit saving of 2 to 50 times. The code will be released upon paper acceptance.

</details>

## 28. Confidence-aware Contrastive Learning for Selective Classification

<details>

<summary>Abstract</summary>

Selective classification enables models to make predictions only when they are sufficiently confident, aiming to enhance safety and reliability, which is important in high-stakes scenarios. Previous methods mainly use deep neural networks and focus on modifying the architecture of classification layers to enable the model to estimate the confidence of its prediction. This work provides a generalization bound for selective classification, disclosing that optimizing feature layers helps improve the performance of selective classification. Inspired by this theory, we propose to explicitly improve the selective classification model at the feature level for the first time, leading to a novel Confidence-aware Contrastive Learning method for Selective Classification, CCL-SC, which similarizes the features of homogeneous instances and differentiates the features of heterogeneous instances, with the strength controlled by the model's confidence. The experimental results on typical datasets, i.e., CIFAR-10, CIFAR-100, CelebA, and ImageNet, show that CCL-SC achieves significantly lower selective risk than state-of-the-art methods, across almost all coverage degrees. Moreover, it can be combined with existing methods to bring further improvement.

</details>

## 29. Convergence Guarantees for the DeepWalk Embedding on Block Models

<details>

<summary>Abstract</summary>

Graph embeddings have emerged as a powerful tool for understanding the structure of graphs. Unlike classical spectral methods, recent methods such as DeepWalk, Node2Vec, etc. are based on solving non-linear optimization problems on the graph, using local information obtained by performing random walks. These techniques have empirically been shown to produce ``better'' embeddings than their classical counterparts. However, due to their reliance on solving a non-convex optimization problem, obtaining theoretical guarantees on the properties of the solution has remained a challenge, even for simple classes of graphs. In this work, we show convergence properties for the DeepWalk algorithm on graphs obtained from the Stochastic Block Model (SBM). Despite being simplistic, the SBM is a classic model for analyzing the behavior of algorithms on large graphs. Our results mirror the existing ones for spectral embeddings on SBMs, showing that even in the case of one-dimensional embeddings, the output of the DeepWalk algorithm recovers the cluster structure with high probability.

</details>

## 30. Cooperative Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Graph neural networks are popular architectures for graph machine learning, based on iterative computation of node representations of an input graph through a series of invariant transformations. A large class of graph neural networks follow a standard message-passing paradigm: at every layer, each node state is updated based on an aggregate of messages from its neighborhood. In this work, we propose a novel framework for training graph neural networks, where every node is viewed as a player that can choose to either `listen`, `broadcast`, `listen and broadcast`, or to `isolate`. The standard message propagation scheme can then be viewed as a special case of this framework where every node `listens and broadcasts` to all neighbors. Our approach offers a more flexible and dynamic message-passing paradigm, where each node can determine its own strategy based on their state,  effectively exploring the graph topology while learning. We provide a theoretical analysis of the new message-passing scheme which is further supported by an extensive empirical analysis on a synthetic and real-world datasets.

</details>

## 31. Copula-Nested Spectral Kernel Network

<details>

<summary>Abstract</summary>

Spectral Kernel Networks (SKNs) emerge as a promising approach in machine learning, melding solid theoretical foundations of spectral kernels with the representation power of hierarchical architectures. At its core, the spectral density function plays a pivotal role by revealing essential patterns in data distributions, thereby offering deep insights into the underlying framework in real-world tasks. Nevertheless, prevailing designs of spectral density often overlook the intricate interactions within data structures. This phenomenon consequently neglects expanses of the hypothesis space, thus curtailing the performance of SKNs. This paper addresses the issues through a novel approach, the **Co**pula-Nested Spectral **Ke**rnel **Net**work (**CokeNet**). Concretely, we first redefine the spectral density with the form of copulas to enhance the diversity of spectral densities. Next, the specific expression of the copula module is designed to allow the excavation of complex dependence structures. Finally, the unified kernel network is proposed by integrating the corresponding spectral kernel and the copula module. Through rigorous theoretical analysis and experimental verification, CokeNet demonstrates superior performance and significant advancements over SOTA algorithms in the field.

</details>

## 32. Correlation-Induced Label Prior for Semi-Supervised Multi-Label Learning

<details>

<summary>Abstract</summary>

Semi-supervised multi-label learning (SSMLL) aims to address the challenge of limited labeled data availability in multi-label learning (MLL) by leveraging unlabeled data to improve the model's performance.  Due to the difficulty of estimating the reliable label correlation on minimal multi-labeled data, previous SSMLL methods fail to unlash the power of the correlation among multiple labels to improve the performance of the predictive model in SSMLL. To deal with this problem, we propose a novel SSMLL method named PCLP where the correlation-induced label prior is inferred to enhance the pseudo-labeling instead of dirtily estimating the correlation among labels. Specifically, we construct the correlated label prior probability distribution using structural causal model (SCM), constraining the correlations of generated pseudo-labels to conform to the prior, which can be integrated into a variational label enhancement framework optimized by both labeled and unlabeled instances in a unified manner. Theoretically, we demonstrate the accuracy of the generated pseudo-labels and guarantee the learning consistency of the proposed method. Comprehensive experiments on several benchmark datasets have validated the superiority of the proposed method.

</details>

## 33. CurBench: Curriculum Learning Benchmark 🌟

<details>

<summary>Abstract</summary>

Curriculum learning is a training paradigm where machine learning models are trained in a meaningful order, inspired by the way humans learn curricula. Due to its capability to improve model generalization and convergence, curriculum learning has gained considerable attention and has been widely applied to various research domains. Nevertheless, as new curriculum learning methods continue to emerge, it remains an open issue to benchmark them fairly. Therefore, we develop CurBench, the first benchmark that supports systematic evaluations for curriculum learning. Specifically, it consists of 15 datasets spanning 3 research domains: computer vision, natural language processing, and graph machine learning, along with 3 settings: standard, noise, and imbalance. To facilitate a comprehensive comparison, we establish the evaluation from 2 dimensions: performance and complexity. CurBench also provides a unified pipeline that plugs automatic curricula into general machine learning process, enabling the implementation of 14 core curriculum learning methods. On the basis of this benchmark, we conduct comparative experiments and make empirical analyses of existing methods. CurBench will be open-source and publicly available on GitHub after the review stage.

</details>

## 34. Data-free Distillation of Diffusion Models with Bootstrapping

<details>

<summary>Abstract</summary>

Diffusion models have demonstrated great potential for generating diverse images. However, their performance often suffers from slow generation due to iterative denoising. Knowledge distillation has been recently proposed as a remedy which can reduce the number of inference steps to one or a few, without significant quality degradation. However, existing distillation methods either require significant amounts of offline computation for generating synthetic training data from the teacher model, or need to perform expensive online learning with the help of real data. In this work, we present a novel technique called BOOT, that overcomes these limitations with an efficient data-free distillation algorithm. The core idea is to learn a time-conditioned model that predicts the output of a pre-trained diffusion model teacher given any time-step. Such a model can be efficiently trained based on bootstrapping from two consecutive sampled steps. Furthermore, our method can be easily adapted to large-scale text-to-image diffusion models, which are challenging for previous methods given the fact that the training sets are often large and difficult to access. We demonstrate the effectiveness of our approach on several benchmark datasets in the DDIM setting, achieving comparable generation quality while being orders of magnitude faster than the diffusion teacher. The text-to-image results show that the proposed approach is able to handle highly complex distributions, shedding light on more efficient generative modeling.

</details>

## 35. Delaunay Graph: Addressing Over-Squashing and Over-Smoothing Using Delaunay Triangulation 🌟

<details>

<summary>Abstract</summary>

GNNs rely on the exchange of messages to distribute information along the edges of the graph. This approach makes the efficiency of architectures highly dependent on the specific structure of the input graph. Certain graph topologies lead to inefficient information propagation, resulting in a phenomenon known as over-squashing.While the majority of existing methods address over-squashing by rewiring the input graph, our novel approach involves constructing a graph directly from features using Delaunay Triangulation. We posit that the topological properties of the resulting graph prove advantageous for mitigate oversmoothing and over-squashing. Our extensive experimentation demonstrates that our method consistently outperforms established graph rewiring methods.

</details>

## 36. Denoising Autoregressive Representation Learning

<details>

<summary>Abstract</summary>

While visual representation learning and image generation often use separate techniques, the ability to generate realistic images is intrinsically dependent upon a deep understanding of visual representations. In this paper, we explore the potential of generative pre-training for visual representations. Our method employs a decoder-only Transformer to predict image patches autoregressively. We find that training with Mean Squared Error (MSE) alone leads to strong representations. To bring it one step closer to image generation methods, we replace the MSE loss with the diffusion objective by adding a denoising patch decoder. We show that the representation quality can be improved by using tailored noise schedules and longer training in larger models. However, these schedules differ significantly from the typical schedules used for image generation purpose. Overall, our approach delivers performance remarkably close to state-of-the-art masked prediction models under the fine-tuning protocol. This marks a significant advancement in representation learning through generative approaches.

</details>

## 37. Density-Softmax: Efficient Test-time Model for Uncertainty Estimation and Robustness under Distribution Shifts

<details>

<summary>Abstract</summary>

Sampling-based methods, e.g., Deep Ensembles and Bayesian Neural Nets have become promising approaches to improve the quality of uncertainty estimation and robust generalization. However, they suffer from a large model size and high latency at test-time, which limits the scalability needed for low-resource devices and real-time applications. To resolve these computational issues, we propose Density-Softmax, a sampling-free deterministic framework via combining a density function built on a Lipschitz-constrained feature extractor with the softmax layer. Theoretically, we show that our model is the solution of minimax uncertainty risk and is distance-aware on feature space, thus reducing the over-confidence of the standard softmax under distribution shifts. Empirically, our method achieves competitive results with state-of-the-art techniques in terms of uncertainty and robustness, while having a lower number of model parameters and a lower latency at test-time.

</details>

## 38. DFD: Distillng the Feature Disparity Differently for Detectors

<details>

<summary>Abstract</summary>

Knowledge distillation is a widely adopted model compression technique that has been successfully applied to object detection. In feature distillation, it is common practice for the student model to imitate the feature responses of the teacher model, with the underlying objective of improving its own abilities by reducing the disparity with the teacher.However, it is crucial to recognize that the disparities between the student and teacher are inconsistent, highlighting their varying abilities.In this paper, we explore the inconsistency in the disparity between teacher and student feature maps and analyze their impact on the efficiency of the distillation.We find that regions with varying degrees of difference should be treated separately, with different distillation constraints applied accordingly. We introduce our distillation method called Disparity Feature Distillation(DFD). The core idea behind DFD is to apply different treatments to regions with varying learning difficulties, simultaneously incorporating leniency and strictness. It enables the student to better assimilate the teacher’s knowledge. Through extensive experiments, we demonstrate the effectiveness of our proposed DFD in achieving significant improvements.For instance, when applied to detectors based on ResNet50 such as RetinaNet, FasterRCNN, and RepPoints, our method enhances their performance from 37.4%, 38.4%, 38.6%to 41.7%, 42.4%, 42.7%, respectively. Our approach also demonstrates substantial improvements on YOLO and ViT-based models, and can be extended to segmentation and pose estimation. Our codes will be released after review.

</details>

## 39. DiffAug: Enhance Unsupervised Contrastive Learning with Domain-Knowledge-Free Diffusion-based Data Augmentation

<details>

<summary>Abstract</summary>

Unsupervised Contrastive learning has gained prominence in fields such as vision, and biology, leveraging predefined positive/negative samples for representation learning. Data augmentation, categorized into hand-designed and model-based methods, has been identified as a crucial component for enhancing contrastive learning. However, hand-designed methods require human expertise in domain-specific data while sometimes distorting the meaning of the data. In contrast, generative model-based approaches usually require supervised or large-scale external data, which has become a bottleneck constraining model training in many domains. To address the problems presented above, this paper proposes DiffAug, a novel unsupervised contrastive learning technique with diffusion mode-based positive data generation. DiffAug consists of a semantic encoder and a conditional diffusion model; the conditional diffusion model generates new positive samples conditioned on the semantic encoding to serve the training of unsupervised contrast learning. With the help of iterative training of the semantic encoder and diffusion model, DiffAug improves the representation ability in an uninterrupted and unsupervised manner. Experimental evaluations show that DiffAug outperforms hand-designed and SOTA model-based augmentation methods on DNA sequence, visual, and bio-feature datasets. The code for review is https://anonymous.4open.science/r/diffaug\_review\-804E.

</details>

## 40. Diffusion models encode the intrinsic dimension of data manifolds

<details>

<summary>Abstract</summary>

In this work, we provide a mathematical proof that diffusion models encode data manifolds by approximating their normal bundles. Based on this observation we propose a novel method for extracting the intrinsic dimension of the data manifold from a trained diffusion model. Our insights are based  on the fact that a diffusion model approximates the score function i.e. the gradient of the log density of a noise-corrupted version of the target distribution for varying levels of corruption. We prove that as the level of corruption decreases, the score function points towards the manifold, as this direction becomes the direction of maximal likelihood increase. Therefore, at low noise levels, the diffusion model provides us with an approximation of the manifold's normal bundle, allowing for an estimation of the manifold's intrinsic dimension.  To the best of our knowledge our method is the first estimator of intrinsic dimension based on diffusion models and it outperforms well established estimators in controlled experiments on both Euclidean and image data.

</details>

## 41. Directly Denoising Diffusion Models

<details>

<summary>Abstract</summary>

In this paper, we present Directly Denoising Diffusion Models (DDDMs): a simple and generic approach for generating realistic images with few-step sampling, while multistep sampling is still preserved for better performance. DDDMs require no delicately designed samplers nor distillation on pre-trained distillation models. DDDMs train the diffusion model conditioned on an estimated target that was generated from previous training iterations of its own. To generate images, samples generated from previous timestep are also taken into consideration, guiding the generation process iteratively. We further propose Pseudo-LPIPS, a novel metric loss that is more robust to various values of hyperparameter. Despite its simplicity, the proposed approach can achieve strong performance in benchmark datasets. Our model achieves FID scores of 2.57 and 2.33 on CIFAR-10 in one-step and two-step sampling respectively, surpassing those obtained from GANs and distillation-based models. By extending the sampling to 1000 steps, we further reduce FID score to 1.79, aligning with state-of-the-art methods in the literature. For ImageNet 64x64, our approach stands as a competitive contender against leading models.

</details>

## 42. Discounted Adaptive Online Prediction

<details>

<summary>Abstract</summary>

Online learning is not always about memorizing everything. Since the future can be statistically very different from the past, a critical challenge is to gracefully forget the history while new data comes in. To formalize this intuition, we revisit the classical notion of discounted regret using recently developed techniques in adaptive online learning. Our main result is a new algorithm that adapts to the complexity of both the loss sequence and the comparator, improving the widespread non-adaptive algorithm -- gradient descent with a constant learning rate. In particular, our theoretical guarantee does not require any structural assumption beyond convexity, and the algorithm is provably robust to suboptimal hyperparameter tuning. We further demonstrate such benefits through online conformal prediction, a downstream online learning task with set-membership decisions.

</details>

## 43. Disentangled Graph Self-supervised Learning under Distribution Shifts 🌟

<details>

<summary>Abstract</summary>

Graph out-of-distribution (OOD) generalization, aiming to generalize graph neural networks (GNNs) under distribution shifts between training and testing environments, has gained increasing significance recently. However, existing literature heavily relies on sufficient task-dependent graph labels, which are often scarce or even unavailable, limiting their applications in real-world scenarios. In this paper, we study self-supervised graph OOD generalization problem, \ie, learning GNNs capable of achieving relatively stable performances under distribution shifts without graph labels. However, the problem remains largely unexplored in literature, with the following critical challenge that the invariant and variant information are highly entangled in the graphs. To solve this problem, we propose an OOD generalized disentangled graph contrastive learning model (\modelnosp), which is capable of learning disentangled graph-level representations with self-supervision that can handle distribution shifts between training and testing graph data. Specifically, we first design a disentangled graph encoder to map each input graph into the factorized graph representation. Then we propose a tailored disentangled invariant self-supervised learning module tomaximize predictive ability of the representations and make sure the representations other than one specific channel are invariant to this latent factor for excluding the information to this latent factor for disentanglement. We provide comprehensive theoretical analyses to show that our model can learn disentangled graph representations and achieve OOD generalization. Extensive experiments on real-world datasets demonstrate the superiority of our model against state-of-the-art baselines under distribution shifts for graph classification tasks.

</details>

## 44. Mitigating Oversmoothing Through Reverse Process of GNNs for Heterophilic Graphs 🌟

<details>

<summary>Abstract</summary>

Graph Neural Network (GNN) resembles the diffusion process, leading to the over-smoothing of learned representations when stacking many layers. Hence, the reverse process of message passing can sharpen the node representations by inverting the forward message propagation. The sharpened representations can help us to better distinguish neighboring nodes with different labels, such as in heterophilic graphs.In this work, we apply the design principle of the reverse process to the three variants of the GNNs.Through the experiments on heterophilic graph data, where adjacent nodes need to have different representations for successful classification, we show that the reverse process significantly improves the prediction performance in many cases. Additional analysis reveals that the reverse mechanism can mitigate the over-smoothing over hundreds of layers.

</details>

## 45. Distributionally Robust Data Valuation

<details>

<summary>Abstract</summary>

Data valuation quantifies the contribution of each data point to the performance of a machine learning model. Existing works typically define the value of data by its improvement of the validation performance of the trained model. However, this approach can be impractical to apply in collaborative machine learning and data marketplace since it is difficult for the parties/buyers to agree on a common validation dataset or determine the exact validation distribution *a priori*. To address this, we propose a *distributionally robust data valuation* approach to perform data valuation without known/fixed validation distributions. Our approach defines the value of data by its improvement of the distributionally robust generalization error (DRGE), thus providing a worst-case performance guarantee *without* a known/fixed validation distribution. However, since computing DRGE directly is infeasible, we propose using *model deviation* as a proxy for the marginal improvement of DRGE (for kernel regression and neural networks) to compute data values. Furthermore, we identify a notion of uniqueness where low uniqueness characterizes low-value data. We empirically demonstrate that our approach outperforms existing data valuation approaches in data subset selection and data removal tasks on real-world datasets (e.g., housing price prediction, diabetes hospitalization prediction).

</details>

## 46. Domain Generalisation via Imprecise Learning 🌟

<details>

<summary>Abstract</summary>

Out-of-distribution (OOD) generalisation is challenging because it involves not only learning from empirical data, but also deciding among various notions of generalisation, e.g. optimise based on the average-case risk, worst-case risk, or interpolations thereof. While this decision should in principle be decided by the model operator like medical doctors in practice, this information might not always be available at training time. This situation leads to arbitrary commitments to specific generalisation strategies by machine learners due to these deployment uncertainties. We introduce the Imprecise Domain Generalisation framework to mitigate this, featuring an imprecise risk optimisation that allows learners to stay imprecise by optimising against a continuous spectrum of generalisation strategies during training, and a model framework that allows operators to specify their generalisation preference at deployment. Our work, supported by theoretical and empirical evidence, showcases the benefits of integrating imprecision into domain generalisation.

</details>

## 47. Don’t Label Twice: Quantity Beats Quality for Comparing Binary Classifiers on a Budget

<details>

<summary>Abstract</summary>

We study how to best spend a budget of noisy labels to compare the accuracy of two binary classifiers. It’s common practice to collect and aggregate multiple noisy labels for a given data point into a less noisy label via a majority vote. We prove a theorem that runs counter to conventional wisdom. If the goal is to identify the better of two classifiers, we show it’s best to spend the budget on collecting a single label for more samples. Our result follows from a non-trivial application of Cramér’s theorem, a staple in the theory of large deviations. We discuss the implications of our work for the design of machine learning benchmarks, where they overturn some time-honored recommendations. In addition, our results provide sample size bounds superior to what follows from Hoeffding’s bound.

</details>

## 48. Don't trust your eyes: on the (un)reliability of feature visualizations

<details>

<summary>Abstract</summary>

How do neural networks extract patterns from pixels? Feature visualizations attempt to answer this important question by visualizing highly activating patterns through optimization. Today, visualization methods form the foundation of our knowledge about the internal workings of neural networks, as a type of mechanistic interpretability. Here we ask: How reliable are feature visualizations? We start our investigation by developing network circuits that trick feature visualizations into showing arbitrary patterns that are completely disconnected from normal network behavior on natural input. We then provide evidence for a similar phenomenon occurring in standard, unmanipulated networks: feature visualizations are processed very differently from standard input, casting doubt on their ability to "explain" how neural networks process natural images. This can be used as a sanity check for feature visualizations. We underpin our empirical findings by theory proving that the set of functions that can be reliably understood by feature visualization is extremely small and does not include general black-box neural networks. Therefore, a promising way forward could be the development of networks that enforce certain structures in order to ensure more reliable feature visualizations.

</details>

## 49. Do Topological Characteristics Help in Knowledge Distillation?

<details>

<summary>Abstract</summary>

Knowledge distillation (KD) aims to transfer knowledge from larger (teacher) to smaller (student) networks. Previous studies focus on point-to-point or pairwise relationships in embedding features as knowledge and struggle to efficiently transfer relationships of complex latent spaces. To tackle this issue, we propose a novel KD method called TopKD, which considers the global topology of the latent spaces. We define global topology knowledge using the persistence diagram (PD) that captures comprehensive geometric structures such as shape of distribution, multiscale structure and connectivity, and the topology distillation loss for teaching this knowledge. To make the PD transferable within reasonable computational time, we employ approximated persistence images of PDs. Through experiments, we support the benefits of using global topology as knowledge and demonstrate the potential of TopKD.

</details>

## 50. DUPLEX: Dual GAT for Complex Embedding of Directed Graphs

<details>

<summary>Abstract</summary>

Current directed graph embedding methods build upon undirected techniques but often inadequately capture directed edge information, leading to challenges such as: (1) Suboptimal representations for nodes with low in/out-degrees, due to the insufficient neighbor interactions; (2) Limited inductive ability for representing new nodes post-training; (3) Narrow generalizability, as training is overly coupled with specific tasks. In response, we propose DUPLEX, an inductive framework for complex embeddings of directed graphs. It (1) leverages Hermitian adjacency matrix decomposition for comprehensive neighbor integration, (2) employs a dual GAT encoder for directional neighbor modeling, and (3) features two parameter-free decoders to decouple training from particular tasks. DUPLEX outperforms state-of-the-art models, especially for nodes with sparse connectivity, and demonstrates robust inductive capability and adaptability across various tasks. The code will be available upon publication.

</details>

## 51. Efficient Contrastive Learning for Fast and Accurate Inference on Graphs 🌟

<details>

<summary>Abstract</summary>

Graph contrastive learning has made remarkable advances in settings where there is a scarcity of task-specific labels. Despite these advances, the significant computational overhead for representation inference incurred by existing methods that rely on intensive message passing makes them unsuitable for latency-constrained applications.  To address this problem, we present GraphECL, a simple and efficient contrastive learning for fast inference on graphs. GraphECL does away with the need for expensive message passing during inference.  Specifically, it introduces a novel coupling of the MLP and GNN models, where the former learns to computationally efficiently mimic the computations performed by the latter. We provide a theoretical analysis showing why MLP can capture essential structural information in neighbors well enough to match the performance of GNN in downstream tasks. We present results of extensive experiments on widely used real-world benchmarks that show that GraphECL achieves superior performance and inference efficiency compared to state-of-the-art  graph constrastive learning (GCL) methods on homophilous and heterophilous graphs. On large-scale graphs, such as Snap-patents and Ogbn-papers100M, GraphECL is 200.00x faster than current methods.

</details>

## 52. EiG-Search: Generating Edge-Induced Subgraphs for GNN Explanation in Linear Time

<details>

<summary>Abstract</summary>

Understanding and explaining the predictions of Graph Neural Networks (GNNs), is crucial for enhancing their safety and trustworthiness. Subgraph-level explanations are gaining attention for their intuitive appeal. However, most existing subgraph-level explainers face efficiency challenges in explaining GNNs due to complex search processes. The key challenge is to find a balance between intuitiveness and efficiency while ensuring transparency. Additionally, these explainers usually induce subgraphs by nodes, which may introduce less-intuitive disconnected nodes in the subgraph-level explanations or omit many important subgraph structures. In this paper, we reveal that inducing subgraph explanations by edges is more comprehensive than other subgraph inducing techniques. We also emphasize the need of determining the subgraph explanation size for each data instance, as different data instances may involve different important substructures. Building upon these considerations, we introduce a training-free approach, named EiG-Search. We employ an efficient linear-time search algorithm over the edge-induced subgraphs, where the edges are ranked by an enhanced gradient-based importance. We conduct extensive experiments on a total of seven datasets, demonstrating its superior performance and efficiency both quantitatively and qualitatively over the leading baselines.

</details>

## 53. Emergent Equivariance in Deep Ensembles

<details>

<summary>Abstract</summary>

We demonstrate that deep ensembles are secretly equivariant models. More precisely, we show that deep ensembles become equivariant for all inputs and at all training times by simply using data augmentation. Crucially, equivariance holds off-manifold and for any architecture in the infinite width limit. The equivariance is emergent in the sense that predictions of individual ensemble members are not equivariant but their collective prediction is. Neural tangent kernel theory is used to derive this result and we verify our theoretical insights using detailed numerical experiments.

</details>

## 54. Empowering Graph Invariance Learning with Deep Spurious Infomax 🌟

<details>

<summary>Abstract</summary>

Recently, there has been a surge of interest in enabling graph neural networks to generalize to data from unseen environments. However, a significant challenge for these algorithms is the presuming assumptions on the correlation strengths between spurious features and class label, which can lead to potential failures when these assumptions do not hold in real-world scenarios. To bridge this gap, we introduce a novel learning paradigm for graph invariance learning, which induces a robust inductive bias without the reliance on presuming correlation strengths between spurious features and class labels. We further propose a flexible learning framework EQuAD to realize this learning paradigm and introduce a new learning objective tailored for EQuAD that provably elicits invariant representations. Notably, our approach shows stable and enhanced performance across different degrees of bias in synthetic datasets and outperforms state-of-the-art baseline methods by an average of **31.76%**. Additionally, EQuAD establishes new state-of-the-art benchmarks on multiple real-world datasets, demonstrating the effectiveness and robustness of our proposed framework.

</details>

## 55. Enhancing Class-Imbalanced Learning with Pre-trained Guidance through Class-Conditional Knowledge Distillation

<details>

<summary>Abstract</summary>

In class-imbalanced learning, the scarcity of information on minority classes presents challenges in obtaining generalizable features for these classes. Leveraging large-scale pre-trained models with powerful generalization capabilities as teacher models can be employed to fill the information gap. Knowledge distillation transfers the knowledge of the teacher model by learning the label distribution $p(\boldsymbol{y}|\boldsymbol{x})$ predicted by the teacher model. However, on imbalanced data, this method falls short in capturing the teacher model's knowledge about the class-conditional probability distribution $p(\boldsymbol{x}|\boldsymbol{y})$, which is crucial for enhancing generalization. Therefore, we propose Class-Conditional Knowledge Distillation (CCKD), directly learning the teacher model's class-conditional probability distribution by minimizing the KL divergence between the $p(\boldsymbol{x}|\boldsymbol{y})$ of the student and teacher model. we further present Augmented CCKD (ACCKD), which includes distillation on the constructed class-balanced data (formed through data mixing in training samples) and feature imitation on the entire dataset to further facilitate the learning of $p(\boldsymbol{x}|\boldsymbol{y})$. Results from experiments on various imbalanced datasets show an average accuracy enhancement of 7.5\% with the application of our method.

</details>

## 56. Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning

<details>

<summary>Abstract</summary>

Although most graph neural networks (GNNs) can operate on graphs of any size, their classification performance often declines on graphs larger than those encountered during training. Existing methods insufficiently address the removal of size information from graph representations, resulting in sub-optimal performance and reliance on backbone models. In response, we propose DISGEN, a novel and model-agnostic framework designed to disentangle size factors from graph representations. DISGEN employs an augmentation strategy and introduces a decoupling loss that minimizes shared information in hidden representations, with theoretical guarantees for its efficacy. Our empirical results show that DISGEN outperforms the state-of-the-art models by up to 7% on real-world datasets, underscoring its effectiveness in enhancing the size generalizability of GNNs.

</details>

## 57. EvoluNet: Advancing Dynamic Non-IID Transfer Learning on Graphs 🌟

<details>

<summary>Abstract</summary>

Non-IID transfer learning on graphs is crucial in many high-stakes domains. The majority of existing works assume stationary distribution for both source and target domains. However, real-world graphs are intrinsically dynamic, presenting challenges in terms of domain evolution and dynamic discrepancy between source and target domains. To bridge the gap, we shift the problem to the dynamic setting and pose the question: given the *label-rich* source graphs and the *label-scarce* target graphs both observed in previous $T$ timestamps, how can we effectively characterize the evolving domain discrepancy and optimize the generalization performance of the target domain at the incoming $T+1$ timestamp? To answer it, we propose a generalization bound for *dynamic non-IID transfer learning on graphs*, which implies the generalization performance is dominated by domain evolution and domain discrepancy between source and target graphs. Inspired by the theoretical results, we introduce a novel generic framework named EvoluNet. It leverages a transformer-based temporal encoding module to model temporal information of the evolving domains, and then uses a dynamic domain unification module to efficiently learn domain-invariant representations across the source and target domains. Finally, EvoluNet outperforms the state-of-the-art models by up to 12.1\%, demonstrating its effectiveness in transferring knowledge from dynamic source graphs to dynamic target graphs.

</details>

## 58. Explaining Graph Neural Networks via Structure-aware Interaction Index

<details>

<summary>Abstract</summary>

The Shapley value is a prominent tool for interpreting black-box machine learning models thanks to its strong theoretical foundation. However, for models with structured inputs, such as graph neural networks, existing Shapley-based explainability approaches either focus solely on node-wise importance or neglect the graph structure when perturbing the input instance. This paper introduces the Myerson-Taylor interaction index that internalizes the graph structure into attributing the node values and the interaction values among nodes. Unlike the Shapley-based methods, the Myerson-Taylor index decomposes coalitions into components satisfying a pre-chosen connectivity criterion. We prove that the Myerson-Taylor index is the unique one that satisfies a system of five natural axioms accounting for graph structure and high-order interaction among nodes. Leveraging these properties, we propose Myerson-Taylor Structure-Aware Graph Explainer (MAGE), a novel explainer that uses the second-order Myerson-Taylor index to identify the most important motifs influencing the model prediction, both positively and negatively. Extensive experiments on various graph datasets and models demonstrate that our method consistently provides superior subgraph explanations compared to state-of-the-art methods.

</details>

## 59. Exploring Correlations of Self-Supervised Tasks for Graphs 🌟

<details>

<summary>Abstract</summary>

Graph self-supervised learning has sparked a research surge in training informative representations without accessing any labeled data. However, our understanding of graph self-supervised learning remains limited, and the inherent relationships between various self-supervised tasks are still unexplored. Our paper aims to provide a fresh understanding of graph self-supervised learning based on task correlations. Specifically, we evaluate the performance of the representations trained by one specific task on other tasks and define correlation values to quantify task correlations. Through this process, we unveil the task correlations between various self-supervised tasks and can measure their expressive capabilities, which are closely related to downstream performance. By analyzing the correlation values between tasks across various datasets, we reveal the complexity of task correlations and the limitations of existing multi-task learning methods. To obtain more capable representations, we propose Graph Task Correlation Modeling (GraphTCM) to illustrate the task correlations and utilize it to enhance graph self-supervised training. The experimental results indicate that our method significantly outperforms existing methods across various downstream tasks.

</details>

## 60. Fast Algorithms for Hypergraph PageRank with Applications to Semi-Supervised Learning 🌟

<details>

<summary>Abstract</summary>

A fundamental approach to semi-supervised learning is to leverage the structure of the sample space to diffuse label information from annotated examples to unlabeled points. Traditional methods model the input data points as a graph and rely on fast algorithms for solving Laplacian systems of equations, such as those defining PageRank. However, previous work has demonstrated that graph-based models fail to capture higher-order relations, such as group membership, which are better modeled by hypergraphs. Unfortunately, the scalable application of hypergraph models has been hampered by the non-linearity of the hypergraph Laplacian. In this paper, we present highly scalable algorithms for hypergraph primitives, such as hypergraph PageRank vectors and hypergraph Laplacian systems, over general families of hypergraphs. In addition to giving strong theoretical guarantees, we empirically showcase the speed of our algorithms on benchmark instances of semi-supervised learning on categorical data. We exploit their generality to improve semi-supervised manifold clustering via hypergraph models. By providing significant speed-ups on fundamental hypergraph tasks, our algorithms enable the deployment of hypergraph  models on a massive scale.

</details>

## 61. Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective 🌟

<details>

<summary>Abstract</summary>

How would randomly shuffling feature vectors among nodes from the same class affect graph neural networks (GNNs)?The feature shuffle, intuitively, perturbs the dependence between graph topology and features (A-X dependence) for GNNs to learn from.Surprisingly, we observe a consistent and significant improvement in GNN performance following the feature shuffle.Having overlooked the impact of A-X dependence on GNNs, the prior literature does not provide a satisfactory understanding of the phenomenon. Thus, we raise two research questions. First, how should A-X dependence be measured, while controlling for potential confounds? Second, how does A-X dependence affect GNNs? In response, we (i) propose a principled measure for A-X dependence, (ii) design a random graph model that controls A-X dependence, (iii) establish a theory on how A-X dependence relates to graph convolution, and (iv) present empirical analysis on real-world graphs that aligns with the theory. We conclude that A-X dependence mediates the effect of graph convolution, such that smaller dependence improves GNN-based node classification.

</details>

## 62. From Biased Selective Labels to Pseudo-Labels: An Expectation-Maximization Framework for Learning from Biased Decisions

<details>

<summary>Abstract</summary>

Selective labels occur when label observations are subject to a decision-making process; e.g., diagnoses that depend on the administration of laboratory tests. We study a clinically-inspired selective label problem called disparate censorship, where labeling biases vary across subgroups, and unlabeled individuals are imputed as “negative” (i.e., no diagnostic test = no illness). Inspired by causal models of selective labels, we propose Disparate Censorship Expectation Maximization (DCEM). We theoretically analyze how DCEM mitigates disparate censorship. We validate DCEM on synthetic data, showing that it improves bias mitigation (area between ROC curves) without sacrificing discriminative performance (AUC) compared to baselines. We achieve similar results in a sepsis classification task using clinical data.

</details>

## 63. From Coarse to Fine: Enable Comprehensive Graph Self-supervised Learning with Multi-granular Semantic Ensemble 🌟

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) has gained increasing attention in the graph learning community, owing to its capability of enabling powerful models pre-trained on large unlabeled graphs for general purposes, facilitating quick adaptation to specific domains. Though promising, existing graph SSL frameworks often struggle to capture both high-level abstract features and fine-grained features simultaneously, leading to sub-optimal generalization abilities across different downstream tasks. To bridge this gap, we present Multi-granularity Graph Semantic Ensemble via Knowledge Distillation, namely MGSE, a plug-and-play graph knowledge distillation framework that can be applied to any existing graph SSL framework to enhance its performance by incorporating the concept of multi-granularity. Specifically, MGSE captures multi-granular knowledge by employing multiple student models to learn from a single teacher model, conditioned by probability distributions with different granularities. We apply it to six state-of-the-art graph SSL frameworks and evaluate their performances over multiple graph datasets across different domains, the experimental results show that MGSE can consistently boost the performance of these existing graph SSL frameworks with up to 9.2% improvement.

</details>

## 64. GATE: How to Keep Out Intrusive Neighbors 🌟

<details>

<summary>Abstract</summary>

Graph Attention Networks (GATs) are designed to provide flexible neighborhood aggregation that assigns weights to neighbors according to their importance. In practice, however, GATs are often unable to switch off task-irrelevant neighborhood aggregation, as we show experimentally and analytically. To address this challenge, we propose GATE, a GAT extension that holds three major advantages: i) It alleviates over-smoothing by addressing its root cause of unnecessary neighborhood aggregation. ii) Similarly to perceptrons, it benefits from higher depth as it can still utilize additional layers for (non-)linear feature transformations in case of (nearly) switched-off neighborhood aggregation. iii) By down-weighting connections to unrelated neighbors, it often outperforms GATs on real-world heterophilic datasets. To further validate our claims, we construct a synthetic test bed to analyze a model's ability to utilize the appropriate amount of neighborhood aggregation, which could be of independent interest.

</details>

## 65. Generating In-Distribution Proxy Graphs for Explainable Graph Neural Networks

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have become a building block in graph data processing, with wide applications in critical domains. The growing needs to deploy GNNs in high-stakes applications necessitate explainability for users in the decision-making processes. A popular paradigm for the explainability of GNNs is to identify explainable subgraphs by comparing their labels with the ones of original graphs. This task is challenging due to the substantial distributional shift from the original graphs in the training set to the set of explainable subgraphs, which prevents accurate prediction of labels with the subgraphs. To address it, in this paper, we propose a novel method that generates proxy graphs for explainable subgraphs that are in the distribution of training data. We introduce a parametric method that employs graph generators to produce proxy graphs. A new training objective based on information theory is designed to ensure that proxy graphs not only adhere to the distribution of training data but also preserve essential explanatory factors. Such generated proxy graphs can be reliably used for approximating the predictions of the true labels of explainable subgraphs. Empirical evaluations across various datasets demonstrate our method achieves more accurate explanations for GNNs.

</details>

## 66. GNNs Also Deserve Editing, and They Need It More Than Once 🌟

<details>

<summary>Abstract</summary>

Suppose a self-driving car is crashing into pedestrians, or a chatbot is instructing its users to conduct criminal wrongdoing; the stakeholders of such products will undoubtedly want to patch these catastrophic errors as soon as possible. To address such concerns, *Model Editing:* the study of efficiently patching model behaviors without significantly altering their general performance, has seen considerable activity, with hundreds of editing techniques developed in various domains such as CV and NLP. However, **the graph learning community has objectively fallen behind with only a few Graph Neural Network-compatible — and just one GNN-specific — model editing methods available**, where all of which are limited in their practical scope. We argue that the impracticality of these methods lies in their lack of *Sequential Editing Robustness:* the ability to edit multiple errors sequentially, and therefore fall short in effectiveness, as this approach mirrors how errors are addressed in the real world. In this paper, we delve into the specific reasons behind the difficulty of editing GNNs in succession and observe the root cause to be model overfitting. We subsequently propose a simple yet effective solution by leveraging overfit-prevention techniques in a GNN-specific context to derive the first — and only — GNN model editing method that scales practically. Additionally, we formally frame the task paradigm of GNN editing and hope to inspire future research in this crucial but currently overlooked field.

</details>

## 67. Graph Automorphism Group Equivariant Neural Networks

<details>

<summary>Abstract</summary>

Permutation equivariant neural networks are typically used to learn from data that lives on a graph. However, for any graph $G$ that has $n$ vertices, using the symmetric group $S_n$ as its group of symmetries does not take into account the relations that exist between the vertices. Given that the actual group of symmetries is the automorphism group Aut$(G)$, we show how to construct neural networks that are equivariant to Aut$(G)$ by obtaining a full characterisation of the learnable, linear, Aut$(G)$-equivariant functions between layers that are some tensor power of $\mathbb{R}^{n}$. In particular, we find a spanning set of matrices for these layer functions in the standard basis of $\mathbb{R}^{n}$. This result has important consequences for learning from data whose group of symmetries is a finite group because a theorem by Frucht (1938) showed that any finite group is isomorphic to the automorphism group of a graph.

</details>

## 68. Graph Distillation with Eigenbasis Matching

<details>

<summary>Abstract</summary>

The increasing amount of graph data places requirements on the efficient training of graph neural networks (GNNs). The emerging graph distillation (GD) tackles this challenge by distilling a small synthetic graph to replace the real large graph, ensuring GNNs trained on real and synthetic graphs exhibit comparable performance. However, existing methods rely on GNN-related information as supervision, including gradients, representations, and trajectories, which have two limitations. First, GNNs can affect the spectrum (i.e., eigenvalues) of the real graph, causing spectrum bias in the synthetic graph. Second, the variety of GNN architectures leads to the creation of different synthetic graphs, requiring traversal to obtain optimal performance. To tackle these issues, we propose Graph Distillation with Eigenbasis Matching (GDEM), which only aligns the eigenbasis and node features of real and synthetic graphs. Meanwhile, it directly replicates the spectrum of the real graph and thus prevents the influence of GNNs. Moreover, we design a discrimination constraint to balance the effectiveness and generalization of GDEM. Theoretically, the synthetic graphs distilled by GDEM are restricted spectral approximations of the real graphs. Extensive experiments demonstrate that GDEM outperforms state-of-the-art GD methods with powerful cross-architecture generalization ability and significant distillation efficiency.

</details>

## 69. Graph External Attention Enhanced Transformer 🌟

<details>

<summary>Abstract</summary>

The Transformer architecture has recently gained considerable attention in the field of graph representation learning, as it naturally overcomes several limitations of graph neural networks (GNNs) with customized attention mechanisms or positional and structural encodings. Despite making some progress, existing works tend to overlook external information of graphs, specifically the correlation between graphs. Intuitively, graphs with similar structures should have similar representations. Therefore, we propose Graph ExternalAttention (GEA) — a novel attention mechanism that leverages multiple external node/edge keyvalue units to capture inter-graph correlations implicitly. On this basis, we design an effective architecture called Graph External Attention Enhanced Transformer (GEAET), which integrates local structure and global interaction information for more comprehensive graph representations. Extensive experiments on benchmark datasets demonstrate that GEAET achieves state-of-theart empirical performance.

</details>

## 70. Graph Generation with Diffusion Mixture

<details>

<summary>Abstract</summary>

Generation of graphs is a major challenge for real-world tasks that require understanding the complex nature of their non-Euclidean structures. Although diffusion models have achieved notable success in graph generation recently, they are ill-suited for modeling the topological properties of graphs since learning to denoise the noisy samples does not explicitly learn the graph structures to be generated. To tackle this limitation, we propose a generative framework that models the topology of graphs by explicitly learning the final graph structures of the diffusion process. Specifically, we design the generative process as a mixture of endpoint-conditioned diffusion processes which is driven toward the predicted graph that results in rapid convergence. We further introduce a simple parameterization of the mixture process and develop an objective for learning the final graph structure, which enables maximum likelihood training. Through extensive experimental validation on general graph and 2D/3D molecule generation tasks, we show that our method outperforms previous generative models, generating graphs with correct topology with both continuous (e.g. 3D coordinates) and discrete (e.g. atom types) features.

</details>

## 71. Graph Geometry-Preserving Autoencoders

<details>

<summary>Abstract</summary>

When using an autoencoder to learn the low-dimensional manifold of high-dimensional data, it is crucial to find the latent representations that preserve the geometry of the data manifold. However, most existing studies assume a Euclidean nature for the high-dimensional data space, which is arbitrary and often does not precisely reflect the underlying semantic or domain-specific attributes of the data. In this paper we propose a novel autoencoder regularization framework based on the premise that the geometry of the data manifold can often be better captured with a well-designed similarity graph associated with data points. Given such a graph, we utilize a Riemannian geometric distortion measure as a regularizer to preserve the geometry derived from the graph Laplacian and make it suitable for larger-scale autoencoder training. Through extensive experiments compared to existing state-of-the-art geometry-preserving and graph-based autoencoders, we show that our method learns the most accurate graph geometry-preserving latent structures and is particularly effective in learning dynamics in the latent space.

</details>

## 72. Graph Neural Networks Use Graphs When They Shouldn't 🌟

<details>

<summary>Abstract</summary>

Predictions over graphs play a crucial role in various domains, including social networks and medicine.Graph Neural Networks (GNNs) have emerged as the dominant approach for learning on graph data.Although a graph-structure is provided as input to the GNN, in some cases the best solution can be obtained by ignoring it.While GNNs have the ability to ignore the graph-structure in such cases, it is not clear that they will.In this work, we show that GNNs actually tend to overfit the given graph-structure in the sense that they use it even when a better solution can be obtained by ignoring it.We analyze the implicit bias of gradient-descent learning of GNNs and prove that when the ground truth function does not use the graphs, GNNs are not guaranteed to learn a solution that ignores the graph, even with infinite data.We examine this phenomenon with respect to different graph distributions and find that regular graphs are more robust to this overfitting.  We also prove that within the family of regular graphs, GNNs are guaranteed to extrapolate when learning with gradient descent.Finally, based on our empirical and theoretical findings, we demonstrate on real-data how regular graphs can be leveraged to reduce graph overfitting and enhance performance.

</details>

## 73. Graph Positional and Structural Encoder 🌟

<details>

<summary>Abstract</summary>

Positional and structural encodings (PSE) enable better identifiability of nodes within a graph, as in general graphs lack a canonical node ordering. This renders PSEs essential tools for empowering modern GNNs, and in particular graph Transformers.However, designing PSEs that work optimally for all graph prediction tasks is a challenging and unsolved problem.Here, we present the graph positional and structural encoder (GPSE), the first-ever graph encoder designed to capture rich PSE representations for augmenting any GNN.GPSE learns an efficient common latent representation for multiple PSEs, and is highly transferable: The encoder trained on a particular graph dataset can be used effectively on datasets drawn from markedly different distributions and modalities. We show that across a wide range of benchmarks, GPSE-enhanced models can significantly outperform those that employ explicitly computed PSEs, and at least match their performance in others. Our results pave the way for the development of foundational pre-trained graph encoders for extracting positional and structural information, and highlight their potential as a more powerful and efficient alternative to explicitly computed PSEs and existing self-supervised pre-training approaches.

</details>

## 74. Graph Structure Extrapolation for Out-of-Distribution Generalization

<details>

<summary>Abstract</summary>

Out-of-distribution (OOD) generalization deals with the prevalent learning scenario where test distribution shifts from training distribution. With rising application demands and inherent complexity, graph OOD problems call for specialized solutions. While data-centric methods exhibit performance enhancements on many generic machine learning tasks, there is a notable absence of data augmentation methods tailored for graph OOD generalization. In this work, we propose to achieve graph OOD generalization with the novel design of non-Euclidean-space linear extrapolation. The proposed augmentation strategy extrapolates structure spaces to generate OOD graph data. Our design tailors OOD samples for specific shifts without corrupting underlying causal mechanisms.Theoretical analysis and empirical results evidence the effectiveness of our method in solving target shifts, showing substantial and constant improvements across various graph OOD tasks.

</details>

## 75. Grokking Happens All the Time and Here is Why

<details>

<summary>Abstract</summary>

Grokking or delayed generalization, is a phenomenon where generalization in a Deep Neural Network (DNN) occurs long after achieving near zero training error. Previous studies have reported the occurrence of grokking in controlled settings, e.g., for transformers trained on algorithmic datasets (power2022grokking), or for DNNs initialized with large-norm parameters (liu2022omnigrok). We instead observe that for a large number of standard and practical settings, e.g., while training a CNN on CIFAR10, or a Resnet on Imagenette, DNNs grok adversarial examples, i.e., adversarial robustness emerges long after interpolation and/or generalization.We present a theoretically motivated explanation behind the emergence of delayed generalization and delayed robustness. We find that both phenomenon are tied, originating from a phase transition in the DNN's input space partition geometry during training. We provide the first evidence that a migration of DNN 'linear regions' occurs, making the function progresively linear around training samples and non-linear around the decision boundary during the latest phase of training. This migration provably induces grokking, as the emergence of a robust partition widens the linear regions around the training samples.

</details>

## 76. How Graph Neural Networks Learn: Lessons from Training Dynamics

<details>

<summary>Abstract</summary>

A long-standing goal in deep learning has been to characterize the learning behavior of black-box models in a more interpretable manner. For graph neural networks (GNNs), considerable advances have been made in formalizing what functions they can represent, but whether GNNs will learn desired functions during the optimization process remains less clear. To fill this gap, we study their training dynamics in function space. In particular, we find that the optimization of GNNs through gradient descent implicitly leverages the graph structure to update the learned function. This phenomenon is dubbed as kernel-graph alignment, which has been empirically and theoretically corroborated. This new analytical framework from the optimization perspective enables interpretable explanations of when and why the learned GNN functions generalize, which are relevant to their limitations on heterophilic graphs. From a practical standpoint, it also provides high-level principles for designing new algorithms. We exemplify this by showing that a simple and efficient non-parametric algorithm, obtained by explicitly using graph structure to update the learned function, can consistently compete with nonlinear GNNs.

</details>

## 77. How Interpretable Are Interpretable Graph Neural Networks?

<details>

<summary>Abstract</summary>

Interpretable graph neural networks (XGNNs) are widely adopted in various scientific applications involving graph-structured data. Existing XGNNs predominantly adopt the attention-based mechanism to learn edge or node importance for extracting and making predictions with the interpretable subgraph. However, the representational properties and limitations of these methods remain inadequately explored. In this work, we present a theoretical framework that formulates interpretable subgraph learning with the multilinear extension of the subgraph distribution, which we term as subgraph multilinear extension (SubMT). Extracting the desired interpretable subgraph requires an accurate approximation of SubMT, yet we find that the existing XGNNs can have a huge gap in fitting SubMT. Consequently, the SubMT approximation failure will lead to the degenerated interpretability of the extracted subgraphs. To mitigate the issue, we design a new XGNN architecture called Graph Multilinear neT (GMT), which is provably more powerful in approximating SubMT. We empirically validate our theoretical findings on a number of graph classification benchmarks. The results demonstrate that GMT outperforms the state-of-the-art up to 10% in terms of both interpretability and generalizability across 12 regular and geometric graph benchmarks.

</details>

## 78. How Universal Polynomial Bases Enhance Spectral Graph Neural Networks: Heterophily, Over-smoothing, and Over-squashing 🌟

<details>

<summary>Abstract</summary>

Spectral Graph Neural Networks (GNNs), alternatively known as *graph filters*, have gained increasing prevalence for heterophily graphs. Optimal graph filters rely on Laplacian eigendecomposition for Fourier transform. In an attempt to avert prohibitive computations, numerous polynomial filters have been proposed. However, polynomials in the majority of these filters are *predefined* and remain *fixed* across different graphs, failing to accommodate the varying degrees of heterophily. Addressing this gap, we demystify the intrinsic correlation between the spectral property of desired polynomial bases and the heterophily degrees via thorough theoretical analyses. Subsequently, we develop a novel adaptive heterophily basis wherein the basis vectors mutually form angles reflecting the heterophily degree of the graph. We integrate this heterophily basis with the homophily basis to construct a universal polynomial basis *UniBasis*, which devises a polynomial filter based graph neural network – *UniFilter*. It optimizes the convolution and propagation in GNN, thus effectively limiting over-smoothing and alleviating over-squashing. Our extensive experiments, conducted on a diverse range of real-world and synthetic datasets with varying degrees of heterophily, support the superiority of UniFilter. These results not only demonstrate the universality of UniBasis but also highlight its proficiency in graph explanation.

</details>

## 79. Hypergraph-enhanced Dual Semi-supervised Graph Classification 🌟

<details>

<summary>Abstract</summary>

In this paper, we study semi-supervised graph classification, which aims at accurately predicting the categories of graphs in scenarios with limited labeled graphs and abundant unlabeled graphs. Despite the promising capability of graph neural networks (GNNs), they typically require a large number of costly labeled graphs, while a wealth of unlabeled graphs fail to be effectively utilized. Moreover, GNNs are inherently limited to encoding local neighborhood information using message-passing mechanisms, thus lacking the ability to model higher-order dependencies among nodes. To tackle these challenges, we propose a Hypergraph-Enhanced DuAL framework named HEAL for semi-supervised graph classification, which captures graph semantics from the perspective of the hypergraph and the line graph, respectively. Specifically, to better explore the higher-order relationships among nodes, we design a hypergraph structure learning to adaptively learn complex node dependencies beyond pairwise relations. Meanwhile, based on the learned hypergraph, we introduce a line graph to capture the interaction between hyperedges, thereby better mining the underlying semantic structures. Finally, we develop a relational consistency learning to facilitate knowledge transfer between the two branches and provide better mutual guidance. Extensive experiments on real-world graph datasets verify the effectiveness of the proposed method against existing state-of-the-art methods.

</details>

## 80. Improving Equivariant Graph Neural Networks on Large Geometric Graphs via Virtual Nodes Learning 🌟

<details>

<summary>Abstract</summary>

Equivariant Graph Neural Networks (GNNs)  have made remarkable success in a variety of scientific applications. However, existing equivariant GNNs encounter the efficiency issue for large geometric graphs and perform poorly if the input is reduced to sparse local graph for speed acceleration. In this paper, we propose FastEGNN, an enhanced model of equivariant GNNs on large geometric graphs. The central idea is leveraging a small ordered set of virtual nodes to approximate the large unordered graph of real nodes. In particular, we distinguish the message passing and aggregation for different virtual node to encourage the mutual distinctiveness, and minimize the Maximum Mean Discrepancy (MMD) between virtual and real coordinates to realize the global distributedness. FastEGNN meets all necessary E(3) symmetries, with certain universal expressivity assurance as well. Our experiments on N-body systems (100 nodes), proteins (800 nodes) and water-3D (8000 nodes), demonstrate that FastEGNN achieves a promising balance between accuracy and efficiency, and outperforms EGNN in accuracy even after dropping all edges in real systems like proteins and water-3D.

</details>

## 81. Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization

<details>

<summary>Abstract</summary>

We study the problem of training an unbiased and accurate model given a dataset with multiple biases. This problem is challenging since the multiple biases cause multiple undesirable shortcuts during training, and even worse, mitigating one may exacerbate the other. We propose a novel training method to tackle this challenge. Our method first groups training data so that different groups induce different shortcuts, and then optimizes a linear combination of group-wise losses while adjusting their weights dynamically to alleviate conflicts between the groups in performance; this approach, rooted in the multi-objective optimization theory, enables to achieve a Pareto-stationary solution. We also present a new benchmark with multiple biases, dubbed MultiCelebA, for evaluating debiased training methods under realistic and challenging scenarios. Our method achieved the best on three datasets with multiple biases, and also showed superior performance on conventional single-bias datasets.

</details>

## 82. Information Flow in Self-Supervised Learning

<details>

<summary>Abstract</summary>

In this paper, we conduct a comprehensive analysis of two dual-branch (Siamese architecture) self-supervised learning approaches, namely Barlow Twins and spectral contrastive learning, through the lens of matrix mutual information. We prove that the loss functions of these methods implicitly optimize both matrix mutual information and matrix joint entropy. This insight prompts us to further explore the category of single-branch algorithms, specifically MAE and U-MAE,  for which mutual information and joint entropy become the entropy. Building on this intuition, we introduce the Matrix Variational Masked Auto-Encoder (M-MAE), a novel method that leverages the matrix-based estimation of entropy as a regularizer and subsumes U-MAE as a special case. The empirical evaluations underscore the effectiveness of M-MAE compared with the state-of-the-art methods, including a 3.9% improvement in linear probing ViT-Base, and a 1% improvement in fine-tuning ViT-Large, both on ImageNet.

</details>

## 83. InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning 🌟

<details>

<summary>Abstract</summary>

Semi-supervised learning (SSL) seeks to enhance task performance by training on both labeled and unlabeled data. Mainstream SSL image classification methods mostly optimize a loss that additively combines a supervised classification objective with a regularization term derived solely from unlabeled data. This formulation neglects the potential for interaction between labeled and unlabeled images. In this paper, we introduce InterLUDE, a new approach to enhance SSL made of two parts that each benefit from labeled-unlabeled interaction. The first part, embedding fusion, interpolates between labeled and unlabeled embeddings to improve representation learning.The second part is a new loss, grounded in the principle of consistency regularization, that aims to minimize discrepancies in the model's predictions between labeled versus unlabeled inputs. Experiments on standard closed-set SSL benchmarks and a medical SSL task with an uncurated unlabeled set show clear benefits to our approach. On the STL-10 dataset with only 40 labels, InterLUDE achieves 3.2% error rate, while the best previous method reports 14.9%.

</details>

## 84. Knowledge Distillation with Auxiliary Variable 🌟

<details>

<summary>Abstract</summary>

Knowledge distillation (KD) provides an efficient framework for transferring knowledge from a teacher model to a student model by aligning their predictive distributions. The existing KD methods adopt the same strategy as the teacher to formulate the student's predictive distribution. However, employing the same distribution-modeling strategy typically causes sub-optimal knowledge transfer due to the discrepancy in model capacity between teacher and student models. Designing student-friendly teachers contributes to alleviating the capacity discrepancy, while it requires either complicated or student-specific training schemes. To cast off this dilemma, we propose to introduce an auxiliary variable to promote the ability of the student to model predictive distribution. The auxiliary variable is defined to be related to target variables, which will boost the model prediction. Specifically, we reformulate the predictive distribution with the auxiliary variable, deriving a novel objective function of KD. Theoretically, we provide insights to explain why the proposed objective function can outperform the existing KD methods. Experimentally, we demonstrate that the proposed objective function can considerably and consistently outperform existing KD methods.

</details>

## 85. Learning-Efficient Yet Generalizable Collaborative Filtering for Item Recommendation

<details>

<summary>Abstract</summary>

The implicit Alternating Least Squares algorithm (iALS) is widely recognized as an efficient approach for recommender systems, consistently delivering competitive performance when compared to recent approaches. However, a notable challenge arises from the fact that iALS utilizes a quadratic regression loss function, which lacks a clear connection to the ranking objective, such as DCG. This discrepancy poses a fundamental difficulty in explaining the algorithm's exceptional ranking performance.In this work, we make a breakthrough by establishing a connection between quadratic regression loss and ranking metrics through a Taylor expansion of the DCG-consistent surrogate loss —— softmax. We also remarkably discovered a new surrogate quadratic loss function and conducted thorough theoretical analyses, specifically focusing on the DCG-consistency and generalization properties of this newly proposed loss function. These analyses provide solid theoretical foundations and enhance the reliability and applicability of our approach.Moreover, we generalize the original ALS method to incorporate our novel loss function, resulting in a more efficient and effective ranking algorithm. The experimental results over three public datasets demonstrate the effectiveness of the proposed method, i.e., GALS. The results showcased comparable ranking performance to softmax while achieving faster convergence due to the optimization with closed-form solutions.This significant advancement presents a practical alternative to the widely used softmax function, representing a substantial leap forward in our understanding of objective functions in recommendation systems.

</details>

## 86. Learning Graph Representation via Graph Entropy Maximization 🌟

<details>

<summary>Abstract</summary>

Graph representation learning aims to represent graphs as vectors that can be utilized in downstream tasks such as graph classification. In this work, we focus on learning diverse representations that can capture the graph information as much as possible. We propose quantifying graph information using graph entropy, where we define a probability distribution of a graph based on its nodes' representations and global-graph representation. However, the computation of graph entropy is NP-hard due to the complex vertex-packing polytope involved in its definition.  To address this challenge, we provide an approximation method leveraging orthonormal representations for graph entropy maximization.The proposed method is implemented via graph neural networks, resulting in informative node-level and graph-level representations.Experimental results demonstrate the effectiveness of our method in comparison to many baselines in unsupervised learning and semi-supervised learning tasks.

</details>

## 87. Learning High-Order Relationships of Brain Regions

<details>

<summary>Abstract</summary>

Discovering reliable and informative relationships among brain regions from functional magnetic resonance imaging (fMRI) signals is essential in phenotypic predictions in neuroscience. Most of the current methods fail to accurately characterize those interactions because they only focus on pairwise connections and overlook the high-order relationships of brain regions. We propose that these high-order relationships should be *maximally informative and minimally redundant* (MIMR). However, identifying such high-order relationships is challenging and under-explored due to the exponential search space and the absence of a tractable objective. In response to this gap, we propose a novel method named HyBRiD, which aims to extract MIMR high-order relationships from fMRI data. HyBRiD employs a Constructor to identify hyperedge structures, and a Weighter to compute a weight for each hyperedge, which avoids searching in exponential space. HyBRiD achieves the MIMR objective through an innovative information bottleneck framework named multi-head drop-bottleneck with theoretical guarantees. Our comprehensive experiments demonstrate the effectiveness of our model. Our model outperforms the state-of-the-art predictive model by an average of 11.2%, regarding the quality of hyperedges measured by CPM, a standard protocol for studying brain connections.

</details>

## 88. Learning in Deep Factor Graphs with Gaussian Belief Propagation

<details>

<summary>Abstract</summary>

We propose an approach to do learning in Gaussian factor graphs. We treat all relevant quantities (inputs, outputs, parameters, activations) as random variables in a graphical model, and view training and prediction as inference problems with different observed nodes. Our experiments show that these problems can be efficiently solved with belief propagation (BP), whose updates are inherently local, presenting exciting opportunities for distributed and asynchronous training. Our approach can be scaled to deep networks and provides a natural means to do continual learning: use the BP-estimated posterior of the current task as a prior for the next. On a video denoising task we demonstrate the benefit of learnable parameters over a classical factor graph approach and we show encouraging performance of deep factor graphs for continual image classification.

</details>

## 89. Learning Latent Space Hierarchical EBM Diffusion Models

<details>

<summary>Abstract</summary>

This work studies the learning problem of the energy-based prior model and the multi-layer generator model. The multi-layer generator model, which contains multiple layers of latent variables organized in a top-down hierarchical structure, typically assumes the Gaussian prior model. Such a prior model can be limited in modelling expressivity, which results in a gap between the generator posterior and the prior model, known as the prior hole problem. Recent works have explored learning the energy-based (EBM) prior model as a second-stage, complementary model to bridge the gap. However, the EBM defined on a multi-layer latent space can be highly multi-modal, which makes sampling from such marginal EBM prior challenging in practice, resulting in ineffectively learned EBM. To tackle the challenge, we propose to leverage the diffusion probabilistic scheme to mitigate the burden of EBM sampling and thus facilitate EBM learning. Our extensive experiments demonstrate a superior performance of our diffusion-learned EBM prior on various challenging tasks.

</details>

## 90. Less is More: on the Over-Globalizing Problem in Graph Transformers 🌟

<details>

<summary>Abstract</summary>

Graph Transformer, due to its global attention mechanism, has emerged as a new tool in dealing with graph-structured data. It is well recognized that the global attention mechanism considers a wider receptive field in a fully connected graph, leading many to believe that useful information can be extracted from all the nodes. In this paper, we challenge this belief: does the globalizing property always benefit Graph Transformers? We reveal the over-globalizing problem in Graph Transformer by presenting both empirical evidence and theoretical analysis, i.e., the current attention mechanism overly focuses on those distant nodes, while the near nodes, which actually contain most of the useful information, are relatively weakened. Then we propose a novel Bi-Level Global Graph Transformer with Collaborative Training (CoBFormer), including the inter-cluster and intra-cluster Transformers, to prevent the over-globalizing problem while keeping the ability to extract valuable information from distant nodes. Moreover, the collaborative training is proposed to improve the model's generalization ability with a theoretical guarantee. Extensive experiments on various graphs well validate the effectiveness of our proposed CoBFormer.

</details>

## 91. LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering 🌟

<details>

<summary>Abstract</summary>

Graph clustering is a fundamental problem in machine learning. Deep learning methods achieve the state-of-the-art results in recent years, but they still cannot work without predefined cluster numbers. Such limitation motivates us to pose a more challenging problem of graph clustering with unknown cluster number. We propose to address this problem from a fresh perspective of graph information theory, (i.e., structural information). In the literature, structural information has not yet been introduced to deep clustering, and its classic definition falls short of discrete formulation and modeling node features. In this work, we first formulate a differentiable structural information (DSI) in the continuous realm, accompanied by several theoretical results. By minimizing DSI, we construct the optimal partitioning tree, where densely connected nodes in the graph tend to have the same assignment, revealing the cluster structure. DSI is also theoretically presented as a new graph clustering objective, not requiring the predefined cluster number. Furthermore, we design a neural LSEnet in the Lorentz model of hyperbolic space, where we integrate node features to structural information via manifold-valued graph convolution. Extensive empirical results on real graphs show the superiority of our approach.

</details>

## 92. MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models

<details>

<summary>Abstract</summary>

Multi-agent interactions between Large Language Model (LLM) agents have shown major improvements on diverse reasoning tasks. However, these involve long generations from multiple models across several rounds, making them expensive. Moreover, these multi-agent approaches fail to provide a final, single model for efficient inference. To address this, we introduce MAGDi, a new method for structured distillation of the reasoning interactions between multiple LLMs into smaller LMs. MAGDi teaches smaller models by representing multi-agent interactions using graphs, augmenting a base student model, and distilling knowledge using three objective functions: next-token prediction, a contrastive loss between correct and incorrect reasoning, and a graph-based objective to model the interaction structure. Experiments on seven widely-used commonsense and math reasoning benchmarks show that MAGDi improves the reasoning capabilities of smaller models, outperforming several methods that distill from a single teacher and multiple teachers. We conduct extensive analyses to show that MAGDi (1) scales positively with better base student models, (2) enhances the generalizability to out-of-domain tasks, and (3) obtains larger improvements when applying the inference technique of self-consistency, which relies on model diversity.

</details>

## 93. Matrix Information Theory for Self-Supervised Learning

<details>

<summary>Abstract</summary>

The maximum entropy encoding framework provides a unified perspective for many non-contrastive learning methods like SimSiam, Barlow Twins, and MEC. Inspired by this framework, we introduce Matrix-SSL, a novel approach that leverages matrix information theory to interpret the maximum entropy encoding framework as a matrix uniformity loss. Furthermore, Matrix-SSL enhances the maximum entropy encoding by seamlessly incorporating matrix alignment loss, directly aligning covariance matrices in different views.Experimental results reveal that Matrix-SSL outperforms state-of-the-art methods on the ImageNet dataset under linear evaluation settings and on MS-COCO for transfer learning tasks. Specifically, when performing transfer learning tasks on MS-COCO, our method outperforms previous SOTA methods such as MoCo v2 and BYOL up to 3.3\% with only 400 epochs compared to 800 epochs pre-training. We also try to introduce representation learning into the language modeling regime, achieving 72.3\% on the GSM8K dataset by fine-tuning a 7B model using matrix cross-entropy loss, with a margin of 3.1\% over the standard cross-entropy loss.

</details>

## 94. Mitigating Label Noise on Graphs via Topological Sample Selection

<details>

<summary>Abstract</summary>

Despite the success of the carefully-annotated benchmarks, the effectiveness of existing graph neural networks (GNNs) can be considerably impaired in practice when the real-world graph data is noisily labeled.  Previous explorations in sample selection have been demonstrated as an effective way for robust learning with noisy labels, however, the conventional studies focus on i.i.d data, and when moving to non-iid graph data and GNNs, two notable challenges remain: (1) nodes located near topological class boundaries are very informative for classification but cannot be successfully distinguished by the heuristic sample selection. (2) there is no available measure that considers the graph topological information to promote sample selection in a graph. To address this dilemma, we propose a $\textit{Topological Sample Selection}$ (TSS) method that boosts the informative sample selection process in a graph by utilising topological information. We theoretically prove that our procedure minimizes an upper bound of the expected risk under target clean distribution, and experimentally show the superiority of our method compared with state-of-the-art baselines.

</details>

## 95. Multi-group Learning for Hierarchical Groups

<details>

<summary>Abstract</summary>

The multi-group learning model formalizes the learning scenario in which a single predictor must generalize well on multiple, possibly overlapping subgroups of interest. We extend the study of multi-group learning to the natural case where the groups are hierarchically structured. We design an algorithm for this setting that outputs an interpretable and deterministic decision tree predictor with near-optimal sample complexity. We then conduct an empirical evaluation of our algorithm and find that it achieves attractive generalization properties on real datasets with hierarchical group structure.

</details>

## 96. Multigroup Robustness

<details>

<summary>Abstract</summary>

To address the shortcomings of real-world datasets, robust learning algorithms have been designed to overcome arbitrary and indiscriminate data corruption. However, practical processes of gathering data may lead to patterns of data corruption that are localized to specific partitions of the training dataset. Motivated by critical applications where the learned model is deployed to make predictions about people from a rich collection of overlapping subpopulations, we initiate the study of \emph{multigroup robust} algorithms whose robustness guarantees for each subpopulation only degrade with the amount of data corruption \emph{inside} that subpopulation. When the data corruption is not distributed uniformly over subpopulations, our algorithms provide more meaningful robustness guarantees than standard guarantees that are oblivious to how the data corruption and the affected subpopulations are related. Our techniques establish a new connection between multigroup fairness and robustness.

</details>

## 97. Multi-Track Message Passing: Tackling Oversmoothing and Oversquashing in Graph Learning via Preventing Heterophily Mixing 🌟

<details>

<summary>Abstract</summary>

The advancement toward deeper graph neural networks is currently obscured by two inherent issues in message passing, *oversmoothing*  and *oversquashing*. We identify the root cause of these issues as information loss due to *heterophily mixing* in aggregation, where messages of diverse category semantics are mixed. We propose a novel multi-track graph convolutional network to address oversmoothing and oversquashing effectively. Our basic idea is intuitive: if messages are separated and independently propagated according to their category semantics, heterophilic mixing can be prevented. Consequently, we present a novel multi-track message passing scheme capable of preventing heterophilic mixing, enhancing long-distance information flow, and improving separation condition. Empirical validations show that our model achieved state-of-the-art performance on several graph datasets and effectively tackled oversmoothing and oversquashing, setting a new benchmark of $86.4$ accuracy on Cora.

</details>

## 98. Multi-View Stochastic Block Models 🌟

<details>

<summary>Abstract</summary>

Graph clustering is a central topic in unsupervised learning with a multitude of practical applications. In recent years, multi-view graph clustering has gained a lot of attention for its applicability to real-world instances where one often has access to multiple data sources. In this paper we formalize a new family of models, called \textit{multi-view stochastic block models} that capture this setting.For this model, we first study efficient algorithms that naively work on the union of multiple graphs. Then, we introduce a new efficient algorithm that provably outperforms previous approaches by analyzing the structure of each graph separately.Finally, we complement our results with an information-theoretic lower bound studying the limits of what can be done in this model.

</details>

## 99. Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching

<details>

<summary>Abstract</summary>

Graph condensation aims to reduce the size of a large-scale graph dataset by synthesizing a compact counterpart without sacrificing the performance of Graph Neural Networks (GNNs) trained on it, which has shed light on reducing the computational cost for training GNNs. Nevertheless, existing methods often fall short of accurately replicating the original graph for certain datasets, thereby failing to achieve the objective of lossless condensation. To understand this phenomenon, we investigate the potential reasons and reveal that the previous state-of-the-art trajectory matching method provides biased and restricted supervision signals from the original graph when optimizing the condensed one. This constraint significantly limits both the scale and efficacy of the condensed graph. In this paper, we make the first attempt toward \textit{lossless graph condensation} by bridging the previously neglected supervision signals. Specifically, we employ a curriculum learning strategy to train expert trajectories with more diverse supervision signals from the original graph, and then effectively transfer the information into the condensed graph with expanding window matching. Moreover, we design a loss function to further extract knowledge from the expert trajectories with a new perspective. Theoretical analysis justifies the design of our method and extensive experiments verify its superiority across different datasets.

</details>

## 100. Networked Inequality: Preferential Attachment Bias in Graph Neural Network Link Prediction 🌟

<details>

<summary>Abstract</summary>

Graph neural network (GNN) link prediction is increasingly deployed in citation, collaboration, and online social networks to recommend academic literature, collaborators, and friends. While prior research has investigated the dyadic fairness of GNN link prediction, the within-group (e.g., queer women) fairness and ``rich get richer'' dynamics of link prediction remain underexplored. However, these aspects have significant consequences for degree and power imbalances in networks. In this paper, we shed light on how degree bias in networks affects Graph Convolutional Network (GCN) link prediction. In particular, we theoretically uncover that GCNs with a symmetric normalized graph filter have a within-group preferential attachment bias. We validate our theoretical analysis on real-world citation, collaboration, and online social networks. We further bridge GCN's preferential attachment bias with unfairness in link prediction and propose a new within-group fairness metric. This metric quantifies disparities in link prediction scores within social groups, towards combating the amplification of degree and power disparities. Finally, we propose a simple training-time strategy to alleviate within-group unfairness, and we show that it is effective on citation, social, and credit networks.

</details>

## 101. Not all distributional shifts are equal: Fine-grained robust conformal inference

<details>

<summary>Abstract</summary>

We introduce a fine-grained framework for uncertain quantification of predictive models in the presence of distributional shifts. This framework distinguishes between the shift in covariate distributions and that of the conditional relationship between the outcome ($Y$) and the covariate ($X$), prescribing a corresponding treatment for each type. Since the covariate shift is often identifiable but the conditional distributional shift is not, we propose to reweight the training samples according to the covariate shift while defending against the worst-case conditional distribution shift bounded in an $f$-divergence ball. Based on ideas from conformal inference and distributionally robust learning, we present an algorithm that outputs (approximately) valid and efficient prediction sets in the presence of distributional shifts. As a special use case, we show that the framework can be applied to sensitivity analysis of individual treatment effects under hidden confounding. The proposed methods are evaluated in the simulation studies and real data applications, demonstrating superior robustness and efficiency compared with existing benchmarks.

</details>

## 102. On the Effectiveness of Supervision in Non-Contrastive Representation Learning

<details>

<summary>Abstract</summary>

Supervised contrastive representation learning has shown to be effective in many transfer learning scenarios.However, while non-contrastive learning often outperforms its contrastive learning counterpart in self-supervised representation learning, the extension of non-contrastive representation learning to supervised scenarios is less explored.To bridge the gap, we study non-contrastive learning for supervised representation learning, coined SupBYOL and SupSiam, which leverages labels in non-contrastive learning to achieve better representations.The proposed supervised non-contrastive learning framework improves representation learning while avoiding collapse.Our theoretical analysis reveals that providing supervision to non-contrastive learning reduces intra-class variance, and the contribution of supervision should be adjusted to achieve the best performance.In experiments, we show the superiority of supervised non-contrastive learning across various datasets and tasks.The code will be released.

</details>

## 103. On the Embedding Collapse when Scaling up Recommendation Models

<details>

<summary>Abstract</summary>

Recent advances in foundation models have led to a promising trend of developing large recommendation models to leverage vast amounts of available data. Still, mainstream models remain embarrassingly small in size and naive enlarging does not lead to sufficient performance gain, suggesting a deficiency in the model scalability. In this paper, we identify the embedding collapse phenomenon as the inhibition of scalability, wherein the embedding matrix tends to occupy a low-dimensional subspace. Through empirical and theoretical analysis, we demonstrate a two-sided effect of feature interaction specific to recommendation models. On the one hand, interacting with collapsed embeddings restricts embedding learning and exacerbates the collapse issue. On the other hand, interaction is crucial in mitigating the fitting of spurious features as a scalability guarantee. Based on our analysis, we propose a simple yet effective multi-embedding design incorporating embedding-set-specific interaction modules to learn embedding sets with large diversity and thus reduce collapse. Extensive experiments demonstrate that this proposed design provides consistent scalability and effective collapse mitigation for various recommendation models.

</details>

## 104. On the Expressive Power of Spectral Invariant Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Incorporating spectral information to enhance Graph Neural Networks (GNNs) has shown promising results but raises a fundamental challenge due to the inherent ambiguity of eigenvectors. Various architectures have been proposed to address this ambiguity, referred to as spectral invariant architectures. Notable examples include GNNs and Graph Transformers that use spectral distances, spectral projection matrices, or other invariant spectral features. However, the potential expressive power of these spectral invariant architectures remains largely unclear. The goal of this work is to gain a deep theoretical understanding of the expressive power obtainable when using spectral features. We first introduce a novel message-passing framework for designing spectral invariant GNNs, called Eigenspace Projection GNN (EPNN). Our comprehensive analysis shows that EPNN essentially unifies all prior spectral invariant architectures, in that they are either strictly less expressive or equivalent to EPNN. A fine-grained expressiveness hierarchy among different architectures is also established. On the other hand, we present a surprising result that EPNN itself is bounded by a recently proposed class of Subgraph GNNs, implying that all these spectral invariant architectures are strictly less expressive than 3-WL. Finally, we demonstrate that these spectral features offer no additional advantage when combined with more expressive GNNs.

</details>

## 105. On the Role of Edge Dependency in Graph Generative Models 🌟

<details>

<summary>Abstract</summary>

We investigate the trade-off between the representation power of graph generative models and model *overlap*, i.e., the degree to which the model generates diverse outputs versus regurgitating its training data. In particular, we delineate a nested hierarchy of graph generative models categorized into three levels of complexity: edge independent, node independent, and arbitrarily dependent models. This hierarchy encapsulates a wide range of prevalent methods. We derive theoretical bounds on the number of triangles and other short-length cycles producible by each level of the hierarchy, finding that more complex dependency structure allows an improved trade-off between representation power and overlap. We provide instances demonstrating the asymptotic optimality of our bounds. Furthermore, we introduce new generative models for each of the three hierarchical levels, leveraging dense subgraph discovery. Our evaluation, conducted on real-world datasets, focuses on assessing the output quality and overlap of our proposed models in comparison to other popular models. Our results indicate that our simple, interpretable models provide competitive baselines to popular generative models. Through this investigation, we offer a structured and robust evaluation scheme, thereby facilitating the development of models capable of generating accurate and edge-diverse graphs.

</details>

## 106. On Which Nodes Does GCN Fail? Enhancing GCN From the Node Perspective 🌟

<details>

<summary>Abstract</summary>

The label smoothness assumption is at the core of Graph Convolutional Networks (GCNs): nodes in a local region have similar labels. Thus, GCN performs local feature smoothing operation to adhere to this assumption. However, there exist some nodes whose labels obtained by feature smoothing conflict with the label smoothness assumption. We find that the label smoothness assumption and the process of feature smoothing are both problematic on these nodes, we call these nodes out of GCN's control (OOC nodes). In this paper, first, we design the corresponding algorithm to locate the OOC nodes, then we summarize the characteristics of OOC nodes that affect their representation learning, and based on their characteristics, we propose the Dual augmented GCN (DaGCN) that can facilitate the OOC nodes. Extensive experiments verify the superiority of the proposed method and demonstrate that current advanced GCNs are improvements specifically on OOC nodes; the remaining nodes under GCN's control (UC nodes) are already optimally represented by vanilla GCN on most datasets.

</details>

## 107. OODRobustBench: a benchmark and large-scale analysis of adversarial robustness under distribution shift

<details>

<summary>Abstract</summary>

Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness in a positive linear way. The latter enables the prediction of OOD robustness from ID robustness. We then predict and verify that existing methods are unlikely to achieve high OOD robustness. Novel methods are therefore required to achieve OOD robustness beyond our prediction. To facilitate the development of these methods, we investigate a wide range of techniques and identify several promising directions. Code is provided in the supplementary material.

</details>

## 108. Pairwise Alignment Improves Graph Domain Adaptation

<details>

<summary>Abstract</summary>

Graph-based methods, pivotal for label inference over interconnected objects in many real-world applications, often encounter generalization challenges when the graph used for model training differs significantly from the graph used for testing. This work delves into Graph Domain Adaptation (GDA) to address the unique complexities of distribution shifts over graph data, where interconnected data points experience shifts in features, labels, and in particular, connecting patterns. We propose a novel, theoretically principled method, Pairwise Alignment (Pair-Align) to counter graph structure shift by mitigating conditional structure shift (CSS) and label shift (LS). Pair-Align uses edge weights to recalibrate the influence among neighboring nodes to handle CSS and adjusts the classification loss with label weights to handle LS. Our method demonstrates superior performance in real-world applications, including node classification with region shift in social networks, and the pileup mitigation task in particle colliding experiments. For the first application, we also curate the largest graphs by far for GDA studies. Our method shows strong performance in synthetic and other existing benchmark datasets.

</details>

## 109. PANDA: Expanded Width-Aware Message Passing Beyond Rewiring 🌟

<details>

<summary>Abstract</summary>

Recent research in the field of graph neural network (GNN) has identified a critical issue known as "over-squashing," resulting from the bottleneck phenomenon in graph structures, which impedes the propagation of long-range information. Prior works have proposed a variety of graph rewiring concepts that aim at optimizing the spatial or spectral properties of graphs to promote the signal propagation. However, such approaches inevitably deteriorate the original graph topology, which may incur a distortion of information flow. To address this, we introduce ex**pand**ed width-**a**ware (**PANDA**) message passing, a new message passing paradigm where nodes with high centrality, a potential source of over-squashing, are selectively expanded in width to encapsulate the growing influx of signals from distant nodes. Experimental results show that our method outperforms existing rewiring methods, suggesting that selectively expanding the hidden state of nodes can be a compelling alternative to graph rewiring for addressing the over-squashing.

</details>

## 110. Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and Prototype Modeling

<details>

<summary>Abstract</summary>

The difficulty of partial multi-view multi-label learning lies in coupling the consensus of multi-view data with the task relevance of multi-label classification, under the condition where partial views and labels are unavailable. In this paper, we seek to compress cross-view representation to maximize the proportion of shared information, based that the shared information is sufficient to predict semantic tags. To achieve this, we establish a model consistent with the information bottleneck theory for learning cross-view shared representation, minimizing non-shared information while maintaining data validity to help the model increase the purity of task-relevant information. Furthermore, we model multi-label prototype instances in the latent space and learn label correlations in a data-driven manner. Our method outperforms existing state-of-the-art methods on multiple public datasets while enjoy good compatibility with both partial and complete data. Finally, we experimentally reveal the importance of condensing shared information through information balancing in the process of multi-view information encoding and compression.

</details>

## 111. Perfect Alignment May be Poisonous to Graph Contrastive Learning 🌟

<details>

<summary>Abstract</summary>

Graph Contrastive Learning (GCL) aims to learn node representations by aligning positive pairs and separating negative ones. However, few of researchers have focused on the inner law behind specific augmentations used in graph-based learning. What kind of augmentation will help downstream performance, how does contrastive learning actually influence downstream tasks, and why the magnitude of augmentation matters so much? This paper seeks to address these questions by establishing a connection between augmentation and downstream performance. Our findings reveal that GCL contributes to downstream tasks mainly by separating different classes rather than gathering nodes of the same class. So perfect alignment and augmentation overlap which draw all intra-class samples the same can not fully explain the success of contrastive learning. Therefore, in order to understand how augmentation aids the contrastive learning process, we conduct further investigations into the generalization, finding that perfect alignment that draw positive pair the same could help contrastive loss but is poisonous to generalization, as a result, perfect alignment may not lead to best downstream performance, so specifically designed augmentation is needed to achieve appropriate alignment performance and improve downstream accuracy. To show how should we conduct augmentation and achieve better performance, we analyse the result by information theory and graph spectrum theory and propose two simple but effective methods to verify the theories. The two methods could be easily applied to various GCL algorithms and extensive experiments are conducted to prove its effectiveness.

</details>

## 112. Pi-DUAL: Using privileged information to distinguish clean from noisy labels

<details>

<summary>Abstract</summary>

Label noise is a pervasive problem in deep learning that often compromises the generalization performance of trained models. Recently, leveraging privileged information (PI) -- information available only during training but not at test time -- has emerged as an effective approach to mitigate this issue. Yet, existing PI-based methods have failed to consistently outperform their no-PI counterparts in terms of preventing overfitting to label noise. To address this deficiency, we introduce Pi-DUAL, an architecture designed to harness PI to distinguish clean from wrong labels. Pi-DUAL decomposes the output logits into a prediction term, based on conventional input features, and a noise-fitting term influenced solely by PI. A gating mechanism steered by PI adaptively shifts focus between these terms, allowing the model to implicitly separate the learning paths of clean and wrong labels. Empirically, Pi-DUAL achieves significant performance improvements on key PI benchmarks (e.g., +6.8% on ImageNet-PI), establishing a new state-of-the-art test set accuracy. Additionally, Pi-DUAL is a potent method for identifying noisy samples post-training, outperforming other strong methods at this task. Overall, Pi-DUAL is a simple, scalable and practical approach for mitigating the effects of label noise in a variety of real-world scenarios with PI.

</details>

## 113. Pseudo-Calibration: Improving Predictive Uncertainty Estimation in Unsupervised Domain Adaptation

<details>

<summary>Abstract</summary>

Unsupervised domain adaptation (UDA) has seen substantial efforts to improve model accuracy for an unlabeled target domain with the help of a labeled source domain. However, UDA models often exhibit poorly calibrated predictive uncertainty on target data, a problem that remains under-explored and poses risks in safety-critical UDA applications. The calibration problem in UDA is particularly challenging due to the absence of labeled target data and severe distribution shifts between domains. In this paper, we approach UDA calibration as a target-domain-specific unsupervised problem, different from mainstream solutions based on \emph{covariate shift}. We introduce Pseudo-Calibration (PseudoCal), a novel post-hoc calibration framework. Our innovative use of inference-stage \emph{mixup} synthesizes a labeled pseudo-target set capturing the structure of the real unlabeled target data. This turns the unsupervised calibration problem into a supervised one, easily solvable with \emph{temperature scaling}.Extensive empirical evaluations across 5 diverse UDA scenarios involving 10 UDA methods consistently demonstrate the superior performance and versatility of PseudoCal over existing solutions.

</details>

## 114. Recurrent Distance Filtering for Graph Representation Learning 🌟

<details>

<summary>Abstract</summary>

Graph neural networks based on iterative one-hop message passing have been shown to struggle in harnessing the information from distant nodes effectively. Conversely, graph transformers allow each node to attend to all other nodes directly, but lack graph inductive bias and have to rely on ad-hoc positional encoding. In this paper, we propose a new architecture to reconcile these challenges. Our approach stems from the recent breakthroughs in long-range modeling provided by deep state-space models on sequential data: for a given target node, our model aggregates other nodes by their shortest distances to the target and uses a linear RNN to encode the sequence of hop representations. The linear RNN is parameterized in a particular diagonal form for stable long-range signal propagation and is theoretically expressive enough to encode the neighborhood hierarchy. With no need for positional encoding, we empirically show that the performance of our model is highly competitive compared with that of state-of-the-art graph transformers on various benchmarks, with a significantly reduced computational cost.

</details>

## 115. Reducing Item Discrepancy via Differentially Private Robust Embedding Alignment for Privacy-Preserving Cross Domain Recommendation

<details>

<summary>Abstract</summary>

Cross-Domain Recommendation (CDR) have become increasingly appealing by leveraging useful information to tackle the data sparsity problem across domains. Most of latest CDR models assume that domain-shareable user-item information (e.g., rating and review on overlapped users or items) are accessible across domains. However, these assumptions become impractical due to the strict data privacy protection policy. In this paper, we propose Reducing Item Discrepancy (RidCDR) model on solving Privacy-Preserving Cross-Domain Recommendation (PPCDR) problem. Specifically, we aim to enhance the model performance on both source and target domainswithout overlapped users and items while protecting the data privacy. We innovatively propose private-robust embedding alignment module in RidCDR for knowledge sharing across domains while avoiding negative transfer privately. Our empirical study on Amazon and Douban datasets demonstrates that RidCDR significantly outperforms the state-of-the-art models under the PPCDR without overlapped users and items.

</details>

## 116. Regularizing with Pseudo-Negatives for Continual Self-Supervised Learning

<details>

<summary>Abstract</summary>

We introduce a novel Pseudo-Negative Regularization (PNR) framework for effective continual self-supervised learning (CSSL). Our PNR leverages pseudo-negative samples obtained through model-based augmentation in a way that newly learned representations may not contradict with what have been learned in the past. Specifically, for the InfoNCE-based contrastive learning methods, we define symmetric pseudo-negatives obtained from current and previous models and utilize them in both main and regularization loss terms. Furthermore, we extend this idea to non-contrastive learning methods that do not necessarily use negative samples. The pseudo-negative in this case is defined as the outcome of previous model for differently augmented sample of the anchor and is asymmetrically applied to the regularization term. Through extensive experimental evaluations, our PNR is shown to  achieve state-of-the-art representation learning performance through attaining improved plasticity and stability trade-off.

</details>

## 117. Relational Learning in Pre-Trained Models: A Theory from Hypergraph Recovery Perspective

<details>

<summary>Abstract</summary>

Foundation Models (FMs) have demonstrated remarkable insights into the relational dynamics of the world, leading to the crucial question: *how do these models acquire an understanding of world hybrid relations?*Traditional statistical learning, particularly for prediction problems, may overlook the rich and inherently structured information from the data, especially regardingthe relationships between objects. We introduce a mathematical model that formalizes relational learning as hypergraph recovery to study pre-training of FMs. In our framework, the world is represented as a hypergraph, with data abstracted as random samples from hyperedges. We theoretically examine the feasibility of a Pre-Trained Model (PTM) to recover this hypergraph and analyze the data efficiency in a minimax near-optimal style.By integrating rich graph theories into the realm of PTMs, our  mathematical framework offers powerful tools for an in-depth understanding of pre-training from a unique perspective and can be used  under various scenarios. As an example, we  extend the framework to entity alignment in multimodal learning.

</details>

## 118. Relaxing the Accurate Imputation Assumption in Doubly Robust Learning for Debiased Recommendation

<details>

<summary>Abstract</summary>

Recommender system aims to recommend items or information that may be of interest to users based on their behaviors and preferences. However, there may be sampling selection bias in the process of data collection, i.e., the collected data is not a representative of the target population. Many debiasing methods are developed based on pseudo-labelings. Nevertheless, the effectiveness of these methods relies heavily on accurate pseudo-labelings (i.e., the imputed labels), which is difficult to satisfy in practice. In this paper, we theoretically propose several novel doubly robust estimators that are unbiased when either (a) the pseudo-labelings deviate from the true labels with an arbitrary user-specific inductive bias, item-specific inductive bias, or a combination of both, or (b) the learned propensities are accurate. We further propose a principled propensity reconstruction learning approach that adaptively updates the constraint weights using an attention mechanism and effectively controls the variance. Extensive experiments show that our approach outperforms the state-of-the-art on one semi-synthetic and three real-world datasets.

</details>

## 119. Repeat After Me: Transformers are Better than State Space Models at Copying

<details>

<summary>Abstract</summary>

Transformers are the dominant architecture for sequence modeling, but there is growing interest in models that use a fixed-size latent state that does not depend on the sequence length, which we refer to as ''generalized state space models'' (GSSMs). In this paper we show that while GSSMs are promising in terms of inference-time efficiency, they are limited compared to transformer models on tasks that require copying from the input context.We start with a theoretical analysis of the simple task of string copying and prove that a two layer transformer can copy strings of exponential length while GSSMs are fundamentally limited by their fixed-size latent state. Empirically, we find that transformers outperform GSSMs in terms of efficiency and generalization on synthetic tasks that require copying the context.Finally, we evaluate pretrained large language models and find that transformer models dramatically outperform state space models at copying and retrieving information from context.Taken together, these results suggest a fundamental gap between transformers and GSSMs on tasks of practical interest.

</details>

## 120. Rethinking Guidance Information to Utilize Unlabeled Samples: A Label Encoding Perspective

<details>

<summary>Abstract</summary>

Empirical Risk Minimization (ERM) has achieved great success in scenarios with sufficient labeled samples. However, many practical scenarios suffer from insufficient labeled samples. Under those scenarios, the ERM does not yield good performance as it cannot unleash the potential of unlabeled samples. In this paper, we rethink the guidance information to utilize unlabeled samples for handling those scenarios. By analyzing the learning objective of the ERM, we find that the guidance information for the labeled samples in a specific category is the corresponding *label encoding*. Inspired by this finding, we propose a Label-Encoding Risk Minimization (LERM) to mine the potential of unlabeled samples. It first estimates the label encodings through prediction means of unlabeled samples and then aligns them with their corresponding ground-truth label encodings. As a result, the LERM ensures both prediction discriminability and diversity and can be integrated into existing methods as a plugin. Theoretically, we analyze the relationship between the LERM and ERM. Empirically, we verify the superiority of the LERM under several label insufficient scenarios, including semi-supervised learning, unsupervised domain adaptation, and semi-supervised heterogeneous domain adaptation.

</details>

## 121. Rethinking Independent Cross-Entropy Loss For Graph-Structured Data 🌟

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) have exhibited prominent performance in learning graph-structured data. Considering node classification task, the individual label distribution conditioned on node representation is used to predict its classes. Based on the i.i.d assumption among node labels, the traditional supervised learning simply sums up cross-entropy losses of the independent training nodes and applies the average loss to optimize GNNs' weights. But different from other data formats, the nodes are naturally connected and their classes are correlated to neighbors at the same cluster. It is found that the independent distribution modeling of node labels restricts GNNs' capability to generalize over the entire graph and defend adversarial attacks. In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. Rather than assuming the node labels are independent, we learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data-label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. The extensive experiments on 12 benchmark datasets and 7 backbone models demonstrate that our joint-cluster supervised learning can effectively bolster GNNs' node classification accuracy. Furthermore, being benefited from the reference signals which may be free from spiteful interference, our learning paradigm significantly protects the node classification from being affected by the adversarial attack.

</details>

## 122. Rethinking Momentum Knowledge Distillation in Online Continual Learning

<details>

<summary>Abstract</summary>

Online Continual Learning (OCL) addresses the problem of training neural networks on a continuous data stream where multiple classification tasks emerge in sequence. In contrast to offline Continual Learning, data can be seen only once in OCL, which is a very severe constraint. In this context, replay-based strategies have achieved impressive results and most state-of-the-art approaches heavily depend on them. While Knowledge Distillation (KD) has been extensively used in offline Continual Learning, it remains under-exploited in OCL, despite its high potential. In this paper, we theoretically analyze the challenges in applying KD to OCL. We introduce a direct yet effective methodology for applying Momentum Knowledge Distillation (MKD) to many flagship OCL methods and demonstrate its capabilities to enhance existing approaches. In addition to improving existing state-of-the-arts accuracy by more than $10\%$ points on ImageNet100, we shed light on MKD internal mechanics and impacts during training in OCL. We argue that similar to replay, MKD should be considered a central component of OCL.

</details>

## 123. Revisit the Essence of Distilling Knowledge through Calibration 🌟

<details>

<summary>Abstract</summary>

Knowledge Distillation (KD) has evolved into a practical technology for transferring knowledge from a well-performing model (teacher) to a weak model (student). A counter-intuitive phenomenon known as capacity mismatch has been identified, wherein KD performance may not be good when a better teacher instructs the student. Various preliminary methods have been proposed to alleviate capacity mismatch, but a unifying explanation for its cause remains lacking. In this paper, we propose \textit{a unifying analytical framework to pinpoint the core of capacity mismatch based on calibration}. Through extensive analytical experiments, we observe a positive correlation between the calibration of the teacher model and the KD performance with original KD methods. As this correlation arises due to the sensitivity of metrics (e.g., KL divergence) to calibration, we recommend employing measurements insensitive to calibration such as ranking-based loss. Our experiments demonstrate that ranking-based loss can effectively replace KL divergence, aiding large models with poor calibration to teach better.

</details>

## 124. Robust Graph Matching when Nodes are Corrupt

<details>

<summary>Abstract</summary>

Two models are introduced to study the problem of matching two correlated graphs when some of the nodes are corrupt. In the weak model, a random subset of nodes in one or both graphs can interact randomly with their network. For this model, it is shown that no estimator can correctly recover a positive fraction of the corrupt nodes. Necessary conditions for any estimator to correctly identify and match all the uncorrupt nodes are derived, and it is shown that these conditions are also sufficient for the k-core estimator. In the strong model, an adversarially selected subset of nodes in one or both graphs can interact arbitrarily with their network. For this model, detection of corrupt nodes is impossible. Even so, we show that if only one of the networks is compromised, then under appropriate conditions, the maximum overlap estimator can correctly match a positive fraction of nodes albeit without explicitly identifying them.

</details>

## 125. S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning 🌟

<details>

<summary>Abstract</summary>

Graph Contrastive Learning (GCL) has emerged as a highly effective self-supervised approach in graph representation learning. However, prevailing GCL methods confront two primary challenges: 1) They predominantly operate under homophily assumptions, focusing on low-frequency signals in node features while neglecting heterophilic edges that connect nodes with dissimilar features. 2) Their reliance on neighborhood aggregation for inference leads to scalability challenges and hinders deployment in real-time applications. In this paper, we introduce S3GCL,  an innovative framework designed to tackle these challenges. Inspired by spectral GNNs, we initially demonstrate the correlation between frequency and homophily levels. Then, we propose a novel cosine-parameterized Chebyshev polynomial as low/high-pass filters to generate biased graph views. To resolve the inference dilemma, we incorporate an MLP encoder and enhance its awareness of graph context by introducing structurally and semantically neighboring nodes as positive pairs in the spatial domain. Finally, we formulate a cross-pass GCL objective between full-pass MLP and biased-pass GNN filtered features, eliminating the need for augmentation. Extensive experiments on real-world tasks validate S3GCL proficiency in generalization to diverse homophily levels and its superior inference efficiency.

</details>

## 126. SAM as the Guide: Mastering Pseudo-Label Refinement in Semi-Supervised Referring Expression Segmentation

<details>

<summary>Abstract</summary>

In this paper, we introduce SemiRES, a semi-supervised framework that effectively leverages a combination of labeled and unlabeled data to perform RES. A significant hurdle in applying semi-supervised techniques to RES is the prevalence of noisy pseudo-labels, particularly at the boundaries of objects. SemiRES incorporates the Segment Anything Model (SAM), renowned for its precise boundary demarcation, to improve the accuracy of these pseudo-labels. Within SemiRES, we offer two alternative matching strategies: IoU-based Optimal Matching (IOM) and Composite Parts Integration (CPI). These strategies are designed to extract the most accurate masks from SAM's output, thus guiding the training of the student model with enhanced precision. In instances where a precise mask cannot be matched from the available candidates,  we develop the Pixel-Wise Adjustment (PWA) strategy, guiding the student model's training directly by the pseudo-labels. Extensive experiments on three RES benchmarks—RefCOCO, RefCOCO+, and G-Ref reveal its superior performance compared to fully supervised methods, especially in low-data scenarios. Remarkably, with only 1\% labeled data, our SemiRES outperforms the supervised baseline by a large margin, e.g. +18.64\% gains on RefCOCO val set.

</details>

## 127. Scene Graph Generation Strategy with Co-occurrence Knowledge and Learnable Term Frequency

<details>

<summary>Abstract</summary>

Scene graph generation (SGG) is an important task in image understanding because it represents the relationships between objects in an image as a graph structure, making it possible to understand the semantic relationships between objects intuitively. Previous SGG studies used a message-passing neural networks (MPNN) to update features, which can effectively reflect information about surrounding objects. However, these studies have failed to reflect the co-occurrence of objects during SGG generation. In addition, they only addressed the long-tail problem of the training dataset from the perspectives of sampling and learning methods. To address these two problems, we propose CooK, which reflects the Co-occurrence Knowledge between objects, and the learnable term frequency-inverse document frequency (TF-$l$-IDF) to solve the long-tail problem. By combining CooK and TF-$l$-IDF, we successfully perform SGG, which can simultaneously enhance our understanding of object co-occurrence and mitigate the long-tail problem. We applied the proposed model to the SGG benchmark dataset, and the results showed a performance improvement of up to 3.8\% compared with existing state-of-the-art models in SGGen subtask. In addition, the proposed method can be easily applied to existing MPNN-based models. The proposed method exhibits generalization ability from the results obtained, showing uniform performance improvement for all MPNN models.

</details>

## 128. Semantic-Aware Distribution Matching for Semi-Supervised Learning

<details>

<summary>Abstract</summary>

Semi-supervised learning has made remarkable strides by effectively utilizing a limited amount of labeled data while capitalizing on the abundant information present in unlabeled data. However, current algorithms often prioritize aligning image predictions with specific classes generated through self-training techniques, thereby neglecting the inherent relationships that exist within these classes. In this paper, we present a new approach called SaMatch, which leverages semantic relationships among classes by employing an optimal transport loss function to match distributions. We conduct extensive experiments on vision datasets like CIFAR 10/100, STL-10, and ImageNet and language datasets like Amazon Review, and Yelp Review. The empirical results show substantial improvements in our method above baseline, this demonstrates the effectiveness and superiority of our approach in harnessing semantic relationships to enhance learning performance in a semi-supervised setting.

</details>

## 129. Sign is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have gained significant attention as a powerful modeling and inference method, especially for homophilic graph-structured data. To empower GNNs in heterophilic graphs, where adjacent nodes exhibit dissimilar labels or features, Signed Message Passing (SMP) has been widely adopted. However, there is a lack of theoretical and empirical analysis regarding the limitations of SMP. In this work, we unveil the potential pitfalls of SMP and their remedies. We first identify two limitations of SMP: undesirable representation update for multi-hop neighbors and vulnerability against oversmoothing issues. To overcome these challenges, we propose a novel message-passing function called Multiset to Multiset GNN (M2M-GNN). Our theoretical analyses and extensive experiments demonstrate that M2M-GNN effectively alleviates the limitations of SMP, yielding superior performance in comparison.

</details>

## 130. Sign Rank Limitations for Inner Product Graph Decoders

<details>

<summary>Abstract</summary>

Inner product-based decoders are among the most influential frameworks used to extract meaningful data from latent embeddings. However, such decoders have shown limitations in representation capacity in numerous works within the literature, which have been particularly notable in graph reconstruction problems. In this paper, we provide the first theoretical elucidation of this pervasive phenomenon in graph data, and suggest straightforward modifications to circumvent this issue without deviating from the inner product framework.

</details>

## 131. SLOG: An Inductive Spectral Graph Neural Network Beyond Polynomial Filter 🌟

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) have exhibited superb power in many graph related tasks. Existing GNNs can be categorized into spatial GNNs and spectral GNNs. The spatial GNNs primarily capture the local information around each node, while the spectral GNNs are able to operate on the frequency signals of the entire graph. However, most, if not all, existing spectral GNNs are faced with two limitations: (1) the polynomial limitation that for most spectral GNNs, the expressive power in the spectral domain is limited to polynomial filters; and (2) the transductive limitation that most spectral GNNs can only be applied to the transductive setting on relatively small-scale graphs. In this paper, we propose a novel spectral graph neural network named SLOG to solve the above two limitations. For the polynomial limitation, SLOG proposes a novel real-valued filter with geometric interpretability, mathematical feasibility and adaptive filtering ability to go beyond polynomial. For the transductive limitation, SLOG combines the subgraph sampling technique in spatial GNNs and the signal processing technique in spectral GNNs together to make itself tailored to the inductive setting on large-scale graphs. Extensive experimental results on 16 datasets demonstrate the superiority of SLOG in inductive homophilic and heterophilic node classification task.

</details>

## 132. Subgraphormer: Unifying Subgraph GNNs and Graph Transformers via Graph Products 🌟

<details>

<summary>Abstract</summary>

In the realm of Graph Neural Networks (GNNs), two exciting research directions have recently emerged: Subgraph GNNs and Graph Transformers. In this paper, we propose an architecture that integrates both approaches, dubbed *Subgraphormer*, which combines the enhanced expressive power, message-passing mechanisms, and aggregation schemes from Subgraph GNNs with attention and positional encodings, arguably the most important components in Graph Transformers. Our method is based on an intriguing new connection we reveal between Subgraph GNNs and product graphs, suggesting that Subgraph GNNs can be formulated as Message Passing Neural Networks (MPNNs) operating on a product of the graph with itself. We use this formulation to design our architecture: first, we devise an attention mechanism based on the connectivity of the product graph. Following this, we propose a novel and efficient positional encoding scheme for Subgraph GNNs, which we derive as a positional encoding for the product graph. Our experimental results demonstrate significant performance improvements over both Subgraph GNNs and Graph Transformers on a wide range of datasets.

</details>

## 133. The Entropy Enigma: Success and Failure of Entropy Minimization 🌟

<details>

<summary>Abstract</summary>

Entropy minimization (EM) is frequently used to increase the accuracy of classification models when they're faced with new data at test time. EM is a self-supervised learning method that optimizes classifiers to assign even higher probabilities to their top predicted classes. In this paper, we analyze why EM works when adapting a model for a few steps and why it eventually fails after adapting for many steps. We show that, at first, EM causes the model to embed test images close to training images, thereby increasing model accuracy. After many steps of optimization, EM makes the model embed test images far away from the embeddings of training images, which results in a degradation of accuracy. Building upon our insights, we present a method for solving a practical problem: estimating a model's accuracy on a given arbitrary dataset without having access to its labels. Our method estimates accuracy by looking at how the embeddings of input images change as the model is optimized to minimize entropy. Experiments on 23 challenging datasets show that our method sets the SoTA with a mean absolute error of 5.75%, an improvement of 29.62% over the previous SoTA on this task.

</details>

## 134. The Expressive Power of Path based Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

We systematically investigate the expressive power of path-based graph neural networks. While it has been shown that they can achieve strong empirical results, an investigation into their expressive power is lacking. Therefore, we propose PATH-WL, a general class of color refinement algorithms based on paths and geodesic distance information. We characterize families of graphs that can be distinguished by PATH-WL. For a sufficient path length, PATH-WL is incomparable to a wide range of expressive graph neural networks, can count cycles, and achieves strong results on the notoriously difficult family of strongly regular graphs. Our theoretical results indicate that PATH-WL forms a new hierarchy of highly expressive graph neural networks.

</details>

## 135. Topological Neural Networks go Persistent, Equivariant and Continuous

<details>

<summary>Abstract</summary>

Topological Neural Networks (TNNs) have enabled representations using higher dimensional simplicial complexes. Concurrently, persistence homology methods have undergone rapid strides, offering rich topological descriptors that improve the expressivity of GNNs. However, the integration of these methods to increase the expressivity of TNNs, and adaptation in handling geometric complexes, remains an unexplored frontier. We introduce TopNets, extending the concept of TNNs by unifying them with persistent homology (PH), equivariance and making them continuous. This framework provides a generalized approach that encompasses various methods at the intersection of PH and TNNs. TopNets enhances the expressiveness of Equivariant Message Passing (MP) simplicial networks, allowing them to acquire high-dimensional simplex features alongside topological embeddings generated through geometric color filtrations in an $\mathrm{E}(n)$-equivariant manner. Empirical evaluation demonstrates the efficacy of the proposed method across diverse tasks such as graph classification, drug property prediction, and generative design.

</details>

## 136. Towards General Algorithm Discovery for Combinatorial Optimization: Learning Symbolic Branching Policy from Bipartite Graph

<details>

<summary>Abstract</summary>

Machine learning (ML) approaches have been successfully applied to accelerating exact combinatorial optimization (CO) solvers.However, many of them fail to explain what patterns they have learned that accelerate the CO algorithms due to the black-box nature of ML models like neural networks, and thus they prevent researchers from further understanding the tasks they are interested in. To tackle this problem, we propose the *first* graph-based algorithm discovery framework---namely, graph symbolic discovery for exact combinatorial optimization solver (GS4CO)---that learns interpretable branching policies directly from the *general* bipartite graph representation of CO problems. Specifically, we design a unified  representation for symbolic policies with graph inputs, and then we employ a Transformer with multiple tree-structural encodings to generate symbolic trees end-to-end, which effectively reduces the cumulative error from iteratively distilling graph neural networks. Experiments show that GS4CO learned interpretable and lightweight policies outperform all the baselines on CPU machines, including both the human-designed and the learning-based. GS4CO shows an encouraging step towards general algorithm discovery on modern CO solvers.

</details>

## 137. Towards Theoretical Understanding of Learning Large-scale Dependent Data via Random Features

<details>

<summary>Abstract</summary>

Random feature (RF) mapping is an attractive and powerful technique for solving large-scale nonparametric regression. Yet, the existing theoretical analysis crucially relies on the i.i.d. assumption that individuals in the data are independent and identically distributed. It is still unclear whether learning accuracy will be compromised when such an assumption is violated.  This paper aims to provide theoretical understanding of the kernel ridge regression (KRR) with RFs for large-scale dependent data. Specifically, we consider two types of data dependence structure, where one is the  $\tau$-mixing process with exponential decay coefficient, and another is the $\tau$-mixing process with polynomial decay coefficient. Theoretically,  we prove that the kernel ridge estimator with RFs achieves the minimax optimality under the exponential decay case, but yields a sub-optimal result under the polynomial decay case. Our analysis further reveals how the decay rate of the $\tau$-mixing coefficient impacts the learning accuracy of the kernel ridge estimator with RFs, which, to the best of our knowledge, is new.  Extensive numerical experiments on both synthetic and real examples further validate our theoretical findings and support the effectiveness of the KRR with RFs in dealing with dependent data.

</details>

## 138. Towards Understanding Inductive Bias in Transformers: A View From Infinity 🌟

<details>

<summary>Abstract</summary>

We study inductive bias in Transformers in the infinitely over-parameterized Gaussian process limit and argue transformers are tend to be biased towards more permutation symmetric functions in sequence space. We show that the representation theory of the symmetric group can be used to give quantitative analytical predictions when the dataset is symmetric to permutations between tokens.We present a simplified transformer block and solve the model at the limit, including accurate predictions for the learning curves and network outputs. We show that in common setups, one can derive tight bounds in the form of a scaling law for the learnability as a function of the context length. Finally, we argue WikiText dataset, does indeed possess a degree of permutation symmetry.

</details>

## 139. Translating Subgraphs to Nodes Makes Simple GNNs Strong and Efficient for Subgraph Representation Learning 🌟

<details>

<summary>Abstract</summary>

Subgraph representation learning has emerged as an important problem, but it is by default approached with specialized graph neural networks on a large global graph. These models demand extensive memory and computational resources but challenge modeling hierarchical structures of subgraphs. In this paper, we propose Subgraph-To-Node (S2N) translation, a novel formulation for learning representations of subgraphs. Specifically, given a set of subgraphs in the global graph, we construct a new graph by coarsely transforming subgraphs into nodes. Demonstrating both theoretical and empirical evidence, S2N not only significantly reduces memory and computational costs compared to state-of-the-art models but also outperforms them by capturing both local and global structures of the subgraph. By leveraging graph coarsening methods, our method outperforms baselines even in a data-scarce setting with insufficient subgraphs. Our experiments on eight benchmarks demonstrate that fined-tuned models with S2N translation can process 183 -- 711 times more subgraph samples than state-of-the-art models at a better or similar performance level.

</details>

## 140. Triplet Interaction Improves Graph Transformers: Accurate Molecular Graph Learning with Triplet Graph Transformers

<details>

<summary>Abstract</summary>

Graph transformers typically lack direct pair-to-pair communication, instead forcing neighboring pairs to exchange information via a common node. We propose the Triplet Graph Transformer (TGT) that enables direct communication between two neighboring pairs in a graph via novel triplet attention and aggregation mechanisms. TGT is applied to molecular property prediction by first predicting interatomic distances from 2D graphs and then using these distances for downstream tasks. A novel three-stage training procedure and stochastic inference further improve training efficiency and model performance. Our model achieves new state-of-the-art (SOTA) results on open challenge benchmarks PCQM4Mv2 and OC20 IS2RE. We also obtain SOTA results on QM9, MOLPCBA, and LIT-PCBA molecular property prediction benchmarks via transfer learning. We also demonstrate the generality of TGT with SOTA results on the traveling salesman problem (TSP).

</details>

## 141. Two Heads are Actually Better than One: Towards Better Adversarial Robustness via Transduction and Rejection

<details>

<summary>Abstract</summary>

Both transduction and rejection have emerged as important techniques for defending against adversarial perturbations. A recent work by Goldwasser et. al showed that rejection combined with transduction can give *provable* guarantees (for certain problems) that cannot be achieved otherwise. Nevertheless, under recent strong adversarial attacks (GMSA), Goldwasser et al.'s work was shown to have low performance in a practical deep-learning setting.  In this paper, we take a step towards realizing the promise of transduction+rejection in more realistic scenarios. Our key observation is that a novel application of a reduction technique by Tramèr, which was until now only used to demonstrate the vulnerability of certain defenses, can be used to actually construct effective defenses. Theoretically, we show that a careful application of this technique in the transductive setting can give significantly improved sample-complexity for robust generalization. Our theory guides us to design a new transductive algorithm for learning a selective model; extensive experiments using state of the art attacks (AutoAttack, GMSA) show that our approach provides significantly better robust accuracy (81.6\% on CIFAR-10 and 57.9\% on CIFAR-100 under $l_\infty$ with budget 8/255) than existing techniques.

</details>

## 142. Two Heads Are Better Than One: Boosting Graph Sparse Training via Semantic and Topological Awareness

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) excel in various graph learning tasks but face computational challenges when applied to large-scale graphs. A promising solution is to remove non-essential edges to reduce the computational overheads in GNN. Previous literature generally falls into two categories: topology-guided and semantic-guided. The former maintains certain graph topological properties yet often underperforms on GNNs due to low integration with neural network training. The latter performs well at lower sparsity on GNNs but faces performance collapse at higher sparsity levels. With this in mind, we take the \underline{first} step to propose a new research line and concept termed \textbf{Graph Sparse Training} \textbf{(GST)}, which dynamically manipulates sparsity at the data level. Specifically, GST initially constructs a topology \& semantic anchor at a low training cost, followed by performing dynamic sparse training to align the sparse graph with the anchor. We introduce the \textbf{Equilibria Sparsification Principle} to guide this process, effectively balancing the preservation of both topological and semantic information. Ultimately, GST produces a sparse graph with maximum topological integrity and no performance degradation.Extensive experiments on 6 datasets and 5 backbones showcase that GST \textbf{(I)} identifies subgraphs at higher graph sparsity levels ($1.67\%\sim15.85\%$$\uparrow$) than state-of-the-art sparsification methods, \textbf{(II)} preserves more key spectral properties, \textbf{(III)} achieves $1.27-3.42\times$ speedup in GNN inference and \textbf{(IV)} successfully helps graph adversarial defense and graph lottery tickets. The source code is available at \url{https://anonymous.4open.science/r/GST-0F15}.

</details>

## 143. Uncertainty for Active Learning on Graphs

<details>

<summary>Abstract</summary>

Uncertainty Sampling is an Active Learning strategy that aims to improve the data efficiency of machine learning models by iteratively acquiring labels of data points with the highest uncertainty. While it has proven effective for independent data its applicability to graphs remains under-explored. We propose the first extensive study of Uncertainty Sampling for node classification: **(1)** We benchmark Uncertainty Sampling beyond predictive uncertainty and highlight a significant performance gap to other Active Learning strategies. **(2)** We develop ground-truth Bayesian uncertainty estimates in terms of the data generating process and prove their effectiveness in guiding Uncertainty Sampling toward optimal queries. We confirm our results on synthetic data and design an approximate approach that consistently outperforms other uncertainty estimators on real datasets. **(3)** Based on this analysis, we relate pitfalls in modeling uncertainty to existing methods. Our analysis enables and informs the development of principled uncertainty estimation on graphs.

</details>

## 144. Understanding Heterophily for Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Graphs with heterophily have been regarded as challenging scenarios for Graph Neural Networks (GNNs), where nodes are connected with dissimilar neighbors through various patterns. In this paper, we present theoretical understandings of heterophily for GNNs by incorporating the graph convolution (GC) operations into fully connected networks via the proposed Heterophilous Stochastic Block Models (HSBM), a general random graph model that can accommodate diverse heterophily patterns. Our theoretical investigation comprehensively analyze the impact of heterophily from three critical aspects. Firstly, for the impact of different heterophily patterns, we show that the separability gains are determined by two factors, i.e., the Euclidean distance of the neighborhood distributions and $\sqrt{\mathbb{E}\left[\operatorname{deg}\right]}$, where $\mathbb{E}\left[\operatorname{deg}\right]$ is the averaged node degree. Secondly, we show that the neighborhood inconsistency has a detrimental impact on separability, which is similar to degrading $\mathbb{E}\left[\operatorname{deg}\right]$ by a specific factor. Finally, for the impact of stacking multiple layers, we show that the separability gains are determined by the normalized distance of the $l$-powered neighborhood distributions, indicating that nodes still possess separability in various regimes, even when over-smoothing occurs. Extensive experiments on both synthetic and real-world data verify the effectiveness of our theory.

</details>

## 145. Unlocking Exact Recovery in Semi-Supervised Learning: Analysis of Spectral Method and Graph Convolution Network 🌟

<details>

<summary>Abstract</summary>

We delve into the challenge of semi-supervised node classification on the Contextual Stochastic Block Model (CSBM) dataset. Here, nodes from the two-cluster stochastic block model (SBM) are coupled with feature vectors, which are derived from a Gaussian Mixture Model (GMM) that corresponds to their respective node labels. With only a subset of the CSBM node labels accessible for training, our primary objective becomes the accurate classification of the remaining nodes. Venturing into the transductive learning landscape, we, for the first time, pinpoint the information-theoretical threshold for the exact recovery of all test nodes in CSBM. Concurrently, we design an optimal spectral estimator inspired by Principal Component Analysis (PCA) with the training labels and essential data from both the adjacency matrix and feature vectors. We also evaluate the efficacy of graph ridge regression and Graph Convolutional Networks (GCN) on this synthetic dataset. Our findings underscore that graph ridge regression and GCN possess the ability to achieve the information threshold of exact recovery in a manner akin to the optimal estimator when using the optimal weighted self-loops. This highlights the potential role of feature learning in augmenting the proficiency of GCN, especially in the realm of semi-supervised learning.

</details>

## 146. Unraveling the Impact of Heterophilic Structures on Graph Positive-Unlabeled Learning 🌟

<details>

<summary>Abstract</summary>

While Positive-Unlabeled (PU) learning is vital in many real-world scenarios, its application to graph data still remains under-explored. We unveil that a critical challenge for PU learning on graph lies on the edge heterophily, which directly violates the $\textit{irreducibilityassumption}$ for $\textit{Class-Prior Estimation}$ (class prior is essential for building PU learning algorithms) and degenerates the latent label inference on unlabeled nodes during classifier training. In response to this challenge, we introduce a new method, named $\textit{$\underline{G}$raph $\underline{P}$U Learning with $\underline{L}$abel Propagation Loss}$ (GPL). Specifically, GPL considers learning from PU nodes along with an intermediate heterophily reduction, which helps mitigate the negative impact of the heterophilic structure. We formulate this procedure as a bilevel optimization that reduces heterophily in the inner loop and efficiently learns a classifier in the outer loop. Extensive experiments across a variety of datasets have shown that GPL significantly outperforms baseline methods, confirming its effectiveness and superiority.

</details>

## 147. Unsupervised Concept Discovery Mitigates Spurious Correlations

<details>

<summary>Abstract</summary>

Models susceptible to spurious correlations in their training data often produce brittle predictions and introduce unintended biases. Addressing this challenge typically involves methods relying on prior knowledge and group annotation to remove spurious correlations, which may not be readily available in many applications. In this paper, we establish a novel connection between unsupervised object-centric learning and mitigation of spurious correlations. Instead of inferring subgroups with varying correlations with the labels, our approach focuses on discovering concepts: discrete ideas that are shared across input samples. Leveraging existing object-centric representation learning, we propose a method that effectively mitigates spurious correlations without requiring human labeling of subgroups. Evaluation on diverse benchmark datasets for subpopulation shifts, without relying on ground-truth or human-annotated groups, demonstrates improvements of 1–2% on the challenging ImageNet-9 background challenge and overall competitive performance in the absence of human-annotated groups.

</details>

## 148. Unsupervised Episode Generation for Graph Meta-learning

<details>

<summary>Abstract</summary>

We propose Unsupervised Episode Generation method called **Neighbors as Queries (NaQ)** to solve the Few-Shot Node-Classification (FSNC) task by *unsupervised Graph Meta-learning*.Doing so enables full utilization of the information of all nodes in a graph, which is not possible in current supervised meta-learning methods for FSNC due to the label-scarcity problem.In addition, unlike unsupervised Graph Contrastive Learning (GCL) methods that overlook the downstream task to be solved at the training phase resulting in vulnerability to class imbalance of a graph, we adopt the episodic learning framework that allows the model to be aware of the downstream task format, i.e., FSNC.The proposed NaQ is a simple but effective *unsupervised* episode generation method that randomly samples nodes from a graph to make a support set, followed by similarity-based sampling of nodes to make the corresponding query set.Since NaQ is *model-agnostic*, any existing supervised graph meta-learning methods can be trained in an unsupervised manner, while not sacrificing much of their performance or sometimes even improving them.Extensive experimental results demonstrate the effectiveness of our proposed unsupervised episode generation method for graph meta-learning towards the FSNC task.Our code is available at: https://github.com/JhngJng/NaQ-PyTorch.

</details>

## 149. Verifying message-passing neural networks via topology-based bounds tightening 🌟

<details>

<summary>Abstract</summary>

Since graph neural networks (GNNs) are often vulnerable to attack, we need to know when we can trust them. We develop a computationally effective approach towards providing robust certificates for message-passing neural networks (MPNNs) using a Rectified Linear Unit (ReLU) activation function. Because our work builds on mixed-integer optimization, it encodes a wide variety of subproblems, for example it admits (i) both adding and removing edges, (ii) both global and local budgets, and (iii) both topological perturbations and feature modifications. Our key technology, topology-based bounds tightening, uses graph structure to tighten bounds. We also experiment with aggressive bounds tightening to dynamically change the optimization constraints by tightening variable bounds. To demonstrate the effectiveness of these strategies, we implement an extension to the open-source branch-and-cut solver SCIP. We test on both node and graph classification problems and consider topological attacks that both add and remove edges.

</details>

## 150. VNNs: Verification-Friendly Neural Networks with Hard Robustness Guarantees

<details>

<summary>Abstract</summary>

Machine learning techniques often lack formal correctness guarantees, evidenced by the widespread adversarial examples that plague most deep-learning applications. This lack of formal guarantees resulted in several research efforts that aim at verifying Deep Neural Networks (DNNs), with a particular focus on safety-critical applications. However, formal verification techniques still face major scalability and precision challenges. The over-approximation introduced during the formal verification process to tackle the scalability challenge often results in inconclusive analysis. To address this challenge, we propose a novel framework to generate Verification-friendly Neural Networks (VNNs). We present a post-training optimization framework to achieve a balance between preserving prediction performance and verification-friendliness. Our proposed framework results in VNNs that are comparable to the original DNNs in terms of prediction performance, while amenable to formal verification techniques. This essentially enables us to establish robustness for more VNNs than their DNN counterparts, in a time-efficient manner.

</details>

## 151. Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation 🌟

<details>

<summary>Abstract</summary>

Uncertainty estimation (UE), as an effective means of quantifying predictive uncertainty, is crucial for safe and reliable decision-making, especially in high-risk scenarios. Existing UE schemes usually assume that there are completely-labeled samples to support fully-supervised learning. In practice, however, many UE tasks often have no sufficiently-labeled data to use, such as the Multiple Instance Learning (MIL) with only weak instance annotations. To bridge this gap, this paper, for the first time, addresses the weakly-supervised issue of *Multi-Instance UE* (MIUE) and proposes a new baseline scheme, *Multi-Instance Residual Evidential Learning* (MIREL). Particularly, at the fine-grained instance UE with only weak supervision, we derive a multi-instance residual operator through the Fundamental Theorem of Symmetric Functions. On this operator derivation, we further propose MIREL to jointly model the high-order predictive distribution at bag and instance levels for MIUE. Extensive experiments empirically demonstrate that our MIREL not only could often make existing MIL networks perform better in MIUE, but also could surpass representative UE methods by large margins, especially in instance-level UE tasks. Our source code is available at https://github.com/liupei101/MIREL.

</details>

## 152. Weisfeiler-Leman at the margin: When more expressivity matters

<details>

<summary>Abstract</summary>

The Weisfeiler--Leman algorithm ($1\textsf{-WL}$) is a well-studied heuristic for the graph isomorphism problem. Recently, the algorithm has played a prominent role in understanding the expressive power of message-passing graph neural networks (MPNNs) and being effective as a graph kernel. Despite its success, the $1\textsf{-WL}$ faces challenges in distinguishing non-isomorphic graphs, leading to the development of more expressive MPNN and kernel architectures. However, the relationship between enhanced expressivity and improved generalization performance remains unclear. Here, we show that an architecture's expressivity offers limited insights into its generalization performance when viewed through graph isomorphism. Moreover, we focus on augmenting $1\textsf{-WL}$ and MPNNs with subgraph information and employ classical margin theory to investigate the conditions under which an architecture's increased expressivity aligns with improved generalization performance. In addition, we introduce variations of expressive \wlone-based kernel and MPNN architectures with provable generalization properties. Our empirical study confirms the validity of our theoretical findings.

</details>

## 153. What Can Transformer Learn with Varying Depth? Case Studies on Sequence Learning Tasks

<details>

<summary>Abstract</summary>

We study the capabilities of the transformer architecture with varying depth. Specifically, we designed a novel set of sequence learning tasks to systematically evaluate and comprehend how the depth of transformer affects its ability to perform memorization, reasoning, generalization, and contextual generalization. We show a transformer with only one attention layer can excel in memorization but falls short in other tasks. Then, we show that exhibiting reasoning and generalization ability requires the transformer to have at least two attention layers, while context generalization ability may necessitate three attention layers. Additionally, we identify a class of simple operations that a single attention layer can execute, and show that the complex tasks can be approached as the combinations of these simple operations and thus can be resolved by stacking multiple attention layers. This sheds light on studying more practical and complex tasks beyond our design. Numerical experiments corroborate our theoretical findings.

</details>

## 154. What Improves the Generalization of Graph Transformer? A Theoretical Dive into Self-attention and Positional Encoding

<details>

<summary>Abstract</summary>

Graph Transformers, which incorporate self-attention and positional encoding, have recently emerged as a powerful architecture for various graph learning tasks. Despite their impressive performance, the complex non-convex interactions across layers and the recursive graph structure have made it challenging to establish a theoretical foundation for learning and generalization. This study introduces the first theoretical investigation of a shallow Graph Transformer for semi-supervised node classification, comprising a self-attention layer with relative positional encoding and a two-layer perception. Focusing on a graph data model with discriminative nodes that determine node labels and non-discriminative nodes that are class-irrelevant, we characterize the sample complexity required to achieve a desirable generalization error by training with stochastic gradient descent (SGD). This paper provides the quantitative characterization of the sample complexity and number of iterations for convergence dependent on the fraction of discriminative nodes, the dominant patterns,and the initial model errors. Furthermore, we demonstrate that self-attention and positional encoding enhance generalization by making the attention map sparse and promoting the core neighborhood during training, which explains the superior feature representation of Graph Transformers. Our theoretical results are supported by empirical experiments on synthetic and real-world benchmarks.

</details>

## 155. What is Dataset Distillation Learning?

<details>

<summary>Abstract</summary>

Dataset distillation has emerged as a strategy to overcome the hurdles associated with large datasets by learning a compact set of synthetic data that retains essential information from the original dataset. While distilled data can be used to train high performing models, little is understood how information is stored. In this study, we posit and answer three questions: is dataset distillation more analogous to the compression of model parameters or data statistics? how does dataset distillation improve expressiveness compared to classical compression techniques? do distilled data points individually carry meaningful information? We reveal that distilled data cannot be simply characterized as either model or data compression. Additionally, the distillation process works by compressing the early dynamics of real models. Finally, we provide an interpretable framework for analyzing distilled data and uncover the fact that individual distilled data points do contain meaningful semantic information. This investigation sheds light on the intricate nature of distilled data, providing a better understanding on how this data can be effectively utilized.

</details>

## 156. When is Transfer Learning Possible?

<details>

<summary>Abstract</summary>

We present a general framework for transfer learning that is flexible enough to capture transfer in supervised, reinforcement, and imitation learning. Our framework enables new insights into the fundamental question of \emph{when} we can successfully transfer learned information across problems. We model the learner as interacting with a sequence of problem instances, or \textit{environments}, each of which is generated from a common structural causal model (SCM) by choosing the SCM's parameters from restricted sets. We derive a procedure that can propagate restrictions on SCM parameters through the SCM's graph structure to other parameters that we are trying to learn. The propagated restrictions then enable more efficient learning (i.e., transfer). By analyzing the procedure, we are able to challenge widely-held beliefs about transfer learning. First, we show that having \textit{sparse} changes across environments is neither necessary nor sufficient for transfer. Second, we show an example where the common heuristic of \textit{freezing} a layer in a network causes poor transfer performance. We then use our procedure to select a more refined set of parameters to freeze, leading to successful transfer learning.

</details>

## 157. Why Do Animals Need Shaping? A Theory of Task Composition and Curriculum Learning 🌟

<details>

<summary>Abstract</summary>

Diverse studies in systems neuroscience begin with extended periods of training known as ‘shaping’procedures. These involve progressively studying component parts of more complex tasks, and can make the difference between learning a task quickly, slowly or not at all. Despite the importance of shaping to the acquisition of complex tasks, there is as yet no theory that can help guide the design of shaping procedures, or more fundamentally, explain its key role in learning and provide conceptual insight and clarity. Modern deep reinforcement learning systems might implicitly learn compositional primitives within their multilayerpolicy networks. Inspired by these models, we propose and analyse a model of deep policy gradient learning of simple compositional reinforcement learning tasks. Using the tools of statistical physics, we solve for exact learning dynamics and characterise different learning strategies including primitives pre-training, in which task primitives are studied individually before learning compositional tasks. We find a complex interplay between task complexity and the efficacy of shaping strategies. Overall, our theory provides an analytical understanding of the benefits of shaping in a class of compositional tasks and a quantitative account of how training protocols can disclose useful task primitives, ultimately yielding faster and more robust learning.

</details>

## 158. Why do Variational Autoencoders Really Promote Disentanglement?

<details>

<summary>Abstract</summary>

Despite not being designed for this purpose, the use of variational autoencoders (VAEs) has proven remarkably effective for disentangled representation learning (DRL). Recent research attributes this success to certain characteristics of the loss function that prevent latent space rotation, or hypothesize about the orthogonality properties of the decoder by drawing parallels with principal component analysis (PCA). This hypothesis, however, has only been tested experimentally for linear VAEs, and the theoretical justification still remains an open problem. Moreover, since real-world VAEs are often inherently non-linear due to the use of neural architectures, understanding DRL capabilities of real-world VAEs remains a critical task. Our work takes a step towards understanding disentanglement in real-world VAEs to theoretically establish how the orthogonality properties of the decoder promotes disentanglement in practical applications. Complementary to our theoretical contributions, our experimental results corroborate our analysis yields a more precise approximation of the error.

</details>

## 159. Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition

<details>

<summary>Abstract</summary>

We present a theoretical explanation of the “grokking” phenomenon (Power et al., 2022), where a model generalizes long after overfitting, for the originally-studied problem of modular addition. First, we show that early in gradient descent, so that the “kernel regime” approximately holds, no permutation-equivariant model can achieve small population error on modular addition unless it sees at least a constant fraction of all possible data points. Eventually, however, models escape the kernel regime. We show that one-hidden-layer quadratic networks that achieve zero training loss with bounded $\ell_\infty$ norm generalize well with substantially fewer training points,and further show such networks exist and can be found by gradient descent with small $\ell_\infty$ regularization. We further provide empirical evidence that these networks leave the kernel regime only after initially overfitting. Taken together, our results strongly support the case for grokking as a consequence of the transition from kernel-like behavior to limiting behavior of gradient descent on deep networks.

</details>

## 160. Learning Divergence Fields for Shift-Robust Graph Representations 🌟

<details>

<summary>Abstract</summary>

Real-world data generation often involves certain geometries (e.g., graphs) that induce instance-level interdependence. This characteristic makes the generalization of learning models more difficult due to the intricate interdependent patterns that impact data-generative distributions and can vary from training to testing. In this work, we propose a geometric diffusion model with learnable divergence fields for the challenging generalization problem with interdependent data. We generalize the diffusion equation with stochastic diffusivity at each time step, which aims to capture the multi-faceted information flows among interdependent data. Furthermore, we derive a new learning objective through causal inference, which can guide the model to learn generalizable patterns of interdependence that are insensitive across domains. Regarding practical implementation, we introduce three model instantiations that can be considered as the generalized versions of GCN, GAT, and Transformers, respectively, which possess advanced robustness against distribution shifts. We demonstrate their promising efficacy for out-of-distribution generalization on diverse real-world datasets.

</details>

## 161. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution 🌟

<details>

<summary>Abstract</summary>

Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel loss that naturally extends score matching to discrete spaces, integrates seamlessly to build discrete diffusion models, and significantly boosts performance. Experimentally, we test our Score Entropy Discrete Diffusion models (SEDD) on standard language modeling tasks. For comparable model sizes, SEDD beats existing language diffusion paradigms (reducing perplexity by 25-75\%) and is competitive with autoregressive models, in particular outperforming GPT-2. Furthermore, compared to autoregressive mdoels, SEDD generates faithful text without requiring distribution annealing techniques like temperature scaling (around 6-8× better generative perplexity than un-annealed GPT-2), can trade compute and quality (similar quality with 32× fewer network evaluations), and enables controllable infilling (matching nucleus sampling quality while enabling other strategies besides left to right prompting).

</details>

## 162. Position: Graph Foundation Models Are Already Here 🌟

<details>

<summary>Abstract</summary>

Graph Foundation Models (GFMs) are emerging as a significant research topic in the graph domain, aiming to develop graph models trained on extensive and diverse data to enhance their applicability across various tasks and domains. Developing GFMs presents unique challenges over traditional Graph Neural Networks (GNNs), which are typically trained from scratch for specific tasks on particular datasets. The primary challenge in constructing GFMs lies in effectively leveraging vast and diverse graph data to achieve positive transfer. Drawing inspiration from existing foundation models in the CV and NLP domains, we propose a novel perspective for the GFM development by advocating for a "graph vocabulary'', in which the basic transferable units underlying graphs encode the invariance on graphs. We ground the graph vocabulary construction from essential aspects including network analysis, expressiveness, and stability. Such a vocabulary perspective can potentially advance the future GFM design in line with the neural scaling laws. All relevant resources with GFM design can be found here.

</details>

## 163. Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization 🌟

<details>

<summary>Abstract</summary>

Graph out-of-distribution (OOD) generalization, aiming to generalize graph neural networks (GNNs) under distribution shifts between training and testing environments, has attracted ever-increasing attention recently. However, existing literature heavily relies on sufficient task-dependent graph labels, which are often scarce or even unavailable, limiting their applications in real-world scenarios. In this paper, we study the self-supervised graph OOD generalization problem, i.e., learning GNNs capable of achieving relatively stable performances under distribution shifts without graph labels. However, the problem remains largely unexplored, with the critical challenge that the invariant and variant information are highly entangled in graphs. To solve this problem, we propose an OOD generalized disentangled graph contrastive learning model (OOD-GCL), which is capable of learning disentangled graph-level representations with self-supervision that can handle distribution shifts between training and testing graph data. Specifically, we first introduce a disentangled graph encoder to map each input graph into the factorized graph representation. Then we propose a tailored disentangled invariant self-supervised learning module to maximize predictive ability of the representations and make sure the representations other than from one specific channel are invariant to the environments partitioned by this latent factor for excluding the information corresponding to this latent factor for disentanglement. Finally, the disentangled graph representations are fed into a linear predictor and finetuned for the downstream tasks. We provide comprehensive theoretical analyses to show that our model can learn disentangled graph representations and achieve OOD generalization. Extensive experiments on real-world datasets demonstrate the superiority of our model against state-of-the-art baselines under distribution shifts for graph classification tasks.

</details>

