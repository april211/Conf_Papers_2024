# IJCAI 2024

## 0. Guidance Graph Optimization for Lifelong Multi-Agent Path Finding

<details>

<summary>Abstract</summary>

We study how to use guidance to improve the throughput of lifelong Multi-Agent Path Finding (MAPF). Previous studies have demonstrated that, while incorporating guidance, such as highways, can accelerate MAPF algorithms, this often results in a trade-off with solution quality. In addition, how to generate good guidance automatically remains largely unexplored, with current methods falling short of surpassing manually designed ones. In this work, we introduce the guidance graph as a versatile representation of guidance for lifelong MAPF, framing Guidance Graph Optimization as the task of optimizing its edge weights. We present two GGO algorithms to automatically generate guidance for arbitrary lifelong MAPF algorithms and maps. The first method directly optimizes edge weights, while the second method optimizes an update model capable of generating edge weights. Empirically, we show that (1) our guidance graphs improve the throughput of three representative lifelong MAPF algorithms in eight benchmark maps, and (2) our update model can generate guidance graphs for as large as 93 x 91 maps and as many as 3,000 agents. We include the source code at: https://github.com/lunjohnzhang/ggo_public. All optimized guidance graphs are available online at: https://yulunzhang.net/publication/zhang2024ggo.

</details>

## 1. Machine Unlearning via Null Space Calibration

<details>

<summary>Abstract</summary>

Machine unlearning aims to enable models to forget specific data instances when receiving deletion requests. Current research centers on efficient unlearning to erase the influence of data from the model and neglects the subsequent impacts on the remaining data. Consequently, existing unlearning algorithms degrade the model's performance after unlearning, known as over-unlearning. This paper addresses this critical yet under-explored issue by introducing machine Unlearning via Null Space Calibration (UNSC), which can accurately unlearn target samples without over-unlearning. On the contrary, by calibrating the decision space during unlearning, UNSC can significantly improve the model's performance on the remaining samples. In particular, our approach hinges on confining the unlearning process to a specified null space tailored to the remaining samples, which is augmented by strategically pseudo-labeling the unlearning samples. Comparison against several established baselines affirms the superiority of our approach.

</details>

## 2. Are Logistic Models Really Interpretable?

<details>

<summary>Abstract</summary>

The demand for open and trustworthy AI models points towards widespread publishing of model weights. Consumers of these model weights must be able to act accordingly with the information provided. That said, one of the simplest AI classification models, Logistic Regression (LR), has an unwieldy interpretation of its model weights, with greater difficulties when extending LR to generalised additive models. In this work, we show via a User Study that skilled participants are unable to reliably reproduce the action of small LR models given the trained parameters. As an antidote to this, we define Linearised Additive Models (LAMs), an optimal piecewise linear approximation that augments any trained additive model equipped with a sigmoid link function, requiring no retraining. We argue that LAMs are more interpretable than logistic models -- survey participants are shown to solve model reasoning tasks with LAMs much more accurately than with LR given the same information. Furthermore, we show that LAMs do not suffer from large performance penalties in terms of ROC-AUC and calibration with respect to their logistic counterparts on a broad suite of public financial modelling data.

</details>

## 3. The Impact of Features Used by Algorithms on Perceptions of Fairness

<details>

<summary>Abstract</summary>

We investigate perceptions of fairness in the choice of features that algorithms use about individuals in a simulated gigwork employment experiment. First, a collection of experimental participants (the selectors) were asked to recommend an algorithm for making employment decisions. Second, a different collection of participants (the workers) were told about the setup, and a subset were ostensibly selected by the algorithm to perform an image labeling task. For both selector and worker participants, algorithmic choices differed principally in the inclusion of features that were non-volitional, and either directly relevant to the task, or for which relevance is not evident except for these features resulting in higher accuracy. We find that the selectors had a clear predilection for the more accurate algorithms, which they also judged as more fair. Worker sentiments were considerably more nuanced. Workers who were hired were largely indifferent among the algorithms. In contrast, workers who were not hired exhibited considerably more positive sentiments for algorithms that included non-volitional but relevant features. However, workers with disadvantaged values of non-volitional features exhibited more negative sentiment towards their use than the average, although the extent of this appears to depend considerably on the nature of such features.

</details>

## 4. FairGT: A Fairness-aware Graph Transformer 🌟

<details>

<summary>Abstract</summary>

The design of Graph Transformers (GTs) often neglects considerations for fairness, resulting in biased outcomes against certain sensitive subgroups. Since GTs encode graph information without relying on message-passing mechanisms, conventional fairness-aware graph learning methods are not directly applicable to address these issues. To tackle this challenge, we propose FairGT, a Fairness-aware Graph Transformer explicitly crafted to mitigate fairness concerns inherent in GTs. FairGT incorporates a meticulous structural feature selection strategy and a multi-hop node feature integration method, ensuring independence of sensitive features and bolstering fairness considerations. These fairness-aware graph information encodings seamlessly integrate into the Transformer framework for downstream tasks. We also prove that the proposed fair structural topology encoding with adjacency matrix eigenvector selection and multi-hop integration are theoretically effective. Empirical evaluations conducted across five real-world datasets demonstrate FairGT's superiority in fairness metrics over existing graph transformers, graph neural networks, and state-of-the-art fairness-aware graph learning approaches.

</details>

## 5. A Self-explaining Neural Architecture for Generalizable Concept Learning

<details>

<summary>Abstract</summary>

With the wide proliferation of Deep Neural Networks in high-stake applications, there is a growing demand for explainability behind their decision-making process. Concept learning models attempt to learn high-level 'concepts' - abstract entities that align with human understanding, and thus provide interpretability to DNN architectures. However, in this paper, we demonstrate that present SOTA concept learning approaches suffer from two major problems - lack of concept fidelity wherein the models fail to learn consistent concepts among similar classes and limited concept interoperability wherein the models fail to generalize learned concepts to new domains for the same task. Keeping these in mind, we propose a novel self-explaining architecture for concept learning across domains which - i) incorporates a new concept saliency network for representative concept selection, ii) utilizes contrastive learning to capture representative domain invariant concepts, and iii) uses a novel prototype-based concept grounding regularization to improve concept alignment across domains. We demonstrate the efficacy of our proposed approach over current SOTA concept learning approaches on four widely used real-world datasets. Empirical results show that our method improves both concept fidelity measured through concept overlap and concept interoperability measured through domain adaptation performance. An appendix of the paper with more comprehensive results can also be viewed at https://arxiv.org/abs/2405.00349.

</details>

## 6. On the Effects of Fairness to Adversarial Vulnerability

<details>

<summary>Abstract</summary>

Fairness and robustness are two important notions of learning models. Fairness ensures that models do not disproportionately harm (or benefit) some groups over others, while robustness measures the models' resilience against small input perturbations. While equally important properties, this paper illustrates a dichotomy between fairness and robustness, and analyzes when striving for fairness decreases the model robustness to adversarial samples. The reported analysis sheds light on the factors causing such contrasting behavior, suggesting that distance to the decision boundary across groups as a key factor. Experiments on non-linear models and different architectures validate the theoretical findings. In addition to the theoretical analysis, the paper also proposes a simple, yet effective, solution to construct models achieving good tradeoffs between fairness and robustness.

</details>

## 7. BADFSS: Backdoor Attacks on Federated Self-Supervised Learning

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) is capable of learning remarkable representations from centrally available data. Recent works further implement federated learning with SSL to learn from rapidly growing decentralized unlabeled images (e.g., from cameras and phones), often resulting from privacy constraints. Extensive attention has been paid to designing new frameworks or methods that achieve better performance for the SSL-based FL. However, such an effort has not yet taken the security of SSL-based FL into consideration. We aim to explore backdoor attacks in the context of SSL-based FL via an in-depth empirical study. In this paper, we propose a novel backdoor attack BADFSS against SSL-based FL. First, BADFSS learns a backdoored encoder via supervised contrastive learning on poison datasets constructed based on local datasets. Then, BADFSS employs attention alignment to enhance the backdoor effect and maintain the consistency between backdoored and global encoders. Moreover, we perform empirical evaluations of the proposed backdoor attacks on four datasets and compared BADFSS with three existing backdoor attacks that are transferred into federated self-supervised learning. The experiments demonstrate that BADFSS outperforms baseline methods and is effective under various settings.

</details>

## 8. PRASS: Probabilistic Risk-averse Robust Learning with Stochastic Search

<details>

<summary>Abstract</summary>

Deep learning models, despite their remarkable success in various tasks, have been shown to be vulnerable to adversarial perturbations. Although robust learning techniques that consider adversarial risks against worst-case perturbations can effectively increase a model's robustness, they may not always be the most suitable approach. This is due to the fact that in certain scenarios, perturbations are more likely to occur probabilistically rather than being intentionally crafted by attackers. To address this challenge, we propose a novel risk-averse robust learning method based on entropic value-at-risk, called PRASS (Probabilistical Risk-Averse Robust Learning with Stochastic Search). Our approach leverages principles of stochastic optimisation and considers perturbing distributions rather than solely worst-case adversaries. By applying adaptive stochastic search to parameterised distributions, we further enhance the scalability of PRASS to handle distributional robustness. Empirical experiments demonstrate that PRASS outperforms existing state-of-the-art baselines.

</details>

## 9. A General Black-box Adversarial Attack on Graph-based Fake News Detectors

<details>

<summary>Abstract</summary>

Graph Neural Network (GNN)-based fake news detectors apply various methods to construct graphs, aiming to learn distinctive news embeddings for classification. Since the construction details are unknown for attackers in a black-box scenario, it is unrealistic to conduct the classical adversarial attacks that require a specific adjacency matrix. In this paper, we propose the first general black-box adversarial attack framework, i.e., General Attack via Fake Social Interaction (GAFSI), against detectors based on different graph structures. Specifically, as sharing is an important social interaction for GNN-based fake news detectors to construct the graph, we simulate sharing behaviors to fool the detectors. Firstly, we propose a fraudster selection module to select engaged users leveraging local and global information. In addition, a post injection module guides the selected users to create shared relations by sending posts. The sharing records will be added to the social context, leading to a general attack against different detectors. Experimental results on empirical datasets demonstrate the effectiveness of GAFSI.

</details>

## 10. Bring Metric Functions into Diffusion Models

<details>

<summary>Abstract</summary>

We introduce a Cascaded Diffusion Model (Cas-DM) that improves a Denoising Diffusion Probabilistic Model (DDPM) by effectively incorporating additional metric functions in training. Metric functions such as the LPIPS loss have been proven highly effective in consistency models derived from the score matching. However, for the diffusion counterparts, the methodology and efficacy of adding extra metric functions remain unclear. One major challenge is the mismatch between the noise predicted by a DDPM at each step and the desired clean image that the metric function works well on. To address this problem, we propose Cas-DM, a network architecture that cascades two network modules to effectively apply metric functions to the diffusion model training. The first module, similar to a standard DDPM, learns to predict the added noise and is unaffected by the metric function. The second cascaded module learns to predict the clean image, thereby facilitating the metric function computation. Experiment results show that the proposed diffusion model backbone enables the effective use of the LPIPS loss, improving the image quality (FID, sFID) of diffusion models on various established benchmarks.

</details>

## 11. D3ETR: Decoder Distillation for Detection Transformer

<details>

<summary>Abstract</summary>

Although various knowledge distillation (KD) methods for CNN-based detectors have been proven effective in improving small students, build- ing baselines and recipes for DETR-based detec- tors remains a challenge. This paper concentrates on the transformer decoder of DETR-based detec- tors and explores KD methods suitable for them. However, the random order of the decoder outputs poses a challenge for knowledge distillation as it provides no direct correspondence between the pre- dictions of the teacher and the student. To this end, we propose MixMatcher that aligns the de- coder outputs of DETR-based teacher and student, by mixing two teacher-student matching strategies for combined advantages. The first strategy, Adap- tive Matching, applies bipartite matching to adap- tively match the outputs of the teacher and the stu- dent in each decoder layer. The second strategy, Fixed Matching, fixes the correspondence between the outputs of the teacher and the student with the same object queries as input, which alleviates in- stability of bipartite matching in Adaptive Match- ing. Using both strategies together produces bet- ter results than using either strategy alone. Based on MixMatcher, we devise Decoder Distillation for DEtection TRansformer (D3ETR), which dis- tills knowledge in decoder predictions and attention maps from the teacher to student. D3ETR shows superior performance on various DETR-based de- tectors with different backbones. For instance, D3ETR improves Conditional DETR-R50-C5 by 8.3 mAP under 12 epochs training setting with Conditional DETR-R101-C5 serving as the teacher. The code will be released.

</details>

## 12. Hybrid Frequency Modulation Network for Image Restoration

<details>

<summary>Abstract</summary>

Image restoration involves recovering a high-quality image from its corrupted counterpart. This paper presents an effective and efficient framework for image restoration, termed CSNet, based on ``channel + spatial" hybrid frequency modulation. Different feature channels include different degradation patterns and degrees, however, most current networks ignore the importance of channel interactions. To alleviate this issue, we propose a frequency-based channel feature modulation module to facilitate channel interactions through the channel-dimension Fourier transform. Furthermore, based on our observations, we develop a multi-scale frequency-based spatial feature modulation module to refine the direct-current component of features using extremely lightweight learnable parameters. This module contains a densely connected coarse-to-fine learning paradigm for enhancing multi-scale representation learning. In addition, we introduce a frequency-inspired loss function to achieve omni-frequency learning. Extensive experiments on nine datasets demonstrate that the proposed network achieves state-of-the-art performance for three image restoration tasks, including image dehazing, image defocus deblurring, and image desnowing. The code and models are available at https://github.com/c-yn/CSNet.

</details>

## 13. Improving Adversarial Robustness via Feature Pattern Consistency Constraint

<details>

<summary>Abstract</summary>

Convolutional Neural Networks (CNNs) are well-known for their vulnerability to adversarial attacks, posing significant security concerns. In response to these threats, various defense methods have emerged to bolster the model's robustness. However, most existing methods either focus on learning from adversarial perturbations, leading to overfitting to the adversarial examples, or aim to eliminate such perturbations during inference, inevitably increasing computational burdens. Conversely, clean training, which strengthens the model's robustness by relying solely on clean examples, can address the aforementioned issues. In this paper, we align with this methodological stream and enhance its generalizability to unknown adversarial examples. This enhancement is achieved by scrutinizing the behavior of latent features within the network. Recognizing that a correct prediction relies on the correctness of the latent feature's pattern, we introduce a novel and effective Feature Pattern Consistency Constraint (FPCC) method to reinforce the latent feature's capacity to maintain the correct feature pattern. Specifically, we propose Spatial-wise Feature Modification and Channel-wise Feature Selection to enhance latent features. Subsequently, we employ the Pattern Consistency Loss to constrain the similarity between the feature pattern of the latent features and the correct feature pattern. Our experiments demonstrate that the FPCC method empowers latent features to uphold correct feature patterns even in the face of adversarial examples, resulting in inherent adversarial robustness surpassing state-of-the-art models.

</details>

## 14. Rethinking Correlation Learning via Label Prior for Open Set Domain Adaptation

<details>

<summary>Abstract</summary>

Open Set Domain Adaptation (OSDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain, where known classes exist across domains while unknown classes are present only in the target domain. Existing methods rely on the clustering structure to identify the unknown classes, which empirically induces a large identification error if the unknown classes are a mixture of multiple components. To break through this barrier, we formulate OSDA from the view of correlation and propose a correlation metric-based framework called Balanced Correlation Learning (BCL). BCL employs Hilbert-Schmidt Independence Criterion (HSIC) to characterize the separation between unknown and known classes, where HSIC is reformulated as the nodes’ relation on graph. By considering the label prior as variable, theoretical results are derived to analytically show a sufficient condition for desired learning direction for OSDA. Methodologically, the class-balanced HSIC is proposed to preserve domain-invariant and class-discriminative features. With the guarantee of correlation learning, the entropy-based principle can effectively identify the unknown classes via uncertainty. Empirically, extensive evaluations are conducted, where BCL achieves significant performance improvements.

</details>

## 15. Revealing the Two Sides of Data Augmentation: An Asymmetric Distillation-based Win-Win Solution for Open-Set Recognition 🌟

<details>

<summary>Abstract</summary>

In this paper, we reveal the two sides of data augmentation: enhancements in closed-set recognition correlate with a significant decrease in open-set recognition. Through empirical investigation, we find that multi-sample-based augmentations would contribute to reducing feature discrimination, thereby diminishing the open-set criteria. Although knowledge distillation could impair the feature via imitation, the mixed feature with ambiguous semantics hinders the distillation. To this end, we propose an asymmetric distillation framework by feeding teacher model extra raw data to enlarge the benefit of teacher. Moreover, a joint mutual information loss and a selective relabel strategy are utilized to alleviate the influence of hard mixed samples. Our method successfully mitigates the decline in open-set and outperforms SOTAs by 2%~3% AUROC on the Tiny-ImageNet dataset and experiments on large-scale dataset ImageNet-21K demonstrate the generalization of our method.

</details>

## 16. TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism

<details>

<summary>Abstract</summary>

Table Structure Recognition (TSR) is a task aimed at converting table images into a machine-readable format (e.g. HTML), to facilitate other applications such as information retrieval. Recent works tackle this problem by identifying the HTML tags and text regions, where the latter is used for text extraction from the table document. These works however, suffer from misalignment issues when mapping text into the identified text regions. In this paper, we introduce a new TSR framework, called TFLOP (TSR Framework with LayOut Pointer mechanism), which reformulates the conventional text region prediction and matching into a direct text region pointing problem. Specifically, TFLOP utilizes text region information to identify both the table's structure tags and its aligned text regions, simultaneously. Without the need for region prediction and alignment, TFLOP circumvents the additional text region matching stage, which requires finely-calibrated post-processing. TFLOP also employs span-aware contrastive supervision to enhance the pointing mechanism in tables with complex structure. As a result, TFLOP achieves the state-of-the-art performance across multiple benchmarks such as PubTabNet, FinTabNet, and SynthTabNet. In our extensive experiments, TFLOP not only exhibits competitive performance but also shows promising results on industrial document TSR scenarios such as documents with watermarks or in non-English domain. Source code of our work is publicly available at: https://github.com/UpstageAI/TFLOP.

</details>

## 17. Invertible Residual Rescaling Models

<details>

<summary>Abstract</summary>

Invertible Rescaling Networks (IRNs) and their variants have witnessed remarkable achievements in various image processing tasks like image rescaling. However, we observe that IRNs with deeper networks are difficult to train, thus hindering the representational ability of IRNs. To address this issue, we propose Invertible Residual Rescaling Models (IRRM) for image rescaling by learning a bijection between a high-resolution image and its low-resolution counterpart with a specific distribution. Specifically, we propose IRRM to build a deep network, which contains several Residual Downscaling Modules (RDMs) with long skip connections. Each RDM consists of several Invertible Residual Blocks (IRBs) with short connections. In this way, RDM allows rich low-frequency information to be bypassed by skip connections and forces models to focus on extracting high-frequency information from the image. Extensive experiments show that our IRRM performs significantly better than other state-of-the-art methods with much fewer parameters and complexity. Particularly, our IRRM has respectively PSNR gains of at least 0.3 dB over HCFlow and IRN in the x4 rescaling while only using 60% parameters and 50% FLOPs. The code will be available at https://github.com/THU-Kingmin/IRRM.

</details>

## 18. Probabilistic Contrastive Learning for Domain Adaptation 🌟

<details>

<summary>Abstract</summary>

Contrastive learning has shown impressive success in enhancing feature discriminability for various visual tasks in a self-supervised manner, but the standard contrastive paradigm (features+l2 normalization) has limited benefits when applied in domain adaptation. We find that this is mainly because the class weights (weights of the final fully connected layer) are ignored in the domain adaptation optimization process, which makes it difficult for features to cluster around the corresponding class weights. To solve this problem, we propose the simple but powerful Probabilistic Contrastive Learning (PCL), which moves beyond the standard paradigm by removing l2 normalization and replacing the features with probabilities. PCL can guide the probability distribution towards a one-hot configuration, thus minimizing the discrepancy between features and class weights. We conduct extensive experiments to validate the effectiveness of PCL and observe consistent performance gains on five tasks, i.e., Unsupervised/Semi-Supervised Domain Adaptation (UDA/SSDA), Semi-Supervised Learning (SSL), UDA Detection and Semantic Segmentation. Notably, for UDA Semantic Segmentation on SYNTHIA, PCL surpasses the sophisticated CPSL-D by 2% in terms of mean IoU with a much lower training cost (PCL: 1*3090, 5 days v.s. CPSL-D: 4*V100, 11 days). Code is available at https://github.com/ljjcoder/Probabilistic-Contrastive-Learning.

</details>

## 19. Enhancing Boundary Segmentation for Topological Accuracy with Skeleton-based Methods

<details>

<summary>Abstract</summary>

Topological consistency plays a crucial role in the task of boundary segmentation for reticular images, such as cell membrane segmentation in neuron electron microscopic images, grain boundary segmentation in material microscopic images and road segmentation in aerial images. In these fields, topological changes in segmentation results have a serious impact on the downstream tasks, which can even exceed the misalignment of the boundary itself. To enhance the topology accuracy in segmentation results, we propose the Skea-Topo Aware loss, which is a novel loss function that takes into account the shape of each object and topological significance of the pixels. It consists of two components. First, a skeleton-aware weighted loss improves the segmentation accuracy by better modeling the object geometry with skeletons. Second, a boundary rectified term effectively identifies and emphasizes topological critical pixels in the prediction errors using both foreground and background skeletons in the ground truth and predictions. Experiments prove that our method improves topological consistency by up to 7 points in VI compared to 13 state-of-art methods, based on objective and subjective assessments across three different boundary segmentation datasets. The code is available at https://github.com/clovermini/Skea_topo.

</details>

## 20. Cross-Domain Feature Augmentation for Domain Generalization 🌟

<details>

<summary>Abstract</summary>

Domain generalization aims to develop models that are robust to distribution shifts. Existing methods focus on learning invariance across domains to enhance model robustness, and data augmentation has been widely used to learn invariant predictors, with most methods performing augmentation in the input space. However, augmentation in the input space has limited diversity whereas in the feature space is more versatile and has shown promising results. Nonetheless, feature semantics is seldom considered and existing feature augmentation methods suffer from a limited variety of augmented features. We decompose features into class-generic, class-specific, domain-generic, and domain-specific components. We propose a cross-domain feature augmentation method named XDomainMix that enables us to increase sample diversity while emphasizing the learning of invariant representations to achieve domain generalization. Experiments on widely used benchmark datasets demonstrate that our proposed method is able to achieve state-of-the-art performance. Quantitative analysis indicates that our feature augmentation approach facilitates the learning of effective models that are invariant across different domains.

</details>

## 21. Self-Promoted Clustering-based Contrastive Learning for Brain Networks Pretraining 🌟

<details>

<summary>Abstract</summary>

Rapid advancements in neuroimaging techniques, such as magnetic resonance imaging (MRI), have facilitated the acquisition of the structural and functional characteristics of the brain. Brain network analysis is one of the essential tools for exploring brain mechanisms from MRI, providing valuable insights into the brain's organization, and stimulating the understanding of brain cognition and pathology of neurodegenerative diseases. Graph Neural Networks (GNNs) are commonly used for brain network analysis, but they are limited by the scarcity of medical data. Although Graph Contrastive Learning methods have been developed to address this, they often involve graph augmentations that distort the anatomical brain structures. To address these challenges, an augmentation-free contrastive learning method, named Self-Promoted Clustering-based Contrastive Learning(SPCCL), is proposed in this paper. Specifically, by introducing a clustering-based contrastive Learning loss and a self-promoted contrastive pairs creation scheme, the proposed SPCCL can be pre-trained from additional healthy subjects' data that are relatively easier to acquire than disorder ones. The proposed SPCCL leverages these additional data with respect to the integrity of the original brain structure, making it a promising approach for effective brain network analysis. Comprehensive experiments are conducted on an open-access schizophrenic dataset, demonstrating the effectiveness of the proposed method.

</details>

## 22. Explore Internal and External Similarity for Single Image Deraining with Graph Neural Networks

<details>

<summary>Abstract</summary>

Patch-level non-local self-similarity is an important property of natural images. However, most existing methods do not consider this property into neural networks for image deraining, thus affecting recovery performance. Motivated by this property, we find that there exists significant patch recurrence property of a rainy image, that is, similar patches tend to recur many times in one image and its multi-scale images and external images. To better model this property for image detaining, we develop a multi-scale graph network with exemplars, called MSGNN, that contains two branches: 1) internal data-based supervised branch is used to model the internal relations of similar patches from the rainy image itself and its multi-scale images and 2) external data-participated unsupervised branch is used to model the external relations of the similar patches in the rainy image and exemplar. Specifically, we construct a graph model by searching the k-nearest neighboring patches from both the rainy images in a multi-scale framework and the exemplar. After obtaining the corresponding k neighboring patches from the multi-scale images and exemplar, we build a graph and aggregate them in an attentional manner so that the graph can provide more information from similar patches for image deraining. We embed the proposed graph in a deep neural network and train it in an end-to-end manner. Extensive experiments demonstrate that the proposed algorithm performs favorably against eight state-of-the-art methods on five public synthetic datasets and one real-world dataset. The source codes will be available at https://github.com/supersupercong/MSGNN.

</details>

## 23. Optimal Graph Learning and Nuclear Norm Maximization for Deep Cross-Domain Robust Label Propagation 🌟

<details>

<summary>Abstract</summary>

Domain adaptation aims to achieve label transfer from a labeled source domain to an unlabeled target domain, where the two domains exhibit different distributions. Existing methods primarily concentrate on designing a feature extractor to learn better domain-invariant features, along with developing an effective classifier for reliable predictions. In this paper, we introduce optimal graph learning to generate a cross-domain graph that effectively connects the two domains, and two domain-specific graphs to capture domain-specific structures. On the one hand, we incorporate the three graphs into the label propagation (LP) classifier to enhance its robustness to distribution difference. On the other hand, we leverage the three graphs to introduce graph embedding losses, promoting the learning of locally discriminative and domain-invariant features. Furthermore, we maximize the nuclear norm of predictions in LP to enhance class diversity, thereby improving its robustness to class imbalance problem. Correspondingly, we develop an efficient algorithm to solve the associated optimization problem. Finally, we integrate the proposed LP and graph embedding losses into a deep neural network, resulting in our proposed deep cross-domain robust LP. Extensive experiments conducted on three cross-domain benchmark datasets demonstrate that our proposed approach could outperform existing state-of-the-art domain adaptation methods.

</details>

## 24. Boosting Diffusion Models with an Adaptive Momentum Sampler

<details>

<summary>Abstract</summary>

Diffusion probabilistic models (DPMs) have been shown to generate high-quality images without the need for delicate adversarial training. The sampling process of DPMs is mathematically similar to Stochastic Gradient Descent (SGD), with both being iteratively updated with a function increment. Building on this, we present a novel reverse sampler for DPMs in this paper, drawing inspiration from the widely-used Adam optimizer. Our proposed sampler can be readily applied to a pre-trained diffusion model, utilizing momentum mechanisms and adaptive updating to enhance the generated image's quality. By effectively reusing update directions from early steps, our proposed sampler achieves a better balance between high-level semantics and low-level details. Additionally, this sampler is flexible and can be easily integrated into pre-trained DPMs regardless of the sampler used during training. Our experimental results on multiple benchmarks demonstrate that our proposed reverse sampler yields remarkable improvements over different baselines.

</details>

## 25. Spear: Evaluate the Adversarial Robustness of Compressed Neural Models

<details>

<summary>Abstract</summary>

As Artificial Intelligence evolves, the neural models vulnerable to adversarial attacks may produce fatal results in critical applications. This paper mainly discusses the robustness of the compressed neural models facing adversarial attacks. A few studies discuss the interaction between model compression and adversarial attack. However, they focus on the robustness against the traditional attacks designed for the dense models, not the attacks intended explicitly for the compressed models, using sparsity and quantization techniques. Compressed models often have fewer parameters and smaller sizes that are more friendly to resource-limited devices than dense models, so they are widely deployed in various edge and mobile devices. However, introducing the sparsity and quantization into neural models further imposes higher attack risks. A specific adversarial attack method (Spear) is proposed to generate the particular adversarial attack samples for evaluating the robustness of the compressed models. The Spear attack finds minimal perturbations to create the attack samples to maximize the different behaviors between the compressed and dense reference models. We demonstrate the proposed Spear attack technique can generally be applied to various networks and tasks through quantitative and ablation experiments.

</details>

## 26. Towards Dynamic-Prompting Collaboration for Source-Free Domain Adaptation

<details>

<summary>Abstract</summary>

In domain adaptation, challenges such as data privacy constraints can impede access to source data, catalyzing the development of source-free domain adaptation (SFDA) methods. However, current approaches heavily rely on models trained on source data, posing the risk of overfitting and suboptimal generalization.This paper introduces a dynamic prompt learning paradigm that harnesses the power of large-scale vision-language models to enhance the semantic transfer of source models. Specifically, our approach fosters robust and adaptive collaboration between the source-trained model and the vision-language model, facilitating the reliable extraction of domain-specific information from unlabeled target data, while consolidating domain-invariant knowledge. Without the need for accessing source data, our method amalgamates the strengths inherent in both traditional SFDA approaches and vision-language models, formulating a collaborative framework for addressing SFDA challenges. Extensive experiments conducted on three benchmark datasets showcase the superiority of our framework over previous SOTA methods.

</details>

## 27. A Fourier Perspective of Feature Extraction and Adversarial Robustness

<details>

<summary>Abstract</summary>

Adversarial robustness and interpretability are longstanding challenges of computer vision. Deep neural networks are vulnerable to adversarial perturbations that are incomprehensible and imperceptible to humans. However, the opaqueness of networks prevents one from theoretically addressing adversarial robustness. As a human-comprehensible approach, the frequency perspective has been adopted in recent works to investigate the properties of neural networks and adversarial examples. In this paper, we investigate the frequency properties of feature extraction and analyze the stability of different frequency features when attacking different frequencies. Therefore, we propose an attack method, F-PGD, based on the projected gradient descent to attack the specified frequency bands. Utilizing this method, we find many intriguing properties of neural networks and adversarial perturbations. We experimentally show that contrary to the low-frequency bias of neural networks, the effective features of the same class are distributed across all frequency bands. Meanwhile, the high-frequency features often dominate when the neural networks make conflicting decisions on different frequency features. Furthermore, the attack experiments show that the low-frequency features are more robust to the attacks on different frequencies, but the interference to the high frequencies makes the network unable to make the right decision. These properties indicate that the decision-making process of neural networks tends to use as few low-frequency features as possible and cannot integrate features of different frequencies.

</details>

## 28. Continual Compositional Zero-Shot Learning

<details>

<summary>Abstract</summary>

Compositional Zero-Shot Learning (CZSL) aims to recognize unseen compositions with the knowledge learned from seen compositions, where each composition is composed of two primitives (attribute and object). However, existing CZSL methods are designed to learn compositions from fixed primitive set, which cannot handle the continually expanding primitive set in real-world applications. In this paper, we propose a new CZSL setting, named Continual Compositional Zero-Shot Learning (CCZSL), which requires the model to recognize unseen compositions composed of learned primitive set while continually increasing the size of learned primitive set. Contextuality and catastrophic forgetting are the main issues to be addressed in this setting. Specifically, we capture similar contextuality in compositions through several learnable Super-Primitives that can modify the invariant primitive embedding to better adapt the contextuality in the corresponding composition. Then we introduce a dual knowledge distillation loss which aims at maintaining old knowledge learned from previous sessions and avoiding overfitting of new session. We design the CCZSL evaluation protocol and conduct extensive experiments on widely used benchmarks, demonstrating the superiority of our method compared to the state-of-the-art CZSL methods.

</details>

## 29. Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation

<details>

<summary>Abstract</summary>

Stock investment recommendation is crucial for guiding investment decisions and managing portfolios. Recent studies have demonstrated the potential of temporal-relational models (TRM) to yield excess investment returns. However, in the complicated finance ecosystem, the current TRM suffer from both the intrinsic temporal bias from the low signal-to-noise ratio (SNR) and the relational bias caused by utilizing inappropriate relational topologies and propagation mechanisms. Moreover, the distribution shifts behind macro-market scenarios invalidate the underlying i.i.d. assumption and limit the generalization ability of TRM. In this paper, we pioneer the impact of the above issues on the effective learning of temporal-relational patterns and propose an Automatic De-Biased Temporal-Relational Model (ADB-TRM) for stock recommendation. Specifically, ADB-TRM consists of three main components, i.e., (i) a meta-learned architecture forms a dual-stage training process, with the inner part ameliorating temporal-relational bias and the outer meta-learner counteracting distribution shifts, (ii) automatic adversarial sample generation guides the model adaptively to alleviate bias and enhance its profiling ability through adversarial training, and (iii) global-local interaction helps seek relative invariant stock embeddings from local and global distribution perspectives to mitigate distribution shifts. Experiments on three datasets from distinct stock markets show that ADB-TRM excels state-of-the-arts over 28.41% and 9.53% in terms of cumulative and risk-adjusted returns.

</details>

## 30. Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting

<details>

<summary>Abstract</summary>

Occupational skill demand (OSD) forecasting seeks to predict dynamic skill demand specific to occupations, beneficial for employees and employers to grasp occupational nature and maintain a competitive edge in the rapidly evolving labor market. Although recent research has proposed data-driven techniques for forecasting skill demand, the focus has remained predominantly on overall trends rather than occupational granularity. In this paper, we propose a novel Pre-training Enhanced Dynamic Graph Autoencoder (Pre-DyGAE), forecasting skill demand from an occupational perspective. Specifically, we aggregate job descriptions (JDs) by occupation and segment them into several timestamps. Subsequently, in the initial timestamps, we pre-train a graph autoencoder (GAE), consisting of a semantically-aware cross-attention enhanced uncertainty-aware encoder and decoders for link prediction and edge regression to achieve graph reconstruction. In particular, we utilize contrastive learning on skill cooccurrence clusters to solve the data sparsity and a unified Tweedie and ranking loss for predicting the imbalanced distribution. Afterward, we incorporate an adaptive temporal encoding unit and a temporal shift module into GAE to achieve a dynamic GAE (DyGAE). Furthermore, we fine-tune the DyGAE with a two-stage optimization strategy and infer future representations. Extensive experiments on four real-world datasets validate the effectiveness of Pre-DyGAE compared with state-of-the-art baselines.

</details>

## 31. DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives 🌟

<details>

<summary>Abstract</summary>

Graph Convolutional Networks (GCNs) have become pivotal in recommendation systems for learning user and item embeddings by leveraging the user-item interaction graph's node information and topology. However, these models often face the famous over-smoothing issue, leading to indistinct user and item embeddings and reduced personalization. Traditional desmoothing methods in GCN-based systems are model-specific, lacking a universal solution. This paper introduces a novel, model-agnostic approach named Desmoothing Framework for GCN-based Recommendation Systems (DGR). It effectively addresses over-smoothing on general GCN-based recommendation models by considering both global and local perspectives. Specifically, we first introduce vector perturbations during each message passing layer to penalize the tendency of node embeddings approximating overly to be similar with the guidance of the global topological structure. Meanwhile, we further develop a tailored-design loss term for the readout embeddings to preserve the local collaborative relations between users and their neighboring items. In particular, items that exhibit a high correlation with neighboring items are also incorporated to enhance the local topological information. To validate our approach, we conduct extensive experiments on 5 benchmark datasets based on 5 well-known GCN-based recommendation models, demonstrating the effectiveness and generalization of our proposed framework. Our code is available at https://github.com/me-sonandme/DGR.

</details>

## 32. Real-World Networks Are Low-Dimensional: Theoretical and Practical Assessment

<details>

<summary>Abstract</summary>

Recent empirical evidence suggests that real-world networks have very low underlying dimensionality. We provide a theoretical explanation for this phenomenon as well as develop a linear-time algorithm for detecting the underlying dimensionality of such networks. Our theoretical analysis considers geometric inhomogeneous random graphs (GIRGs), a geometric random graph model, which captures a variety of properties observed in real-world networks. These properties include a heterogeneous degree distribution and non-vanishing clustering coefficient, which is the probability that two random neighbors of a vertex are adjacent. Our first result shows that the clustering coefficient of GIRGs scales inverse exponentially with respect to the number of dimensions d, when the latter is at most logarithmic in n, the number of vertices. Hence, for a GIRG to behave like many real-world networks and have a non-vanishing clustering coefficient, it must come from a geometric space of o(log n) dimensions. Our analysis on GIRGs allows us to obtain a linear-time algorithm for determining the dimensionality of a network. Our algorithm bridges the gap between theory and practice, as it comes with a rigorous proof of correctness and yields results comparable to prior empirical approaches, as indicated by our experiments on real-world instances. The efficiency of our algorithm makes it applicable to very large data-sets. We conclude that very low dimensionalities (from 1 to 10) are needed to explain properties of real-world networks.

</details>

## 33. SVD-AE: Simple Autoencoders for Collaborative Filtering

<details>

<summary>Abstract</summary>

Collaborative filtering (CF) methods for recommendation systems have been extensively researched, ranging from matrix factorization and autoencoder-based to graph filtering-based methods. Recently, lightweight methods that require almost no training have been recently proposed to reduce overall computation. However, existing methods still have room to improve the trade-offs among accuracy, efficiency, and robustness. In particular, there are no well-designed closed-form studies for balanced CF in terms of the aforementioned trade-offs. In this paper, we design SVD-AE, a simple yet effective singular vector decomposition (SVD)-based linear autoencoder, whose closed-form solution can be defined based on SVD for CF. SVD-AE does not require iterative training processes as its closed-form solution can be calculated at once. Furthermore, given the noisy nature of the rating matrix, we explore the robustness against such noisy interactions of existing CF methods and our SVD-AE. As a result, we demonstrate that our simple design choice based on truncated SVD can be used to strengthen the noise robustness of the recommendation while improving efficiency. Code is available at https://github.com/seoyoungh/svd-ae.

</details>

## 34. Exploring the Role of Node Diversity in Directed Graph Representation Learning

<details>

<summary>Abstract</summary>

Many methods of Directed Graph Neural Networks (DGNNs) are designed to equally treat nodes in the same neighbor set (i.e., out-neighbor set and in-neighbor set) for every node, without considering the node diversity in directed graphs, so they are often unavailable to adaptively acquire suitable information from neighbors of different directions. To alleviate this issue, in this paper, we investigate a new way to first consider node diversity for representation learning on directed graphs, i.e., neighbor diversity and degree diversity, and then propose a new NDDGNN framework to adaptively assign weights to both outgoing information and incoming information at the node level. Extensive experiments on seven real-world datasets validate the superior performance of our method compared to state-of-the-art methods in terms of both node classification and link prediction tasks.

</details>

## 35. Multiplex Graph Representation Learning via Bi-level Optimization 🌟

<details>

<summary>Abstract</summary>

Many multiplex graph representation learning (MGRL) methods have been demonstrated to 1) ignore the globally positive and negative relationships among node features; and 2) usually utilize the node classification task to train both graph structure learning and representation learning parameters, and thus resulting in the problem of edge starvation. To address these issues, in this paper, we propose a new MGRL method based on the bi-level optimization. Specifically, in the inner level, we optimize the self-expression matrix to capture the globally positive and negative relationships among nodes, as well as complement them with the local relationships in graph structures. In the outer level, we optimize the parameters of the graph convolutional layer to obtain discriminative node representations. As a result, the graph structure optimization does not depend on the node classification task, which solves the edge starvation problem. Extensive experiments show that our model achieves the superior performance on node classification tasks on all datasets.

</details>

## 36. Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs 🌟

<details>

<summary>Abstract</summary>

Graph classification benchmarks, vital for assessing and developing graph neural network (GNN) models, have recently been scrutinized, as simple methods like MLPs have demonstrated comparable performance. This leads to an important question: Do these benchmarks effectively distinguish the advancements of GNNs over other methodologies? If so, how do we quantitatively measure this effectiveness? In response, we first propose an empirical protocol based on a fair benchmarking framework to investigate the performance discrepancy between simple methods and GNNs. We further propose a novel metric to quantify the dataset effectiveness by considering both dataset complexity and model performance. To the best of our knowledge, our work is the first to thoroughly study and provide an explicit definition for dataset effectiveness in the graph learning area. Through testing across 16 real-world datasets, we found our metric to align with existing studies and intuitive assumptions. Finally, we explore the causes behind the low effectiveness of certain datasets by investigating the correlation between intrinsic graph properties and class labels, and we developed a novel technique supporting the correlation-controllable synthetic dataset generation. Our findings shed light on the current understanding of benchmark datasets, and our new platform could fuel the future evolution of graph classification benchmarks.

</details>

## 37. Gradformer: Graph Transformer with Exponential Decay 🌟

<details>

<summary>Abstract</summary>

Graph Transformers (GTs) have demonstrated their advantages across a wide range of tasks. However, the self-attention mechanism in GTs overlooks the graph's inductive biases, particularly biases related to structure, which are crucial for the graph tasks. Although some methods utilize positional encoding and attention bias to model inductive biases, their effectiveness is still suboptimal analytically. Therefore, this paper presents Gradformer, a method innovatively integrating GT with the intrinsic inductive bias by applying an exponential decay mask to the attention matrix. Specifically, the values in the decay mask matrix diminish exponentially, correlating with the decreasing node proximities within the graph structure. This design enables Gradformer to retain its ability to capture information from distant nodes while focusing on the graph's local details. Furthermore, Gradformer introduces a learnable constraint into the decay mask, allowing different attention heads to learn distinct decay masks. Such an design diversifies the attention heads, enabling a more effective assimilation of diverse structural information within the graph. Extensive experiments on various benchmarks demonstrate that Gradformer consistently outperforms the Graph Neural Network and GT baseline models in various graph classification and regression tasks. Additionally, Gradformer has proven to be an effective method for training deep GT models, maintaining or even enhancing accuracy compared to shallow models as the network deepens, in contrast to the significant accuracy drop observed in other GT models. Codes are available at https://github.com/LiuChuang0059/Gradformer.

</details>

## 38. Where to Mask: Structure-Guided Masking for Graph Masked Autoencoders 🌟

<details>

<summary>Abstract</summary>

Graph masked autoencoders (GMAE) have emerged as a significant advancement in self-supervised pre-training for graph-structured data. Previous GMAE models primarily utilize a straightforward random masking strategy for nodes or edges during training. However, this strategy fails to consider the varying significance of different nodes within the graph structure. In this paper, we investigate the potential of leveraging the graph's structural composition as a fundamental and unique prior in the masked pre-training process. To this end, we introduce a novel structure-guided masking strategy (i.e., StructMAE), designed to refine the existing GMAE models. StructMAE involves two steps: 1) Structure-based Scoring: Each node is evaluated and assigned a score reflecting its structural significance. Two distinct types of scoring manners are proposed: predefined and learnable scoring. 2) Structure-guided Masking: With the obtained assessment scores, we develop an easy-to-hard masking strategy that gradually increases the structural awareness of the self-supervised reconstruction task. Specifically, the strategy begins with random masking and progresses to masking structure-informative nodes based on the assessment scores. This design gradually and effectively guides the model in learning graph structural information. Furthermore, extensive experiments consistently demonstrate that our StructMAE method outperforms existing state-of-the-art GMAE models in both unsupervised and transfer learning tasks. Codes are available at https: //github.com/LiuChuang0059/StructMAE.

</details>

## 39. DGCD: An Adaptive Denoising GNN for Group-level Cognitive Diagnosis 🌟

<details>

<summary>Abstract</summary>

Group-level cognitive diagnosis, pivotal in intelligent education, aims to effectively assess group-level knowledge proficiency by modeling the learning behaviors of individuals within the group. Existing methods typically conceptualize the group as an abstract entity or aggregate the knowledge levels of all members to represent the group’s overall ability. However, these methods neglect the high-order connectivity among groups, students, and exercises within the context of group learning activities, along with the noise present in their interactions, resulting in less robust and suboptimal diagnosis performance. To this end, in this paper, we propose DGCD, an adaptive Denoising graph neural network for realizing effective Group-level Cognitive Diagnosis. Specifically, we first construct a group-student-exercise (GSE) graph to explicitly model higher-order connectivity among groups, students, and exercises, contributing to the acquisition of informative representations. Then, we carefully design an adaptive denoising module, integrated into the graph neural network, to model the reliability distribution of student-exercise edges for mining purer interaction features. In particular, edges of lower reliability are more prone to exclusion, thereby reducing the impact of noisy interactions. Furthermore, recognizing the relational imbalance in the GSE graph, which could potentially introduce bias during message passing, we propose an entropy-weighted balance module to mitigate such bias. Finally, extensive experiments conducted on four real-world educational datasets clearly demonstrate the effectiveness of our proposed DGCD model. The code is available at https://github.com/BIMK/Intelligent-Education/tree/main/DGCD.

</details>

## 40. SeeDRec: Sememe-based Diffusion for Sequential Recommendation 🌟

<details>

<summary>Abstract</summary>

Inspired by the power of Diffusion Models (DM) verified in various fields, some pioneering works have started to explore DM in recommendation. However, these prevailing endeavors commonly implement diffusion on item indices, leading to the increasing time complexity, the lack of transferability, and the inability to fully harness item semantic information. To tackle these challenges, we propose SeeDRec, a sememe-based diffusion framework for sequential recommendation (SR). Specifically, inspired by the notion of sememe in NLP, SeeDRec first defines a similar concept of recommendation sememe to represent the minimal interest unit and upgrades the specific diffusion objective from the item level to the sememe level. With the Sememe-to-Interest Diffusion Model (S2IDM), SeeDRec can accurately capture the user's diffused interest distribution learned from both local interest evolution and global interest generalization while maintaining low computational costs. Subsequently, an Interest-aware Prompt-enhanced (IPE) strategy is proposed to better guide each user's sequential behavior modeling via the learned user interest distribution. Extensive experiments on nine SR datasets and four cross-domain SR datasets verify its effectiveness and universality. The code is available in https://github.com/hulkima/SeeDRec.

</details>

## 41. Graph Collaborative Expert Finding with Contrastive Learning

<details>

<summary>Abstract</summary>

In Community Question Answering (CQA) websites, most current expert finding methods often model expert embeddings from textual features and optimize them with expert-question first-order interactions, i.e., this expert has answered this question. In this paper, we try to address the limitation of current models that typically neglect the intrinsic high-order connectivity within expert-question interactions, which is pivotal for collaborative effects. We introduce an innovative and simple approach: by conceptualizing expert-question interactions as a bipartite graph, and then we propose a novel graph-based expert finding method based on contrastive learning to effectively capture both first-order and intricate high-order connectivity, named CGEF. Specifically, we employ a question encoder to model questions from titles and employ the graph attention network to recursively propagate embeddings. Besides, to alleviate the problem of sparse interactions, we devise two auxiliary tasks to enhance expert modeling. First, we generate multiple views of one expert, including: 1) behavior-level augmentation drops interaction edges randomly in the graph; 2) interest-level augmentation randomly replaces question titles with tags in the graph. Then we maximize the agreement between one expert and the corresponding augmented expert on a specific view. In this way, the model can effectively inject collaborative signals into expert modeling. Extensive experiments on six CQA datasets demonstrate significant improvements compared with recent methods.

</details>

## 42. Counterfactual User Sequence Synthesis Augmented with Continuous Time Dynamic Preference Modeling for Sequential POI Recommendation

<details>

<summary>Abstract</summary>

With the proliferation of Location-based Social Networks (LBSNs), user check-in data at Points-of-Interest (POIs) has surged, offering rich insights into user preferences. However, sequential POI recommendation systems always face two pivotal challenges. A challenge lies in the difficulty of modeling time in a discrete space, which fails to accurately capture the dynamic nature of user preferences. Another challenge is the inherent sparsity and noise in continuous POI recommendation, which hinder the recommendation process. To address these challenges, we propose counterfactual user sequence synthesis with continuous time dynamic preference modeling (CussCtpm). CussCtpm innovatively combines Gated Recurrent Unit (GRU) with neural Ordinary Differential Equations (ODEs) to model user preferences in a continuous time framework. CussCtpm captures user preferences at both the POI-level and interest-level, identifying deterministic and non-deterministic preference concepts. Particularly at the interest-level, we employ GRU and neural ODEs to model users' dynamic preferences in continuous space, aiming to capture finer-grained shifts in user preferences over time. Furthermore, CussCtpm utilizes counterfactual data augmentation to generate counterfactual positive and negative user sequences. Our extensive experiments on two widely-used public datasets demonstrate that CussCtpm outperforms several advanced baseline models.

</details>

## 43. Learning Multi-Granularity and Adaptive Representation for Knowledge Graph Reasoning

<details>

<summary>Abstract</summary>

Knowledge graph reasoning (KGR) aims to infer new factual triples from existing knowledge graphs (KGs). Recently, a new category of methods, possessing both transductive and inductive reasoning capabilities, has been proposed to tackle this task via learning entity-independent representations from local neighboring structures. However, these methods are plagued by inefficiency issues and they exclusively capture evidence from well-designed local structures, ignoring the correlation between the query and different structures within KGs. In this work, we first propose a novel multi-granularity and adaptive representation framework, MulGA, exploiting the connectivity subgraph to uniformly and hierarchically model query-related triples, relation paths, and subgraphs without explicitly extracting any graph structure, hence mitigating inefficiency issues. Second, we introduce a message-passing mechanism across connectivity subgraphs, facilitating all entities to attain query-related structural representations of diverse granularity levels, i.e., triple and relation paths of different lengths. Third, we design a self-attention-based merging mechanism that allocates weights to different granularities and then consolidates them into subgraph granularity representations for reasoning. The systematic experiments have been conducted on 15 benchmarks and MulGA achieves a significant improvement in MRR by an average of 1.5% on transductive and 2.7% on inductive tasks than existing state-of-the-art methods. Moreover, MulGA boasts faster convergence speed, competitive inference time, and alleviates the over-smoothing prevalent in graph neural networks.

</details>

## 44. Unsupervised Deep Graph Structure and Embedding Learning 🌟

<details>

<summary>Abstract</summary>

Graph Neural Network (GNN) is powerful in graph embedding learning, but its performance has been shown to be heavily degraded under adversarial attacks. Deep graph structure learning (GSL) is proposed to defend attack by jointly learning graph structure and graph embedding, typically in node classification task. Label supervision is expensive in real-world applications, and thus unsupervised GSL is more challenging and still remains less studied. To fulfill this gap, this paper proposes a new unsupervised GSL method, i.e., unsupervised property GNN (UPGNN). UPGNN first refines graph structure by exploring properties of low rank, sparsity, feature smoothness. UPGNN employs graph mutual information loss to learn graph embedding by maximizing its correlation with refined graph. The proposed UPGNN learns graph structure and embedding without label supervision, and thus can be applied various downstream tasks. We further propose Accelerated UPGNN (AUPGNN) to reduce computational complexity, providing a efficient alternative to UPGNN. Our extensive experiments on node classification and clustering demonstrate the effectiveness of the proposed method over the state-of-the-arts especially under heavy perturbation.

</details>

## 45. Anomaly Subgraph Detection through High-Order Sampling Contrastive Learning

<details>

<summary>Abstract</summary>

Anomaly subgraph detection is a crucial task in various real-world applications, including identifying high-risk areas, detecting river pollution, and monitoring disease outbreaks. Early traditional graph-based methods can obtain high-precision detection results in scenes with small-scale graphs and obvious anomaly features. Most existing anomaly detection methods based on deep learning primarily concentrate on identifying anomalies at the node level, while neglecting to detect anomaly groups in the internal structure. In this paper, we propose a novel end-to-end Graph Neural Network (GNN) based anomaly subgraph detection approach(ASD-HC) in graph-structured data. 1)We propose a high-order neighborhood sampling strategy to construct our node and k-order neighbor-subgraph instance pairs. 2)Anomaly features of nodes are captured through a self-supervised contrastive learning model. 3) Detecting the maximum connected anomaly subgraph is performed by integrating the Non-parameter Graph Scan statistics and a Random Walk module. We evaluate ASD-HC against five state-of-the-art baselines using five benchmark datasets. ASD-HC outperforms the baselines by over 13.01% in AUC score. Various experiments demonstrate that our approach effectively detects anomaly subgraphs within large-scale graphs.

</details>

## 46. HeterGCL: Graph Contrastive Learning Framework on Heterophilic Graph 🌟

<details>

<summary>Abstract</summary>

Graph Contrastive Learning (GCL) has attracted significant research attention due to its self-supervised ability to learn robust node representations. Unfortunately, most methods primarily focus on homophilic graphs, rendering them less effective for heterophilic graphs. In addition, the complexity of node interactions in heterophilic graphs poses considerable challenges to augmentation schemes, coding architectures, and contrastive designs for traditional GCL. In this work, we propose HeterGCL, a novel graph contrastive learning framework with structural and semantic learning to explore the true potential of GCL on heterophilic graphs. Specifically, We abandon the random augmentation scheme that leads to the destruction of the graph structure, instead introduce an adaptive neighbor aggregation strategy (ANA) to extract topology-supervised signals from neighboring nodes at different distances and explore the structural information with an adaptive local-to-global contrastive loss. In the semantic learning module, we jointly consider the original nodes' features and the similarity between nodes in the latent feature space to explore hidden associations between nodes. Experimental results on homophilic and heterophilic graphs demonstrate that HeterGCL outperforms existing self-supervised and semi-supervised baselines across various downstream tasks.

</details>

## 47. Spatial-Temporal Perceiving: Deciphering User Hierarchical Intent in Session-Based Recommendation

<details>

<summary>Abstract</summary>

Session-based recommendation (SBR) aims to predict the next-interacted item based on anonymous users' behavior sequences. The main challenge is how to recognize the user intent with limited interactions to achieve a more accurate inference of user behavior. Existing works usually regard several consecutive items in the current session as intent. However, we argue such intent generation based on temporal transition ignores the fact that each item also has its semantically connected items in the feature space, which can be regarded as spatial intent. The limited consideration of intent fails to capture complex behavioral patterns in real-world scenarios, leading to sub-optimal solutions. To address this issue, we propose the Hierarchical Intent Perceiving Contrastive Learning Framework (HearInt) for SBR, which proposes a hierarchical consideration of intents from both temporal and spatial perspective. Specifically, we first propose that the user's temporal intents are mutually exclusive while the spatial intents are mutually compatible. Following these analyses, we design a Temporal Intent Decoupling module to mitigate the mutual influence of long-term and short-term intents, and a Cross-scale Contrastive Learning task to enhance the consistency of intents across different spatial scales. Experimental results on three real-world datasets exhibit that HearInt achieves state-of-the-art performance.

</details>

## 48. WeatherGNN: Exploiting Meteo- and Spatial-Dependencies for Local Numerical Weather Prediction Bias-Correction

<details>

<summary>Abstract</summary>

Due to insufficient local area information, numerical weather prediction (NWP) may yield biases for specific areas. Previous studies correct biases mainly by employing handcrafted features or applying data-driven methods intuitively, overlooking the complicated dependencies between weather factors and between areas. To address this issue, we propose WeatherGNN, a local NWP bias-correction method that utilizes Graph Neural Networks (GNNs) to exploit meteorological dependencies and spatial dependencies under the guidance of domain knowledge. Specifically, we introduce a factor GNN to capture area-specific meteorological dependencies adaptively based on spatial heterogeneity and a fast hierarchical GNN to capture dynamic spatial dependencies efficiently guided by Tobler's first and second laws of geography. Our experimental results on two real-world datasets demonstrate that WeatherGNN achieves the state-of-the-art performance, outperforming the best baseline with an average of 4.75 % on RMSE.

</details>

## 49. Unsupervised Anomaly Detection via Masked Diffusion Posterior Sampling

<details>

<summary>Abstract</summary>

Reconstruction-based methods have been commonly used for unsupervised anomaly detection, in which a normal image is reconstructed and compared with the given test image to detect and locate anomalies. Recently, diffusion models have shown promising applications for anomaly detection due to their powerful generative ability. However, these models lack strict mathematical support for normal image reconstruction and unexpectedly suffer from low reconstruction quality. To address these issues, this paper proposes a novel and highly-interpretable method named Masked Diffusion Posterior Sampling (MDPS). In MDPS, the problem of normal image reconstruction is mathematically modeled as multiple diffusion posterior sampling for normal images based on the devised masked noisy observation model and the diffusion-based normal image prior under Bayesian framework. Using a metric designed from pixel-level and perceptual-level perspectives, MDPS can effectively compute the difference map between each normal posterior sample and the given test image. Anomaly scores are obtained by averaging all difference maps for multiple posterior samples. Exhaustive experiments on MVTec and BTAD datasets demonstrate that MDPS can achieve state-of-the-art performance in normal image reconstruction quality as well as anomaly detection and localization.

</details>

## 50. Robust Heterophilic Graph Learning against Label Noise for Anomaly Detection 🌟

<details>

<summary>Abstract</summary>

Given clean labels, Graph Neural Networks (GNNs) have shown promising abilities for graph anomaly detection. However, real-world graphs are inevitably noisy labeled, which drastically degrades the performance of GNNs. To alleviate it, some studies follow the local consistency (a.k.a homophily) assumption to conduct neighborhood-based label noise correction, and to dense raw graphs using raw features or representations learned by poisoned labels. But for the anomaly detection task, the graph is not always homophilic but more likely to be heterophilic, which would corrupt the above assumption due to complicating connection patterns and impairing the effects of message passing. To this end, we propose a novel label noise-resistant graph learning (NRGL) framework, which facilitates robust graph learning from the perspectives of structure augmentation and fine-grained label governance. Specifically, we first present an investigation to verify that increasing graph homophily could help resist label noise. Based on the observation, an unsupervised contrastive learning paradigm is then introduced so well that it cannot only adaptively extract the dual views from the raw graph as structure augmentation, but also enhance the robustness of node representations. Next, given robust node representations, the noisy labels are divided into three candidate sets based on the small-loss criterion for fine-grained noise governance. Furthermore, a node sampler is designed to take structure importance, class frequency, and confidence score into consideration, which helps select reliable and important nodes for training. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

</details>

## 51. Learning Fair Representations for Recommendation via Information Bottleneck Principle 🌟

<details>

<summary>Abstract</summary>

User-oriented recommender systems (RS) characterize users' preferences based on observed behaviors and are widely deployed in personalized services. However, RS may unintentionally capture biases related to sensitive attributes (e.g., gender) from behavioral data, leading to unfair issues and discrimination against particular groups (e.g., females). Adversarial training is a popular technique for fairness-aware RS, when filtering sensitive information in user modeling. Despite advancements in fairness, achieving a good accuracy-fairness trade-off remains a challenge in adversarial training. In this paper, we investigate fair representation learning from a novel information theory perspective. Specifically, we propose a model-agnostic Fair recommendation method via the Information Bottleneck principle FairIB. The learning objective of FairIB is to maximize the mutual information between user representations and observed interactions, while simultaneously minimizing it between user representations and sensitive attributes. This approach facilitates the capturing of essential collaborative signals in user representations while mitigating the inclusion of unnecessary sensitive information. Empirical studies on two real-world datasets demonstrate the effectiveness of the proposed FairIB, which significantly improves fairness while maintaining competitive recommendation accuracy, either in single or multiple sensitive scenarios. The code is available at https://github.com/jsxie9/IJCAI_FairIB.

</details>

## 52. Graph Attention Network with High-Order Neighbor Information Propagation for Social Recommendation 🌟

<details>

<summary>Abstract</summary>

In recommender systems, graph neural networks (GNN) can integrate interactions between users and items with their attributes, which makes GNN-based methods more powerful. However, directly stacking multiple layers in a graph neural network can easily lead to over-smoothing, hence recommendation systems based on graph neural networks typically underutilize higher-order neighborhoods in their learning. Although some heterogeneous graph random walk methods based on meta-paths can achieve higher-order aggregation, the focus is predominantly on the nodes at the ends of the paths. Moreover, these methods require manually defined meta-paths, which limits the model’s expressiveness and flexibility. Furthermore, path encoding in graph neural networks usually focuses only on the sequence leading to the target node. However, real-world interactions often do not follow this strict sequence, limiting the predictive performance of sequence-based network models. These problems prevent GNN-based methods from being fully effective. We propose a Graph Attention network with Information Propagation path aggregation for Social Recommendation (GAIPSRec). Firstly, we propose a universal heterogeneous graph sampling framework that does not require manually defining meta-paths for path sampling, thereby offering greater flexibility. Moreover, our method takes into account all nodes on the aggregation path and is capable of learning information from higher-order neighbors without succumbing to over-smoothing. Finally, our method utilizes a gate mechanism to fuse sequential and non-sequential dependence in encoding path instances, allowing a more holistic view of the data. Extensive experiments on real-world datasets show that our proposed GAIPSRec improves the performance significantly and outperforms state-of-the-art methods.

</details>

## 53. Joint Domain Adaptive Graph Convolutional Network 🌟

<details>

<summary>Abstract</summary>

In the realm of cross-network tasks, graph domain adaptation is an effective tool due to its ability to transfer abundant labels from nodes in the source domain to those in the target domain. Existing adversarial domain adaptation methods mainly focus on domain-wise alignment. These approaches, while effective in mitigating the marginal distribution shift between the two domains, often ignore the integral aspect of structural alignment, potentially leading to negative transfer. To address this issue, we propose a joint adversarial domain adaptive graph convolutional network (JDA-GCN) that is uniquely augmented with structural graph alignment, so as to enhance the efficacy of knowledge transfer. Specifically, we construct a structural graph to delineate the interconnections among nodes within identical categories across the source and target domains. To further refine node representation, we integrate the local consistency matrix with the global consistency matrix, thereby leveraging the learning of the sub-structure similarity of nodes to enable more robust and effective representation of nodes. Empirical evaluation on diverse real-world datasets substantiates the superiority of our proposed method, marking a significant advancement over existing state-of-the-art graph domain adaptation algorithms.

</details>

## 54. Kernel Readout for Graph Neural Networks

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) for graph classification or representation learning require a pooling operation to convert the nodes' embeddings of each graph to a vector as the graph-level representation and the operation has a significant impact on model accuracy. The paper presents a novel graph pooling method called Kernel Readout (KerRead). KerRead maps the node embeddings from the sample space with limited nodes to an augmented sample space with infinite nodes, and then calculates the inner product between some learnable adaptive centers and the augmented node embeddings, which forms a final graph-level feature vector. We apply the proposed strategy to six supervised and two unsupervised graph neural networks such as GCN, GIN, GUNet, InfoGraph, and GraphCL, and the experiments on eight benchmark datasets show that the proposed readout outperforms classical pooling methods such as Sum and seven state-of-the-art pooling methods such as SRead and Janossy GRU. Code and Appendix are both available at https://github.com/jiajunCAU/KerRead.

</details>

## 55. LG-GNN: Local-Global Adaptive Graph Neural Network for Modeling Both Homophily and Heterophily 🌟

<details>

<summary>Abstract</summary>

Most Graph Neural Networks (GNNs) are based on the homophily assumption, where nodes with the same labels or similar features tend to be connected to each other. However, real-world graphs often do not adhere to this homophily assumption. Currently, most researches aggregate multi-hop neighbor information to discover more potentially relevant nodes. However, in the aggregation process of GNNs, the difference in modeling global and local information is not considered, inevitably leading to information loss. Motivated by this limitation, we propose LG-GNN, a local-global adaptive graph neural network for modeling both homophily and heterophily. Specifically, we model the long-range structural similarity and local feature similarity between nodes from global and local perspectives, in order to capture distant dependencies in highly heterophilic networks while reducing the mixing of locally dissimilar feature nodes, thereby increasing the effectiveness of information aggregation in highly heterophilic graphs. Extensive experiments on a wide range of real-world datasets demonstrate that our proposed approach performs well in both heterophilic and homophilic graphs.

</details>

## 56. Exploring Urban Semantics: A Multimodal Model for POI Semantic Annotation with Street View Images and Place Names

<details>

<summary>Abstract</summary>

Semantic annotation for points of interest (POIs) is the process of annotating a POI with a category label, which facilitates many services related to POIs, such as POI search and recommendation. Most of the existing solutions extract features related to POIs from abundant user-generated content data (e.g., check-ins and user comments). However, such data are often difficult to obtain, especially for newly created POIs. In this paper, we aim to explore semantic annotation for POIs with limited information such as POI (place) names and geographic locations. Additionally, we have found that the street view images provide extensive visual clues about POI attributes and could be an essential supplement to limited information of POIs that enables semantic annotation. To this end, we propose a novel multimodal model for POI semantic annotation, namely M3PA, which achieves enhanced semantic annotation through fusing a POI’s textual and visual representations. Specifically, M3PA extracts visual features from street view images using a pre-trained image encoder and integrates these features to generate the visual representation of a targeted POI based on a geographic attention mechanism. Furthermore, M3PA utilizes the contextual information of neighboring POIs to extract textual features and captures their spatial relationships through geographical encoding to generate the textual representation of a targeted POI. Finally, the visual and textual representations of a POI are fused for semantic annotation. Extensive experiments with POI data from Amap validate the effectiveness of M3PA for POI semantic annotation, compared with several competitive baselines.

</details>

## 57. Make Graph Neural Networks Great Again: A Generic Integration Paradigm of Topology-Free Patterns for Traffic Speed Prediction

<details>

<summary>Abstract</summary>

Urban traffic speed prediction aims to estimate the future traffic speed for improving urban transportation services. Enormous efforts have been made to exploit Graph Neural Networks (GNNs) for modeling spatial correlations and temporal dependencies of traffic speed evolving patterns, regularized by graph topology. While achieving promising results, current traffic speed prediction methods still suffer from ignoring topology-free patterns, which cannot be captured by GNNs. To tackle this challenge, we propose a generic model for enabling the current GNN-based methods to preserve topology-free patterns. Specifically, we first develop a Dual Cross-Scale Transformer (DCST) architecture, including a Spatial Transformer and a Temporal Transformer, to preserve the cross-scale topology-free patterns and associated dynamics, respectively. Then, to further integrate both topology-regularized/-free patterns, we propose a distillation-style learning framework, in which the existing GNN-based methods are considered as the teacher model, and the proposed DCST architecture is considered as the student model. The teacher model would inject the learned topology-regularized patterns into the student model for integrating topology-free patterns. The extensive experimental results demonstrated the effectiveness of our methods.

</details>

## 58. Generalized Taxonomy-Guided Graph Neural Networks

<details>

<summary>Abstract</summary>

Graph neural networks have been demonstrated to be effective analytic apparatus for mining network data. Most real-world networks are inherently hierarchical, offering unique opportunities to acquire latent, intrinsic network organizational properties by utilizing network taxonomies. The existing approaches for learning implicit hierarchical network structures focus on introducing taxonomy to graph neural networks but often run short of exploiting the rich network semantics and structural properties in the taxonomy, resulting in poor generalizability and reusability. To address these issues, we propose generalized Taxonomy-Guided Graph Neural Networks (TG-GNN) to integrate taxonomy into network representation learning. We first construct a taxonomy representation learning module that introduces the concept of ego network to propagate and aggregate rich semantic and structural information in the taxonomy. We then design a taxonomy-guided Markov mechanism, which encapsulates taxonomy knowledge in pairwise potential functions, to refine network embeddings. Extensive experiments on various real-world networks illustrate the effectiveness of TG-GNN over the state-of-the-art methods on scenarios involving incomplete taxonomies and inductive settings.

</details>

## 59. Layered Graph Security Games

<details>

<summary>Abstract</summary>

Security games model strategic interactions in adversarial real-world applications. Such applications often involve extremely large but highly structured strategy sets (e.g., selecting a distribution over all patrol routes in a given graph). In this paper, we represent each player's strategy space using a layered graph whose paths represent an exponentially large strategy space. Our formulation entails not only classic pursuit-evasion games, but also other security games, such as those modeling anti-terrorism and logistical interdiction. We study two-player zero-sum games under two distinct utility models: linear and binary utilities. We show that under linear utilities, Nash equilibrium can be computed in polynomial time, while binary utilities may lead to situations where even computing a best-response is computationally intractable. To this end, we propose a practical algorithm based on incremental strategy generation and mixed integer linear programs. We show through extensive experiments that our algorithm efficiently computes epsilon-equilibrium for many games of interest. We find that target values and graph structure often have a larger influence on running times as compared to the size of the graph per se.

</details>

## 60. Getting More by Knowing Less: Bayesian Incentive Compatible Mechanisms for Fair Division

<details>

<summary>Abstract</summary>

We study fair resource allocation with strategic agents. It is well-known that, across multiple fundamental problems in this domain, truthfulness and fairness are incompatible. For example, when allocating indivisible goods, no truthful and deterministic mechanism can guarantee envy-freeness up to one item (EF1), even for two agents with additive valuations. Or, in cake-cutting, no truthful and deterministic mechanism always outputs a proportional allocation, even for two agents with piecewise constant valuations. Our work stems from the observation that, in the context of fair division, truthfulness is used as a synonym for Dominant Strategy Incentive Compatibility (DSIC), requiring that an agent prefers reporting the truth, no matter what other agents report. In this paper, we instead focus on Bayesian Incentive Compatible (BIC) mechanisms, requiring that agents are better off reporting the truth in expectation over other agents' reports. We prove that, when agents know a bit less about each other, a lot more is possible: using BIC mechanisms we can achieve fairness notions that are unattainable by DSIC mechanisms in both the fundamental problems of allocation of indivisible goods and cake-cutting. We prove that this is the case even for an arbitrary number of agents, as long as the agents' priors about each others' types satisfy a neutrality condition. Notably, for the case of indivisible goods, we significantly strengthen the state-of-the-art negative result for efficient DSIC mechanisms, while also highlighting the limitations of BIC mechanisms, by showing that a very general class of welfare objectives is incompatible with Bayesian Incentive Compatibility. Combined, these results give a near-complete picture of the power and limitations of BIC and DSIC mechanisms for the problem of allocating indivisible goods.

</details>

## 61. Concept-Level Causal Explanation Method for Brain Function Network Classification

<details>

<summary>Abstract</summary>

Using deep models to classify brain functional networks (BFNs) for the auxiliary diagnosis and treatment of brain diseases has become increasingly popular. However, the unexplainability of deep models has seriously hindered their applications in computer-aided diagnosis. In addition, current explanation methods mostly focus on natural images, which cannot be directly used to explain the deep model for BFN classification. In this paper, we propose a concept-level causal explanation method for BFN classification called CLCEM. First, CLCEM employs the causal learning method to extract concepts that are meaningful to humans from BFNs. Second, it aggregates the same concepts to obtain the contribution of each concept to the model output. Finally, CLCEM adds the contribution of each concept to make a diagnosis. The experimental results show that our CLCEM can not only accurately identify brain regions related to specific brain diseases but also make decisions based on the concepts of these brain regions, which enables humans to understand the decision-making process without performance degradation.

</details>

## 62. General Epistemic Abstract Argumentation Framework: Semantics and Complexity

<details>

<summary>Abstract</summary>

Epistemic Abstract Argumentation Framework (EAAF) extends Dung's framework (AAF)---a central formalism in AI for modeling disputes among agents---by allowing the representation of epistemic knowledge. In particular, EAAF augments AAF with weak and strong epistemic attacks whose intuitive meaning is that an argument a defeats an argument b by means of a weak (resp. strong) epistemic attack if a is true in every (resp. at least one) extension. So far, the semantics of EAAF has been defined only for a restricted class of frameworks, namely acyclic EAAF, where epistemic attacks do not occur in any cycle. In this paper, we provide an intuitive semantics for (general) EAAF that naturally extends that for AAF as well as that for acyclic EAAF. After providing some fundamental properties and giving an algorithm that enables the computation of EAAF semantics, by relying on state-of-the-art AAF-solvers, we investigate the complexity of canonical argumentation problems.

</details>

## 63. Revisiting Causal Discovery from a Complexity-Theoretic Perspective

<details>

<summary>Abstract</summary>

Causal discovery seeks to unveil causal relationships (represented as a so-called causal graph) from observational data. This paper investigates the complex relationship between the graph structure and the efficiency of constraint-based causal discovery algorithms. Our main contributions include (i) a near-tight characterization of which causal graphs admit a small d-separating set for each pair of vertices and thus can potentially be efficiently recovered by a constraint-based causal discovery algorithm, (ii) the explicit construction of a sequence of causal graphs on which the influential PC algorithm might need exponential time, although there is a small d-separating set between every pair of variables, and (iii) the formulation of a new causal discovery algorithm which achieves fixed-parameter running time by considering the maximum number of edge-disjoint paths between variables in the (undirected) super-structure as the parameter. A distinguishing feature of our investigation is that it is carried out within a more fine-grained model which more faithfully captures the infeasibility of performing accurate independence tests for large sets of conditioning variables.

</details>

## 64. Learning Logic Programs by Discovering Higher-Order Abstractions

<details>

<summary>Abstract</summary>

We introduce the higher-order refactoring problem, where the goal is to compress a logic program by discovering higher-order abstractions, such as map, filter, and fold. We implement our approach in Stevie, which formulates the refactoring problem as a constraint optimisation problem. Our experiments on multiple domains, including program synthesis and visual reasoning, show that refactoring can improve the learning performance of an inductive logic programming system, specifically improving predictive accuracies by 27% and reducing learning times by 47%. We also show that Stevie can discover abstractions that transfer to multiple domains.

</details>

## 65. Regression Residual Reasoning with Pseudo-labeled Contrastive Learning for Uncovering Multiple Complex Compositional Relations

<details>

<summary>Abstract</summary>

Abstract Visual Reasoning (AVR) has been widely studied in literature. Our study reveals that AVR models tend to rely on appearance matching rather than a genuine understanding of underlying rules. We hence develop a challenging benchmark, Multiple Complex Compositional Reasoning (MC2R), composed of diverse compositional rules on attributes with intentionally increased variations. It aims to identify two outliers from five given images, in contrast to single-answer questions in previous AVR tasks. To solve MC2R tasks, a Regression Residual Reasoning with Pseudo-labeled Contrastive Learning (R3PCL) is proposed, which first transforms the original problem by selecting three images following the same rule, and iteratively regresses one normal image by using the other two, allowing the model to gradually comprehend the underlying rules. The proposed PCL leverages a set of min-max operations to generate more reliable pseudo labels, and exploits contrastive learning with data augmentation on pseudo-labeled images to boost the discrimination and generalization of features. Experimental results on two AVR datasets show that the proposed R3PCL significantly outperforms state-of-the-art models.

</details>

## 66. A Logic for Reasoning about Aggregate-Combine Graph Neural Networks

<details>

<summary>Abstract</summary>

We propose a modal logic in which counting modalities appear in linear inequalities. We show that each formula can be transformed into an equivalent graph neural network (GNN). We also show that a broad class of GNNs can be transformed efficiently into a formula, thus significantly improving upon the literature about the logical expressiveness of GNNs. We also show that the satisfiability problem is PSPACE-complete. These results bring together the promise of using standard logical methods for reasoning about GNNs and their properties, particularly in applications such as GNN querying, equivalence checking, etc. We prove that such natural problems can be solved in polynomial space.

</details>

## 67. Fine-tuning Pre-trained Models for Robustness under Noisy Labels 🌟

<details>

<summary>Abstract</summary>

The presence of noisy labels in a training dataset can significantly impact the performance of machine learning models. In response to this issue, researchers have focused on identifying clean samples and reducing the influence of noisy labels. Recent works in this field have achieved notable success in terms of generalizability, albeit at the expense of extensive computing resources. Therefore, reducing computational costs remains a crucial challenge. Concurrently, in other research areas, there has been a focus on developing fine-tuning techniques to efficiently achieve high generalization performance. Despite their proven efficiently achievable generalization capabilities, these techniques have seen limited exploration from a label noise point of view. In this research, we aim to find an effective approach to fine-tune pre-trained models for noisy labeled datasets. To achieve this goal, we empirically investigate the characteristics of pre-trained models on noisy labels and propose an algorithm, named TURN. We present the results of extensive testing and demonstrate both efficient and improved denoising performance on various benchmarks, surpassing previous methods.

</details>

## 68. Contrastive Learning Is Not Optimal for Quasiperiodic Time Series

<details>

<summary>Abstract</summary>

Despite recent advancements in Self-Supervised Learning (SSL) for Time Series analysis, a noticeable gap persists between the anticipated achievements and actual performance. While these methods have demonstrated formidable generalization capabilities with minimal labels in various domains, their effectiveness in distinguishing between different classes based on a limited number of annotated records is notably lacking. Our hypothesis attributes this bottleneck to the prevalent use of Contrastive Learning, a shared training objective in previous state-of-the-art (SOTA) methods. By mandating distinctiveness between representations for negative pairs drawn from separate records, this approach compels the model to encode unique record-based patterns but simultaneously neglects changes occurring across the entire record. To overcome this challenge, we introduce Distilled Embedding for Almost-Periodic Time Series (DEAPS) in this paper, offering a non-contrastive method tailored for quasiperiodic time series, such as electrocardiogram (ECG) data. By avoiding the use of negative pairs, we not only mitigate the model's blindness to temporal changes but also enable the integration of a "Gradual Loss (L_gra)" function. This function guides the model to effectively capture dynamic patterns evolving throughout the record. The outcomes are promising, as DEAPS demonstrates a notable improvement of +10% over existing SOTA methods when just a few annotated records are presented to fit a Machine Learning (ML) model based on the learned representation.

</details>

## 69. Contrastive General Graph Matching with Adaptive Augmentation Sampling

<details>

<summary>Abstract</summary>

Graph matching has important applications in pattern recognition and beyond. Current approaches predominantly adopt supervised learning, demanding extensive labeled data which can be limited or costly. Meanwhile, self-supervised learning methods for graph matching often require additional side information such as extra categorical information and input features, limiting their application to the general case. Moreover, designing the optimal graph augmentations for self-supervised graph matching presents another challenge to ensure robustness and efficacy. To address these issues, we introduce a novel Graph-centric Contrastive framework for Graph Matching (GCGM), capitalizing on a vast pool of graph augmentations for contrastive learning, yet without needing any side information. Given the variety of augmentation choices, we further introduce a Boosting-inspired Adaptive Augmentation Sampler (BiAS), which adaptively selects more challenging augmentations tailored for graph matching. Through various experiments, our GCGM surpasses state-of-the-art self-supervised methods across various datasets, marking a significant step toward more effective, efficient and general graph matching.

</details>

## 70. Towards Exact Computation of Inductive Bias

<details>

<summary>Abstract</summary>

Much research in machine learning involves finding appropriate inductive biases (e.g. convolutional neural networks, momentum-based optimizers, transformers) to promote generalization on tasks. However, quantification of the amount of inductive bias associated with these architectures and hyperparameters has been limited. We propose a novel method for efficiently computing the inductive bias required for generalization on a task with a fixed training data budget; formally, this corresponds to the amount of information required to specify well-generalizing models within a specific hypothesis space of models. Our approach involves modeling the loss distribution of random hypotheses drawn from a hypothesis space to estimate the required inductive bias for a task relative to these hypotheses. Unlike prior work, our method provides a direct estimate of inductive bias without using bounds and is applicable to diverse hypothesis spaces. Moreover, we derive approximation error bounds for our estimation approach in terms of the number of sampled hypotheses. Consistent with prior results, our empirical results demonstrate that higher dimensional tasks require greater inductive bias. We show that relative to other expressive model classes, neural networks as a model class encode large amounts of inductive bias. Furthermore, our measure quantifies the relative difference in inductive bias between different neural network architectures. Our proposed inductive bias metric provides an information-theoretic interpretation of the benefits of specific model architectures for certain tasks and provides a quantitative guide to developing tasks requiring greater inductive bias, thereby encouraging the development of more powerful inductive biases.

</details>

## 71. EAT: Self-Supervised Pre-Training with Efficient Audio Transformer

<details>

<summary>Abstract</summary>

Audio self-supervised learning (SSL) pre-training, which aims to learn good representations from unlabeled audio, has made remarkable progress. However, the extensive computational demands during pre-training pose a significant barrier to the potential application and optimization of audio SSL models. In this paper, inspired by the success of data2vec 2.0 in image modality and Audio-MAE in audio modality, we introduce Efficient Audio Transformer (EAT) to further improve the effectiveness and efficiency in audio SSL. The proposed EAT adopts the bootstrap self-supervised training paradigm to the audio domain. A novel Utterance-Frame Objective (UFO) is designed to enhance the modeling capability of acoustic events. Furthermore, we reveal that the masking strategy is critical in audio SSL pre-training, and superior audio representations can be obtained with large inverse block masks. Experiment results demonstrate that EAT achieves state-of-the-art (SOTA) performance on a range of audio-related tasks, including AudioSet (AS-2M, AS-20K), ESC-50, and SPC-2, along with a significant pre-training speedup up to ~15x compared to existing audio SSL models.

</details>

## 72. Boosting Single Positive Multi-label Classification with Generalized Robust Loss

<details>

<summary>Abstract</summary>

Multi-label learning (MLL) requires comprehensive multi-semantic annotations that is hard to fully obtain, thus often resulting in missing labels scenarios. In this paper, we investigate Single Positive Multi-label Learning (SPML), where each image is associated with merely one positive label. Existing SPML methods only focus on designing losses using mechanisms such as hard pseudo-labeling and robust losses, mostly leading to unacceptable false negatives. To address this issue, we first propose a generalized loss framework based on expected risk minimization to provide soft pseudo labels, and point out that the former losses can be seamlessly converted into our framework. In particular, we design a novel robust loss based on our framework, which enjoys flexible coordination between false positives and false negatives, and can additionally deal with the imbalance between positive and negative samples. Extensive experiments show that our approach can significantly improve SPML performance and outperform the vast majority of state-of-the-art methods on all the four benchmarks. Our code is available at https://github.com/yan4xi1/GRLoss.

</details>

## 73. Automated CPU Design by Learning from Input-Output Examples

<details>

<summary>Abstract</summary>

Designing a central processing unit (CPU) requires intensive manual work of talented experts to implement the circuit logic from design specifications. Although considerable progress has been made in electronic design automation (EDA) to relieve human efforts, all existing EDA tools require hand-crafted formal program codes (e.g., Verilog, Chisel, or C) as the input. To automate the CPU design without human programming, we are motivated to learn the CPU design from only input-output (IO) examples. The key challenge is that the learned CPU design should have almost zero tolerance for inaccuracy, which makes well-known approximate algorithms such as neural networks ineffective. We propose a new AI approach to generate the CPU design in the form of a large-scale Boolean function, from only external IO examples instead of formal program code. This approach employs a novel graph structure called Binary Speculative Diagram (BSD) to approximate the CPU-scale Boolean function accurately. We propose an efficient BSD expansion method based on Boolean Distance, a new metric to quantitatively measure the structural similarity between Boolean functions, gradually increasing the design accuracy up to 100%. Our approach generates an industrial-scale RISC-V CPU design within 5 hours, reducing the design cycle by about 1000x without human involvement. The taped-out chip, Enlightenment-1, the world's first CPU designed by AI, successfully runs the Linux operating system and performs comparably against the human-design Intel 80486SX CPU. Our approach even autonomously discovers human knowledge of the von Neumann architecture.

</details>

## 74. Deep Embedding Clustering Driven by Sample Stability

<details>

<summary>Abstract</summary>

Deep clustering methods improve the performance of clustering tasks by jointly optimizing deep representation learning and clustering. While numerous deep clustering algorithms have been proposed, most of them rely on artificially constructed pseudo targets for performing clustering. This construction process requires some prior knowledge, and it is challenging to determine a suitable pseudo target for clustering. To address this issue, we propose a deep embedding clustering algorithm driven by sample stability (DECS), which eliminates the requirement of pseudo targets. Specifically, we start by constructing the initial feature space with an autoencoder and then learn the cluster-oriented embedding feature constrained by sample stability. The sample stability aims to explore the deterministic relationship between samples and all cluster centroids, pulling samples to their respective clusters and keeping them away from other clusters with high determinacy. We analyzed the convergence of the loss using Lipschitz continuity in theory, which verifies the validity of the model. The experimental results on five datasets illustrate that the proposed method achieves superior performance compared to state-of-the-art clustering approaches.

</details>

## 75. Structure-Preserving Physics-Informed Neural Networks with Energy or Lyapunov Structure

<details>

<summary>Abstract</summary>

Recently, there has been growing interest in using physics-informed neural networks (PINNs) to solve differential equations. However, the preservation of structure, such as energy and stability, in a suitable manner has yet to be established. This limitation could be a potential reason why the learning process for PINNs is not always efficient and the numerical results may suggest nonphysical behavior. Besides, there is little research on their applications on downstream tasks. To address these issues, we propose structure-preserving PINNs to improve their performance and broaden their applications for downstream tasks. Firstly, by leveraging prior knowledge about the physical system, a structure‐preserving loss function is designed to assist the PINN in learning the underlying structure. Secondly, a framework that utilizes structure-preserving PINN for robust image recognition is proposed. Here, preserving the Lyapunov structure of the underlying system ensures the stability of the system. Experimental results demonstrate that the proposed method improves the numerical accuracy of PINNs for partial differential equations (PDEs). Furthermore, the robustness of the model against adversarial perturbations in image data is enhanced.

</details>

## 76. VCC-INFUSE: Towards Accurate and Efficient Selection of Unlabeled Examples in Semi-supervised Learning

<details>

<summary>Abstract</summary>

Despite the progress of Semi-supervised Learning (SSL), existing methods fail to utilize unlabeled data effectively and efficiently. Many pseudo-label-based methods select unlabeled examples based on inaccurate confidence scores from the classifier. Most prior work also uses all available unlabeled data without pruning, making it difficult to handle large amounts of unlabeled data. To address these issues, we propose two methods: Variational Confidence Calibration (VCC) and Influence-Function-based Unlabeled Sample Elimination (INFUSE). VCC is a universal plugin for SSL confidence calibration, using a variational autoencoder to select more accurate pseudo labels based on three types of consistency scores. INFUSE is a data pruning method that constructs a core dataset of unlabeled examples under SSL. Our methods are effective in multiple datasets and settings, reducing classification error rates and saving training time. Together, VCC-INFUSE reduces the error rate of FlexMatch on the CIFAR-100 dataset by 1.08% while saving nearly half of the training time.

</details>

## 77. InfoMatch: Entropy Neural Estimation for Semi-Supervised Image Classification 🌟

<details>

<summary>Abstract</summary>

Semi-supervised image classification, leveraging pseudo supervision and consistency regularization, has demonstrated remarkable success. However, the ongoing challenge lies in fully exploiting the potential of unlabeled data. To address this, we employ information entropy neural estimation to utilize the potential of unlabeled samples. Inspired by contrastive learning, the entropy is estimated by maximizing a lower bound on mutual information across different augmented views. Moreover, we theoretically analyze that the information entropy of the posterior of an image classifier is approximated by maximizing the likelihood function of the softmax predictions. Guided by these insights, we optimize our model from both perspectives to ensure that the predicted probability distribution closely aligns with the ground-truth distribution. Given the theoretical connection to information entropy, we name our method InfoMatch. Through extensive experiments, we show its superior performance. The source code is available at https://github.com/kunzhan/InfoMatch.

</details>

## 78. EPIC: Graph Augmentation with Edit Path Interpolation via Learnable Cost

<details>

<summary>Abstract</summary>

Data augmentation plays a critical role in improving model performance across various domains, but it becomes challenging with graph data due to their complex and irregular structure. To address this issue, we propose EPIC (Edit Path Interpolation via learnable Cost), a novel interpolation-based method for augmenting graph datasets. To interpolate between two graphs lying in an irregular domain, EPIC leverages the concept of graph edit distance, constructing an edit path that represents the transformation process between two graphs via edit operations. Moreover, our method introduces a context-sensitive cost model that accounts for the importance of specific edit operations formulated through a learning framework. This allows for a more nuanced transformation process, where the edit distance is not merely count-based but reflects meaningful graph attributes. With randomly sampled graphs from the edit path, we enrich the training set to enhance the generalization capability of classification models. Experimental evaluations across several benchmark datasets demonstrate that our approach outperforms existing augmentation techniques in many tasks.

</details>

## 79. Dynamically Anchored Prompting for Task-Imbalanced Continual Learning

<details>

<summary>Abstract</summary>

Existing continual learning literature relies heavily on a strong assumption that tasks arrive with a balanced data stream, which is often unrealistic in real-world applications. In this work, we explore task-imbalanced continual learning (TICL) scenarios where the distribution of task data is non-uniform across the whole learning process. We find that imbalanced tasks significantly challenge the capability of models to control the trade-off between stability and plasticity from the perspective of recent prompt-based continual learning methods. On top of the above finding, we propose Dynamically Anchored Prompting (DAP), a prompt-based method that only maintains a single general prompt to adapt to the shifts within a task stream dynamically. This general prompt is regularized in the prompt space with two specifically designed prompt anchors, called boosting anchor and stabilizing anchor, to balance stability and plasticity in TICL. Remarkably, DAP achieves this balance by only storing a prompt across the data stream, therefore offering a substantial advantage in rehearsal-free CL. Extensive experiments demonstrate that the proposed DAP results in 4.5% to 15% absolute improvements over state-of-the-art methods on benchmarks under task-imbalanced settings. Our code is available at https://github.com/chenxing6666/DAP.

</details>

## 80. An Efficient Prototype-Based Clustering Approach for Edge Pruning in Graph Neural Networks to Battle Over-Smoothing 🌟

<details>

<summary>Abstract</summary>

Topology augmentation is a popular strategy to address the issue of over-smoothing in graph neural networks (GNNs). To prevent potential distortion of node representations, an essential principle is to enhance the separability between embeddings of nodes from different classes while preserving smoothness among nodes of the same class. However, differentiating between inter-class and intra-class edges becomes arduous when class labels are unavailable or the graph is partially labeled. While clustering offers an alternative for identifying closely connected groups of nodes, traditional clustering methods face challenges when applied to GNNs in terms of accuracy, efficiency, adaptability, and scalability to diverse graphs. To address these limitations, we introduce ClusterDrop, which uses learnable prototypes for efficient clustering and incorporates supervised signals to enhance accuracy and adaptability across different graphs. Experiments on six datasets with varying graph structures demonstrate its effectiveness in alleviating over-smoothing and enhancing GNN performance.

</details>

## 81. QFormer: An Efficient Quaternion Transformer for Image Denoising

<details>

<summary>Abstract</summary>

Since Deep Convolutional Neural Networks (DCNNs) and Vision Transformer perform well in learning generalizable image priors from large-scale data, these models have been widely used in image denoising tasks. However, vanilla DCNNs and Transformer suffer from two problems. First, the vanilla DCNNs and Transformer only accumulate the output along the channel axis, ignoring the internal relationship among channels. This results in the severely inadequate color structure representation retrieved from color images. Secondly, the DCNNs or Transformer-based image denoising models usually have a large number of parameters, high computational complexity, and slow inference speed. To resolve these issues, this paper proposes a highly-efficient Quaternion Transformer (QFormer) for image denoising. Specifically, the proposed Quaternion Transformer Block (QTB) simplifies the typical Transformer from a multi-branch structure to an elaborately sequential structure mainly with quaternion transformations, to alternately capture both long-range dependencies and local contextual features with color structure information. Furthermore, the proposed QTB can also avoid considerable element-wise multiplications of computing the self-attention matrices. Thus, our QTB can significantly reduce the computational complexity and its sequential structure can further improve the practical inference speed. Comprehensive experiments demonstrate that the proposed QFormer produces state-of-the-art results in both denoising performance and efficiency. We hope that our work will encourage further research to explore the Quaternion Transformer architecture for image denoising tasks.

</details>

## 82. Exploiting Multi-Label Correlation in Label Distribution Learning

<details>

<summary>Abstract</summary>

Label Distribution Learning (LDL) is a novel machine learning paradigm that assigns label distribution to each instance. Numerous LDL methods proposed to leverage label correlation in the learning process to solve the exponential-sized output space; among these, many exploited the low-rank structure of label distribution to capture label correlation. However, recent research has unveiled that label distribution matrices typically maintain full rank, posing a challenge to approaches relying on low-rank label correlation. Notably, low-rank label correlation finds widespread adoption in multi-label learning (MLL) literature due to the often low-rank nature of multi-label matrices. Inspired by that, we introduce an auxiliary MLL process within the LDL framework, focusing on capturing low-rank label correlation within this auxiliary MLL component rather than the LDL itself. By doing so, we adeptly exploited low-rank label correlation in our LDL methods. We conduct comprehensive experiments and demonstrate that our methods are superior to existing LDL methods. Besides, the ablation studies justify the advantages of exploiting low-rank label correlation in the auxiliary MLL.

</details>

## 83. Towards a Framework for Learning of Algorithms: The Case of Learned Comparison Sorts

<details>

<summary>Abstract</summary>

Designing algorithms is cumbersome and error-prone. This, among other things, has increasingly led to efforts to extend or even replace designing algorithms with machine learning models. While previous research has demonstrated that some machine learning models possess Turing-completeness, the findings are largely theoretical, and solutions for specific algorithmic tasks remain unclear. With this in mind, we investigate the feasibility of learning representations of classical algorithms from data on their execution, enabling their application to different inputs. We propose a novel and general framework for algorithm learning consisting of a model of computation that facilitates algorithm analysis across various levels of abstraction. We formalize the problem of learning an algorithm using an algebraic approach for graph traversal. We apply this framework to comparison sorts and evaluate the inferred machine learning models' performance, demonstrating the applicability of the approach in terms of accuracy and sensitivity.

</details>

## 84. Deep Neural Networks via Complex Network Theory: A Perspective

<details>

<summary>Abstract</summary>

Deep Neural Networks (DNNs) can be represented as graphs whose links and vertices iteratively process data and solve tasks sub-optimally. Complex Network Theory (CNT), merging statistical physics with graph theory, provides a method for interpreting neural networks by analysing their weights and neuron structures. However, classic works adapt CNT metrics that only permit a topological analysis as they do not account for the effect of the input data. In addition, CNT metrics have been applied to a limited range of architectures, mainly including Fully Connected neural networks. In this work, we extend the existing CNT metrics with measures that sample from the DNNs' training distribution, shifting from a purely topological analysis to one that connects with the interpretability of deep learning. For the novel metrics, in addition to the existing ones, we provide a mathematical formalisation for Fully Connected, AutoEncoder, Convolutional and Recurrent neural networks, of which we vary the activation functions and the number of hidden layers. We show that these metrics differentiate DNNs based on the architecture, the number of hidden layers, and the activation function. Our contribution provides a method rooted in physics for interpreting DNNs that offers insights beyond the traditional input-output relationship and the CNT topological analysis.

</details>

## 85. Hypergraph Self-supervised Learning with Sampling-efficient Signals 🌟

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) provides a promising alternative for representation learning on hypergraphs without costly labels. However, existing hypergraph SSL models are mostly based on contrastive methods with the instance-level discrimination strategy, suffering from two significant limitations: (1) They select negative samples arbitrarily, which is unreliable in deciding similar and dissimilar pairs, causing training bias. (2) They often require a large number of negative samples, resulting in expensive computational costs. To address the above issues, we propose SE-HSSL, a hypergraph SSL framework with three sampling-efficient self-supervised signals. Specifically, we introduce two sampling-free objectives leveraging the canonical correlation analysis as the node-level and group-level self-supervised signals. Additionally, we develop a novel hierarchical membership-level contrast objective motivated by the cascading overlap relationship in hypergraphs, which can further reduce membership sampling bias and improve the efficiency of sample utilization. Through comprehensive experiments on 7 real-world hypergraphs, we demonstrate the superiority of our approach over the state-of-the-art method in terms of both effectiveness and efficiency.

</details>

## 86. Efficiency Calibration of Implicit Regularization in Deep Networks via Self-paced Curriculum-Driven Singular Value Selection

<details>

<summary>Abstract</summary>

The generalization of neural networks has been a major focus of research in deep learning. It is often interpreted as an implicit bias towards solutions with specific properties. Especially, in practical applications, it has been observed that linear neural networks (LNN) tend to favor low-rank solutions for matrix completion tasks. However, most existing methods rely on increasing the depth of the neural network to enhance the low rank of solutions, resulting in higher complexity. In this paper, we propose a new explicit regularization method that calibrates the implicit bias towards low-rank trends in matrix completion tasks. Our approach automatically incorporates smaller singular values into the training process using a self-paced learning strategy, gradually restoring matrix information. By jointly using both implicit and explicit regularization, we effectively capture the low-rank structure of LNN and accelerate its convergence. We also analyze how our proposed penalty term interacts with implicit regularization and provide theoretical guarantees for our new model. To evaluate the effectiveness of our method, we conduct a series of experiments on both simulated and real-world data. Our experimental results clearly demonstrate that our method has better robustness and generalization ability compared with other methods.

</details>

## 87. Towards Counterfactual Fairness-aware Domain Generalization in Changing Environments

<details>

<summary>Abstract</summary>

Recognizing domain generalization as a commonplace challenge in machine learning, data distribution might progressively evolve across a continuum of sequential domains in practical scenarios. While current methodologies primarily concentrate on bolstering model effectiveness within these new domains, they tend to neglect issues of fairness throughout the learning process. In response, we propose an innovative framework known as Disentanglement for Counterfactual Fairness-aware Domain Generalization (DCFDG). This approach adeptly removes domain-specific information and sensitive information from the embedded representation of classification features. To scrutinize the intricate interplay between semantic information, domain-specific information, and sensitive attributes, we systematically partition the exogenous factors into four latent variables. By incorporating fairness regularization, we utilize semantic information exclusively for classification purposes. Empirical validation on synthetic and authentic datasets substantiates the efficacy of our approach, demonstrating elevated accuracy levels while ensuring the preservation of fairness amidst the evolving landscape of continuous domains.

</details>

## 88. Meta-Learning via PAC-Bayesian with Data-Dependent Prior: Generalization Bounds from Local Entropy

<details>

<summary>Abstract</summary>

Meta-learning accelerates the learning process on unseen learning tasks by acquiring prior knowledge through previous related tasks. The PAC-Bayesian theory provides a theoretical framework to analyze the generalization of meta-learning to unseen tasks. However, previous works still encounter two notable limitations: (1) they merely focus on the data-free priors, which often result in inappropriate regularization and loose generalization bounds; (2) more importantly, their optimization process usually involves nested optimization problems, incurring significant computational costs. To address these issues, we derive new generalization bounds and introduce a novel PAC-Bayesian framework for meta-learning that integrates data-dependent priors. This framework enables the extraction of optimal posteriors for each task in closed form, thereby allowing us to minimize generalization bounds incorporated data-dependent priors with only a simple local entropy. The resulting algorithm, which employs SGLD for sampling from the optimal posteriors, is stable, efficient, and computationally lightweight, eliminating the need for nested optimization. Extensive experimental results demonstrate that our proposed method outperforms the other baselines.

</details>

## 89. Implicit Prompt Learning for Image Denoising

<details>

<summary>Abstract</summary>

Recently, various deep denoising methods have been proposed to solve the insufficient feature problem in image denoising. These methods can be mainly classified into two categories: (1) Injecting learnable tensors into denoising backbone to supplement feature, which is effective to some extent but may cause serious over-fitting. (2) Using diverse natural images from large image datasets to synthesize noisy images and pre-train denoising models, which can bring model generalization but require large model size and expensive training costs. To address these issues, this paper proposes Implicit Prompt Learning for Image Denoising (IPLID) method to flexibly generate adaptive prompts without meticulously designing them. Specifically, we first introduce an efficient Linear Prompt (LP) block with ultra-few parameters to produce dynamic prompts for both different stages and samples in denoising procedure. We further propose an efficient Compact Feature Fusion (CFF) block to process previous multi-level prompted denoising feature to reconstruct the denoising images. Finally, to further efficiently and effectively produce satisfactory prompt and denoising performance, a Gradient Accumulation (GA) learning scheme is proposed. Experiments on multiple benchmarks showed that the proposed IPLID achieves competitive results with only 1 percent of pre-trained backbone parameters, outperforming classical denoising methods in both efficiency and quality of restored images.

</details>

## 90. Rank and Align: Towards Effective Source-free Graph Domain Adaptation

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) have achieved impressive performance in graph domain adaptation. However, extensive source graphs could be unavailable in real-world scenarios due to privacy and storage concerns. To this end, we investigate an underexplored yet practical problem of source-free graph domain adaptation, which transfers knowledge from source models instead of source graphs to a target domain. To solve this problem, we introduce a novel GNN-based approach called Rank and Align (RNA), which ranks graph similarities with spectral seriation for robust semantics learning, and aligns inharmonic graphs with harmonic graphs which close to the source domain for subgraph extraction. In particular, to overcome label scarcity, we employ the spectral seriation algorithm to infer the robust pairwise rankings, which can guide semantic learning using a similarity learning objective. To depict distribution shifts, we utilize spectral clustering and the silhouette coefficient to detect harmonic graphs, which the source model can easily classify. To reduce potential domain discrepancy, we extract domain-invariant subgraphs from inharmonic graphs by an adversarial edge sampling process, which guides the invariant learning of GNNs. Extensive experiments on several benchmark datasets demonstrate the effectiveness of our proposed RNA.

</details>

## 91. Deciphering the Projection Head: Representation Evaluation Self-supervised Learning 🌟

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) aims to learn the intrinsic features of data without labels. Despite the diverse SSL architectures, the projection head always plays an important role in improving downstream task performance. In this study, we systematically investigate the role of the projection head in SSL. We find that the projection head targets the uniformity aspect, which maps samples into uniform distribution and enables the encoder to focus on extracting semantic features. Drawing on this insight, we propose a Representation Evaluation Design (RED) in SSL models in which a shortcut connection between the representation and the projection vectors is built. Our extensive experiments with different architectures (including SimCLR, MoCo-V2, and SimSiam) on various datasets demonstrate that the RED-SSL consistently outperforms their baseline counterparts in downstream tasks. Furthermore, the RED-SSL learned representations exhibit superior robustness to previously unseen augmentations and out-of-distribution data.

</details>

## 92. Mean Aggregator Is More Robust than Robust Aggregators under Label Poisoning Attacks 🌟

<details>

<summary>Abstract</summary>

Robustness to malicious attacks is of paramount importance for distributed learning. Existing works often consider the classical Byzantine attacks model, which assumes that some workers can send arbitrarily malicious messages to the server and disturb the aggregation steps of the distributed learning process. To defend against such worst-case Byzantine attacks, various robust aggregators have been proven effective and much superior to the often-used mean aggregator. In this paper, we show that robust aggregators are too conservative for a class of weak but practical malicious attacks, as known as label poisoning attacks, where the sample labels of some workers are poisoned. Surprisingly, we are able to show that the mean aggregator is more robust than the state-of-the-art robust aggregators in theory, given that the distributed data are sufficiently heterogeneous. In fact, the learning error of the mean aggregator is proven to be optimal in order. Experimental results corroborate our theoretical findings, demonstrating the superiority of the mean aggregator under label poisoning attacks.

</details>

## 93. What Makes Models Compositional? A Theoretical View

<details>

<summary>Abstract</summary>

Compositionality is thought to be a key component of language, and various compositional benchmarks have been developed to empirically probe the compositional generalization of existing sequence processing models. These benchmarks often highlight failures of existing models, but it is not clear why these models fail in this way. In this paper, we seek to theoretically understand the role the compositional structure of the models plays in these failures and how this structure relates to their expressivity and sample complexity. We propose a general neuro-symbolic definition of compositional functions and their compositional complexity. We then show how various existing general and special purpose sequence processing models (such as recurrent, convolution and attention-based ones) fit this definition and use it to analyze their compositional complexity. Finally, we provide theoretical guarantees for the expressivity and systematic generalization of compositional models that explicitly depend on our proposed definition and highlighting factors which drive poor empirical performance.

</details>

## 94. A Context-Enhanced Framework for Sequential Graph Reasoning

<details>

<summary>Abstract</summary>

The paper studies sequential reasoning over graph-structured data, which stands as a fundamental task in various trending fields like automated math problem solving and neural graph algorithm learning, attracting a lot of research interest. Simultaneously managing both sequential and graph-structured information in such tasks presents a notable challenge. Over recent years, many neural architectures in the literature have emerged to tackle the issue. In this work, we generalize the existing architectures and propose a context-enhanced framework. The crucial innovation is that the reasoning of each step does not only rely on the outcome of the preceding step but also leverages the aggregation of information from more historical outcomes. The idea stems from our observation that in sequential graph reasoning, each step's outcome has a much stronger inner connection with each other compared to traditional seq-to-seq tasks. We show that the framework can effectively integrate with the existing methods, enhancing their reasoning abilities. Empirical evaluations are conducted on the challenging CLRS Reasoning Benchmark, and the results demonstrate that the proposed framework significantly improves the performance of existing architectures, yielding state-of-the-art results across the majority of the datasets within the benchmark.

</details>

## 95. Robust Contrastive Multi-view Kernel Clustering

<details>

<summary>Abstract</summary>

Multi-view kernel clustering (MKC) aims to fully reveal the consistency and complementarity of multiple views in a potential Hilbert space, thereby enhancing clustering performance. The clustering results of most MKC methods are highly sensitive to the quality of the constructed kernels, as traditional methods independently compute kernel matrices for each view without fully considering complementary information across views. In previous contrastive multi-view kernel learning, the goal was to bring cross-view instances of the same sample closer during the kernel construction process while pushing apart instances across samples to achieve a comprehensive integration of cross-view information. However, its inherent drawback is the potential inappropriate amplification of distances between different instances of the same clusters (i.e., false negative pairs) during the training process, leading to a reduction in inter-class discriminability. To address this challenge, we propose a Robust Contrastive multi-view kernel Learning approach (R-CMK) against false negative pairs. It partitions negative pairs into different intervals based on distance or similarity, and for false negative pairs, reverses their optimization gradient. This effectively avoids further amplification of distances for false negative pairs while simultaneously pushing true negative pairs farther apart. We conducted comprehensive experiments on various MKC methods to validate the effectiveness of the proposed method. The code is available at https://github.com/Duo-laimi/rcmk_main.

</details>

## 96. Deep Hierarchical Graph Alignment Kernels

<details>

<summary>Abstract</summary>

Typical R-convolution graph kernels invoke the kernel functions that decompose graphs into non-isomorphic substructures and compare them. However, overlooking implicit similarities and topological position information between those substructures limits their performances. In this paper, we introduce Deep Hierarchical Graph Alignment Kernels (DHGAK) to resolve this problem. Specifically, the relational substructures are hierarchically aligned to cluster distributions in their deep embedding space. The substructures belonging to the same cluster are assigned the same feature map in the Reproducing Kernel Hilbert Space (RKHS), where graph feature maps are derived by kernel mean embedding. Theoretical analysis guarantees that DHGAK is positive semi-definite and has linear separability in the RKHS. Comparison with state-of-the-art graph kernels on various benchmark datasets demonstrates the effectiveness and efficiency of DHGAK. The code is available at Github (https://github.com/EWesternRa/DHGAK).

</details>

## 97. Proximal Curriculum with Task Correlations for Deep Reinforcement Learning

<details>

<summary>Abstract</summary>

Curriculum design for reinforcement learning (RL) can speed up an agent's learning process and help it learn to perform well on complex tasks. However, existing techniques typically require domain-specific hyperparameter tuning, involve expensive optimization procedures for task selection, or are suitable only for specific learning objectives. In this work, we consider curriculum design in contextual multi-task settings where the agent's final performance is measured w.r.t. a target distribution over complex tasks. We base our curriculum design on the Zone of Proximal Development concept, which has proven to be effective in accelerating the learning process of RL agents for uniform distribution over all tasks. We propose a novel curriculum, ProCuRL-Target, that effectively balances the need for selecting tasks that are not too difficult for the agent while progressing the agent's learning toward the target distribution via leveraging task correlations. We theoretically justify the task selection strategy of ProCuRL-Target by analyzing a simple learning setting with REINFORCE learner model. Our experimental results across various domains with challenging target task distributions affirm the effectiveness of our curriculum strategy over state-of-the-art baselines in accelerating the training process of deep RL agents.

</details>

## 98. Contrastive and View-Interaction Structure Learning for Multi-view Clustering 🌟

<details>

<summary>Abstract</summary>

Existing Deep Multi-view Clustering (DMVC) approaches typically concentrate on capturing consensus semantics from multiple views, where contrastive learning is widely used to align view-specific representations of each view. Unfortunately, view-specific representations are extracted from the content information of the corresponding instance, neglecting the relationships among different instances. Furthermore, existing contrastive loss imports numerous false negative pairs that conflict with the clustering objectives. In response to these challenges, we propose a contraStive and viEw-interaction stRucture learning framework for multI-viEw cluStering (SERIES). Our method takes into account the structural relations among instances and boosts the contrastive loss to improve intra-class compactness. Meanwhile, a cross-view dual relation generation mechanism is introduced to achieve the consensus structural graph across multiple views for clustering. Specifically, we initially acquire view-specific representations using multiple graph autoencoders to exploit both content information and structural information. Furthermore, to pull together the same cluster instances, a soft negative pair aware contrastive loss is employed to distinguish the dissimilar instances while attracting similar instances. Thereafter, the view-specific representations are fed into cross-view dual relation generation layers to generate the affinity matrices of each other, aiming to reveal a consistent structural graph across various views. Extensive experiments conducted on six benchmarks illustrate the superiority of our method compared to other state-of-the-art approaches.

</details>

## 99. Subgraph Pooling: Tackling Negative Transfer on Graphs 🌟

<details>

<summary>Abstract</summary>

Transfer learning aims to enhance performance on a target task by using knowledge from related tasks. However, when the source and target tasks are not closely aligned, it can lead to reduced performance, known as negative transfer. Unlike in image or text data, we find that negative transfer could commonly occur in graph-structured data, even when source and target graphs have semantic similarities. Specifically, we identify that structural differences significantly amplify the dissimilarities in the node embeddings across graphs. To mitigate this, we bring a new insight in this paper: for semantically similar graphs, although structural differences lead to significant distribution shift in node embeddings, their impact on subgraph embeddings could be marginal. Building on this insight, we introduce Subgraph Pooling (SP) by aggregating nodes sampled from a k-hop neighborhood and Subgraph Pooling++ (SP++) by a random walk, to mitigate the impact of graph structural differences on knowledge transfer. We theoretically analyze the role of SP in reducing graph discrepancy and conduct extensive experiments to evaluate its superiority under various settings. The proposed SP methods are effective yet elegant, which can be easily applied on top of any backbone Graph Neural Networks (GNNs). Our code and data are available at: https://github.com/Zehong-Wang/Subgraph-Pooling.

</details>

## 100. Towards Sharper Generalization Bounds for Adversarial Contrastive Learning

<details>

<summary>Abstract</summary>

Recently, the enhancement on the adversarial robustness of machine learning algorithms has gained significant attention across various application domains. Given the widespread label scarcity issue in real-world data, adversarial contrastive learning (ACL) has been proposed to adversarially train robust models using unlabeled data. Despite the empirical success, its generalization behavior remains poorly understood and far from being well-characterized. This paper aims to address this issue from a learning theory perspective. We establish novel high-probability generalization bounds for the general Lipschitz loss functions. The derived bounds scale O(log(k)) with respect to the number of negative samples k, which improves the existing linear dependency bounds. Our results are generally applicable to many prediction models, including linear models and deep neural networks. In particular, we obtain an optimistic generalization bound O(1/n) under the smoothness assumption of the loss function on the sample size n. To the best of our knowledge, this is the first fast-rate bound valid for ACL. Empirical evaluations on real-world datasets verify our theoretical findings.

</details>

## 101. Trusted Multi-view Learning with Label Noise

<details>

<summary>Abstract</summary>

Multi-view learning methods often focus on improving decision accuracy while neglecting the decision uncertainty, which significantly restricts their applications in safety-critical applications. To address this issue, researchers propose trusted multi-view methods that learn the class distribution for each instance, enabling the estimation of classification probabilities and uncertainty. However, these methods heavily rely on high-quality ground-truth labels. This motivates us to delve into a new generalized trusted multi-view learning problem: how to develop a reliable multi-view learning model under the guidance of noisy labels? We propose a trusted multi-view noise refining method to solve this problem. We first construct view-opinions using evidential deep neural networks, which consist of belief mass vectors and uncertainty estimates. Subsequently, we design view-specific noise correlation matrices that transform the original opinions into noisy opinions aligned with the noisy labels. Considering label noises originating from low-quality data features and easily-confused classes, we ensure that the diagonal elements of these matrices are inversely proportional to the uncertainty, while incorporating class relations into the off-diagonal elements. Finally, we aggregate the noisy opinions and employ a generalized maximum likelihood loss on the aggregated opinion for model training, guided by the noisy labels. We empirically compare TMNR with state-of-the-art trusted multi-view learning and label noise learning baselines on 5 publicly available datasets. Experiment results show that TMNR outperforms baseline methods on accuracy, reliability and robustness. The code and appendix are released at https://github.com/YilinZhang107/TMNR.

</details>

## 102. Dynamic against Dynamic: An Open-Set Self-Learning Framework

<details>

<summary>Abstract</summary>

In open set recognition, existing methods generally learn statically fixed decision boundaries to reject unknown classes. Though they have achieved promising results, such decision boundaries are evidently insufficient for universal unknown classes in dynamic and open scenarios as they can potentially appear at any position in the feature space. Moreover, these methods just simply reject unknown class samples during testing without any effective utilization for them. In fact, such samples completely can constitute the true instantiated representation of the unknown classes to further enhance the model's performance. To address these issues, this paper proposes a novel dynamic against dynamic idea, i.e., dynamic method against dynamic changing open-set world, where an open-set self-learning (OSSL) framework is correspondingly developed. OSSL starts with a good closed-set classifier trained by known classes and utilizes available test samples for model adaptation during testing, thus gaining the adaptability to changing data distributions. In particular, a novel self-matching module is designed for OSSL, which can achieve the adaptation in automatically identifying known class samples while rejecting unknown class samples which are further utilized to enhance the discriminability of the model as the instantiated representation of unknown classes. Our method establishes new performance milestones respectively in almost all standard and cross-data benchmarks.

</details>

## 103. Bridging the Gap: Learning Pace Synchronization for Open-World Semi-Supervised Learning 🌟

<details>

<summary>Abstract</summary>

In open-world semi-supervised learning, a machine learning model is tasked with uncovering novel categories from unlabeled data while maintaining performance on seen categories from labeled data. The central challenge is the substantial learning gap between seen and novel categories, as the model learns the former faster due to accurate supervisory information. Moreover, capturing the semantics of unlabeled novel category samples is also challenging due to the missing label information. To address the above issues, we introduce 1) the adaptive synchronizing marginal loss which imposes class-specific negative margins to alleviate the model bias towards seen classes, and 2) the pseudo-label contrastive clustering which exploits pseudo-labels predicted by the model to group unlabeled data from the same category together in the output space. Extensive experiments on benchmark datasets demonstrate that previous approaches may significantly hinder novel class learning, whereas our method strikingly balances the learning pace between seen and novel classes, achieving a remarkable 3% average accuracy increase on the ImageNet dataset. Importantly, we find that fine-tuning the self-supervised pre-trained model significantly boosts the performance, which is overlooked in prior literature. Our code is available at https://github.com/yebo0216best/LPS-main.

</details>

## 104. Efficient Multi-view Unsupervised Feature Selection with Adaptive Structure Learning and Inference 🌟

<details>

<summary>Abstract</summary>

As data with diverse representations become high-dimensional, multi-view unsupervised feature selection has been an important learning paradigm. Generally, existing methods encounter the following challenges: (i) traditional solutions either concatenate different views or introduce extra parameters to weight them, affecting the performance and applicability; (ii) emphasis is typically placed on graph construction, yet disregarding the clustering information of data; (iii) exploring the similarity structure of all samples from the original features is suboptimal and extremely time-consuming. To solve this dilemma, we propose an efficient multi-view unsupervised feature selection (EMUFS) to construct bipartite graphs between samples and anchors. Specifically, a parameter-free manner is devised to collaboratively fuse the membership matrices and graphs to learn the compatible structure information across all views, naturally balancing different views. Moreover, EMUFS leverages the similarity relations of data in the feature subspace induced by l2,0-norm to dynamically update the graph. Accordingly, the cluster information of anchors can be accurately propagated to samples via the graph structure and further guide feature selection, enhancing the quality of selected features and the computational costs in solution processes. A convergent optimization is developed to solve the formulated problem, and experiments demonstrate the effectiveness and efficiency of EMUFS.

</details>

## 105. LSPAN: Spectrally Localized Augmentation for Graph Consistency Learning

<details>

<summary>Abstract</summary>

Graph-based consistency principle has been successfully applied to many semi-supervised problems in machine learning. Its performance largely depends on the quality of augmented graphs, which has been recently proven that revealing graph properties and maintaining the invariance of graphs are crucial for good performance. However, existing topology- or feature-based augmentation methods are spectrally non-localized -- important spectrums are disturbed throughout the entire frequency range, and their invariance may not be well preserved. Efforts on this issue remain to be limited. This paper proposes a simple yet effective model called Localized SPectral AugmentatioN (LSPAN), which perturbs a concentrated part of graph spectrum with equivalent intensity using Fourier orthogonality, so as to enhance graph spectrum preservation as well as model prediction. Moreover, it also avoids the significant training time of inverse Fourier transform. Extensive empirical evaluation on real-world datasets clearly shows the performance gain of spectrally localized augmentation, as well as its good convergence and efficiency compared to existing graph methods.

</details>

## 106. Learning from Long-Tailed Noisy Data with Sample Selection and Balanced Loss 🌟

<details>

<summary>Abstract</summary>

The success of deep learning depends on large-scale and well-curated training data, while data in real-world applications are commonly long-tailed and noisy. Existing methods are usually dependent on label frequency to tackle class imbalance, while the model bias on different classes is not directly related to label frequency and the true label frequency is inaccessible under label noise. To solve this, we propose a robust method for learning from long-tailed noisy data with sample selection and balanced loss. Specifically, we separate the noisy training data into clean labeled set and unlabeled set with sample selection, and train the deep neural network in a semi-supervised manner with a balanced loss based on model bias. Extensive experiments on benchmarks demonstrate that our method outperforms existing state-of-the-art methods.

</details>

## 107. CONC: Complex-noise-resistant Open-set Node Classification with Adaptive Noise Detection 🌟

<details>

<summary>Abstract</summary>

As a popular task in graph learning, node classification seeks to assign labels to nodes, taking into account both their features and connections. However, an important challenge for its application in real-world scenarios is the presence of newly-emerged out-of-distribution samples and noisy samples, which affect the quality and robustness of learned classifiers. Out-of-distribution (OOD) samples are often found in both the training and testing phases. Such samples don’t belong to any known categories. These OOD samples are considered as outliers (OOD noise) when they appear during training, and are recognized as open-set samples during the testing. Meanwhile, in-distribution (IND) noisy data, i.e., known class samples with wrong labels, are also prevalent and inevitably degrade a model’s performance. The challenge of open-set learning with complex IND and OOD noise remains largely unexplored, particularly when dealing with non-IID graph data. To address these challenges, this paper introduces a novel complex-noise-resistant open-set node classification approach, designed for open-set graph data containing both IND and OOD noisy nodes. Specifically, a trustworthiness learner is adopted to learn the trustworthiness rates of the feature and label for each node while a decoder and an open-set classifier are trained to reconstruct the structure of a node and to predict its category simultaneously with the guidance of node trustworthiness. The experimental results demonstrate the superiority of our method.

</details>

## 108. Pre-training General User Representation with Multi-type APP Behaviors

<details>

<summary>Abstract</summary>

In numerous user-centric services on mobile applications (apps), accurately mining user interests and generating effective user representations are paramount. Traditional approaches, which often involve training task-specific user representations, are becoming increasingly impractical due to their high computational costs and limited adaptability. This paper introduces a novel solution to this challenge: the Multi-type App-usage Fusion Network (MAFN). MAFN innovatively pre-trains universal user representations, leveraging multi-type app behaviors to overcome key limitations in existing methods. We address two primary challenges: 1) the varying frequency of user behaviors (ranging from low-frequency actions like (un)installations to high-frequency yet insightful app launches); and 2) the integration of multi-type behaviors to form a cohesive representation. Our approach involves the creation of novel pre-training tasks that harness self-supervised signals from diverse app behaviors, capturing both long-term and short-term user interests. MAFN's unique fusion approach effectively amalgamates these interests into a unified vector space, facilitating the development of a versatile, general-purpose user representation. With a practical workflow, extensive experiments with three typical downstream tasks on real-world datasets verify the effectiveness of our approach.

</details>

## 109. Reconfigurability-Aware Selection for Contrastive Active Domain Adaptation

<details>

<summary>Abstract</summary>

Active domain adaptation (ADA) aims to label a small portion of target samples to drastically improve the adaptation performance. The existing ADA methods mostly rely on the output of domain discriminator or the original prediction probability to design sample selection strategies and do not fully explore the semantic information of source and target domain features, which may lead to selecting the valueless target samples. Moreover, most of them require complex network structures (such as introducing additional domain discriminator, multiple classifiers, or loss predictors) and multiple query functions. In this work, we propose a concise but effective ADA method called Reconfigurability-Aware Selection for Contrastive active domain adaptation (RASC). With the reconfigurability-aware sample selection strategy, RASC can select the most valuable target samples for annotation in the presence of domain shift. To better utilize the selected target samples, we further design a contrastive learning-based gradual active domain adaptation framework. In addition, we propose a variant of RASC called RASC-Ob, which uses a simpler sample annotation method and supplements the learning of misclassified samples. Extensive experimental results on multiple benchmarks demonstrate the superiority of RASC.

</details>

## 110. Active Deep Multi-view Clustering

<details>

<summary>Abstract</summary>

Deep multi-view clustering has been widely studied. However, since it is an unsupervised task, where no labels are used to guide the training, it is still unreliable especially when handling complicated data. Although deep semi-supervised multi-view clustering can alleviate this problem by using some supervised information, the supervised information is often pregiven or randomly selected. Unfortunately, as we know, the clustering performance highly depends on the quality of the supervised information and most of the semi-supervised methods ignore the supervised information selection. To tackle this problem, in this paper, we propose a novel active deep multi-view clustering method, which can actively select important data for querying human annotations. In this method, we carefully design a fusion module, an active selection module, a supervised module, and an unsupervised module, and integrate them into a unified framework seamlessly. In this framework, we can obtain a more reliable clustering result with as few annotations as possible. The extensive experiments on benchmark data sets show that our method can outperform state-of-the-art unsupervised and semi-supervised methods, demonstrating the effectiveness and superiority of the proposed method. The code is available at https://github.com/wodedazhuozi/ADMC .

</details>

## 111. Towards Robust Multi-Label Learning against Dirty Label Noise

<details>

<summary>Abstract</summary>

In multi-label learning, one of the major challenges is that the data are associated with label noise including the random noisy labels (e.g., data encoding errors) and noisy labels created by annotators (e.g., missing, extra, or error label), where noise is promoted by different structures (e.g., gaussian, sparse or subjective). Existing methods are tailored to handle noise with one specific structure. However, they lack of consideration of the fact that the data are always with dirty noisy labels, simutaneously gaussian, sparse and subjective, in real applications. In this paper, we formalize the multi-label learning with dirty noise as a new learning problem, namely Noisy Multi-label Learning (NML). To solve the NML problem, we decompose a corrupted label matrix as the noise matrix plus a true label matrix (maybe high-rank). For the noise matrix, a mixed norm penalty is developed as regularizer for dirty noise distribution. Under this norm, the conditions required for exact noise recovery are provided theoretically. For the true label matrix that is not necessarily low-rank, we apply a non-linear mapping to ensure its low-rankness such that the high-order label correlation can be utilized. Experimental results show that the proposed method outperforms the state-of-the-art methods significantly.

</details>

## 112. Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation

<details>

<summary>Abstract</summary>

Deep neural classifiers tend to rely on spurious correlations between spurious attributes of inputs and targets to make predictions, which could jeopardize their generalization capability. Training classifiers robust to spurious correlations typically relies on annotations of spurious correlations in data, which are often expensive to get. In this paper, we tackle an annotation-free setting and propose a self-guided spurious correlation mitigation framework. Our framework automatically constructs fine-grained training labels tailored for a classifier obtained with empirical risk minimization to improve its robustness against spurious correlations. The fine-grained training labels are formulated with different prediction behaviors of the classifier identified in a novel spuriousness embedding space. We construct the space with automatically detected conceptual attributes and a novel spuriousness metric which measures how likely a class-attribute correlation is exploited for predictions. We demonstrate that training the classifier to distinguish different prediction behaviors reduces its reliance on spurious correlations without knowing them a priori and outperforms prior methods on five real-world datasets.

</details>

## 113. Denoising-Aware Contrastive Learning for Noisy Time Series

<details>

<summary>Abstract</summary>

Time series self-supervised learning (SSL) aims to exploit unlabeled data for pre-training to mitigate the reliance on labels. Despite the great success in recent years, there is limited discussion on the potential noise in the time series, which can severely impair the performance of existing SSL methods. To mitigate the noise, the de facto strategy is to apply conventional denoising methods before model training. However, this pre-processing approach may not fully eliminate the effect of noise in SSL for two reasons: (i) the diverse types of noise in time series make it difficult to automatically determine suitable denoising methods; (ii) noise can be amplified after mapping raw data into latent space. In this paper, we propose denoising-aware contrastive learning (DECL), which uses contrastive learning objectives to mitigate the noise in the representation and automatically selects suitable denoising methods for every sample. Extensive experiments on various datasets verify the effectiveness of our method. The code is open-sourced.

</details>

## 114. Boosting Model Resilience via Implicit Adversarial Data Augmentation 🌟

<details>

<summary>Abstract</summary>

Data augmentation plays a pivotal role in enhancing and diversifying training data. Nonetheless, consistently improving model performance in varied learning scenarios, especially those with inherent data biases, remains challenging. To address this, we propose to augment the deep features of samples by incorporating their adversarial and anti-adversarial perturbation distributions, enabling adaptive adjustment in the learning difficulty tailored to each sample’s specific characteristics. We then theoretically reveal that our augmentation process approximates the optimization of a surrogate loss function as the number of augmented copies increases indefinitely. This insight leads us to develop a meta-learning-based framework for optimizing classifiers with this novel loss, introducing the effects of augmentation while bypassing the explicit augmentation process. We conduct extensive experiments across four common biased learning scenarios: long-tail learning, generalized long-tail learning, noisy label learning, and subpopulation shift learning. The empirical results demonstrate that our method consistently achieves state-of-the-art performance, highlighting its broad adaptability.

</details>

## 115. Rethinking Centered Kernel Alignment in Knowledge Distillation

<details>

<summary>Abstract</summary>

Knowledge distillation has emerged as a highly effective method for bridging the representation discrepancy between large-scale models and lightweight models. Prevalent approaches involve leveraging appropriate metrics to minimize the divergence or distance between the knowledge extracted from the teacher model and the knowledge learned by the student model. Centered Kernel Alignment (CKA) is widely used to measure representation similarity and has been applied in several knowledge distillation methods. However, these methods are complex and fail to uncover the essence of CKA, thus not answering the question of how to use CKA to achieve simple and effective distillation properly. This paper first provides a theoretical perspective to illustrate the effectiveness of CKA, which decouples CKA to the upper bound of Maximum Mean Discrepancy (MMD) and a constant term. Drawing from this, we propose a novel Relation-Centered Kernel Alignment (RCKA) framework, which practically establishes a connection between CKA and MMD. Furthermore, we dynamically customize the application of CKA based on the characteristics of each task, with less computational source yet comparable performance than the previous methods. The extensive experiments on the CIFAR-100, ImageNet-1k, and MS-COCO demonstrate that our method achieves state-of-the-art performance on almost all teacher-student pairs for image classification and object detection, validating the effectiveness of our approaches. Our code is available in https://github.com/Klayand/PCKA.

</details>

## 116. Predictive Modeling with Temporal Graphical Representation on Electronic Health Records

<details>

<summary>Abstract</summary>

Deep learning-based predictive models, leveraging Electronic Health Records (EHR), are receiving increasing attention in healthcare. An effective representation of a patient's EHR should hierarchically encompass both the temporal relationships between historical visits and medical events, and the inherent structural information within these elements. Existing patient representation methods can be roughly categorized into sequential representation and graphical representation. The sequential representation methods focus only on the temporal relationships among longitudinal visits. On the other hand, the graphical representation approaches, while adept at extracting the graph-structured relationships between various medical events, fall short in effectively integrate temporal information. To capture both types of information, we model a patient's EHR as a novel temporal heterogeneous graph. This graph includes historical visits nodes and medical events nodes. It propagates structured information from medical event nodes to visit nodes and utilizes time-aware visit nodes to capture changes in the patient's health status. Furthermore, we introduce a novel temporal graph transformer (TRANS) that integrates temporal edge features, global positional encoding, and local structural encoding into heterogeneous graph convolution, capturing both temporal and structural information. We validate the effectiveness of TRANS through extensive experiments on three real-world datasets. The results show that our proposed approach achieves state-of-the-art performance.

</details>

## 117. Shadow-Free Membership Inference Attacks: Recommender Systems Are More Vulnerable Than You Thought 🌟

<details>

<summary>Abstract</summary>

Recommender systems have been successfully applied in many applications. Nonetheless, recent studies demonstrate that recommender systems are vulnerable to membership inference attacks (MIAs), leading to the leakage of users’ membership privacy. However, existing MIAs relying on shadow training suffer a large performance drop when the attacker lacks knowledge of the training data distribution and the model architecture of the target recommender system. To better understand the privacy risks of recommender systems, we propose shadow-free MIAs that directly leverage a user’s recommendations for membership inference. Without shadow training, the proposed attack can conduct MIAs efficiently and effectively under a practice scenario where the attacker is given only black-box access to the target recommender system. The proposed attack leverages an intuition that the recommender system personalizes a user’s recommendations if his historical interactions are used by it. Thus, an attacker can infer membership privacy by determining whether the recommendations are more similar to the interactions or the general popular items. We conduct extensive experiments on benchmark datasets across various recommender systems. Remarkably, our attack achieves far better attack accuracy with low false positive rates than baselines while with a much lower computational cost.

</details>

## 118. Towards Geometric Normalization Techniques in SE(3) Equivariant Graph Neural Networks for Physical Dynamics Simulations

<details>

<summary>Abstract</summary>

SE(3) equivariance is a fundamental property that is highly desirable to maintain in physical dynamics modeling. This property ensures neural outputs to remain robust when the inputs are translated or rotated. Recently, there have been several proposals for SE(3) equivariant graph neural networks (GNNs) that have shown promising results in simulating particle dynamics. However, existing works have neglected an important issue that current SE(3) equivariant GNNs cannot scale to large particle systems. Although some simple normalization techniques are already in use to stabilize the training dynamics of equivariant graph networks, they actually break the SE(3) equivariance of the architectures. In this work, we first show the numerical instability of training equivariant GNNs on large particle systems and then analyze some existing normalization strategies adopted in modern works. We propose a new normalization layer called GeoNorm, which can satisfy the SE(3) equivariance and simultaneously stabilize the training process. We conduct comprehensive experiments on N-body system simulation tasks with larger particle system sizes. The experimental results demonstrate that GeoNorm successfully preserves the SE(3) equivariance compared to baseline techniques and stabilizes the training dynamics of SE(3) equivariant GNNs on large systems.

</details>

## 119. Improving Pseudo Labels with Global-Local Denoising Framework for Cross-lingual Named Entity Recognition

<details>

<summary>Abstract</summary>

Cross-lingual named entity recognition (NER) aims to train an NER model for the target language leveraging only labeled source language data and unlabeled target language data. Prior approaches either perform label projection on translated source language data or employ a source model to assign pseudo labels for target language data and train a target model on these pseudo-labeled data to generalize to the target language. However, these automatic labeling procedures inevitably introduce noisy labels, thus leading to a performance drop. In this paper, we propose a Global-Local Denoising framework (GLoDe) for cross-lingual NER. Specifically, GLoDe introduces a progressive denoising strategy to rectify incorrect pseudo labels by leveraging both global and local distribution information in the semantic space. The refined pseudo-labeled target language data significantly improves the model's generalization ability. Moreover, previous methods only consider improving the model with language-agnostic features, however, we argue that target language-specific features are also important and should never be ignored. To this end, we employ a simple auxiliary task to achieve this goal. Experimental results on two benchmark datasets with six target languages demonstrate that our proposed GLoDe significantly outperforms current state-of-the-art methods.

</details>

## 120. Finding Increasingly Large Extremal Graphs with AlphaZero and Tabu Search

<details>

<summary>Abstract</summary>

This work proposes a new learning-to-search benchmark and uses AI to discover new mathematical knowledge related to an open conjecture of Erdos (1975) in extremal graph theory. The problem is to find graphs with a given size (number of nodes) that maximize the number of edges without having 3- or 4-cycles. We formulate this as a sequential decision-making problem and compare AlphaZero, a neural network-guided tree search, with tabu search, a heuristic local search method. Using either method, by introducing a curriculum---jump-starting the search for larger graphs using good graphs found at smaller sizes---we improve the state-of-the-art lower bounds for several sizes. We also propose a flexible graph-generation environment and a permutation-invariant network architecture for learning to search in the space of graphs.

</details>

## 121. A New Paradigm for Counterfactual Reasoning in Fairness and Recourse

<details>

<summary>Abstract</summary>

Counterfactuals underpin numerous techniques for auditing and understanding artificial intelligence (AI) systems. The traditional paradigm for counterfactual reasoning in this literature is the interventional counterfactual, where hypothetical interventions are imagined and simulated. For this reason, the starting point for causal reasoning about legal protections and demographic data in AI is an imagined intervention on a legally-protected characteristic, such as ethnicity, race, gender, disability, age, etc. We ask, for example, what would have happened had your race been different? An inherent limitation of this paradigm is that some demographic interventions — like interventions on race — may not be well-defined or translate into the formalisms of interventional counterfactuals. In this work, we explore a new paradigm based instead on the backtracking counterfactual, where rather than imagine hypothetical interventions on legally-protected characteristics, we imagine alternate initial conditions while holding these characteristics fixed. We ask instead, what would explain a counterfactual outcome for you as you actually are or could be? This alternate framework allows us to address many of the same social concerns, but to do so while asking fundamentally different questions that do not rely on demographic interventions.

</details>

## 122. Individual Causal Structure Learning from Population Data

<details>

<summary>Abstract</summary>

Learning the causal structure of each individual plays a crucial role in neuroscience, biology, and so on. Existing methods consider data from each individual separately, which may yield inaccurate causal structure estimations in limited samples. To leverage more samples, we consider incorporating data from all individuals as population data. We observe that the variables of all individuals are influenced by the common environment variables they share. These shared environment variables can be modeled as latent variables and serve as a bridge connecting data from different individuals. In particular, we propose an Individual Linear Acyclic Model (ILAM) for each individual from population data, which models the individual's variables as being linearly influenced by their parents, in addition to environment variables and noise terms. Theoretical analysis shows that the model is identifiable when all environment variables are non-Gaussian, or even if some are Gaussian with an adequate diversity in the variance of noises for each individual. We then develop an individual causal structures learning method based on the Share Independence Component Analysis technique. Experimental results on synthetic and real-world data demonstrate the correctness of the method even when the sample size of each individual's data is small.

</details>

## 123. CGAP: Urban Region Representation Learning with Coarsened Graph Attention Pooling

<details>

<summary>Abstract</summary>

The explosion of massive urban data recently has provided us with a valuable opportunity to gain deeper insights into urban regions and the daily lives of residents. Urban region representation learning emerges as a crucial realm for fulfilling this task. Among deep learning approaches, graph neural networks (GNNs) have shown promise, given that city elements can be naturally represented as nodes with various connections between them as edges. However, many existing GNN approaches encounter challenges such as over-smoothing and limitations in capturing information from nodes in other regions, resulting in the loss of crucial urban information and a decline in region representation performance. To address these challenges, we leverage urban graph structure information and introduce a hierarchical graph pooling process called Coarsened Graph Attention Pooling (CGAP). CGAP features local attention units to create coarsened intermediate graphs and global features. Additionally, by incorporating urban region graphs and global features into a global attention layer, we harness relational information to enhance representation effectiveness. Furthermore, CGAP integrates region attributes such as Points of Interest (POIs) and inter-regional contexts like human mobility, enabling the exploitation of multi-modal urban data for more comprehensive representation learning. Experiments on three downstream tasks related to the UN Sustainable Development Goals validate the effectiveness of region representations learned by our approach. Experimental results and analyses demonstrate that CGAP excels in various socioeconomic prediction tasks compared to competitive baselines.

</details>

## 124. A Survey on Cross-Domain Sequential Recommendation 🌟

<details>

<summary>Abstract</summary>

Cross-domain sequential recommendation (CDSR) shifts the modeling of user preferences from flat to stereoscopic by integrating and learning interaction information from multiple domains at different granularities (ranging from inter-sequence to intra-sequence and from single-domain to cross-domain). In this survey, we initially define the CDSR problem using a four-dimensional tensor and then analyze its multi-type input representations under multidirectional dimensionality reductions. Following that, we provide a systematic overview from both macro and micro views. From a macro view, we abstract the multi-level fusion structures of various models across domains and discuss their bridges for fusion. From a micro view, focusing on the existing models, we specifically discuss the basic technologies and then explain the auxiliary learning technologies. Finally, we exhibit the available public datasets and the representative experimental results as well as provide some insights into future directions for research in CDSR.

</details>

## 125. A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation 🌟

<details>

<summary>Abstract</summary>

Many real-world datasets can be naturally represented as graphs, spanning a wide range of domains. However, the increasing complexity and size of graph datasets present significant challenges for analysis and computation. In response, graph reduction techniques have gained prominence for simplifying large graphs while preserving essential properties. In this survey, we aim to provide a comprehensive understanding of graph reduction methods, including graph sparsification, graph coarsening, and graph condensation. Specifically, we establish a unified definition for these methods and introduce a hierarchical taxonomy to categorize the challenges they address. Our survey then systematically reviews the technical details of these methods and emphasizes their practical applications across diverse scenarios. Furthermore, we outline critical research directions to ensure the continued effectiveness of graph reduction techniques.

</details>

## 126. Graph Neural Networks for Brain Graph Learning: A Survey 🌟

<details>

<summary>Abstract</summary>

Exploring the complex structure of the human brain is crucial for understanding its functionality and diagnosing brain disorders. Thanks to advancements in neuroimaging technology, a novel approach has emerged that involves modeling the human brain as a graph-structured pattern, with different brain regions represented as nodes and the functional relationships among these regions as edges. Moreover, graph neural networks (GNNs) have demonstrated a significant advantage in mining graph-structured data. Developing GNNs to learn brain graph representations for brain disorder analysis has recently gained increasing attention. However, there is a lack of systematic survey work summarizing current research methods in this domain. In this paper, we aim to bridge this gap by reviewing brain graph learning works that utilize GNNs. We first introduce the process of brain graph modeling based on common neuroimaging data. Subsequently, we systematically categorize current works based on the type of brain graph generated and the targeted research problems. To make this research accessible to a broader range of interested researchers, we provide an overview of representative methods and commonly used datasets, along with their implementation sources. Finally, we present our insights on future research directions. The repository of this survey is available at https://github.com/XuexiongLuoMQ/Awesome-Brain-Graph-Learning-with-GNNs.

</details>

## 127. Knowledge Distillation in Federated Learning: A Practical Guide

<details>

<summary>Abstract</summary>

Federated Learning (FL) enables the training of Deep Learning models without centrally collecting possibly sensitive raw data. The most used algorithms for FL are parameter-averaging based schemes (e.g., Federated Averaging) that, however, have well known limits, i.e., model homogeneity, high communication cost, poor performance in presence of heterogeneous data distributions. Federated adaptations of regular Knowledge Distillation (KD) can solve or mitigate the weaknesses of parameter-averaging FL algorithms while possibly introducing other trade-offs. In this article, we originally present a focused review of the state-of-the-art KD-based algorithms specifically tailored for FL, by providing both a novel classification of the existing approaches and a detailed technical description of their pros, cons, and tradeoffs.

</details>

## 128. Learning Structural Causal Models through Deep Generative Models: Methods, Guarantees, and Challenges

<details>

<summary>Abstract</summary>

This paper provides a comprehensive review of deep structural causal models (DSCMs), particularly focusing on their ability to answer counterfactual queries using observational data within known causal structures. It delves into the characteristics of DSCMs by analyzing the hypotheses, guarantees, and applications inherent to the underlying deep learning components and structural causal models, fostering a finer understanding of their capabilities and limitations in addressing different counterfactual queries. Furthermore, it highlights the challenges and open questions in the field of deep structural causal modeling. It sets the stages for researchers to identify future work directions and for practitioners to get an overview in order to find out the most appropriate methods for their needs.

</details>

## 129. Building Expressive and Tractable Probabilistic Generative Models: A Review 🌟

<details>

<summary>Abstract</summary>

We present a comprehensive survey of the advancements and techniques in the field of tractable probabilistic generative modeling, primarily focusing on Probabilistic Circuits (PCs). We provide a unified perspective on the inherent trade-offs between expressivity and tractability, highlighting the design principles and algorithmic extensions that have enabled building expressive and efficient PCs, and provide a taxonomy of the field. We also discuss recent efforts to build deep and hybrid PCs by fusing notions from deep neural models, and outline the challenges and open questions that can guide future research in this evolving field.

</details>

## 130. A Survey on Extractive Knowledge Graph Summarization: Applications, Approaches, Evaluation, and Future Directions

<details>

<summary>Abstract</summary>

With the continuous growth of large Knowledge Graphs (KGs), extractive KG summarization becomes a trending task. Aiming at distilling a compact subgraph with condensed information, it facilitates various downstream KG-based tasks. In this survey paper, we are among the first to provide a systematic overview of its applications and define a taxonomy for existing methods from its interdisciplinary studies. Future directions are also laid out based on our extensive and comparative review.

</details>

## 131. On the Essence and Prospect: An Investigation of Alignment Approaches for Big Models

<details>

<summary>Abstract</summary>

Big models have achieved revolutionary breakthroughs in the field of AI, but they also pose potential ethical and societal risks to humans. Addressing such problems, alignment technologies were introduced to make these models conform to human preferences and values. Despite the considerable advancements in the past year, various challenges lie in establishing the optimal alignment strategy, such as data cost and scalable oversight, and how to align remains an open question. In this survey paper, we comprehensively investigate value alignment approaches. We first unpack the historical context of alignment tracing back to the 1920s (where it comes from), then delve into the mathematical essence of alignment (what it is), shedding light on the inherent challenges. Following this foundation, we provide a detailed examination of existing alignment methods, which fall into three categories: RL-based Alignment, SFT-based Alignment, and Inference-Time Alignment, and demonstrate their intrinsic connections, strengths, and limitations, helping readers better understand this research area. In addition, two emerging topics, alignment goal and multimodal alignment, are also discussed as novel frontiers in the field. Looking forward, we discuss potential alignment paradigms and how they could handle remaining challenges, prospecting where future alignment will go.

</details>

## 132. AI-Enhanced Virtual Reality in Medicine: A Comprehensive Survey

<details>

<summary>Abstract</summary>

With the rapid advance of computer graphics and artificial intelligence technologies, the ways we interact with the world have undergone a transformative shift. Virtual Reality (VR) technology, aided by artificial intelligence (AI), has emerged as a dominant interaction media in multiple application areas, thanks to its advantage of providing users with immersive experiences. Among those applications, medicine is considered one of the most promising areas. In this paper, we present a comprehensive examination of the burgeoning field of AI-enhanced VR applications in medical care and services. By introducing a systematic taxonomy, we meticulously classify the pertinent techniques and applications into three well-defined categories based on different phases of medical diagnosis and treatment: Visualization Enhancement, VR-related Medical Data Processing, and VR-assisted Intervention. This categorization enables a structured exploration of the diverse roles that AI-powered VR plays in the medical domain, providing a framework for a more comprehensive understanding and evaluation of these technologies.nTo our best knowledge, this work is the first systematic survey of AI-powered VR systems in medical settings, laying a foundation for future research in this interdisciplinary domain.

</details>

## 133. More is Better: Deep Domain Adaptation with Multiple Sources 🌟

<details>

<summary>Abstract</summary>

In many practical applications, it is often difficult and expensive to obtain large-scale labeled data to train state-of-the-art deep neural networks. Therefore, transferring the learned knowledge from a separate, labeled source domain to an unlabeled or sparsely labeled target domain becomes an appealing alternative. However, direct transfer often results in significant performance decay due to domain shift. Domain adaptation (DA) aims to address this problem by aligning the distributions between the source and target domains. Multi-source domain adaptation (MDA) is a powerful and practical extension in which the labeled data may be collected from multiple sources with different distributions. In this survey, we first define various MDA strategies. Then we systematically summarize and compare modern MDA methods in the deep learning era from different perspectives, followed by commonly used datasets and a brief benchmark. Finally, we discuss future research directions for MDA that are worth investigating.

</details>

## 134. Continual Learning with Pre-Trained Models: A Survey

<details>

<summary>Abstract</summary>

Nowadays, real-world applications often face streaming data, which requires the learning system to absorb new knowledge as data evolves. Continual Learning (CL) aims to achieve this goal and meanwhile overcome the catastrophic forgetting of former knowledge when learning new ones. Typical CL methods build the model from scratch to grow with incoming data. However, the advent of the pre-trained model (PTM) era has sparked immense research interest, particularly in leveraging PTMs' robust representational capabilities. This paper presents a comprehensive survey of the latest advancements in PTM-based CL. We categorize existing methodologies into three distinct groups, providing a comparative analysis of their similarities, differences, and respective advantages and disadvantages. Additionally, we offer an empirical study contrasting various state-of-the-art methods to highlight concerns regarding fairness in comparisons. The source code to reproduce these evaluations is available at: https://github.com/sun-hailong/LAMDA-PILOT

</details>

## 135. MPGraf: a Modular and Pre-trained Graphformer for Learning to Rank at Web-scale (Extended Abstract)

<details>

<summary>Abstract</summary>

Both Transformer and Graph Neural Networks (GNNs) have been used in learning to rank (LTR), however, they adhere to two distinct yet complementary problem formulations, i.e., ranking score regression based on query-webpage pairs and link prediction within query-webpage bipartite graphs, respectively. Though it is possible to pre-train GNNs or Transformers on source datasets and fine-tune them subject to sparsely annotated LTR datasets separately, the source-target distribution shifts across the pairs and bipartite graphs domains make it extremely difficult to integrate these diverse models into a single LTR framework at a web-scale. We introduce the novel MPGraf model, which utilizes a modular and capsule-based pre-training approach, aiming to incorporate regression capacities from Transformers and link prediction capabilities of GNNs cohesively. We conduct extensive experiments to evaluate the performance of MPGraf using real-world datasets collected from large-scale search engines. The results show that MPGraf can outperform baseline algorithms on several major metrics. Further, we deploy and evaluate MPGraf atop a large-scale search engine with realistic web traffic via A/B tests, where we can still observe significant improvement. MPGraf performs consistently in both offline and online evaluations.

</details>

## 136. Content Matters: A Computational Investigation into the Effectiveness of Retrieval Practice and Worked Examples (Extended Abstract)

<details>

<summary>Abstract</summary>

In this paper, we argue that computational models of learning can contribute precise theory to explain surprising student learning phenomena. In some past studies, practice produces better learning than studying examples, whereas other studies show the opposite result. We explain this contradiction by suggesting that retrieval practice and example study involve different learning cognitive processes, memorization and induction, and each process is optimal for different types of knowledge. We implement and test this theoretical explanation by extending an AI model of human cognition to include both memory and induction processes and comparing the behavior of the simulated learners to those of human participants. We show that the behavior of simulated learners with forgetting matches that of human participants better than simulated learners without forgetting. Simulated learners with forgetting learn best using retrieval practice in situations that emphasize memorization (such as learning facts), whereas studying examples improves learning when multiple pieces of information are available, so induction and generalization are necessary (such as learning skills).

</details>

## 137. All in One: Multi-task Prompting for Graph Neural Networks (Extended Abstract)

<details>

<summary>Abstract</summary>

This paper is an extended abstract of our original work published in KDD23, where we won the best research paper award. The paper introduces a novel approach to bridging the gap between pre-trained graph models and the diverse tasks they’re applied to, inspired by the success of prompt learning in NLP. Recognizing the challenge of aligning pre-trained models with varied graph tasks (node level, edge level, and graph level), which can lead to negative transfer and poor performance, we propose a multi-task prompting method for graphs. This method involves unifying graph and language prompt formats, enabling NLP’s prompting strategies to be adapted for graph tasks. By analyzing the task space of graph applications, we reformulate problems to fit graph-level tasks and apply meta-learning to improve prompt initialization for multiple tasks. Experiments show our method’s effectiveness in enhancing model performance across different graph tasks. Beyond the original work, in this extended abstract, we further discuss the graph prompt from a bigger picture and provide some of the latest work toward this area.

</details>

## 138. Human-AI Interaction Generation: A Connective Lens for Generative AI and Procedural Content Generation

<details>

<summary>Abstract</summary>

Generative AI has recently gained popularity as a paradigm for content generation. In this paper, we link this paradigm to an older one: Procedural Content Generation (PCG). We propose a lens to identify the commonalities between both paradigms that we call human-AI interactive generation. Using this lens, we identify three beneficial attributes then survey recent related work and summarize relevant findings.

</details>

## 139. A Little of That Human Touch: Achieving Human-Centric Explainable AI via Argumentation

<details>

<summary>Abstract</summary>

As data-driven AI models achieve unprecedented feats across previously unthinkable tasks, the diminishing levels of interpretability of their increasingly complex architectures can often be sidelined in place of performance. If we are to comprehend and trust these AI models as they advance, it is clear that symbolic methods, given their unparalleled strengths in knowledge representation and reasoning, can play an important role in explaining AI models. In this paper, I discuss some of the ways in which one branch of such methods, computational argumentation, given its human-like nature, can be used to tackle this problem. I first outline a general paradigm for this area of explainable AI, before detailing a prominent methodology therein which we have pioneered. I then illustrate how this approach has been put into practice with diverse AI models and types of explanations, before looking ahead to challenges, future work and the outlook in this field.

</details>

## 140. Computational Argumentation: Reasoning, Dynamics, and Supporting Explainability

<details>

<summary>Abstract</summary>

This overview accompanies the author's Early Career Track presentation. We survey recent research and research agenda of the author, focusing on contributions in the area of computational argumentation. Contributions span from foundations of static and dynamic forms of argumentative reasoning and approaches to support explainability, e.g., analysis of the computational complexity of argumentative reasoning and algorithmic approaches.

</details>

## 141. PyXAI: An XAI Library for Tree-Based Models

<details>

<summary>Abstract</summary>

PyXAI (Python eXplainable AI) is a Python library designed for providing explanations and cor- recting tree-based Machine Learning (ML) models. It is suited to decision trees, random forests, and boosted trees, when used for regression or classification tasks. In contrast to many model-agnostic approaches to XAI, PyXAI exploits the model it- self to generate explanations, ensuring them to be faithful. PyXAI includes several algorithms for the generation of explanations, which can be abductive or contrastive. PyXAI also includes algorithms for correcting tree-based models when their predictions conflict with pieces of user knowledge.

</details>

