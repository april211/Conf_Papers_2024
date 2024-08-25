# KDD 2024

## 0. A Learned Generalized Geodesic Distance Function-Based Approach for Node Feature Augmentation on Graphs

<details>

<summary>Abstract</summary>

Geodesic distances on manifolds have numerous applications in image processing, computer graphics and computer vision. In this work, we introduce an approach called 'LGGD' (Learned Generalized Geodesic Distances). This method involves generating node features by learning a generalized geodesic distance function through a training pipeline that incorporates training data, graph topology and the node content features. The strength of this method lies in the proven robustness of the generalized geodesic distances to noise and outliers. Our contributions encompass improved performance in node classification tasks, competitive results with state-of-the-art methods on real-world graph datasets, the demonstration of the learnability of parameters within the generalized geodesic equation on graph, and dynamic inclusion of new labels.

</details>

## 1. Graph Mamba: Towards Learning on Graphs with State Space Models 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have shown promising potential in graph representation learning. The majority of GNNs define a local message-passing mechanism, propagating information over the graph by stacking multiple layers. These methods, however, are known to suffer from two major limitations: over-squashing and poor capturing of long-range dependencies. Recently, Graph Transformers (GTs) emerged as a powerful alternative to Message-Passing Neural Networks (MPNNs). GTs, however, have quadratic computational cost, lack inductive biases on graph structures, and rely on complex Positional Encodings (PE). In this paper, we show that while Transformers, complex message-passing, and PE are sufficient for good performance in practice, neither is necessary. Motivated by the recent success of State Space Models (SSMs), we present Graph Mamba Networks (GMNs), a framework for a new class of GNNs based on selective SSMs. We discuss the new challenges when adapting SSMs to graph-structured data, and present four required steps to design GMNs, where we choose (1) Neighborhood Tokenization, (2) Token Ordering, (3) Architecture of SSM Encoder, and (4) Local Encoding. We provide theoretical justification for the power of GMNs, and experimentally show that GMNs attain an outstanding performance in various benchmark datasets. The code is available in this link.

</details>

## 2. Where Have You Been? A Study of Privacy Risk for Point-of-Interest Recommendation

<details>

<summary>Abstract</summary>

As location-based services (LBS) have grown in popularity, more human mobility data has been collected. The collected data can be used to build machine learning (ML) models for LBS to enhance their performance and improve overall experience for users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges.

</details>

## 3. Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias

<details>

<summary>Abstract</summary>

Collaborative Filtering~(CF) typically suffers from the significant challenge of popularity bias due to the uneven distribution of items in real-world datasets. This bias leads to a significant accuracy gap between popular and unpopular items. It not only hinders accurate user preference understanding but also exacerbates the Matthew effect in recommendation systems. To alleviate popularity bias, existing efforts focus on emphasizing unpopular items or separating the correlation between item representations and their popularity. Despite the effectiveness, existing works still face two persistent challenges: (1) how to extract common supervision signals from popular items to improve the unpopular item representations, and (2) how to alleviate the representation separation caused by popularity bias. In this work, we conduct an empirical analysis of popularity bias and propose &lt;u&gt;P&lt;/u&gt;opularity-&lt;u&gt;A&lt;/u&gt;ware &lt;u&gt;A&lt;/u&gt;lignment and &lt;u&gt;C&lt;/u&gt;ontrast (PAAC) to address two challenges. Specifically, we use the common supervisory signals modeled in popular item representations and propose a novel popularity-aware supervised alignment module to learn unpopular item representations. Additionally, we suggest re-weighting the contrastive learning loss to mitigate the representation separation from a popularity-centric perspective. Finally, we validate the effectiveness and rationale of PAAC in mitigating popularity bias through extensive experiments on three real-world datasets. Our code is available at https://github.com/miaomiao-cai2/KDD2024-PAAC.

</details>

## 4. Tackling Instance-Dependent Label Noise with Class Rebalance and Geometric Regularization

<details>

<summary>Abstract</summary>

In label-noise learning, accurately identifying the transition matrix is crucial for developing statistically consistent classifiers. This task is complicated by instance-dependent noise, which introduces identifiability challenges in the absence of stringent assumptions. Existing methods use neural networks to estimate the transition matrix by initially extracting confident clean instances. However, this extraction process is hindered by severe inter-class imbalance and a bias toward selecting unambiguous intra-class instances, leading to a distorted understanding of noise patterns. To tackle these challenges, our paper introduces a Class Rebalance and Geometric Regularization-based Framework (CRGR). CRGR employs a smoothed, noise-tolerant reweighting mechanism to equilibrate inter-class representation, thereby mitigating the risk of model overfitting to dominant classes. Additionally, recognizing that instances with similar characteristics often exhibit parallel noise patterns, we propose that the transition matrix should mirror the similarity of the feature space. This insight promotes the inclusion of ambiguous instances in training, serving as a form of geometric regularization. Such a strategy enhances the model's ability to navigate diverse noise patterns and strengthens its generalization capabilities. By addressing both inter-class and intra-class biases, CRGR offers a more balanced and robust classification model. Extensive experiments on both synthetic and real-world datasets demonstrate CRGR's superiority over existing state-of-the-art methods, significantly boosting classification accuracy and showcasing its effectiveness in handling instance-dependent noise.

</details>

## 5. DiffusionE: Reasoning on Knowledge Graphs via Diffusion-based Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have demonstrated powerful capabilities in reasoning within Knowledge Graphs (KGs), gathering increasing attention. Our idea stems from the observation that the prior work typically employs hand-designed or sample-designed paradigms in the process of message propagation, engaging a set of adjacent entities at each step of propagation. As a result, such methods struggle with the increasing number of entities involved as propagation steps extend. Moreover, they neglect the message interactions between adjacent entities and propagation relations in KG reasoning, leading to semantic inconsistency during the message aggregation phase. To address these issues, we introduce a novel knowledge graph embedding method through a diffusion process, termed DiffusionE. Specifically, we reformulate the message propagation in knowledge reasoning as a diffusion process, regarding the message semantics as the diffusion signal. In this sense, guided by semantic information, messages can be transmitted between nodes effectively and adaptively. Furthermore, the theoretical analysis suggests our method can leverage an optimal diffusivity for message propagation in the semantic interactions of KGs. It shows that DiffusionE effectively leverages message interactions between entities and propagation relations, ensuring semantic consistency in KG reasoning. Comprehensive experiments reveal that our method attains state-of-the-art performance compared to prior work on several well-established benchmarks.

</details>

## 6. Harm Mitigation in Recommender Systems under User Preference Dynamics

<details>

<summary>Abstract</summary>

We consider a recommender system that takes into account the interplay between recommendations, the evolution of user interests, and harmful content. We model the impact of recommendations on user behavior, particularly the tendency to consume harmful content. We seek recommendation policies that establish a tradeoff between maximizing click-through rate (CTR) and mitigating harm. We establish conditions under which the user profile dynamics have a stationary point, and propose algorithms for finding an optimal recommendation policy at stationarity. We experiment on a semi-synthetic movie recommendation setting initialized with real data and observe that our policies outperform baselines at simultaneously maximizing CTR and mitigating harm.

</details>

## 7. QGRL: Quaternion Graph Representation Learning for Heterogeneous Feature Data Clustering

<details>

<summary>Abstract</summary>

Clustering is one of the most commonly used techniques for unsupervised data analysis. As real data sets are usually composed of numerical and categorical features that are heterogeneous in nature, the heterogeneity in the distance metric and feature coupling prevents deep representation learning from achieving satisfactory clustering accuracy. Currently, supervised Quaternion Representation Learning (QRL) has achieved remarkable success in efficiently learning informative representations of coupled features from multiple views derived endogenously from the original data. To inherit the advantages of QRL for unsupervised heterogeneous feature representation learning, we propose a deep QRL model that works in an encoder-decoder manner. To ensure that the implicit couplings of heterogeneous feature data can be well characterized by representation learning, a hierarchical coupling encoding strategy is designed to convert the data set into an attributed graph to be the input of QRL. We also integrate the clustering objective into the model training to facilitate a joint optimization of the representation and clustering. Extensive experimental evaluations illustrate the superiority of the proposed Quaternion Graph Representation Learning (QGRL) method in terms of clustering accuracy and robustness to various data sets composed of arbitrary combinations of numerical and categorical features. The source code is opened at https://github.com/Juny-Chen/QGRL.git.

</details>

## 8. Explicit and Implicit Modeling via Dual-Path Transformer for Behavior Set-informed Sequential Recommendation 🌟

<details>

<summary>Abstract</summary>

Sequential recommendation (SR) and multi-behavior sequential recommendation (MBSR) both come from real-world scenarios. Compared with SR, MBSR takes into account the dependencies of different behaviors. We find that most existing works on MBSR are studied in the context of e-commerce scenarios. In terms of the data format of the behavior types, we observe that the conventional label-formatted data carries limited information and is inadequate for scenarios like social media. With this observation, we introducebehavior set and extend MBSR to behavior set-informed sequential recommendation (BSSR). In BSSR, behavior dependencies become more complex and personalized, and user interest arousal may lack explicit contextual associations. To delve into the dynamics inhered within a behavior set and adaptively tailor recommendation lists upon its variability, we propose a novel solution called Explicit and Implicit modeling via Dual-Path Transformer (EIDP) for BSSR. Our EIDP adopts a dual-path architecture, distinguishing between explicit modeling path (EMP) and implicit modeling path (IMP) based on whether to directly incorporate the behavior representations. EMP features the personalized behavior set-wise transition pattern extractor (PBS-TPE) as its core component. It couples behavioral representations with both the items and positions to explore intra-behavior dynamics within a behavior set at a fine granularity. IMP utilizes light multi-head self-attention blocks (L-MSAB) as encoders under specific behavior types. The obtained multi-view representations are then aggregated by cross-behavior attention fusion (CBAF), using the behavior set of the next time step as a guidance to extract collaborative semantics at the behavioral level. Extensive experiments on two real-world datasets demonstrate the effectiveness of our EIDP. We release the implementation code at: https://github.com/OshiNoCSMA/EIDP.

</details>

## 9. Shopping Trajectory Representation Learning with Pre-training for E-commerce Customer Understanding and Recommendation 🌟

<details>

<summary>Abstract</summary>

Understanding customer behavior is crucial for improving service quality in large-scale E-commerce. This paper proposes C-STAR, a new framework that learns compact representations from customer shopping journeys, with good versatility to fuel multiple downstream customer-centric tasks. We define the notion of shopping trajectory that encompasses customer interactions at the level of product categories, capturing the overall flow of their browsing and purchase activities. C-STAR excels at modeling both inter-trajectory distribution similarity-the structural similarities between different trajectories, and intra-trajectory semantic correlation-the semantic relationships within individual ones. This coarse-to-fine approach ensures informative trajectory embeddings for representing customers. To enhance embedding quality, we introduce a pre-training strategy that captures two intrinsic properties within the pre-training data. Extensive evaluation on large-scale industrial and public datasets demonstrates the effectiveness of C-STAR across three diverse customer-centric tasks. These tasks empower customer profiling and recommendation services for enhancing personalized shopping experiences on our E-commerce platform.

</details>

## 10. DyGKT: Dynamic Graph Learning for Knowledge Tracing

<details>

<summary>Abstract</summary>

Knowledge Tracing aims to assess student learning states by predicting their performance in answering questions. Different from the existing research which utilizes fixed-length learning sequence to obtain the student states and regards KT as a static problem, this work is motivated by three dynamical characteristics: 1) The scales of students answering records are constantly growing; 2) The semantics of time intervals between the records vary; 3) The relationships between students, questions and concepts are evolving. The three dynamical characteristics above contain the great potential to revolutionize the existing knowledge tracing methods. Along this line, we propose a Dynamic Graph-based Knowledge Tracing model, namely DyGKT. In particular, a continuous-time dynamic question-answering graph for knowledge tracing is constructed to deal with the infinitely growing answering behaviors, and it is worth mentioning that it is the first time dynamic graph learning technology is used in this field. Then, a dual time encoder is proposed to capture long-term and short-term semantics among the different time intervals. Finally, a multiset indicator is utilized to model the evolving relationships between students, questions, and concepts via the graph structural feature. Numerous experiments are conducted on five real-world datasets, and the results demonstrate the superiority of our model. All the used resources are publicly available at https://github.com/PengLinzhi/DyGKT.

</details>

## 11. Resurrecting Label Propagation for Graphs with Heterophily and Label Noise 🌟

<details>

<summary>Abstract</summary>

Label noise is a common challenge in large datasets, as it can significantly degrade the generalization ability of deep neural networks. Most existing studies focus on noisy labels in computer vision; however, graph models encompass both node features and graph topology as input, and become more susceptible to label noise through message-passing mechanisms. Recently, only a few works have been proposed to tackle the label noise on graphs. One significant limitation is that they operate under the assumption that the graph exhibits homophily and that the labels are distributed smoothly. However, real-world graphs can exhibit varying degrees of heterophily, or even be dominated by heterophily, which results in the inadequacy of the current methods.In this paper, we study graph label noise in the context of arbitrary heterophily, with the aim of rectifying noisy labels and assigning labels to previously unlabeled nodes. We begin by conducting two empirical analyses to explore the impact of graph homophily on graph label noise. Following observations, we propose a efficient algorithm, denoted as R2LP. Specifically, R2LP is an iterative algorithm with three steps: (1) reconstruct the graph to recover the homophily property, (2) utilize label propagation to rectify the noisy labels, (3) select high-confidence labels to retain for the next iteration. By iterating these steps, we obtain a set of ''correct'' labels, ultimately achieving high accuracy in the node classification task. The theoretical analysis is also provided to demonstrate its remarkable denoising effect. Finally, we perform experiments on ten benchmark datasets with different levels of graph heterophily and various types of noise. In these experiments, we compare the performance of R2LP against ten typical baseline methods. Our results illustrate the superior performance of the proposed \o{}urs. The code and data of this paper can be accessed at: https://github.com/cy623/R2LP.git.

</details>

## 12. Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction

<details>

<summary>Abstract</summary>

Accurately predicting the popularity of multimodal user-generated content (UGC) is fundamental for many real-world applications such as online advertising and recommendation. Existing approaches generally focus on limited contextual information within individual UGCs, yet overlook the potential benefit of exploiting meaningful knowledge in relevant UGCs. In this work, we propose RAGTrans, an aspect-aware retrieval-augmented multi-modal hypergraph transformer that retrieves pertinent knowledge from a multi-modal memory bank and enhances UGC representations via neighborhood knowledge aggregation on multi-model hypergraphs. In particular, we initially retrieve relevant multimedia instances from a large corpus of UGCs via the aspect information and construct a knowledge-enhanced hypergraph based on retrieved relevant instances. This allows capturing meaningful contextual information across the data. We then design a novel bootstrapping hypergraph transformer on multimodal hypergraphs to strengthen UGC representations across modalities via customizing a propagation algorithm to effectively diffuse information across nodes and edges. Additionally, we propose a user-aware attention-based fusion module to comprise the enriched UGC representations for popularity prediction. Extensive experiments on real-world social media datasets demonstrate that RAGTrans outperforms state-of-the-art popularity prediction models across settings.

</details>

## 13. Enhancing Contrastive Learning on Graphs with Node Similarity 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNN) have proven successful for graph-related tasks. However, many GNNs methods require labeled data, which is challenging to obtain. To tackle this, graph contrastive learning (GCL) have gained attention. GCL learns by contrasting similar nodes (positives) and dissimilar nodes (negatives). Current GCL methods, using data augmentation for positive samples and random selection for negative samples, can be sub-optimal due to limited positive samples and the possibility of false-negative samples. In this study, we propose an enhanced objective addressing these issues. We first introduce an ideal objective with all positive and no false-negative samples, then transform it probabilistically based on sampling distributions. We next model these distributions with node similarity and derive an enhanced objective. Comprehensive experiments have shown the effectiveness of the proposed enhanced objective for a broad set of GCL models.

</details>

## 14. Relevance Meets Diversity: A User-Centric Framework for Knowledge Exploration Through Recommendations

<details>

<summary>Abstract</summary>

Providing recommendations that are both relevant and diverse is a key consideration of modern recommender systems. Optimizing both of these measures presents a fundamental trade-off, as higher diversity typically comes at the cost of relevance, resulting in lower user engagement. Existing recommendation algorithms try to resolve this trade-off by combining the two measures, relevance and diversity, into one aim and then seeking recommendations that optimize the combined objective, for a given number of items. Traditional approaches, however, do not consider the user interaction with the suggested items. In this paper, we put the user at the central stage, and build on the interplay between relevance, diversity, and user behavior. In contrast to applications where the goal is solely to maximize engagement, we focus on scenarios aiming at maximizing the total amount of knowledge encountered by the user. We use diversity as a surrogate for the amount of knowledge obtained by the user while interacting with the system, and we seek to maximize diversity. We propose a probabilistic user-behavior model in which users keep interacting with the recommender system as long as they receive relevant suggestions, but they may stop if the relevance of the recommended items drops. Thus, for a recommender system to achieve a high-diversity measure, it will need to produce recommendations that are both relevant and diverse. Finally, we propose a novel recommendation strategy that combines relevance and diversity by a copula function. We conduct an extensive evaluation of the proposed methodology over multiple datasets, and we show that our strategy outperforms several state-of-the-art competitors. Our implementation is publicly available at https://github.com/EricaCoppolillo/EXPLORE.

</details>

## 15. Leveraging Pedagogical Theories to Understand Student Learning Process with Graph-based Reasonable Knowledge Tracing

<details>

<summary>Abstract</summary>

Knowledge tracing (KT) is a crucial task in intelligent education, focusing on predicting students' performance on given questions to trace their evolving knowledge. The advancement of deep learning in this field has led to deep-learning knowledge tracing (DLKT) models that prioritize high predictive accuracy. However, many existing DLKT methods overlook the fundamental goal of tracking students' dynamical knowledge mastery. These models do not explicitly model knowledge mastery tracing processes or yield unreasonable results that educators find difficulty to comprehend and apply in real teaching scenarios. In response, our research conducts a preliminary analysis of mainstream KT approaches to highlight and explain such unreasonableness. We introduce GRKT, a graph-based reasonable knowledge tracing method to address these issues. By leveraging graph neural networks, our approach delves into the mutual influences of knowledge concepts, offering a more accurate representation of how the knowledge mastery evolves throughout the learning process. Additionally, we propose a fine-grained and psychological three-stage modeling process as knowledge retrieval, memory strengthening, and knowledge learning/forgetting, to conduct a more reasonable knowledge tracing process. Comprehensive experiments demonstrate that GRKT outperforms eleven baselines across three datasets, not only enhancing predictive accuracy but also generating more reasonable knowledge tracing results. This makes our model a promising advancement for practical implementation in educational settings. The source code is available at https://github.com/JJCui96/GRKT.

</details>

## 16. AGS-GNN: Attribute-guided Sampling for Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

We propose AGS-GNN, a novel attribute-guided sampling algorithm for Graph Neural Networks (GNNs). AGS-GNN exploits the node features and the connectivity structure of a graph while simultaneously adapting for both homophily and heterophily in graphs. In homophilic graphs, vertices of the same class are more likely to be adjacent, but vertices of different classes tend to be adjacent in heterophilic graphs. GNNs have been successfully applied to homophilic graphs, but their utility to heterophilic graphs remains challenging. The state-of-the-art GNNs for heterophilic graphs use the full neighborhood of a node instead of sampling it, and hence do not scale to large graphs and are not inductive. We develop dual-channel sampling techniques based on feature-similarity and feature-diversity to select subsets of neighbors for a node that capture adaptive information from homophilic and heterophilic neighborhoods. Currently, AGS-GNN is the only algorithm that explicitly controls homophily in the sampled subgraph through similar and diverse neighborhood samples. For diverse neighborhood sampling, we employ submodularity, a novel contribution in this context. We pre-compute the sampling distribution in parallel, achieving the desired scalability. Using an extensive dataset consisting of 35 small (&lt; 100K nodes) and large (- 100K nodes) homophilic and heterophilic graphs, we demonstrate the superiority of AGS-GNN compared to the state-of-the-art approaches. AGS-GNN achieves test accuracy comparable to the best-performing heterophilic GNNs, even outperforming methods that use the entire graph for node classification. AGS-GNN converges faster than methods that sample neighborhoods randomly, and can be incorporated into existing GNN models that employ node or graph sampling.

</details>

## 17. Divide and Denoise: Empowering Simple Models for Robust Semi-Supervised Node Classification against Label Noise 🌟

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) based on message passing have achieved remarkable performance in graph machine learning. By combining it with the power of pseudo labeling, one can further push forward the performance on the task of semi-supervised node classification. However, most existing works assume that the training node labels are purely noise-free, while this strong assumption usually does not hold in practice. GNNs will overfit the noisy training labels and the adverse effects of mislabeled nodes can be exaggerated by being propagated to the remaining nodes through the graph structure, exacerbating the model failure. Worse still, the noisy pseudo labels could also largely undermine the model's reliability without special treatment. In this paper, we revisit the role of (1) message passing and (2) pseudo labels in the studied problem and try to address two denoising subproblems from the model architecture and algorithm perspective, respectively. Specifically, we first develop a label-noise robust GNN that discards the coupled message-passing scheme. Despite its simple architecture, this learning backbone prevents overfitting to noisy labels and also inherently avoids the noise propagation issue. Moreover, we propose a novel reliable graph pseudo labeling algorithm that can effectively leverage the knowledge of unlabeled nodes while mitigating the adverse effects of noisy pseudo labels. Based on those novel designs, we can attain exceptional effectiveness and efficiency in solving the studied problem. We conduct extensive experiments on benchmark datasets for semi-supervised node classification with different levels of label noise and show new state-of-the-art performance. The code is available at https://github.com/DND-NET/DND-NET.

</details>

## 18. Unsupervised Alignment of Hypergraphs with Different Scales

<details>

<summary>Abstract</summary>

People usually interact in groups, and such groups may appear on different platforms. For instance, people often create various group chats on messaging apps (e.g., Facebook Messenger and WhatsApp) to communicate with families, friends, or colleagues. How do we identify the same people across the two platforms based on the information about the groups? This gives rise to the hypergraph alignment problem, whose objective is to find the correspondences between the sets of nodes of two hypergraphs. In a hypergraph, a node represents a person, and each hyperedge represents a group of several people. In addition, the two sets of hyperedges in the two hypergraphs can vary significantly in scales as people may use different apps at different time periods.In this work, we propose and tackle the problem of unsupervised hypergraph alignment. Given two hypergraphs with potentially different scales and without any side information or prior ground-truth correspondences, we develop \O{}urMethod, a learning framework, to find node correspondences across the two hypergraphs. \O{}urMethod directly addresses each challenge of the problem. In particular, it (a) extracts node features from the hypergraph topology, (b) employs contrastive learning, as a "supervised pseudo-alignment'' task to pre-train the learning model (c) applies topological augmentation to help a generative adversarial network to align the two embedding spaces from the two hypergraphs. The purpose of augmentation is to add virtual hyperedges from one hypergraph in order to the other to resolve the scale difference and share information across the two hypergraphs. Our extensive experiments on 12 real-world datasets demonstrate the significant and consistent superiority of \O{}urMethod over the baseline approaches.

</details>

## 19. IDEA: A Flexible Framework of Certified Unlearning for Graph Neural Networks

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have been increasingly deployed in a plethora of applications. However, the graph data used for training may contain sensitive personal information of the involved individuals. Once trained, GNNs typically encode such information in their learnable parameters. As a consequence, privacy leakage may happen when the trained GNNs are deployed and exposed to potential attackers. Facing such a threat, machine unlearning for GNNs has become an emerging technique that aims to remove certain personal information from a trained GNN. Among these techniques, certified unlearning stands out, as it provides a solid theoretical guarantee of the information removal effectiveness. Nevertheless, most of the existing certified unlearning methods for GNNs are only designed to handle node and edge unlearning requests. Meanwhile, these approaches are usually tailored for either a specific design of GNN or a specially designed training objective. These disadvantages significantly jeopardize their flexibility. In this paper, we propose a principled framework named IDEA to achieve flexible and certified unlearning for GNNs. Specifically, we first instantiate four types of unlearning requests on graphs, and then we propose an approximation approach to flexibly handle these unlearning requests over diverse GNNs. We further provide theoretical guarantee of the effectiveness for the proposed approach as a certification. Different from existing alternatives, IDEA is not designed for any specific GNNs or optimization objectives to perform certified unlearning, and thus can be easily generalized. Extensive experiments on real-world datasets demonstrate the superiority of IDEA in multiple key perspectives.

</details>

## 20. DisCo: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation

<details>

<summary>Abstract</summary>

Recommender systems play important roles in various applications such as e-commerce, social media, etc. Conventional recommendation methods usually model the collaborative signals within the tabular representation space. Despite the personalization modeling and the efficiency, the latent semantic dependencies are omitted. Methods that introduce semantics into recommendation then emerge, injecting knowledge from the semantic representation space where the general language understanding are compressed. However, existing semantic-enhanced recommendation methods focus on aligning the two spaces, during which the representations of the two spaces tend to get close while the unique patterns are discarded and not well explored. In this paper, we propose DisCo to Disentangle the unique patterns from the two representation spaces and Collaborate the two spaces for recommendation enhancement, where both the specificity and the consistency of the two spaces are captured. Concretely, we propose 1) a dual-side attentive network to capture the intra-domain patterns and the inter-domain patterns, 2) a sufficiency constraint to preserve the task-relevant information of each representation space and filter out the noise, and 3) a disentanglement constraint to avoid the model from discarding the unique information. These modules strike a balance between disentanglement and collaboration of the two representation spaces to produce informative pattern vectors, which could serve as extra features and be appended to arbitrary recommendation backbones for enhancement. Experiment results validate the superiority of our method against different models and the compatibility of DisCo over different backbones. Various ablation studies and efficiency analysis are also conducted to justify each model component.

</details>

## 21. Disentangled Multi-interest Representation Learning for Sequential Recommendation 🌟

<details>

<summary>Abstract</summary>

Recently, much effort has been devoted to modeling users' multi-interests (aka multi-faceted preferences) based on their behaviors, aiming to accurately capture users' complex preferences. Existing methods attempt to model each interest of users through a distinct representation, but these multi-interest representations easily collapse into similar ones due to a lack of effective guidance. In this paper, we propose a generic multi-interest method for sequential recommendation, achieving disentangled representation learning of diverse interests technically and theoretically. To alleviate the collapse issue of multi-interests, we propose to conduct item partition guided by their likelihood of being co-purchased in a global view. It can encourage items in each group to focus on a discriminated interest, thus achieving effective disentangled learning of multi-interests. Specifically, we first prove the theoretical connection between item partition and spectral clustering, demonstrating its effectiveness in alleviating item-level and facet-level collapse issues that hinder existing disentangled methods. To efficiently optimize this problem, we then propose a Markov Random Field (MRF)-based method that samples small-scale sub-graphs from two separate MRFs, thus it can be approximated with a cross-entropy loss and optimized through contrastive learning. Finally, we perform multi-task learning to seamlessly align item partition learning with multi-interest modeling for more accurate recommendation. Experiments on three real-world datasets show that our method significantly outperforms state-of-the-art methods and can flexibly integrate with existing multi-interest models as a plugin to enhance their performances.

</details>

## 22. Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation

<details>

<summary>Abstract</summary>

Self-supervised Heterogeneous Graph Representation (SSHGRL) learning is widely used in data mining. The latest SSHGRL methods normally use metapaths to describe the heterogeneous information (multiple relations and node types) to learn the heterogeneous graph representation and achieve impressive results. However, establishing metapaths requires lofty computational costs that are too high for the medium and large graphs. To this end, this paper proposes a Reserving-Masking-Reconstruction (RMR) model that can fully consider heterogeneous information without relying on the metapaths. In detail, we propose a reserving method to reserve to-be-masked nodes' (target nodes) information before graph masking. Second, we split the reserved graph into relation subgraphs according to the type of relations that require much less computational overheads than metapath. Then, the target nodes in each relation subgraph are randomly masked with minimal topology information loss. After, a novel reconstruction method is proposed to reconstruct the masked nodes on different relation subgraphs to establish the self-supervised signal. The proposed method requires low computational complexity and can establish a self-supervised signal without deeply changing the graph topology. Experimental results show the proposed method achieves state-of-the-art records on medium and large-scale heterogeneous graphs and competitive records on small-scale heterogeneous graphs. The code is available at https://github.com/DuanhaoranCC/RMR.

</details>

## 23. Label Shift Correction via Bidirectional Marginal Distribution Matching

<details>

<summary>Abstract</summary>

Due to the timeliness and uncertainty of data acquisition, label shift, which assumes that the source (training) and target (test) label distributions differ, occurs with the changing environment and reduces the generalization ability of traditional models. To correct the label shift, existing methods estimate the true label distribution by prediction of target data from a source classifier, which results in high variance, especially with large label shift. In this paper, we tackle this problem by proposing a novel approach termed as Label Shift Correction via Bidirectional Marginal Distribution Matching (BMDM). Our approach matchs the label and feature marginal distributions simultaneously to ensure the stability of estimated class proportions. We prove theoretically that there is a unique optimal solution, i.e., true target label distribution, for our approach under mild conditions, and an efficient optimization strategy is also proposed. On this basis, in multi-shot scenario where label distribution changes continuously, we extend BMDM by designing a new distribution matching mechanism and constructing a regularization term that constrains the direction of label distribution change. Extensive experimental results validate the effectiveness of our approach over existing state-of-the-arts methods.

</details>

## 24. GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models

<details>

<summary>Abstract</summary>

This work studies self-supervised graph learning for text-attributed graphs (TAGs) where nodes are represented by textual attributes. Unlike traditional graph contrastive methods that perturb the numerical feature space and alter the graph's topological structure, we aim to improve view generation through language supervision. This is driven by the prevalence of textual attributes in real applications, which complement graph structures with rich semantic information. However, this presents challenges because of two major reasons. First, text attributes often vary in length and quality, making it difficulty to perturb raw text descriptions without altering their original semantic meanings. Second, although text attributes complement graph structures, they are not inherently well-aligned. To bridge the gap, we introduce GAugLLM, a novel framework for augmenting TAGs. It leverages advanced large language models like Mistral to enhance self-supervised graph learning. Specifically, we introduce a mixture-of-prompt-expert technique to generate augmented node features. This approach adaptively maps multiple prompt experts, each of which modifies raw text attributes using prompt engineering, into numerical feature space. Additionally, we devise a collaborative edge modifier to leverage structural and textual commonalities, enhancing edge augmentation by examining or building connections between nodes. Empirical results across five benchmark datasets spanning various domains underscore our framework's ability to enhance the performance of leading contrastive methods (e.g., BGRL, GraphCL, and GBT) as a plug-in tool. Notably, we observe that the augmented features and graph structure can also enhance the performance of standard generative methods (e.g., GraphMAE and S2GAE), as well as popular graph neural networks (e.g., GCN and GAT). The open-sourced implementation of our GAugLLM is available at https://github.com/NYUSHCS/GAugLLM.

</details>

## 25. Influence Maximization via Graph Neural Bandits

<details>

<summary>Abstract</summary>

We consider a ubiquitous scenario in the study of Influence Maximization (IM), in which there is limited knowledge about the topology of the diffusion network. We set the IM problem in a multi-round diffusion campaign, aiming to maximize the number of distinct users that are influenced. Leveraging the capability of bandit algorithms to effectively balance the objectives of exploration and exploitation, as well as the expressivity of neural networks, our study explores the application of neural bandit algorithms to the IM problem. We propose the framework IM-GNB (Influence Maximization with Graph Neural Bandits), where we provide an estimate of the users' probabilities of being influenced by influencers (also known as diffusion seeds). This initial estimate forms the basis for constructing both an exploitation graph and an exploration one. Subsequently, IM-GNB handles the exploration-exploitation tradeoff, by selecting seed nodes in real-time using Graph Convolutional Networks (GCN), in which the pre-estimated graphs are employed to refine the influencers' estimated rewards in each contextual setting. Through extensive experiments on two large real-world datasets, we demonstrate the effectiveness of IM-GNB compared with other baseline methods, significantly improving the spread outcome of such diffusion campaigns, when the underlying network is unknown.

</details>

## 26. DIET: Customized Slimming for Incompatible Networks in Sequential Recommendation

<details>

<summary>Abstract</summary>

Due to the continuously improving capabilities of mobile edges, recommender systems start to deploy models on edges to alleviate network congestion caused by frequent mobile requests. Several studies have leveraged the proximity of edge-side to real-time data, fine-tuning them to create edge-specific models. Despite their significant progress, these methods require substantial on-edge computational resources and frequent network transfers to keep the model up to date. The former may disrupt other processes on the edge to acquire computational resources, while the latter consumes network bandwidth, leading to a decrease in user satisfaction. In response to these challenges, we propose a customizeD slImming framework for incompatiblE neTworks(DIET). DIET deploys the same generic backbone (potentially incompatible for a specific edge) to all devices. To minimize frequent bandwidth usage and storage consumption in personalization, DIET tailors specific subnets for each edge based on its past interactions, learning to generate slimming subnets(diets) within incompatible networks for efficient transfer. It also takes the inter-layer relationships into account, empirically reducing inference time while obtaining more suitable diets. We further explore the repeated modules within networks and propose a more storage-efficient framework, DIETING, which utilizes a single layer of parameters to represent the entire network, achieving comparably excellent performance. The experiments across four state-of-the-art datasets and two widely used models demonstrate the superior accuracy in recommendation and efficiency in transmission and storage of our framework.

</details>

## 27. Graph Condensation for Open-World Graph Learning 🌟

<details>

<summary>Abstract</summary>

The burgeoning volume of graph data presents significant computational challenges in training graph neural networks (GNNs), critically impeding their efficiency in various applications. To tackle this challenge, graph condensation (GC) has emerged as a promising acceleration solution, focusing on the synthesis of a compact yet representative graph for efficiently training GNNs while retaining performance. Despite the potential to promote scalable use of GNNs, existing GC methods are limited to aligning the condensed graph with merely the observed static graph distribution. This limitation significantly restricts the generalization capacity of condensed graphs, particularly in adapting to dynamic distribution changes. In real-world scenarios, however, graphs are dynamic and constantly evolving, with new nodes and edges being continually integrated. Consequently, due to the limited generalization capacity of condensed graphs, applications that employ GC for efficient GNN training end up with sub-optimal GNNs when confronted with evolving graph structures and distributions in dynamic real-world situations. To overcome this issue, we propose open-world graph condensation (OpenGC), a robust GC framework that integrates structure-aware distribution shift to simulate evolving graph patterns and exploit the temporal environments for invariance condensation. This approach is designed to extract temporal invariant patterns from the original graph, thereby enhancing the generalization capabilities of the condensed graph and, subsequently, the GNNs trained on it. Furthermore, to support the periodic re-condensation and expedite condensed graph updating in life-long graph learning, OpenGC reconstructs the sophisticated optimization scheme with kernel ridge regression and non-parametric graph convolution, significantly accelerating the condensation process while ensuring the exact solutions. Extensive experiments on both real-world and synthetic evolving graphs demonstrate that OpenGC outperforms state-of-the-art (SOTA) GC methods in adapting to dynamic changes in open-world graph environments.

</details>

## 28. An Energy-centric Framework for Category-free Out-of-distribution Node Detection in Graphs

<details>

<summary>Abstract</summary>

Graph neural networks have garnered notable attention for effectively processing graph-structured data. Prevalent models prioritize improving in-distribution (IND) data performance, frequently overlooking the risks from potential out-of-distribution (OOD) nodes during training and inference. In real-world graphs, the automated network construction can introduce noisy nodes from unknown distributions. Previous research into OOD node detection, typically referred to as entropy-based methods, calculates OOD measurements from the prediction entropy alongside category classification training. However, the nodes in the graph might not be pre-labeled with specific categories, rendering entropy-based OOD detectors inapplicable in such category-free situations. To tackle this issue, we propose an energy-centric density estimation framework for OOD node detection, referred to as EnergyDef. Within this framework, we introduce an energy-based GNN to compute node energies that act as indicators of node density and reveal the OOD uncertainty of nodes. Importantly, EnergyDef can efficiently identify OOD nodes with low-resource OOD node annotations, achieved by sampling hallucinated nodes via Langevin Dynamics and structure estimation, along with training through Contrastive Divergence. Our comprehensive experiments on real-world datasets substantiate that our framework markedly surpasses state-of-the-art methods in terms of detection quality, even under conditions of scarce or entirely absent OOD node annotations.

</details>

## 29. Topology-Driven Multi-View Clustering via Tensorial Refined Sigmoid Rank Minimization

<details>

<summary>Abstract</summary>

Benefiting from the effective exploitation of the high-order correlations across multiple views, tensor-based multi-view clustering (TMVC) has garnered considerable attention in recent years. Nevertheless, prior TMVC techniques commonly involve assembling multiple view-specific spatial similarity graphs into a three-dimensional tensor, overlooking the intrinsic topological structure essential for precise clustering of data within a manifold. Additionally, mainstream techniques are constrained by equally shrinking all singular values to recover a low-rank tensor, limiting their capacity to distinguish significant variations among different singular values. In this investigation, we present an innovative TMVC framework termed toPology-driven multi-view clustering viA refined teNsorial sigmoiD rAnk minimization (PANDA ). Specifically, PANDA extracts view-specific topological structures from Euclidean graphs and intricately integrates them into a low-rank three-dimensional tensor, facilitating the concurrent utilization of intra-view topological connectivity and inter-view high-order correlations. Moreover, we develop a refined sigmoid function as the tighter surrogate to tensor rank, enabling the exploration of significant information of heterogeneous singular values. Meanwhile, the topological structures are merged into a unified structure with varying weights, associated with a connectivity constraint, empowering the significant divergence among views and the explicit cluster structure of the target graph are simultaneously leveraged. Extensive experiments demonstrate the superiority of PANDA, outperforming SOTA methods.

</details>

## 30. Investigating Out-of-Distribution Generalization of GNNs: An Architecture Perspective 🌟

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) have exhibited remarkable performance under the assumption that test data comes from the same distribution of training data. However, in real-world scenarios, this assumption may not always be valid. Consequently, there is a growing focus on exploring the Out-of-Distribution (OOD) problem in the context of graphs. Most existing efforts have primarily concentrated on improving graph OOD generalization from two model-agnostic perspectives: data-driven methods and strategy-based learning. However, there has been limited attention dedicated to investigating the impact of well-known GNN model architectures on graph OOD generalization, which is orthogonal to existing research. In this work, we provide the first comprehensive investigation of OOD generalization on graphs from an architecture perspective, by examining the common building blocks of modern GNNs. Through extensive experiments, we reveal that both the graph self-attention mechanism and the decoupled architecture contribute positively to graph OOD generalization. In contrast, we observe that the linear classification layer tends to compromise graph OOD generalization capability. Furthermore, we provide in-depth theoretical insights and discussions to underpin these discoveries. These insights have empowered us to develop a novel GNN backbone model, DGat, designed to harness the robust properties of both graph self-attention mechanism and the decoupled architecture. Extensive experimental results demonstrate the effectiveness of our model under graph OOD, exhibiting substantial and consistent enhancements across various training strategies. Our codes are available at https://github.com/KaiGuo20/DGAT **REMOVE 2nd URL**://github.com/KaiGuo20/DGAT.

</details>

## 31. Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations

<details>

<summary>Abstract</summary>

Tripartite graph-based recommender systems markedly diverge from traditional models by recommending unique combinations such as user groups and item bundles. Despite their effectiveness, these systems exacerbate the long-standing cold-start problem in traditional recommender systems, because any number of user groups or item bundles can be formed among users or items. To address this issue, we introduce a &lt;u&gt;C&lt;/u&gt;onsistency and &lt;u&gt;D&lt;/u&gt;iscrepancy-based graph contrastive learning method for tripartite graph-based &lt;u&gt;R&lt;/u&gt;ecommendation (CDR). This approach leverages two novel meta-path-based metrics-consistency and discrepancy-to capture nuanced, implicit associations between the recommended objects and the recommendees. These metrics, indicative of high-order similarities, can be efficiently calculated with infinite graph convolutional networks (GCN) layers under a multi-objective optimization framework, using the limit theory of GCN. Additionally, we introduce a novel Contrastive Divergence (CD) loss, which can seamlessly integrate the consistency and discrepancy metrics into the contrastive objective as the positive and contrastive supervision signals to learn node representations, enhancing the pairwise ranking of recommended objects and proving particularly valuable in severe cold-start scenarios. Extensive experiments demonstrate the effectiveness of the proposed CDR. The code is released at https://github.com/foodfaust/CDR.

</details>

## 32. AnyLoss: Transforming Classification Metrics into Loss Functions 🌟

<details>

<summary>Abstract</summary>

Many evaluation metrics can be used to assess the performance of models in binary classification tasks. However, most of them are derived from a confusion matrix in a non-differentiable form, making it very difficult to generate a differentiable loss function that could directly optimize them. The lack of solutions to bridge this challenge not only hinders our ability to solve difficult tasks, such as imbalanced learning, but also requires the deployment of computationally expensive hyperparameter search processes in model selection. In this paper, we propose a general-purpose approach that transforms any confusion matrix-based metric into a loss function, AnyLoss, that is available in optimization processes. To this end, we use an approximation function to make a confusion matrix represented in a differentiable form, and this approach enables any confusion matrix-based metric to be directly used as a loss function. The mechanism of the approximation function is provided to ensure its operability and the differentiability of our loss functions is proved by suggesting their derivatives. We conduct extensive experiments under diverse neural networks with many datasets, and we demonstrate their general availability to target any confusion matrix-based metrics. Our method, especially, shows outstanding achievements in dealing with imbalanced datasets, and its competitive learning speed, compared to multiple baseline models, underscores its efficiency.

</details>

## 33. Adapting Job Recommendations to User Preference Drift with Behavioral-Semantic Fusion Learning

<details>

<summary>Abstract</summary>

Job recommender systems are crucial for aligning job opportunities with job-seekers in online job-seeking. However, users tend to adjust their job preferences to secure employment opportunities continually, which limits the performance of job recommendations. The inherent frequency of preference drift poses a challenge to promptly and precisely capture user preferences. To address this issue, we propose a novel session-based framework, BISTRO, to timely model user preference through fusion learning of semantic and behavioral information. Specifically, BISTRO is composed of three stages: 1) coarse-grained semantic clustering, 2) fine-grained job preference extraction, and 3) personalized top-k job recommendation. Initially, BISTRO segments the user interaction sequence into sessions and leverages session-based semantic clustering to achieve broad identification of person-job matching. Subsequently, we design a hypergraph wavelet learning method to capture the nuanced job preference drift. To mitigate the effect of noise in interactions caused by frequent preference drift, we innovatively propose an adaptive wavelet filtering technique to remove noisy interaction. Finally, a recurrent neural network is utilized to analyze session-based interaction for inferring personalized preferences. Extensive experiments on three real-world offline recruitment datasets demonstrate the significant performances of our framework. Significantly, BISTRO also excels in online experiments, affirming its effectiveness in live recruitment settings. This dual success underscores the robustness and adaptability of BISTRO. The source code is available at https://github.com/Applied-Machine-Learning-Lab/BISTRO.

</details>

## 34. Expander Hierarchies for Normalized Cuts on Graphs

<details>

<summary>Abstract</summary>

Expander decompositions of graphs have significantly advanced the understanding of many classical graph problems and led to numerous fundamental theoretical results. However, their adoption in practice has been hindered due to their inherent intricacies and large hidden factors in their asymptotic running times. Here, we introduce the first practically efficient algorithm for computing expander decompositions and their hierarchies and demonstrate its effectiveness and utility by incorporating it as the core component in a novel solver for the normalized cut graph clustering objective.Our extensive experiments on a variety of large graphs show that our expander-based algorithm outperforms state-of-the-art solvers for normalized cut with respect to solution quality by a large margin on a variety of graph classes such as citation, e-mail, and social networks or web graphs while remaining competitive in running time.

</details>

## 35. A Unified Core Structure in Multiplex Networks: From Finding the Densest Subgraph to Modeling User Engagement

<details>

<summary>Abstract</summary>

In many complex systems, the interactions between objects span multiple aspects. Multiplex networks are accurate paradigms to model such systems, where each edge is associated with a type. A key graph mining primitive is extracting dense subgraphs, and this has led to interesting notions such as k-cores, known as building blocks of complex networks. Despite recent attempts to extend the notion of core to multiplex networks, existing studies suffer from a subset of the following limitations: They (1) force all nodes to exhibit their high degree in the same set of relation types while in multiplex networks some connection types can be noisy for some nodes, (2) either require high computational cost or miss the complex information of multiplex networks, and (3) assume the same importance for all relation types. We introduce Score, a novel and unifying family of dense structures in multiplex networks that uses a function S(.) to summarize the degree vector of each node. We then discuss how one can choose a proper S(.) from the data. To demonstrate the usefulness of Scores, we focus on finding the densest subgraph as well as modeling user engagement in multiplex networks. We present a new density measure in multiplex networks and discuss its advantages over existing density measures. We show that the problem of finding the densest subgraph in multiplex networks is NP-hard and design an efficient approximation algorithm based on Scores. Finally, we present a new mathematical model of user engagement in the presence of different relation types. Our experiments shows the efficiency and effectiveness of our algorithms and supports the proposed mathematical model of user engagement.

</details>

## 36. Model-Agnostic Random Weighting for Out-of-Distribution Generalization 🌟

<details>

<summary>Abstract</summary>

Despite the encouraging successes in numerous applications, machine learning methods grounded on the i.i.d. assumption often experience performance deterioration when confronted with the distribution shift between training and test data. This challenge has instigated recent research endeavors focusing on out-of-distribution (OOD) generalization. A particularly pervasive and intricate OOD problem is to enhance the model's generalization ability by training it on samples drawn from a single environment. In response to the problem, we propose a simple model-agnostic method tailored for a practical OOD scenario in this paper. Our approach centers on pursuing robust weighted empirical risks, utilizing randomly shifted training distributions derived through a specific sample-based weighting strategy. Furthermore, we theoretically establish that the expected risk of the shifted training distribution can bound the expected risk of the test distribution. This theoretical foundation ensures the improved prediction performance of our method when employed in uncertain test distributions. Extensive experiments conducted on diverse real-world datasets affirm the effectiveness of our method, highlighting its potential to address the distribution shifts in machine learning applications.

</details>

## 37. Double Correction Framework for Denoising Recommendation

<details>

<summary>Abstract</summary>

As its availability and generality in online services, implicit feedback is more commonly used in recommender systems. However, implicit feedback usually presents noisy samples in real-world recommendation scenarios (such as misclicks or non-preferential behaviors), which will affect precise user preference learning. To overcome the noisy samples problem, a popular solution is based on dropping noisy samples in the model training phase, which follows the observation that noisy samples have higher training losses than clean samples. Despite the effectiveness, we argue that this solution still has limits. (1) High training losses can result from model optimization instability or hard samples, not just noisy samples. (2) Completely dropping of noisy samples will aggravate the data sparsity, which lacks full data exploitation.To tackle the above limitations, we propose a Double Correction Framework for Denoising Recommendation (DCF), which contains two correction components from views of more precise sample dropping and avoiding more sparse data. In the sample dropping correction component, we use the loss value of the samples over time to determine whether it is noise or not, increasing dropping stability. Instead of averaging directly, we use the damping function to reduce the bias effect of outliers. Furthermore, due to the higher variance exhibited by hard samples, we derive a lower bound for the loss through concentration inequality to identify and reuse hard samples. In progressive label correction, we iteratively re-label highly deterministic noisy samples and retrain them to further improve performance. Finally, extensive experimental results on three datasets and four backbones demonstrate the effectiveness and generalization of our proposed framework.

</details>

## 38. RoutePlacer: An End-to-End Routability-Aware Placer with Graph Neural Network

<details>

<summary>Abstract</summary>

Placement is a critical and challenging step of modern chip design, with routability being an essential indicator of placement quality. Current routability-oriented placers typically apply an iterative two-stage approach, wherein the first stage generates a placement solution, and the second stage provides non-differentiable routing results to heuristically improve the solution quality. This method hinders jointly optimizing the routability aspect during placement. To address this problem, this work introduces RoutePlacer, an end-to-end routability-aware placement method. It trains RouteGNN, a customized graph neural network, to efficiently and accurately predict routability by capturing and fusing geometric and topological representations of placements. Well-trained RouteGNN then serves as a differentiable approximation of routability, enabling end-to-end gradient-based routability optimization. In addition, RouteGNN can improve two-stage placers as a plug-and-play alternative to external routers. Our experiments on DREAMPlace, an open-source AI4EDA platform, show that RoutePlacer can reduce Total Overflow by up to 16\% while maintaining routed wirelength, compared to the state-of-the-art; integrating RouteGNN within two-stage placers leads to a 44\% reduction in Total Overflow without compromising wirelength.

</details>

## 39. Can Modifying Data Address Graph Domain Adaptation? 🌟

<details>

<summary>Abstract</summary>

Graph neural networks (GNNs) have demonstrated remarkable success in numerous graph analytical tasks. Yet, their effectiveness is often compromised in real-world scenarios due to distribution shifts, limiting their capacity for knowledge transfer across changing environments or domains. Recently, Unsupervised Graph Domain Adaptation (UGDA) has been introduced to resolve this issue. UGDA aims to facilitate knowledge transfer from a labeled source graph to an unlabeled target graph. Current UGDA efforts primarily focus on model-centric methods, such as employing domain invariant learning strategies and designing model architectures. However, our critical examination reveals the limitations inherent to these model-centric methods, while a data-centric method allowed to modify the source graph provably demonstrates considerable potential. This insight motivates us to explore UGDA from a data-centric perspective. By revisiting the theoretical generalization bound for UGDA, we identify two data-centric principles for UGDA: alignment principle and rescaling principle. Guided by these principles, we propose GraphAlign, a novel UGDA method that generates a small yet transferable graph. By exclusively training a GNN on this new graph with classic Empirical Risk Minimization (ERM), GraphAlign attains exceptional performance on the target graph. Extensive experiments under various transfer scenarios demonstrate the GraphAlign outperforms the best baselines by an average of 2.16\%, training on the generated graph as small as 0.25~1\% of the original training graph.

</details>

## 40. EntropyStop: Unsupervised Deep Outlier Detection with Loss Entropy

<details>

<summary>Abstract</summary>

Unsupervised Outlier Detection (UOD) is an important data mining task. With the advance of deep learning, deep Outlier Detection (OD) has received broad interest. Most deep UOD models are trained exclusively on clean datasets to learn the distribution of the normal data, which requires huge manual efforts to clean the real-world data if possible. Instead of relying on clean datasets, some approaches directly train and detect on unlabeled contaminated datasets, leading to the need for methods that are robust to such challenging conditions. Ensemble methods emerged as a superior solution to enhance model robustness against contaminated training sets. However, the training time is greatly increased by the ensemble mechanism.In this study, we investigate the impact of outliers on training, aiming to halt training on unlabeled contaminated datasets before performance degradation. Initially, we noted that blending normal and anomalous data causes AUC fluctuations-a label-dependent measure of detection accuracy. To circumvent the need for labels, we propose a zero-label entropy metric named Loss Entropy for loss distribution, enabling us to infer optimal stopping points for training without labels. Meanwhile, a negative correlation between entropy metric and the label-based AUC score is demonstrated by theoretical proofs. Based on this, an automated early-stopping algorithm called EntropyStop is designed to halt training when loss entropy suggests the maximum model detection capability. We conduct extensive experiments on ADBench (including 47 real datasets), and the overall results indicate that AutoEncoder (AE) enhanced by our approach not only achieves better performance than ensemble AEs but also requires under 2\% of training time. Lastly, loss entropy and EntropyStop are evaluated on other deep OD models, exhibiting their broad potential applicability.

</details>

## 41. RC-Mixup: A Data Augmentation Strategy against Noisy Data for Regression Tasks

<details>

<summary>Abstract</summary>

We study the problem of robust data augmentation for regression tasks in the presence of noisy data. Data augmentation is essential for generalizing deep learning models, but most of the techniques like the popular Mixup are primarily designed for classification tasks on image data. Recently, there are also Mixup techniques that are specialized to regression tasks like C-Mixup. In comparison to Mixup, which takes linear interpolations of pairs of samples, C-Mixup is more selective in which samples to mix based on their label distances for better regression performance. However, C-Mixup does not distinguish noisy versus clean samples, which can be problematic when mixing and lead to suboptimal model performance. At the same time, robust training has been heavily studied where the goal is to train accurate models against noisy data through multiple rounds of model training. We thus propose our data augmentation strategy RC-Mixup, which tightly integrates C-Mixup with multi-round robust training methods for a synergistic effect. In particular, C-Mixup improves robust training in identifying clean data, while robust training provides cleaner data to C-Mixup for it to perform better. A key advantage of RC-Mixup is that it is data-centric where the robust model training algorithm itself does not need to be modified, but can simply benefit from data mixing. We show in our experiments that RC-Mixup significantly outperforms C-Mixup and robust training baselines on noisy data benchmarks and can be integrated with various robust training methods.

</details>

## 42. On (Normalised) Discounted Cumulative Gain as an Off-Policy Evaluation Metric for Top-n Recommendation

<details>

<summary>Abstract</summary>

Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-n recommendation for many years.Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles, highlighting where we deviate from its traditional uses in IR. Importantly, we show that normalising the metric renders it inconsistent, in that even when DCG is unbiased, ranking competing methods by their normalised DCG can invert their relative order. Through a correlation analysis between off- and on-line experiments conducted on a large-scale recommendation platform, we show that our unbiased DCG estimates strongly correlate with online reward, even when some of the metric's inherent assumptions are violated. This statement no longer holds for its normalised variant, suggesting that nDCG's practical utility may be limited.

</details>

## 43. Killing Two Birds with One Stone: Cross-modal Reinforced Prompting for Graph and Language Tasks

<details>

<summary>Abstract</summary>

In recent years, Graph Neural Networks (GNNs) and Large Language Models (LLMs) have exhibited remarkable capability in addressing different graph learning and natural language tasks, respectively. Motivated by this, integrating LLMs with GNNs has been increasingly studied to acquire transferable knowledge across modalities, which leads to improved empirical performance in language and graph domains. However, existing studies mainly focused on a single-domain scenario by designing complicated integration techniques to manage multimodal data effectively. Therefore, a concise and generic learning framework for multi-domain tasks, i.e., graph and language domains, is highly desired yet remains under-exploited due to two major challenges. First, the language corpus of downstream tasks differs significantly from graph data, making it hard to bridge the knowledge gap between modalities. Second, not all knowledge demonstrates immediate benefits for downstream tasks, potentially introducing disruptive noise to context-sensitive models like LLMs. To tackle these challenges, we propose a novel plug-and-play framework for incorporating a lightweight cross-domain prompting method into both language and graph learning tasks. Specifically, we first convert the textual input into a domain-scalable prompt, which not only preserves the semantic and logical contents of the textual input, but also highlights related graph information as external knowledge for different domains. Then, we develop a reinforcement learning-based method to learn the optimal edge selection strategy for useful knowledge extraction, which profoundly sharpens the multi-domain model capabilities. In addition, we introduce a joint multi-view optimization module to regularize agent-level collaborative learning across two domains. Finally, extensive empirical justifications over 23 public and synthetic datasets demonstrate that our approach can be applied to diverse multi-domain tasks more accurately, robustly, and reasonably, and improve the performances of the state-of-the-art graph and language models in different learning paradigms.

</details>

## 44. Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Leman Go Indifferent

<details>

<summary>Abstract</summary>

Prior attacks on graph neural networks have focused on graph poisoning and evasion, neglecting the network's weights and biases. For convolutional neural networks, however, the risk arising from bit flip attacks is well recognized. We show that the direct application of a traditional bit flip attack to graph neural networks is of limited effectivity. Hence, we discuss the Injectivity Bit Flip Attack, the first bit flip attack designed specifically for graph neural networks. Our attack targets the learnable neighborhood aggregation functions in quantized message passing neural networks, degrading their ability to distinguish graph structures and impairing the expressivity of the Weisfeiler-Leman test. We find that exploiting mathematical properties specific to certain graph neural networks significantly increases their vulnerability to bit flip attacks. The Injectivity Bit Flip Attack can degrade the maximal expressive Graph Isomorphism Networks trained on graph property prediction datasets to random output by flipping only a small fraction of the network's bits, demonstrating its higher destructive power compared to traditional bit flip attacks transferred from convolutional neural networks. Our attack is transparent, motivated by theoretical insights and confirmed by extensive empirical results.

</details>

## 45. Efficient Topology-aware Data Augmentation for High-Degree Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

In recent years, graph neural networks (GNNs) have emerged as a potent tool for learning on graph-structured data and won fruitful successes in varied fields. The majority of GNNs follow the message-passing paradigm, where representations of each node are learned by recursively aggregating features of its neighbors. However, this mechanism brings severe over-smoothing and efficiency issues over high-degree graphs (HDGs), wherein most nodes have dozens (or even hundreds) of neighbors, such as social networks, transaction graphs, power grids, etc. Additionally, such graphs usually encompass rich and complex structure semantics, which are hard to capture merely by feature aggregations in GNNs.Motivated by the above limitations, we propose TADA, an efficient and effective front-mounted data augmentation framework for GNNs on HDGs. Under the hood, TADA includes two key modules: (i) feature expansion with structure embeddings, and (ii) topology- and attribute-aware graph sparsification. The former obtains augmented node features and enhanced model capacity by encoding the graph structure into high-quality structure embeddings with our highly-efficient sketching method. Further, by exploiting task-relevant features extracted from graph structures and attributes, the second module enables the accurate identification and reduction of numerous redundant/noisy edges from the input graph, thereby alleviating over-smoothing and facilitating faster feature aggregations over HDGs. Empirically, algo considerably improves the predictive performance of mainstream GNN models on 8 real homophilic/heterophilic HDGs in terms of node classification, while achieving efficient training and inference processes.

</details>

## 46. Continual Collaborative Distillation for Recommender System 🌟

<details>

<summary>Abstract</summary>

Knowledge distillation (KD) has emerged as a promising technique for addressing the computational challenges associated with deploying large-scale recommender systems. KD transfers the knowledge of a massive teacher system to a compact student model, to reduce the huge computational burdens for inference while retaining high accuracy. The existing KD studies primarily focus on one-time distillation in static environments, leaving a substantial gap in their applicability to real-world scenarios dealing with continuously incoming users, items, and their interactions. In this work, we delve into a systematic approach to operating the teacher-student KD in a non-stationary data stream. Our goal is to enable efficient deployment through a compact student, which preserves the high performance of the massive teacher, while effectively adapting to continuously incoming data. We propose &lt;u&gt;C&lt;/u&gt;ontinual &lt;u&gt;C&lt;/u&gt;ollaborative &lt;u&gt;D&lt;/u&gt;istillation (CCD) framework, where both the teacher and the student continually and collaboratively evolve along the data stream. CCD facilitates the student in effectively adapting to new data, while also enabling the teacher to fully leverage accumulated knowledge. We validate the effectiveness of CCD through extensive quantitative, ablative, and exploratory experiments on two real-world datasets. We expect this research direction to contribute to narrowing the gap between existing KD studies and practical applications, thereby enhancing the applicability of KD in real-world systems.

</details>

## 47. Dynamic Neural Dowker Network: Approximating Persistent Homology in Dynamic Directed Graphs

<details>

<summary>Abstract</summary>

Persistent homology, a fundamental technique within Topological Data Analysis (TDA), captures structural and shape characteristics of graphs, yet encounters computational difficulties when applied to dynamic directed graphs. This paper introduces the Dynamic Neural Dowker Network (DNDN), a novel framework specifically designed to approximate the results of dynamic Dowker filtration, aiming to capture the high-order topological features of dynamic directed graphs. Our approach creatively uses line graph transformations to produce both source and sink line graphs, highlighting the shared neighbor structures that Dowker complexes focus on. The DNDN incorporates a Source-Sink Line Graph Neural Network (SSLGNN) layer to effectively capture the neighborhood relationships among dynamic edges. Additionally, we introduce an innovative duality edge fusion mechanism, ensuring that the results for both the sink and source line graphs adhere to the duality principle intrinsic to Dowker complexes. Our approach is validated through comprehensive experiments on real-world datasets, demonstrating DNDN's capability not only to effectively approximate dynamic Dowker filtration results but also to perform exceptionally in dynamic graph classification tasks.

</details>

## 48. Debiased Recommendation with Noisy Feedback

<details>

<summary>Abstract</summary>

Ratings of a user to most items in recommender systems are usually missing not at random (MNAR), largely because users are free to choose which items to rate. To achieve unbiased learning of the prediction model under MNAR data, three typical solutions have been proposed, including error-imputation-based (EIB), inverse-propensity-scoring (IPS), and doubly robust (DR) methods. However, these methods ignore an alternative form of bias caused by the inconsistency between the observed ratings and the users' true preferences, also known as noisy feedback or outcome measurement errors (OME), e.g., due to public opinion or low-quality data collection process. In this work, we study intersectional threats to the unbiased learning of the prediction model from data MNAR and OME in the collected data. First, we design OME-EIB, OME-IPS, and OME-DR estimators, which largely extend the existing estimators to combat OME in real-world recommendation scenarios. Next, we theoretically prove the unbiasedness and generalization bound of the proposed estimators. We further propose an alternate denoising training approach to achieve unbiased learning of the prediction model under MNAR data with OME. Extensive experiments are conducted on three real-world datasets and one semi-synthetic dataset to show the effectiveness of our proposed approaches. The code is available at https://github.com/haoxuanli-pku/KDD24-OME-DR.

</details>

## 49. Causal Subgraph Learning for Generalizable Inductive Relation Prediction

<details>

<summary>Abstract</summary>

Inductive relation reasoning in knowledge graphs aims at predicting missing triplets involving unseen entities and/or unseen relations. While subgraph-based methods that reason about the local structure surrounding a candidate triplet have shown promise, they often fall short in accurately modeling the causal dependence between a triplet's subgraph and its ground-truth label. This limitation typically results in a susceptibility to spurious correlations caused by confounders, adversely affecting generalization capabilities. Herein, we introduce a novel front-door adjustment-based approach designed to learn the causal relationship between subgraphs and their ground-truth labels, specifically for inductive relation prediction. We conceptualize the semantic information of subgraphs as a mediator and employ a graph data augmentation mechanism to create augmented subgraphs. Furthermore, we integrate a fusion module and a decoder within the front-door adjustment framework, enabling the estimation of the mediator's combination with augmented subgraphs. We also introduce the reparameterization trick in the fusion model to enhance model robustness. Extensive experiments on widely recognized benchmark datasets demonstrate the proposed method's superiority in inductive relation prediction, particularly for tasks involving unseen entities and unseen relations. Additionally, the subgraphs reconstructed by our decoder offer valuable insights into the model's decision-making process, enhancing transparency and interpretability.

</details>

## 50. Privileged Knowledge State Distillation for Reinforcement Learning-based Educational Path Recommendation

<details>

<summary>Abstract</summary>

Educational recommendation seeks to suggest knowledge concepts that match a learner's ability, thus facilitating a personalized learning experience. In recent years, reinforcement learning (RL) methods have achieved considerable results by taking the encoding of the learner's exercise log as the state and employing an RL-based agent to make suitable recommendations. However, these approaches suffer from handling the diverse and dynamic learner's knowledge states. In this paper, we introduce the privileged feature distillation technique and propose the P rivileged K nowledge S tate D istillation (PKSD ) framework, allowing the RL agent to leverage the "actual'' knowledge state as privileged information in the state encoding to help tailor recommendations to meet individual needs. Concretely, our PKSD takes the privileged knowledge states together with the representations of the exercise log for the state representations during training. And through distillation, we transfer the ability to adapt to learners to aknowledge state adapter. During inference, theknowledge state adapter would serve as the estimated privileged knowledge states instead of the real one since it is not accessible. Considering that there are strong connections among the knowledge concepts in education, we further propose to collaborate the graph structure learning for concepts into our PKSD framework. This new approach is termed GEPKSD (Graph-Enhanced PKSD). As our method is model-agnostic, we evaluate PKSD and GEPKSD by integrating them with five different RL bases on four public simulators, respectively. Our results verify that PKSD can consistently improve the recommendation performance with various RL methods, and our GEPKSD could further enhance the effectiveness of PKSD in all the simulations.

</details>

## 51. SimDiff: Simple Denoising Probabilistic Latent Diffusion Model for Data Augmentation on Multi-modal Knowledge Graph

<details>

<summary>Abstract</summary>

In this paper, we address the challenges of data augmentation in Multi-Modal Knowledge Graphs (MMKGs), a relatively under-explored area. We propose a novel diffusion-based generative model, the Simple Denoising Probabilistic Latent Diffusion Model (SimDiff). SimDiff is capable of handling different data modalities including the graph topology in a unified manner by the same diffusion model in the latent space. It enhances the utilization of multi-modal data and encourage the multi-modal fusion and reduces the dependency on limited training data. We validate our method in downstream Entity Alignment (EA) tasks in MMKGs, demonstrating that even when using only half of the seed entities in training, our methods can still achieve superior performance. This work contributes to the field by providing a new data generation or augmentation method for MMKGs, potentially paving the way for more effective use of MMKGs in various applications. Code is made available at https://github.com/ranlislz/SimDiff.

</details>

## 52. Self-Distilled Disentangled Learning for Counterfactual Prediction 🌟

<details>

<summary>Abstract</summary>

The advancements in disentangled representation learning significantly enhance the accuracy of counterfactual predictions by granting precise control over instrumental variables, confounders, and adjustable variables. An appealing method for achieving the independent separation of these factors is mutual information minimization, a task that presents challenges in numerous machine learning scenarios, especially within high-dimensional spaces. To circumvent this challenge, we propose the Self-Distilled Disentanglement framework, referred to as SD2. Grounded in information theory, it ensures theoretically sound independent disentangled representations without intricate mutual information estimator designs for high-dimensional representations. Our comprehensive experiments, conducted on both synthetic and real-world datasets, confirms the effectiveness of our approach in facilitating counterfactual inference in the presence of both observed and unobserved confounders.

</details>

## 53. Toward Structure Fairness in Dynamic Graph Embedding: A Trend-aware Dual Debiasing Approach 🌟

<details>

<summary>Abstract</summary>

Recent studies successfully learned static graph embeddings that are structurally fair by preventing the effectiveness disparity of high- and low-degree vertex groups in downstream graph mining tasks. However, achieving structure fairness in dynamic graph embedding remains an open problem. Neglecting degree changes in dynamic graphs will significantly impair embedding effectiveness without notably improving structure fairness. This is because the embedding performance of high-degree and low-to-high-degree vertices will significantly drop close to the generally poorer embedding performance of most slightly changed vertices in the long-tail part of the power-law distribution. We first identify biased structural evolutions in a dynamic graph based on the evolving trend of vertex degree and then propose FairDGE, the first structurally Fair Dynamic Graph Embedding algorithm. FairDGE learns biased structural evolutions by jointly embedding the connection changes among vertices and the long-short-term evolutionary trend of vertex degrees. Furthermore, a novel dual debiasing approach is devised to encode fair embeddings contrastively, customizing debiasing strategies for different biased structural evolutions. This innovative debiasing strategy breaks the effectiveness bottleneck of embeddings without notable fairness loss. Extensive experiments demonstrate that FairDGE achieves simultaneous improvement in the effectiveness and fairness of embeddings.

</details>

## 54. Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis

<details>

<summary>Abstract</summary>

Hyperbolic neural networks (HNNs) are emerging as a promising tool for representing data embedded in non-Euclidean geometries, yet their adoption has been hindered by challenges related to stability and robustness. In this work, we conduct a rigorous Lipschitz analysis for HNNs and propose using Lipschitz regularization as a novel strategy to enhance their robustness. Our comprehensive investigation spans both the Poincar\'{e} ball model and the hyperboloid model, establishing Lipschitz bounds for HNN layers. Importantly, our analysis provides detailed insights into the behavior of the Lipschitz bounds as they relate to feature norms, particularly distinguishing between scenarios where features have unit norms and those with large norms. Further, we study regularization using the derived Lipschitz bounds. Our empirical validations demonstrate consistent improvements in HNN robustness against noisy perturbations.

</details>

## 55. Rethinking Fair Graph Neural Networks from Re-balancing 🌟

<details>

<summary>Abstract</summary>

Driven by the powerful representation ability of Graph Neural Networks (GNNs), plentiful GNN models have been widely deployed in many real-world applications. Nevertheless, due to distribution disparities between different demographic groups, fairness in high-stake decision-making systems is receiving increasing attention. Although lots of recent works devoted to improving the fairness of GNNs and achieved considerable success, they all require significant architectural changes or additional loss functions requiring more hyper-parameter tuning. Surprisingly, we find that simple re-balancing methods can easily match or surpass existing fair GNN methods. We claim that the imbalance across different demographic groups is a significant source of unfairness, resulting in imbalanced contributions from each group to the parameters updating. However, these simple re-balancing methods have their own shortcomings during training. In this paper, we propose FairGB, &lt;u&gt;Fair&lt;/u&gt; &lt;u&gt;G&lt;/u&gt;raph Neural Network via re-&lt;u&gt;B&lt;/u&gt;alancing, which mitigates the unfairness of GNNs by group balancing. Technically, FairGB consists of two modules: counterfactual node mixup and contribution alignment loss. Firstly, we select counterfactual pairs across inter-domain and inter-class, and interpolate the ego-networks to generate new samples. Guided by analysis, we can reveal the debiasing mechanism of our model by the causal view and prove that our strategy can make sensitive attributes statistically independent from target labels. Secondly, we reweigh the contribution of each group according to gradients. By combining these two modules, they can mutually promote each other. Experimental results on benchmark datasets show that our method can achieve state-of-the-art results concerning both utility and fairness metrics. Code is available at https://github.com/ZhixunLEE/FairGB.

</details>

## 56. Customizing Graph Neural Network for CAD Assembly Recommendation

<details>

<summary>Abstract</summary>

CAD assembly modeling, which refers to using CAD software to design new products from a catalog of existing machine components, is important in the industrial field. The graph neural network (GNN) based recommender system for CAD assembly modeling can help designers make decisions and speed up the design process by recommending the next required component based on the existing components in CAD software. These components can be represented as a graph naturally. However, present recommender systems for CAD assembly modeling adopt fixed GNN architectures, which may be sub-optimal for different manufacturers with different data distribution. Therefore, to customize a well-suited recommender system for different manufacturers, we propose a novel neural architecture search (NAS) framework, dubbed CusGNN, which can design data-specific GNN automatically. Specifically, we design a search space from three dimensions (i.e., aggregation, fusion, and readout functions), which contains a wide variety of GNN architectures. Then, we develop an effective differentiable search algorithm to search high-performing GNN from the search space. Experimental results show that the customized GNNs achieve 1.5-5.1\% higher top-10 accuracy compared to previous manual designed methods, demonstrating the superiority of the proposed approach. Code and data are available at https://github.com/BUPT-GAMMA/CusGNN.

</details>

## 57. TDNetGen: Empowering Complex Network Resilience Prediction with Generative Augmentation of Topology and Dynamics

<details>

<summary>Abstract</summary>

Predicting the resilience of complex networks, which represents the ability to retain fundamental functionality amidst external perturbations or internal failures, plays a critical role in understanding and improving real-world complex systems. Traditional theoretical approaches grounded in nonlinear dynamical systems rely on prior knowledge of network dynamics. On the other hand, data-driven approaches frequently encounter the challenge of insufficient labeled data, a predicament commonly observed in real-world scenarios. In this paper, we introduce a novel resilience prediction framework for complex networks, designed to tackle this issue through generative data augmentation of network topology and dynamics. The core idea is the strategic utilization of the inherent joint distribution present in unlabeled network data, facilitating the learning process of the resilience predictor by illuminating the relationship between network topology and dynamics. Experiment results on three network datasets demonstrate that our proposed framework TDNetGen can achieve high prediction accuracy up to 85\%-95\%. Furthermore, the framework still demonstrates a pronounced augmentation capability in extreme low-data regimes, thereby underscoring its utility and robustness in enhancing the prediction of network resilience. We have open-sourced our code in the following link, https://github.com/tsinghua-fib-lab/TDNetGen.

</details>

## 58. Fast Query of Biharmonic Distance in Networks

<details>

<summary>Abstract</summary>

Thebiharmonic distance (BD) is a fundamental metric that measures the distance of two nodes in a graph. It has found applications in network coherence, machine learning, and computational graphics, among others. In spite of BD's importance, efficient algorithms for the exact computation or approximation of this metric on large graphs remain notably absent. In this work, we provide several algorithms to estimate BD, building on a novel formulation of this metric. These algorithms enjoy locality property (that is, they only read a small portion of the input graph) and at the same time possess provable performance guarantees. In particular, our main algorithms approximate the BD between any node pair with an arbitrarily small additive error ε in time O(1/ε2 poly(log n/ε)). Furthermore, we perform an extensive empirical study on several benchmark networks, validating the performance and accuracy of our algorithms.

</details>

## 59. Asymmetric Beta Loss for Evidence-Based Safe Semi-Supervised Multi-Label Learning

<details>

<summary>Abstract</summary>

The goal of semi-supervised multi-label learning (SSMLL) is to improve model performance by leveraging the information of unlabeled data. Recent studies usually adopt the pseudo-labeling strategy to tackle unlabeled data based on the assumption that labeled and unlabeled data share the same distribution. However, in realistic scenarios, unlabeled examples are often collected through cost-effective methods, inevitably introducing out-of-distribution (OOD) data, leading to a significant decline in model performance. In this paper, we propose a safe semi-supervised multi-label learning framework based on the theory of evidential deep learning (EDL), with the goal of achieving robust and effective unlabeled data exploitation. On one hand, we propose the asymmetric beta loss to not only compensate for the lack of robustness in common MLL losses, but also to solve the inherent positive-negative imbalance problem faced by the EDL losses in MLL. On the other hand, to construct a robust SSMLL framework, we adopt a dual-head structure to generate class probabilities and instance uncertainties. The former are used to generate pseudo-labels, while the latter are utilized to filter OOD examples. To avoid the need for threshold estimation, we develop a dual-measurement weighted loss function to safely perform unlabeled training. Extensive experiments on multiple benchmark datasets verify the effectiveness of the proposed method in both OOD detection and SSMLL tasks.

</details>

## 60. Probabilistic Attention for Sequential Recommendation

<details>

<summary>Abstract</summary>

Sequential Recommendation (SR) navigates users' dynamic preferences through modeling their historical interactions. The incorporation of the popular Transformer framework, which captures long relationships through pairwise dot products, has notably benefited SR. However, prevailing research in this domain faces three significant challenges: (i) Existing studies directly adopt the primary component of Transformer (i.e., the self-attention mechanism), without a clear explanation or tailored definition for its specific role in SR; (ii) The predominant focus on pairwise computations overlooks the global context or relative prevalence of item pairs within the overall sequence; (iii) Transformer primarily pursues relevance-dominated relationships, neglecting another essential objective in recommendation, i.e., diversity. In response, this work introduces a fresh perspective to elucidate the attention mechanism in SR. Here, attention is defined as dependency interactions among items, quantitatively determined under a global probabilistic model by observing the probabilities of corresponding item subsets. This viewpoint offers a precise and context-specific definition of attention, leading to the design of a distinctive attention mechanism tailored for SR. Specifically, we transmute the well-formulated global, repulsive interactions in Determinantal Point Processes (DPPs) to effectively model dependency interactions. Guided by the repulsive interactions, a theoretically and practically feasible DPP kernel is designed, enabling our attention mechanism to directly consider category/topic distribution for enhancing diversity. Consequently, the &lt;u&gt;P&lt;/u&gt;robabilistic &lt;u&gt;Att&lt;/u&gt;ention mechanism (PAtt) for sequential recommendation is developed. Experimental results demonstrate the excellent scalability and adaptability of our attention mechanism, which significantly improves recommendation performance in terms of both relevance and diversity.

</details>

## 61. Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective

<details>

<summary>Abstract</summary>

Graph clustering, a fundamental and challenging task in graph mining, aims to classify nodes in a graph into several disjoint clusters. In recent years, graph contrastive learning (GCL) has emerged as a dominant line of research in graph clustering and advances the new state-of-the-art. However, GCL-based methods heavily rely on graph augmentations and contrastive schemes, which may potentially introduce challenges such as semantic drift and scalability issues. Another promising line of research involves the adoption of modularity maximization, a popular and effective measure for community detection, as the guiding principle for clustering tasks. Despite the recent progress, the underlying mechanism of modularity maximization is still not well understood. In this work, we dig into the hidden success of modularity maximization for graph clustering. Our analysis reveals the strong connections between modularity maximization and graph contrastive learning, where positive and negative examples are naturally defined by modularity. In light of our results, we propose a community-aware graph clustering framework, coined \o{}urs, which leverages modularity maximization as a contrastive pretext task to effectively uncover the underlying information of communities in graphs, while avoiding the problem of semantic drift. Extensive experiments on multiple graph datasets verify the effectiveness of \o{}urs in terms of scalability and clustering performance compared to state-of-the-art graph clustering methods. Notably, \o{}urs easily scales a sufficiently large graph with 100M nodes while outperforming strong baselines.

</details>

## 62. Graph Data Condensation via Self-expressive Graph Structure Reconstruction 🌟

<details>

<summary>Abstract</summary>

With the increasing demands of training graph neural networks (GNNs) on large-scale graphs, graph data condensation has emerged as a critical technique to relieve the storage and time costs during the training phase. It aims to condense the original large-scale graph to a much smaller synthetic graph while preserving the essential information necessary for efficiently training a downstream GNN. However, existing methods concentrate either on optimizing node features exclusively or endeavor to independently learn node features and the graph structure generator. They could not explicitly leverage the information of the original graph structure and failed to construct an interpretable graph structure for the synthetic dataset. To address these issues, we introduce a novel framework named Graph Data Condensation via Self-expressive Graph Structure Reconstruction (GCSR). Our method stands out by (1) explicitly incorporating the original graph structure into the condensing process and (2) capturing the nuanced interdependencies between the condensed nodes by reconstructing an interpretable self-expressive graph structure. Extensive experiments and comprehensive analysis validate the efficacy of the proposed method across diverse GNN models and datasets. Our code is available at https://github.com/zclzcl0223/GCSR.

</details>

## 63. AIM: Attributing, Interpreting, Mitigating Data Unfairness

<details>

<summary>Abstract</summary>

Data collected in the real world often encapsulates historical discrimination against disadvantaged groups and individuals. Existing fair machine learning (FairML) research has predominantly focused on mitigating discriminative bias in the model prediction, with far less effort dedicated towards exploring how to trace biases present in the data, despite its importance for the transparency and interpretability of FairML. To fill this gap, we investigate a novel research problem: discovering samples that reflect biases/prejudices from the training data. Grounding on the existing fairness notions, we lay out a sample bias criterion and propose practical algorithms for measuring and countering sample bias. The derived bias score provides intuitive sample-level attribution and explanation of historical bias in data. On this basis, we further design two FairML strategies via sample-bias-informed minimal data editing. They can mitigate both group and individual unfairness at the cost of minimal or zero predictive utility loss. Extensive experiments and analyses on multiple real-world datasets demonstrate the effectiveness of our methods in explaining and mitigating unfairness. Code is available at https://github.com/ZhiningLiu1998/AIM.

</details>

## 64. Diffusion-Based Cloud-Edge-Device Collaborative Learning for Next POI Recommendations 🌟

<details>

<summary>Abstract</summary>

The rapid expansion of Location-Based Social Networks (LBSNs) has highlighted the importance of effective next Point-of-Interest (POI) recommendations, which leverage historical check-in data to predict users' next POIs to visit. Traditional centralized deep neural networks (DNNs) offer impressive POI recommendation performance but face challenges due to privacy concerns and limited timeliness. In response, on-device POI recommendations have been introduced, utilizing federated learning (FL) and decentralized approaches to ensure privacy and recommendation timeliness. However, these methods often suffer from computational strain on devices and struggle to adapt to new users and regions. This paper introduces a novel collaborative learning framework, Diffusion-Based Cloud-Edge-Device Collaborative Learning for Next POI Recommendations (DCPR), leveraging the diffusion model known for its success across various domains. DCPR operates with a cloud-edge-device architecture to offer region-specific and highly personalized POI recommendations while reducing on-device computational burdens. DCPR minimizes on-device computational demands through a unique blend of global and local learning processes. Our evaluation with two real-world datasets demonstrates DCPR's superior performance in recommendation accuracy, efficiency, and adaptability to new users and regions, marking a significant step forward in on-device POI recommendation technology.

</details>

## 65. AdaGMLP: AdaBoosting GNN-to-MLP Knowledge Distillation 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have revolutionized graph-based machine learning, but their heavy computational demands pose challenges for latency-sensitive edge devices in practical industrial applications. In response, a new wave of methods, collectively known as GNN-to-MLP Knowledge Distillation, has emerged. They aim to transfer GNN-learned knowledge to a more efficient MLP student, which offers faster, resource-efficient inference while maintaining competitive performance compared to GNNs. However, these methods face significant challenges in situations with insufficient training data and incomplete test data, limiting their applicability in real-world applications. To address these challenges, we propose AdaGMLP, an AdaBoosting GNN-to-MLP Knowledge Distillation framework. It leverages an ensemble of diverse MLP students trained on different subsets of labeled nodes, addressing the issue of insufficient training data. Additionally, it incorporates a Node Alignment technique for robust predictions on test data with missing or incomplete features. Our experiments on seven benchmark datasets with different settings demonstrate that AdaGMLP outperforms existing G2M methods, making it suitable for a wide range of latency-sensitive real-world applications. We have submitted our code to the GitHub repository (https://github.com/WeigangLu/AdaGMLP-KDD24).

</details>

## 66. FUGNN: Harmonizing Fairness and Utility in Graph Neural Networks

<details>

<summary>Abstract</summary>

Fairness-aware Graph Neural Networks (GNNs) often face a challenging trade-off, where prioritizing fairness may require compromising utility. In this work, we re-examine fairness through the lens of spectral graph theory, aiming to reconcile fairness and utility within the framework of spectral graph learning. We explore the correlation between sensitive features and spectrum in GNNs, using theoretical analysis to delineate the similarity between original sensitive features and those after convolution under different spectra. Our analysis reveals a reduction in the impact of similarity when the eigenvectors associated with the largest magnitude eigenvalue exhibit directional similarity. Based on these theoretical insights, we propose FUGNN, a novel spectral graph learning approach that harmonizes the conflict between fairness and utility. FUGNN ensures algorithmic fairness and utility by truncating the spectrum and optimizing eigenvector distribution during the encoding process. The fairness-aware eigenvector selection reduces the impact of convolution on sensitive features while concurrently minimizing the sacrifice of utility. FUGNN further optimizes the distribution of eigenvectors through a transformer architecture. By incorporating the optimized spectrum into the graph convolution network, FUGNN effectively learns node representations. Experiments on six real-world datasets demonstrate the superiority of FUGNN over baseline methods. The codes are available at https://github.com/yushuowiki/FUGNN.

</details>

## 67. Cross-Context Backdoor Attacks against Graph Prompt Learning

<details>

<summary>Abstract</summary>

Graph Prompt Learning (GPL) bridges significant disparities between pretraining and downstream applications to alleviate the knowledge transfer bottleneck in real-world graph learning. While GPL offers superior effectiveness in graph knowledge transfer and computational efficiency, the security risks posed by backdoor poisoning effects embedded in pretrained models remain largely unexplored. Our study provides a comprehensive analysis of GPL's vulnerability to backdoor attacks. We introduce CrossBA, the first cross-context backdoor attack against GPL, which manipulates only the pretraining phase without requiring knowledge of downstream applications. Our investigation reveals both theoretically and empirically that tuning trigger graphs, combined with prompt transformations, can seamlessly transfer the backdoor threat from pretrained encoders to downstream applications.Through extensive experiments involving 3 representative GPL methods across 5 distinct cross-context scenarios and 5 benchmark datasets of node and graph classification tasks, we demonstrate that CrossBA consistently achieves high attack success rates while preserving the functionality of downstream applications over clean input. We also explore potential countermeasures against CrossBA and conclude that current defenses are insufficient to mitigate CrossBA. Our study highlights the persistent backdoor threats to GPL systems, raising trustworthiness concerns in the practices of GPL techniques.

</details>

## 68. PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer 🌟

<details>

<summary>Abstract</summary>

Spectral Graph Neural Networks have demonstrated superior performance in graph representation learning. However, many current methods focus on employing shared polynomial coefficients for all nodes, i.e., learning node-unified filters, which limits the filters' flexibility for node-level tasks. The recent DSF attempts to overcome this limitation by learning node-wise coefficients based on positional encoding. However, the initialization and updating process of the positional encoding are burdensome, hindering scalability on large-scale graphs. In this work, we propose a scalable node-wise filter, PolyAttn. Leveraging the attention mechanism, PolyAttn can directly learn node-wise filters in an efficient manner, offering powerful representation capabilities. Building on PolyAttn, we introduce the whole model, named PolyFormer. In the lens of Graph Transformer models, PolyFormer, which calculates attention scores within nodes, shows great scalability. Moreover, the model captures spectral information, enhancing expressiveness while maintaining efficiency. With these advantages, PolyFormer offers a desirable balance between scalability and expressiveness for node-level tasks. Extensive experiments demonstrate that our proposed methods excel at learning arbitrary node-wise filters, showing superior performance on both homophilic and heterophilic graphs, and handling graphs containing up to 100 million nodes. The code is available at https://github.com/air029/PolyFormer.

</details>

## 69. Graph Anomaly Detection with Few Labels: A Data-Centric Approach 🌟

<details>

<summary>Abstract</summary>

Anomalous node detection in a static graph faces significant challenges due to the rarity of anomalies and the substantial cost of labeling their deviant structure and attribute patterns. These challenges give rise to data-centric problems, including extremely imbalanced data distributions and intricate graph learning, which significantly impede machine learning and deep learning methods from discerning the patterns of graph anomalies with few labels. While these issues remain crucial, much of the current research focuses on addressing the induced technical challenges, treating the shortage of labeled data as a given. Distinct from previous efforts, this work focuses on tackling the data-centric problems by generating auxiliary training nodes that conform to the original graph topology and attribute distribution. We categorize this approach as data-centric, aiming to enhance existing anomaly detectors by training them on our synthetic data. However, the methods for generating nodes and the effectiveness of utilizing synthetic data for graph anomaly detection remain unexplored in the realm. To answer these questions, we thoroughly investigate the denoising diffusion model. Drawing from our observations on the diffusion process, we illuminate the shifts in graph energy distribution and establish two principles for designing denoising neural networks tailored to graph anomaly generation. From the insights, we propose a diffusion-based graph generation method to synthesize training nodes, which can be promptly integrated to work with existing anomaly detectors. The empirical results on eight widely-used datasets demonstrate our generated data can effectively enhance the nine state-of-the-art graph detectors' performance.

</details>

## 70. Interpretable Transformer Hawkes Processes: Unveiling Complex Interactions in Social Networks

<details>

<summary>Abstract</summary>

Social networks represent complex ecosystems where the interactions between users or groups play a pivotal role in information dissemination, opinion formation, and social interactions. Effectively harnessing event sequence data within social networks to unearth interactions among users or groups has persistently posed a challenging frontier within the realm of point processes. Current deep point process models face inherent limitations within the context of social networks, constraining both their interpretability and expressive power. These models encounter challenges in capturing interactions among users or groups and often rely on parameterized extrapolation methods when modeling intensity over non-event intervals, limiting their capacity to capture intricate intensity patterns, particularly beyond observed events. To address these challenges, this study proposes modifications to Transformer Hawkes processes (THP), leading to the development of interpretable Transformer Hawkes processes (ITHP). ITHP inherits the strengths of THP while aligning with statistical nonlinear Hawkes processes, thereby enhancing its interpretability and providing valuable insights into interactions between users or groups. Additionally, ITHP enhances the flexibility of the intensity function over non-event intervals, making it better suited to capture complex event propagation patterns in social networks. Experimental results, both on synthetic and real data, demonstrate the effectiveness of ITHP in overcoming the identified limitations. Moreover, they highlight ITHP's applicability in the context of exploring the complex impact of users or groups within social networks. Our code is available at https://github.com/waystogetthere/Interpretable-Transformer- Hawkes-Process.git.

</details>

## 71. Learning Causal Networks from Episodic Data 🌟

<details>

<summary>Abstract</summary>

In numerous real-world domains, spanning from environmental monitoring to long-term medical studies, observations do not arrive in a single batch but rather over time in episodes. This challenges the traditional assumption in causal discovery of a single, observational dataset, not only because each episode may be a biased sample of the population but also because multiple episodes could differ in the causal interactions underlying the observed variables. We address these issues using notions of context switches and episodic selection bias, and introduce a framework for causal modeling of episodic data. We show under which conditions we can apply information-theoretic scoring criteria for causal discovery while preserving consistency. To in practice discover the causal model progressively over time, we propose the CONTINENT algorithm which, taking inspiration from continual learning, discovers the causal model in an online fashion without having to re-learn the model upon arrival of each new episode. Our experiments over a variety of settings including selection bias, unknown interventions, and network changes showcase that CONTINENT works well in practice and outperforms the baselines by a clear margin.

</details>

## 72. Improving the Consistency in Cross-Lingual Cross-Modal Retrieval with 1-to-K Contrastive Learning

<details>

<summary>Abstract</summary>

Cross-lingual Cross-modal Retrieval (CCR) is an essential task in web search, which aims to break the barriers between modality and language simultaneously and achieves image-text retrieval in the multi-lingual scenario with a single model. In recent years, excellent progress has been made based on cross-lingual cross-modal pre-training; particularly, the methods based on contrastive learning on large-scale data have significantly improved retrieval tasks. However, these methods directly follow the existing pre-training methods in the cross-lingual or cross-modal domain, leading to two problems of inconsistency in CCR: The methods with cross-lingual style suffer from the intra-modal error propagation, resulting in inconsistent recall performance across languages in the whole dataset. The methods with cross-modal style suffer from the inter-modal optimization direction bias, resulting in inconsistent rank across languages within each instance, which cannot be reflected by Recall@K. To solve these problems, we propose a simple but effective 1-to-K contrastive learning method, which treats each language equally and eliminates error propagation and optimization bias. In addition, we propose a new evaluation metric, Mean Rank Variance (MRV), to reflect the rank inconsistency across languages within each instance. Extensive experiments on four CCR datasets show that our method improves both recall rates and MRV with smaller-scale pre-trained data, achieving the new state-of-art.

</details>

## 73. TSC: A Simple Two-Sided Constraint against Over-Smoothing 🌟

<details>

<summary>Abstract</summary>

Graph Convolutional Neural Network (GCN), a widely adopted method for analyzing relational data, enhances node discriminability through the aggregation of neighboring information. Usually, stacking multiple layers can improve the performance of GCN by leveraging information from high-order neighbors. However, the increase of the network depth will induce the over-smoothing problem, which can be attributed to the quality and quantity of neighbors changing: (a) neighbor quality, node's neighbors become overlapping in high order, leading to aggregated information becoming indistinguishable, (b) neighbor quantity, the exponentially growing aggregated neighbors submerges the node's initial feature by recursively aggregating operations. Current solutions mainly focus on one of the above causes and seldom consider both at once. Aiming at tackling both causes of over-smoothing in one shot, we introduce a simple Two-Sided Constraint (TSC) for GCNs, comprising two straightforward yet potent techniques: random masking and contrastive constraint. The random masking acts on the representation matrix's columns to regulate the degree of information aggregation from neighbors, thus preventing the convergence of node representations. Meanwhile, the contrastive constraint, applied to the representation matrix's rows, enhances the discriminability of the nodes. Designed as a plug-in module, TSC can be easily coupled with GCN or SGC architectures. Experimental analyses on diverse real-world graph datasets verify that our approach markedly reduces the convergence of node's representation and the performance degradation in deeper GCN.

</details>

## 74. How Powerful is Graph Filtering for Recommendation 🌟

<details>

<summary>Abstract</summary>

It has been shown that the effectiveness of graph convolutional network (GCN) for recommendation is attributed to the spectral graph filtering. Most GCN-based methods consist of a graph filter or followed by a low-rank mapping optimized based on supervised training. However, we show two limitations suppressing the power of graph filtering: (1) Lack of generality. Due to the varied noise distribution, graph filters fail to denoise sparse data where noise is scattered across all frequencies, while supervised training results in worse performance on dense data where noise is concentrated in middle frequencies that can be removed by graph filters without training. (2) Lack of expressive power. We theoretically show that linear GCN (LGCN) that is effective on collaborative filtering (CF) cannot generate arbitrary embeddings, implying the possibility that optimal data representation might be unreachable.To tackle the first limitation, we show close relation between noise distribution and the sharpness of spectrum where a sharper spectral distribution is more desirable causing data noise to be separable from important features without training. Based on this observation, we propose a generalized graph normalization (G2N) with hyperparameters adjusting the sharpness of spectral distribution in order to redistribute data noise to assure that it can be removed by graph filtering without training. As for the second limitation, we propose an individualized graph filter (IGF) adapting to the different confidence levels of the user preference that interactions can reflect, which is proved to be able to generate arbitrary embeddings. By simplifying LGCN, we further propose a simplified graph filtering for CF (SGFCF) which only requires the top-K singular values for recommendation. Finally, experimental results on four datasets with different density settings demonstrate the effectiveness and efficiency of our proposed methods.

</details>

## 75. Unifying Evolution, Explanation, and Discernment: A Generative Approach for Dynamic Graph Counterfactuals

<details>

<summary>Abstract</summary>

We present GRACIE (Graph Recalibration and Adaptive Counterfactual Inspection and Explanation), a novel approach for generative classification and counterfactual explanations of dynamically changing graph data. We study graph classification problems through the lens of generative classifiers. We propose a dynamic, self-supervised latent variable model that updates by identifying plausible counterfactuals for input graphs and recalibrating decision boundaries through contrastive optimization. Unlike prior work, we do not rely on linear separability between the learned graph representations to find plausible counterfactuals. Moreover, GRACIE eliminates the need for stochastic sampling in latent spaces and graph-matching heuristics. Our work distills the implicit link between generative classification and loss functions in the latent space, a key insight to understanding recent successes with this architecture. We further observe the inherent trade-off between validity and pulling explainee instances towards the central region of the latent space, empirically demonstrating our theoretical findings. In extensive experiments on synthetic and real-world graph data, we attain considerable improvements, reaching ~99\% validity when sampling sets of counterfactuals even in the challenging setting of dynamic data landscapes.

</details>

## 76. Reimagining Graph Classification from a Prototype View with Optimal Transport: Algorithm and Theorem

<details>

<summary>Abstract</summary>

Recently, Graph Neural Networks (GNNs) have achieved inspiring performances in graph classification tasks. However, the message passing mechanism in GNNs implicitly utilizes the topological information of the graph, which may lead to a potential loss of structural information. Furthermore, the graph classification decision process based on GNNs resembles a black box and lacks sufficient transparency. The non-linear classifier following the GNNs also defaults to the assumption that each class is represented by a single vector, thereby limiting the diversity of intra-class representations.To address these issues, we propose a novel prototype-based graph classification framework that introduces the Fused Gromov-Wasserstein (FGW) distance in Optimal Transport (OT) as the similarity measure. In this way, the model explicitly exploits the structural information on the graph through OT while leading to a more transparent and straightforward classification process. The introduction of prototypes also inherently addresses the issue of limited within-class representations. Besides, to alleviate the widely acknowledged computational complexity issue of FGW distance calculation, we devise a simple yet effective NN-based FGW distance approximator, which can enable full GPU training acceleration with a marginal performance loss. In theory, we analyze the generalization performance of the proposed method and derive an O (1 over N) generalization bound, where the proof techniques can be extended to a broader range of prototype-based classification frameworks. Experimental results show that the proposed framework achieves competitive and superior performance on several widely used graph classification benchmark datasets. The code is avaliable at https://github.com/ChnQ/PGOT.

</details>

## 77. ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems

<details>

<summary>Abstract</summary>

Cognitive diagnosis models (CDMs) are designed to learn students' mastery levels using their response logs. CDMs play a fundamental role in online education systems since they significantly influence downstream applications such as teachers' guidance and computerized adaptive testing. Despite the success achieved by existing CDMs, we find that they suffer from a thorny issue that the learned students' mastery levels are too similar. This issue, which we refer to as oversmoothing, could diminish the CDMs' effectiveness in downstream tasks. CDMs comprise two core parts: learning students' mastery levels and assessing mastery levels by fitting the response logs. This paper contends that the oversmoothing issue arises from that existing CDMs seldom utilize response signals on exercises in the learning part but only use them as labels in the assessing part. To this end, this paper proposes an oversmoothing-resistant cognitive diagnosis framework (ORCDF) to enhance existing CDMs by utilizing response signals in the learning part. Specifically, ORCDF introduces a novel response graph to inherently incorporate response signals as types of edges. Then, ORCDF designs a tailored response-aware graph convolution network (RGC) that effectively captures the crucial response signals within the response graph. Via ORCDF, existing CDMs are enhanced by replacing the input embeddings with the outcome of RGC, allowing for the consideration of response signals on exercises in the learning part. Extensive experiments on real-world datasets show that ORCDF not only helps existing CDMs alleviate the oversmoothing issue but also significantly enhances the models' prediction and interpretability performance. Moreover, the effectiveness of ORCDF is validated in the downstream task of computerized adaptive testing.

</details>

## 78. A Novel Feature Space Augmentation Method to Improve Classification Performance and Evaluation Reliability 🌟

<details>

<summary>Abstract</summary>

Classification tasks in many real-world domains are exacerbated by class imbalance, relatively small sample sizes compared to high dimensionality, and measurement uncertainty. The problem of class imbalance has been extensively studied, and data augmentation methods based on interpolation of minority class instances have been proposed as a viable solution to mitigate imbalance. It remains to be seen whether augmentation can be applied to improve the overall performance while maintaining stability, especially with a limited number of samples. In this paper, we present a novel feature-space augmentation technique that can be applied to high-dimensional data for classification tasks and address these issues. Our method utilizes uniform random sampling and introduces synthetic instances by taking advantage of the local distributions of individual features in the observed instances. The core augmentation algorithm is class-invariant, which opens up an unexplored avenue of simultaneously improving and stabilizing performance by augmenting unlabeled instances. The proposed method is evaluated using a comprehensive performance analysis involving multiple classifiers and metrics. Comparative analysis with existing feature space augmentation methods strongly suggests that the proposed algorithm can result in improved classification performance while also increasing the overall reliability of the performance evaluation.

</details>

## 79. DPHGNN: A Dual Perspective Hypergraph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Message passing on hypergraphs has been a standard framework for learning higher-order correlations between hypernodes. Recently-proposed hypergraph neural networks (HGNNs) can be categorized into spatial and spectral methods based on their design choices. In this work, we analyze the impact of change in hypergraph topology on the suboptimal performance of HGNNs and propose DPHGNN, a novel dual-perspective HGNN that introduces equivariant operator learning to capture lower-order semantics by inducing topology-aware spatial and spectral inductive biases. DPHGNN employs a unified framework to dynamically fuse lower-order explicit feature representations from the underlying graph into the super-imposed hypergraph structure. We benchmark DPHGNN over eight benchmark hypergraph datasets for the semi-supervised hypernode classification task and obtain superior performance compared to seven state-of-the-art baselines. We also provide a theoretical framework and a synthetic hypergraph isomorphism test to express the power of spatial HGNNs and quantify the expressivity of DPHGNN beyond the Generalized Weisfeiler Leman (1-GWL) test. Finally, DPHGNN was deployed by our partner e-commerce company, Meesho for the Return-to-Origin (RTO) prediction task, which shows ~7\% higher macro F1-Score than the best baseline.

</details>

## 80. Certified Robustness on Visual Graph Matching via Searching Optimal Smoothing Range

<details>

<summary>Abstract</summary>

Deep visual graph matching (GM) is a challenging combinatorial task that involves finding a permutation matrix that indicates the correspondence between keypoints from a pair of images. Like many learning systems, empirical studies have shown that visual GM is susceptible to adversarial attacks, with reliability issues in downstream applications. To the best of our knowledge, certifying robustness for deep visual GM remains an open challenge with two main difficulties: how to handle the paired inputs together with the heavily non-linear permutation output space (especially at large scale), and how to balance the trade-off between certified robustness and matching performance.  Inspired by the randomized smoothing (RS) technique, we propose the Certified Robustness based on the Optimal Smoothing Range Search (CR-OSRS) technique to fulfill the robustness guarantee for deep visual GM. First, unlike conventional RS methods that use isotropic Gaussian distributions for smoothing, we build the smoothed model with paired joint Gaussian distributions, which capture the structural information among keypoints, and mitigate the performance degradation caused by smoothing. For the vast space of the permutation output, we devise a similarity-based partitioning method that can lower the computational complexity and certification difficulty. We then derive a stringent robustness guarantee that links the certified space of inputs to their corresponding fixed outputs. Second, we design a global optimization method to search for optimal joint Gaussian distributions and facilitate a larger certified space and better performance. Third, we apply data augmentation and a similarity-based regularizer in training to enhance smoothed model performance. Lastly, for the high-dimensional and multivariable nature of the certified space, we propose two methods (sampling and marginal radii) to evaluate it. Experimental results on public benchmarks show that our method achieves state-of-the-art certified robustness.

</details>

## 81. Capturing Homogeneous Influence among Students: Hypergraph Cognitive Diagnosis for Intelligent Education Systems

<details>

<summary>Abstract</summary>

Cognitive diagnosis is a vital upstream task in intelligent education systems. It models the student-exercise interaction, aiming to infer the students' proficiency levels on each knowledge concept. This paper observes that most existing methods can hardly effectively capture the homogeneous influence due to its inherent complexity. That is to say, although students exhibit similar performance on given exercises, their proficiency levels inferred by these methods vary significantly, resulting in shortcomings in interpretability and efficacy. Given the complexity of homogeneous influence, a hypergraph could be a choice due to its flexibility and capability of modeling high-order similarity which aligns with the nature of homogeneous influence. However, before incorporating hypergraph, one at first needs to address the challenges of distorted homogeneous influence, sparsity of response logs, and over-smoothing. To this end, this paper proposes a hypergraph cognitive diagnosis model (HyperCDM) to address these challenges and effectively capture the homogeneous influence. Specifically, to avoid distortion, HyperCDM employs a divide-and-conquer strategy to learn student, exercise and knowledge representations in their own hypergraphs respectively, and interconnects them via a feature-based interaction function. To construct hypergraphs based on sparse response logs, the auto-encoder is utilized to preprocess response logs and K-means is applied to cluster students. To mitigate over-smoothing, momentum hypergraph convolution networks are designed to partially keep previous representations during the message propagation. Extensive experiments on both offline and online real-world datasets show that HyperCDM achieves state-of-the-art performance in terms of interpretability and capturing homogeneous influence effectively, and is competitive in generalization. The ablation study verifies the efficacy of each component, and the case study explicitly showcases the homogeneous influence captured by HyperCDM.

</details>

## 82. Optimizing OOD Detection in Molecular Graphs: A Novel Approach with Diffusion Models

<details>

<summary>Abstract</summary>

Despite the recent progress of molecular representation learning, its effectiveness is assumed on the close-world assumptions that training and testing graphs are from identical distribution. The open-world test dataset is often mixed with out-of-distribution (OOD) samples, where the deployed models will struggle to make accurate predictions. The misleading estimations of molecules' properties in drug screening or design can result in the tremendous waste of wet-lab resources and delay the discovery of novel therapies. Traditional detection methods need to trade off OOD detection and in-distribution (ID) classification performance since they share the same representation learning model. In this work, we propose to detect OOD molecules by adopting an auxiliary diffusion model-based framework, which compares similarities between input molecules and reconstructed graphs. Due to the generative bias towards reconstructing ID training samples, the similarity scores of OOD molecules will be much lower to facilitate detection. Although it is conceptually simple, extending this vanilla framework to practical detection applications is still limited by two significant challenges. First, the popular similarity metrics based on Euclidian distance fail to consider the complex graph structure. Second, the generative model involving iterative denoising steps is notoriously time-consuming especially when it runs on the enormous pool of drugs. To address these challenges, our research pioneers an approach of Prototypical Graph Reconstruction for Molecular OOd Detection, dubbed as PGR-MOOD. Specifically, PGR-MOOD hinges on three innovations: i) An effective metric to comprehensively quantify the matching degree of input and reconstructed molecules according to their discrete edges and continuous node features; ii) A creative graph generator to construct a list of prototypical graphs that are in line with ID distribution but away from OOD one; iii) An efficient and scalable OOD detector to compare the similarity between test samples and pre-constructed prototypical graphs and omit the generative process on every new molecule. Extensive experiments on ten benchmark datasets and six baselines are conducted to demonstrate our superiority: PGR-MOOD achieves more than 8\% of average improvement in terms of detection AUC and AUPR accompanied by the reduced cost of testing time and memory consumption. The anonymous code is in: https://github.com/se7esx/PGR-MOOD.

</details>

## 83. LPFormer: An Adaptive Graph Transformer for Link Prediction 🌟

<details>

<summary>Abstract</summary>

Link prediction is a common task on graph-structured data that has seen applications in a variety of domains. Classically, hand-crafted heuristics were used for this task. Heuristic measures are chosen such that they correlate well with the underlying factors related to link formation. In recent years, a new class of methods has emerged that combines the advantages of message-passing neural networks (MPNN) and heuristics methods. These methods perform predictions by using the output of an MPNN in conjunction with a "pairwise encoding" that captures the relationship between nodes in the candidate link. They have been shown to achieve strong performance on numerous datasets. However, current pairwise encodings often contain a strong inductive bias, using the same underlying factors to classify all links. This limits the ability of existing methods to learn how to properly classify a variety of different links that may form from different factors. To address this limitation, we propose a new method, LPFormer, which attempts to adaptively learn the pairwise encodings for each link. LPFormer models the link factors via an attention module that learns the pairwise encoding that exists between nodes by modeling multiple factors integral to link prediction. Extensive experiments demonstrate that LPFormer can achieve SOTA performance on numerous datasets while maintaining efficiency. The code is available at The code is available at https://github.com/HarryShomer/LPFormer.

</details>

## 84. CoLiDR: &lt;u&gt;Co&lt;/u&gt;ncept &lt;u&gt;L&lt;/u&gt;earn&lt;u&gt;i&lt;/u&gt;ng using Aggregated &lt;u&gt;D&lt;/u&gt;isentangled &lt;u&gt;R&lt;/u&gt;epresentations

<details>

<summary>Abstract</summary>

Interpretability of Deep Neural Networks using concept-based models offers a promising way to explain model behavior through human understandable concepts. A parallel line of research focuses on disentangling the data distribution into its underlying generative factors, in turn explaining the data generation process. While both directions have received extensive attention, little work has been done on explaining concepts in terms of generative factors to unify mathematically disentangled representations and human-understandable concepts as an explanation for downstream tasks. In this paper, we propose a novel method CoLiDR - which utilizes a disentangled representation learning setup for learning mutually independent generative factors and subsequently learns to aggregate the said representations into human-understandable concepts using a novel aggregation/decomposition module. Experiments are conducted on datasets with both known and unknown latent generative factors. Our method successfully aggregates disentangled generative factors into concepts while maintaining parity with state-of-the-art concept-based approaches. Quantitative and visual analysis of the learned aggregation procedure demonstrates the advantages of our work compared to commonly used concept-based models over four challenging datasets. Lastly, our work is generalizable to an arbitrary number of concepts and generative factors - making it flexible enough to be suitable for various types of data.

</details>

## 85. Mitigating Negative Transfer in Cross-Domain Recommendation via Knowledge Transferability Enhancement 🌟

<details>

<summary>Abstract</summary>

Cross-Domain Recommendation (CDR) is a promising technique to alleviate data sparsity by transferring knowledge across domains. However, the negative transfer issue in the presence of numerous domains has received limited attention. Most existing methods transfer all information from source domains to the target domain without distinction. This introduces harmful noise and irrelevant features, resulting in suboptimal performance. Although some methods decompose user features into domain-specific and domain-shared components, they fail to consider other causes of negative transfer. Worse still, we argue that simple feature decomposition is insufficient for multi-domain scenarios. To bridge this gap, we propose TrineCDR, the TRIple-level kNowledge transferability Enhanced model for multi-target CDR. Unlike previous methods, TrineCDR captures single domain and targeted cross-domain embeddings to serve multi-domain recommendation. For the latter, we identify three fundamental causes of negative transfer, ranging from micro to macro perspectives, and correspondingly enhance knowledge transferability at three different levels: the feature level, the interaction level, and the domain level. Through these efforts, TrineCDR effectively filters out noise and irrelevant information from source domains, leading to more comprehensive and accurate representations in the target domain. We extensively evaluate the proposed model on real-world datasets, sampled from Amazon and Douban, under both dual-target and multi-target scenarios. The experimental results demonstrate the superiority of TrineCDR over state-of-the-art cross-domain recommendation methods.

</details>

## 86. Fast Computation for the Forest Matrix of an Evolving Graph

<details>

<summary>Abstract</summary>

The forest matrix plays a crucial role in network science, opinion dynamics, and machine learning, offering deep insights into the structure of and dynamics on networks. In this paper, we study the problem of querying entries of the forest matrix in evolving graphs, which more accurately represent the dynamic nature of real-world networks compared to static graphs. To address the unique challenges posed by evolving graphs, we first introduce two approximation algorithms, SFQ and SFQPlus, for static graphs. SFQ employs a probabilistic interpretation of the forest matrix, while SFQPlus incorporates a novel variance reduction technique and is theoretically proven to offer enhanced accuracy. Based on these two algorithms, we further devise two dynamic algorithms centered around efficiently maintaining a list of spanning converging forests. This approach ensures O(1) runtime complexity for updates, including edge additions and deletions, as well as for querying matrix elements, and provides an unbiased estimation of forest matrix entries. Finally, through extensive experiments on various real-world networks, we demonstrate the efficiency and effectiveness of our algorithms. Particularly, our algorithms are scalable to massive graphs with more than forty million nodes.

</details>

## 87. DIVE: Subgraph Disagreement for Graph Out-of-Distribution Generalization 🌟

<details>

<summary>Abstract</summary>

This paper addresses the challenge of out-of-distribution (OOD) generalization in graph machine learning, a field rapidly advancing yet grappling with the discrepancy between source and target data distributions. Traditional graph learning algorithms, based on the assumption of uniform distribution between training and test data, falter in real-world scenarios where this assumption fails, resulting in suboptimal performance. A principal factor contributing to this suboptimal performance is the inherent simplicity bias of neural networks trained through Stochastic Gradient Descent (SGD), which prefer simpler features over more complex yet equally or more predictive ones. This bias leads to a reliance on spurious correlations, adversely affecting OOD performance in various tasks such as image recognition, natural language understanding, and graph classification. Current methodologies, including subgraph-mixup and information bottleneck approaches, have achieved partial success but struggle to overcome simplicity bias, often reinforcing spurious correlations. To tackle this, our study introduces a new learning paradigm for graph OOD issue. We propose DIVE, training a collection of models to focus on all label-predictive subgraphs by encouraging the models to foster divergence on the subgraph mask, which circumvents the limitation of a model solely focusing on the subgraph corresponding to simple structural patterns. Specifically, we employs a regularizer to punish overlap in extracted subgraphs across models, thereby encouraging different models to concentrate on distinct structural patterns. Model selection for robust OOD performance is achieved through validation accuracy. Tested across four datasets from GOOD benchmark and one dataset from DrugOOD benchmark, our approach demonstrates significant improvement over existing methods, effectively addressing the simplicity bias and enhancing generalization in graph machine learning.

</details>

## 88. Self-Supervised Denoising through Independent Cascade Graph Augmentation for Robust Social Recommendation

<details>

<summary>Abstract</summary>

Social Recommendation (SR) typically exploits neighborhood influence in the social network to enhance user preference modeling. However, users' intricate social behaviors may introduce noisy social connections for user modeling and harm the models' robustness. Existing solutions to alleviate social noise either filter out the noisy connections or generate new potential social connections. Due to the absence of labels, the former approaches may retain uncertain connections for user preference modeling while the latter methods may introduce additional social noise. Through data analysis, we discover that (1) social noise likely comes from the connected users with low preference similarity; and (2) Opinion Leaders (OLs) play a pivotal role in influence dissemination, surpassing high-similarity neighbors, regardless of their preference similarity with trusting peers. Guided by these observations, we propose a novel Self-Supervised Denoising approach through Independent Cascade Graph Augmentation, for more robust SR. Specifically, we employ the independent cascade diffusion model to generate an augmented graph view, which traverses the social graph and activates the edges in sequence to simulate the cascading influence spread. To steer the augmentation towards a denoised social graph, we (1) introduce a hierarchical contrastive loss to prioritize the activation of OLs first, followed by high-similarity neighbors, while weakening the low-similarity neighbors; and (2) integrate an information bottleneck based contrastive loss, aiming to minimize mutual information between original and augmented graphs yet preserve sufficient information for improved SR. Experiments conducted on two public datasets demonstrate that our model outperforms the state-of-the-art while also exhibiting higher robustness to different extents of social noise.

</details>

## 89. Hierarchical Linear Symbolized Tree-Structured Neural Processes

<details>

<summary>Abstract</summary>

Traditional Neural Processes (NPs) and their variants aim to learn relationships between context sample points but do not consider multi-level information, resulting in a limited ability to learn complex distributions.This paper draws inspiration from features such as the hierarchical nature and interpretability of tree-like structures. This paper proposes a Hierarchical Linear Symbolized Tree-structured Neural Processes (HLNPs) architecture. This framework utilizes variables to build a top-down hierarchical linear symbolized tree-structured network architecture, enhancing positional representation information in a hierarchical manner along the deterministic path. In the latent distribution, the hierarchical linear symbolized tree-structured network approximates functions discretely through a layered approach. By decomposing the latent complex distribution into several simpler sub-problems using sum and product symbols, the upper bound of optimization is thereby increased. The tree structure discretizes variables to capture model uncertainty in the form of entropy. This approach also imparts a causal effect to the HLNPs model. Finally, we demonstrate the effectiveness of the HLNPs models for 1D data, Bayesian optimization, and 2D data.

</details>

## 90. Learning Attributed Graphlets: Predictive Graph Mining by Graphlets with Trainable Attribute

<details>

<summary>Abstract</summary>

The graph classification problem has been widely studied; however, achieving an interpretable model with high predictive performance remains a challenging issue. This paper proposes an interpretable classification algorithm for attributed graph data, called LAGRA (Learning Attributed GRAphlets). LAGRA learns importance weights for small attributed subgraphs, called attributed graphlets (AGs), while simultaneously optimizing their attribute vectors. This enables us to obtain a combination of subgraph structures and their attribute vectors that strongly contribute to discriminating different classes. A significant characteristics of LAGRA is that all the subgraph structures in the training dataset can be considered as a candidate structures of AGs. This approach can explore all the potentially important subgraphs exhaustively, but obviously, a na\"{\i}ve implementation can require a large amount of computations. To mitigate this issue, we propose an efficient pruning strategy by combining the proximal gradient descent and a graph mining tree search. Our pruning strategy can ensure that the quality of the solution is maintained compared to the result without pruning. We empirically demonstrate that LAGRA has superior or comparable prediction performance to the standard existing algorithms including graph neural networks, while using only a small number of AGs in an interpretable manner.

</details>

## 91. Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning

<details>

<summary>Abstract</summary>

In recent years, graph contrastive learning (GCL) has received increasing attention in recommender systems due to its effectiveness in reducing bias caused by data sparsity. However, most existing GCL models rely on heuristic approaches and usually assume entity independence when constructing contrastive views. We argue that these methods struggle to strike a balance between semantic invariance and view hardness across the dynamic training process, both of which are critical factors in graph contrastive learning.To address the above issues, we propose a novel GCL-based recommendation framework RGCL, which effectively maintains the semantic invariance of contrastive pairs and dynamically adapts as the model capability evolves through the training process. Specifically, RGCL first introduces decision boundary-aware adversarial perturbations to constrain the exploration space of contrastive augmented views, avoiding the decrease of task-specific information. Furthermore, to incorporate global user-user and item-item collaboration relationships for guiding on the generation of hard contrastive views, we propose an adversarial-contrastive learning objective to construct a relation-aware view-generator. Besides, considering that unsupervised GCL could potentially narrower margins between data points and the decision boundary, resulting in decreased model robustness, we introduce the adversarial examples based on maximum perturbations to achieve margin maximization. We also provide theoretical analyses on the effectiveness of our designs. Through extensive experiments on five public datasets, we demonstrate the superiority of RGCL compared against twelve baseline models.

</details>

## 92. Latent Diffusion-based Data Augmentation for Continuous-Time Dynamic Graph Model

<details>

<summary>Abstract</summary>

Continuous-Time Dynamic Graph (CTDG) precisely models evolving real-world relationships, drawing heightened interest in dynamic graph learning across academia and industry. However, existing CTDG models encounter challenges stemming from noise and limited historical data. Graph Data Augmentation (GDA) emerges as a critical solution, yet current approaches primarily focus on static graphs and struggle to effectively address the dynamics inherent in CTDGs. Moreover, these methods often demand substantial domain expertise for parameter tuning and lack theoretical guarantees for augmentation efficacy. To address these issues, we propose Conda, a novel latent diffusion-based GDA method tailored for CTDGs. Conda features a sandwich-like architecture, incorporating a Variational Auto-Encoder (VAE) and a conditional diffusion model, aimed at generating enhanced historical neighbor embeddings for target nodes. Unlike conventional diffusion models trained on entire graphs via pre-training, Conda requires historical neighbor sequence embeddings of target nodes for training, thus facilitating more targeted augmentation. We integrate Conda into the CTDG model and adopt an alternating training strategy to optimize performance. Extensive experimentation across six widely used real-world datasets showcases the consistent performance improvement of our approach, particularly in scenarios with limited historical data.

</details>

## 93. Rotative Factorization Machines

<details>

<summary>Abstract</summary>

Feature interaction learning (FIL) focuses on capturing the complex relationships among multiple features for building predictive models, which is widely used in real-world tasks. Despite the research progress, existing FIL methods suffer from two major limitations. Firstly, they mainly model the feature interactions within a bounded order (e.g., small integer order) due to the exponential growth of the interaction terms. Secondly, the interaction order of each feature is often independently learned, which lacks the flexibility to capture the feature dependencies in varying contexts. To address these issues, we present Rotative Factorization Machines (RFM), based on the key idea that represents each feature as a polar angle in the complex plane. As such, the feature interactions are converted into a series of complex rotations, where the orders are cast into the rotation coefficients, thereby allowing for the learning of arbitrarily large order. Further, we propose a novel self-attentive rotation function that models the rotation coefficients through a rotation-based attention mechanism, which can adaptively learn the interaction orders under different interaction contexts. Moreover, it incorporates a modulus amplification network to learn the modulus of the complex features, which further enhances the expressive capacity. Our proposed approach provides a general FIL framework, and many existing models can be instantiated in this framework, e.g., factorization machines. In theory, it possesses more strong capacities to model complex feature relationships, and can learn arbitrary features from varied contexts. Extensive experiments conducted on five widely used datasets have demonstrated the effectiveness of our approach.

</details>

## 94. Flexible Graph Neural Diffusion with Latent Class Representation Learning

<details>

<summary>Abstract</summary>

In existing graph data, the connection relationships often exhibit uniform weights, leading to the model aggregating neighboring nodes with equal weights across various connection types. However, this uniform aggregation of diverse information diminishes the discriminability of node representations, contributing significantly to the over-smoothing issue in models. In this paper, we propose the Flexible Graph Neural Diffusion (FGND) model, incorporating latent class representation to address the misalignment between graph topology and node features. In particular, we combine latent class representation learning with the inherent graph topology to reconstruct the diffusion matrix during the graph diffusion process. We introduce the sim metric to quantify the degree of mismatch between graph topology and node features. By flexibly adjusting the dependency level on node features through the hyperparameter, we accommodate diverse adjacency relationships. The effective filtering of noise in the topology also allows the model to capture higher order information, significantly alleviating the over-smoothing problem. Meanwhile, we model the graphical diffusion process as a set of differential equations and employ advanced partial differential equation tools to obtain more accurate solutions. Empirical evaluations on five benchmarks reveal that our FGND model outperforms existing popular GNN methods in terms of both overall performance and stability under data perturbations. Meanwhile, our model exhibits superior performance in comparison to models tailored for heterogeneous graphs and those designed to address oversmoothing issues.

</details>

## 95. Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation

<details>

<summary>Abstract</summary>

Cross-market recommendation (CMR) involves selling the same set of items across multiple nations or regions within a transfer learning framework. However, CMR's distinctive characteristics, including limited data sharing due to privacy policies, absence of user overlap, and a shared item set between markets present challenges for traditional recommendation methods. Moreover, CMR experiences market shifts, leading to differences in item popularity and user preferences among different markets. This study focuses on cross-market sequential recommendation (CMSR) and proposes the Cross-market Attention Transferring with Sequential Recommendation (CAT-SR) framework to address these challenges and market shifts. CAT-SR incorporates a pre-training strategy emphasizing item-item correlation, selective self-attention transferring for effective transfer learning, and query and key adapters for market-specific user preferences. Experimental results on real-world cross-market datasets demonstrate the superiority of CAT-SR, and ablation studies validate the benefits of its components across different geographical continents. CAT-SR offers a robust and adaptable solution for cross-market sequential recommendation. The code is available at https://github.com/ChenMetanoia/CATSR-KDD/.

</details>

## 96. Reinforced Compressive Neural Architecture Search for Versatile Adversarial Robustness

<details>

<summary>Abstract</summary>

Prior research on neural architecture search (NAS) for adversarial robustness has revealed that a lightweight and adversarially robust sub-network could exist in a non-robust large teacher network. Such a sub-network is generally discovered based on heuristic rules to perform neural architecture search. However, heuristic rules are inadequate to handle diverse adversarial attacks and different "teacher" network capacity. To address this key challenge, we propose Reinforced Compressive Neural Architecture Search (RC-NAS), aiming to achieve Versatile Adversarial Robustness. Specifically, we define novel task settings that compose datasets, adversarial attacks, and teacher network configuration. Given diverse tasks, we develop an innovative dual-level training paradigm that consists of a meta-training and a fine-tuning phase to effectively expose the RL agent to diverse attack scenarios (in meta-training), and make it adapt quickly to locate an optimal sub-network (in fine-tuning) for previously unseen scenarios. Experiments show that our framework could achieve adaptive compression towards different initial teacher networks, datasets, and adversarial attacks, resulting in more lightweight and adversarially robust architectures. We also provide a theoretical analysis to explain why the reinforcement learning (RL)-guided adversarial architectural search helps adversarial robustness over standard adversarial training methods.

</details>

## 97. Revisiting Local PageRank Estimation on Undirected Graphs: Simple and Optimal 🌟

<details>

<summary>Abstract</summary>

We propose a simple and optimal algorithm, BackMC, for local PageRank estimation in undirected graphs: given an arbitrary target node t in an undirected graph G comprising n nodes and m edges, BackMC accurately estimates the PageRank score of node t while assuring a small relative error and a high success probability. The worst-case computational complexity of BackMC is upper bounded by O(1/dmin ⋅ min(dt, m1/2)), where dmin denotes the minimum degree of G, and dt denotes the degree of t, respectively. Compared to the previously best upper bound of O(log n ⋅ min(dt, m1/2)) (VLDB '23), which is derived from a significantly more complex algorithm and analysis, our BackMC improves the computational complexity for this problem by a factor of Θ(log n/dmin) with a much simpler algorithm. Furthermore, we establish a matching lower bound of Ω(1/dmin ⋅ min(dt, m1/2)) for any algorithm that attempts to solve the problem of local PageRank estimation, demonstrating the theoretical optimality of our BackMC. We conduct extensive experiments on various large-scale real-world and synthetic graphs, where BackMC consistently shows superior performance.

</details>

## 98. Mastering Long-Tail Complexity on Graphs: Characterization, Learning, and Generalization 🌟

<details>

<summary>Abstract</summary>

In the context of long-tail classification on graphs, the vast majority of existing work primarily revolves around the development of model debiasing strategies, intending to mitigate class imbalances and enhance the overall performance. Despite the notable success, there is very limited literature that provides a theoretical tool for characterizing the behaviors of long-tail classes in graphs and gaining insight into generalization performance in real-world scenarios. To bridge this gap, we propose a generalization bound for long-tail classification on graphs by formulating the problem in the fashion of multi-task learning, i.e., each task corresponds to the prediction of one particular class. Our theoretical results show that the generalization performance of long-tail classification is dominated by the overall loss range and the task complexity. Building upon the theoretical findings, we propose a novel generic framework HierTail for long-tail classification on graphs. In particular, we start with a hierarchical task grouping module that allows us to assign related tasks into hypertasks and thus control the complexity of the task space; then, we further design a balanced contrastive learning module to adaptively balance the gradients of both head and tail classes to control the loss range across all tasks in a unified fashion. Extensive experiments demonstrate the effectiveness of HierTail in characterizing long-tail classes on real graphs, which achieves up to 12.9\% improvement over the leading baseline method in balanced accuracy.

</details>

## 99. Unsupervised Heterogeneous Graph Rewriting Attack via Node Clustering

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) has become one of the most popular learning paradigms and has achieved remarkable success in the graph field. Recently, a series of pre-training studies on heterogeneous graphs (HGs) using SSL have been proposed considering the heterogeneity of real-world graph data. However, verification of the robustness of heterogeneous graph pre-training is still a research gap. Most existing researches focus on supervised attacks on graphs, which are limited to a specific scenario and will not work when labels are not available. In this paper, we propose a novel unsupervised heterogeneous graph rewriting attack via node clustering (HGAC) that can effectively attack HG pre-training models without using labels. Specifically, a heterogeneous edge rewriting strategy is designed to ensure the rationality and concealment of the attacks. Then, a tailored heterogeneous graph contrastive learning (HGCL) is used as a surrogate model. Moreover, we leverage node clustering results of the clean HGs as the pseudo-labels to guide the optimization of structural attacks. Extensive experiments exhibit powerful attack performances of our HGAC on various downstream tasks (i.e., node classification, node clustering, metapath prediction, and visualization) under poisoning attack and evasion attack.

</details>

## 100. A Novel Prompt Tuning for Graph Transformers: Tailoring Prompts to Graph Topologies 🌟

<details>

<summary>Abstract</summary>

Deep graph prompt tuning (DeepGPT), which only tunes a set of continuous prompts for graph transformers, significantly decreases the storage usage during training. However, DeepGPT is limited by its uniform prompts to input graphs with various structures. This is because different graph structures dictate various feature interactions between nodes, while the uniform prompts are not dynamic to tailor the feature transformation for the graph topology. In this paper, we propose a &lt;u&gt;T&lt;/u&gt;opo-specific &lt;u&gt;G&lt;/u&gt;raph &lt;u&gt;P&lt;/u&gt;rompt &lt;u&gt;T&lt;/u&gt;uning (TGPT ), which provides topo-specific prompts tailored to the topological structures of input graphs. Specifically, TGPT learns trainable embeddings for graphlets and frequencies, where graphlets are fundamental sub-graphs that describe the structure around specific nodes. Based on the statistic data about graphlets of input graph, topo-specific prompts are generated by graphlet embeddings and frequency embeddings. The topo-specific prompts include node-level topo-specific prompts for specified nodes, a graph-level topo-specific prompt for the entire graph, and a task-specific prompt to learn task-related information. They are all inserted into specific graph nodes to perform feature transformation, providing specified feature transformation for input graphs with different topological structures. Extensive experiments show that our method outperforms existing lightweight fine-tuning methods and DeepGPT in molecular graph classification and regression with comparable parameters.

</details>

## 101. POND: Multi-Source Time Series Domain Adaptation with Information-Aware Prompt Tuning

<details>

<summary>Abstract</summary>

Time series domain adaptation stands as a pivotal and intricate challenge with diverse applications, including but not limited to human activity recognition, sleep stage classification, and machine fault diagnosis. Despite the numerous domain adaptation techniques proposed to tackle this complex problem, they primarily focus on domain adaptation from a single source domain. Yet, it is more crucial to investigate domain adaptation from multiple domains due to the potential for greater improvements. To address this, three important challenges need to be overcome: 1). The lack of exploration to utilize domain-specific information for domain adaptation, 2). The difficulty to learn domain-specific information that changes over time, and 3). The difficulty to evaluate learned domain-specific information. In order to tackle these challenges simultaneously, in this paper, we introduce PrOmpt-based domaiN Discrimination (POND), the first framework to utilize prompts for time series domain adaptation. Specifically, to address Challenge 1, we extend the idea of prompt tuning to time series analysis and learn prompts to capture common and domain-specific information from all source domains. To handle Challenge 2, we introduce a conditional module for each source domain to generate prompts from time series input data. For Challenge 3, we propose two criteria to select good prompts, which are used to choose the most suitable source domain for domain adaptation. The efficacy and robustness of our proposed POND model are extensively validated through experiments across 50 scenarios encompassing four datasets. Experimental results demonstrate that our proposed POND model outperforms all state-of-the-art comparison methods by up to 66\% on the F1-score.

</details>

## 102. The Heterophilic Snowflake Hypothesis: Training and Empowering GNNs for Heterophilic Graphs 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have become pivotal tools for a range of graph-based learning tasks. Notably, most current GNN architectures operate under the assumption of homophily, whether explicitly or implicitly. While this underlying assumption is frequently adopted, it is not universally applicable, which can result in potential shortcomings in learning effectiveness. In this paper, or the first time, we transfer the prevailing concept of "one node one receptive field" to the heterophilic graph. By constructing a proxy label predictor, we enable each node to possess a latent prediction distribution, which assists connected nodes in determining whether they should aggregate their associated neighbors. Ultimately, every node can have its own unique aggregation hop and pattern, much like each snowflake is unique and possesses its own characteristics. Based on observations, we innovatively introduce the Heterophily Snowflake Hypothesis and provide an effective solution to guide and facilitate research on heterophilic graphs and beyond. We conduct comprehensive experiments including (1) main results on 10 graphs with varying heterophily ratios across 10 backbones; (2) scalability on various deep GNN backbones (SGC, JKNet, etc.) across various large number of layers (2,4,6,8,16,32 layers); (3) comparison with conventional snowflake hypothesis; (4) efficiency comparison with existing graph pruning algorithms. Our observations show that our framework acts as a versatile operator for diverse tasks. It can be integrated into various GNN frameworks, boosting performance in-depth and offering an explainable approach to choosing the optimal network depth. The source code is available at https://github.com/bingreeky/HeteroSnoH.

</details>

## 103. A Hierarchical and Disentangling Interest Learning Framework for Unbiased and True News Recommendation

<details>

<summary>Abstract</summary>

In the era of information explosion, news recommender systems are crucial for users to effectively and efficiently discover their interested news. However, most of the existing news recommender systems face two major issues, hampering recommendation quality. Firstly, they often oversimplify users' reading interests, neglecting their hierarchical nature, spanning from high-level event (e.g., US Election) related interests to low-level news article-specifc interests. Secondly, existing work often assumes a simplistic context, disregarding the prevalence of fake news and political bias under the real-world context. This oversight leads to recommendations of biased or fake news, posing risks to individuals and society. To this end, this paper addresses these gaps by introducing a novel framework, the Hierarchical and Disentangling Interest learning framework (HDInt). HDInt incorporates a hierarchical interest learning module and a disentangling interest learning module. The former captures users' high- and low-level interests, enhancing next-news recommendation accuracy. The latter effectively separates polarity and veracity information from news contents and model them more specifcally, promoting fairness- and truth-aware reading interest learning for unbiased and true news recommendations. Extensive experiments on two real-world datasets demonstrate HDInt's superiority over state-of-the-art news recommender systems in delivering accurate, unbiased, and true news recommendations.

</details>

## 104. Optimizing Long-tailed Link Prediction in Graph Neural Networks through Structure Representation Enhancement 🌟

<details>

<summary>Abstract</summary>

Link prediction, as a fundamental task for graph neural networks (GNNs), has boasted significant progress in varied domains. Its success is typically influenced by the expressive power of node representation, but recent developments reveal the inferior performance of low-degree nodes owing to their sparse neighbor connections, known as the degree-based long-tailed problem. Will the degree-based long-tailed distribution similarly constrain the efficacy of GNNs on link prediction? Unexpectedly, our study reveals that only a mild correlation exists between node degree and predictive accuracy, and more importantly, the number of common neighbors between node pairs exhibits a strong correlation with accuracy. Considering node pairs with less common neighbors, i.e., tail node pairs, make up a substantial fraction of the dataset but achieve worse performance, we propose that link prediction also faces the long-tailed problem. Therefore, link prediction of GNNs is greatly hindered by the tail node pairs. After knowing the weakness of link prediction, a natural question is how can we eliminate the negative effects of the skewed long-tailed distribution on common neighbors so as to improve the performance of link prediction? Towards this end, we introduce our long-tailed framework (LTLP), which is designed to enhance the performance of tail node pairs on link prediction by increasing common neighbors. Two key modules in LTLP respectively supplement high-quality edges for tail node pairs and enforce representational alignment between head and tail node pairs within the same category, thereby improving the performance of tail node pairs. Empirical results across five datasets confirm that our approach not only achieves SOTA performance but also greatly reduces the performance bias between the head and tail. These findings underscore the efficacy and superiority of our framework in addressing the long-tailed problem in link prediction.

</details>

## 105. Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions

<details>

<summary>Abstract</summary>

In recommendation systems, new items are continuously introduced, initially lacking interaction records but gradually accumulating them over time. Accurately predicting the click-through rate (CTR) for these items is crucial for enhancing both revenue and user experience. While existing methods focus on enhancing item ID embeddings for new items within general CTR models, they tend to adopt a global feature interaction approach, often overshadowing new items with sparse data by those with abundant interactions. Addressing this, our work introduces EmerG, a novel approach that warms up cold-start CTR prediction by learning item-specific feature interaction patterns. EmerG utilizes hypernetworks to generate an item-specific feature graph based on item characteristics, which is then processed by a Graph Neural Network (GNN). This GNN is specially tailored to provably capture feature interactions at any order through a customized message passing mechanism. We further design a meta learning strategy that optimizes parameters of hypernetworks and GNN across various item CTR prediction tasks, while only adjusting a minimal set of item-specific parameters within each task. This strategy effectively reduces the risk of overfitting when dealing with limited data. Extensive experiments on benchmark datasets validate that EmerG consistently performs the best given no, a few and sufficient instances of new items.

</details>

## 106. EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration

<details>

<summary>Abstract</summary>

Generative retrieval has recently emerged as a promising approach to sequential recommendation, framing candidate item retrieval as an autoregressive sequence generation problem. However, existing generative methods typically focus solely on either behavioral or semantic aspects of item information, neglecting their complementary nature and thus resulting in limited effectiveness. To address this limitation, we introduce EAGER, a novel generative recommendation framework that seamlessly integrates both behavioral and semantic information. Specifically, we identify three key challenges in combining these two types of information: a unified generative architecture capable of handling two feature types, ensuring sufficient and independent learning for each type, and fostering subtle interactions that enhance collaborative information utilization. To achieve these goals, we propose (1) a two-stream generation architecture leveraging a shared encoder and two separate decoders to decode behavior tokens and semantic tokens with a confidence-based ranking strategy; (2) a global contrastive task with summary tokens to achieve discriminative decoding for each type of information; and (3) a semantic-guided transfer task designed to implicitly promote cross-interactions through reconstruction and estimation objectives. We validate the effectiveness of EAGER on four public benchmarks, demonstrating its superior performance compared to existing methods. Our source code will be publicly available on PapersWithCode.com.

</details>

## 107. AsyncET: Asynchronous Representation Learning for Knowledge Graph Entity Typing

<details>

<summary>Abstract</summary>

Knowledge graph entity typing (KGET) aims to predict the missing entity types in knowledge graphs (KG). The relationship between entities and their corresponding types is often expressed using a single relation, hasType. However, hasType has a limited capability for modeling diverse entity-type relationships in the embedding space. In this paper, we first introduce multiple auxiliary relations to model the complex entity-type relationship. We propose an efficient and robust algorithm to group similar entity types together and assign a unique auxiliary relation to each group. Then, with the auxiliary relations, we propose an Asynchronous representation learning framework for KGET, named AsyncET, where entity and type embeddings are updated alternatively. Consequently, the quality of entity embeddings is gradually improved during training by infusing type information. In addition, entity types with different granularities and semantics can be properly modeled in the embedding space. Experimental results show that AsyncET can substantially improve the performance of embedding-based methods on the KGET task and has a significant advantage over state-of-the-art neural network-based methods in terms of model sizes and inference time.

</details>

## 108. Unveiling Global Interactive Patterns across Graphs: Towards Interpretable Graph Neural Networks 🌟

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) have emerged as a prominent framework for graph mining, leading to significant advances across various domains. Stemmed from the node-wise representations of GNNs, existing explanation studies have embraced the subgraph-specific viewpoint that attributes the decision results to the salient features and local structures of nodes. However, graph-level tasks necessitate long-range dependencies and global interactions for advanced GNNs, deviating significantly from subgraph-specific explanations. To bridge this gap, this paper proposes a novel intrinsically interpretable scheme for graph classification, termed as Global Interactive Pattern (GIP) learning, which introduces learnable global interactive patterns to explicitly interpret decisions. GIP first tackles the complexity of interpretation by clustering numerous nodes using a constrained graph clustering module. Then, it matches the coarsened global interactive instance with a batch of self-interpretable graph prototypes, thereby facilitating a transparent graph-level reasoning process. Extensive experiments conducted on both synthetic and real-world benchmarks demonstrate that the proposed GIP yields significantly superior interpretability and competitive performance to the state-of-the-art counterparts. Our code will be made publicly available¹.

</details>

## 109. Self-Supervised Learning for Graph Dataset Condensation 🌟

<details>

<summary>Abstract</summary>

Graph dataset condensation (GDC) reduces a dataset with many graphs into a smaller dataset with fewer graphs while maintaining model training accuracy. GDC saves the storage cost and hence accelerates training. Although several GDC methods have been proposed, they are all supervised and require massive labels for the graphs, while graph labels can be scarce in many practical scenarios. To fill this gap, we propose a self-supervised graph dataset condensation method called SGDC, which does not require label information. Our initial design starts with the classical bilevel optimization paradigm for dataset condensation and incorporates contrastive learning techniques. But such a solution yields poor accuracy due to the biased gradient estimation caused by data augmentation. To solve this problem, we introduce representation matching, which conducts training by aligning the representations produced by the condensed graphs with the target representations generated by a pre-trained SSL model. This design eliminates the need for data augmentation and avoids biased gradient. We further propose a graph attention kernel, which not only improves accuracy but also reduces running time when combined with self-supervised kernel ridge regression (KRR). To simplify SGDC and make it more robust, we adopt a adjacency matrix reusing approach, which reuses the topology of the original graphs for the condensed graphs instead of repeatedly learning topology during training. Our evaluations on seven graph datasets find that SGDC improves model accuracy by up to 9.7\% compared with 5 state-of-the-art baselines, even if they use label information. Moreover, SGDC is significantly more efficient than the baselines.

</details>

## 110. Unveiling Vulnerabilities of Contrastive Recommender Systems to Poisoning Attacks

<details>

<summary>Abstract</summary>

Contrastive learning (CL) has recently gained prominence in the domain of recommender systems due to its great ability to enhance recommendation accuracy and improve model robustness. Despite its advantages, this paper identifies a vulnerability of CL-based recommender systems that they are more susceptible to poisoning attacks aiming to promote individual items. Our analysis indicates that this vulnerability is attributed to the uniform spread of representations caused by the InfoNCE loss. Furthermore, theoretical and empirical evidence shows that optimizing this loss favors smooth spectral values of representations. This finding suggests that attackers could facilitate this optimization process of CL by encouraging a more uniform distribution of spectral values, thereby enhancing the degree of representation dispersion. With these insights, we attempt to reveal a potential poisoning attack against CL-based recommender systems, which encompasses a dual-objective framework: one that induces a smoother spectral value distribution to amplify the InfoNCE loss's inherent dispersion effect, named dispersion promotion; and the other that directly elevates the visibility of target items, named rank promotion. We validate the threats of our attack model through extensive experimentation on four datasets. By shedding light on these vulnerabilities, our goal is to advance the development of more robust CL-based recommender systems. The code is available at https://github.com/CoderWZW/ARLib.

</details>

## 111. Neural Manifold Operators for Learning the Evolution of Physical Dynamics

<details>

<summary>Abstract</summary>

Modeling the evolution of physical dynamics is a foundational problem in science and engineering, and it is regarded as the modeling of an operator mapping between infinite-dimensional functional spaces. Operator learning methods, learning the underlying infinite-dimensional operator in a high-dimensional latent space, have shown significant potential in modeling physical dynamics. However, there remains insufficient research on how to approximate an infinite-dimensional operator using a finite-dimensional parameter space. Inappropriate dimensionality representation of the underlying operator leads to convergence difficulties, decreasing generalization capability, and violating the physical consistency. To address the problem, we present Neural Manifold Operator (NMO) to learn the invariant subspace with the intrinsic dimension to parameterize infinite-dimensional underlying operators. NMO achieves state-of-the-art performance in statistical and physical metrics and gains 23.35\% average improvement on three real-world scenarios and four equation-governed scenarios across a wide range of multi-disciplinary fields. Our paradigm has demonstrated universal effectiveness across various model structure implementations, including Multi-Layer Perceptron, Convolutional Neural Networks, and Transformers. Experimentally, we prove that the intrinsic dimension calculated by our paradigm is the optimal dimensional representation of the underlying operators. We release our code at https://github.com/AI4EarthLab/Neural-Manifold-Operators.

</details>

## 112. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks

<details>

<summary>Abstract</summary>

It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38\% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.

</details>

## 113. Distributional Network of Networks for Modeling Data Heterogeneity

<details>

<summary>Abstract</summary>

Heterogeneous data widely exists in various high-impact applications. Domain adaptation and out-of-distribution generalization paradigms have been formulated to handle the data heterogeneity across domains. However, most existing domain adaptation and out-of-distribution generalization algorithms do not explicitly explain how the label information can be adaptively propagated from the source domains to the target domain. Furthermore, little effort has been devoted to theoretically understanding the convergence of existing algorithms based on neural networks.To address these problems, in this paper, we propose a generic distributional network of networks (TENON) framework, where each node of the main network represents an individual domain associated with a domain-specific network. In this case, the edges within the main network indicate the domain similarity, and the edges within each network indicate the sample similarity. The crucial idea of TENON is to characterize the within-domain label smoothness and cross-domain parameter smoothness in a unified framework. The convergence and optimality of TENON are theoretically analyzed. Furthermore, we show that based on the TENON framework, domain adaptation and out-of-distribution generalization can be naturally formulated as transductive and inductive distribution learning problems, respectively. This motivates us to develop two instantiated algorithms (TENON-DA and TENON-OOD) of the proposed TENON framework for domain adaptation and out-of-distribution generalization. The effectiveness and efficiency of TENON-DA and TENON-OOD are verified both theoretically and empirically.

</details>

## 114. Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering 🌟

<details>

<summary>Abstract</summary>

Graph-based models and contrastive learning have emerged as prominent methods in Collaborative Filtering (CF). While many existing models in CF incorporate these methods in their design, there seems to be a limited depth of analysis regarding the foundational principles behind them. This paper bridges graph convolution, a pivotal element of graph-based models, with contrastive learning through a theoretical framework. By examining the learning dynamics and equilibrium of the contrastive loss, we offer a fresh lens to understand contrastive learning via graph theory, emphasizing its capability to capture high-order connectivity. Building on this analysis, we further show that the graph convolutional layers often used in graph-based models are not essential for high-order connectivity modeling and might contribute to the risk of oversmoothing. Stemming from our findings, we introduce Simple Contrastive Collaborative Filtering (SCCF), a simple and effective algorithm based on a naive embedding model and a modified contrastive loss. The efficacy of the algorithm is demonstrated through extensive experiments across four public datasets. The experiment code is available at https://github.com/wu1hong/SCCF.

</details>

## 115. DFGNN: Dual-frequency Graph Neural Network for Sign-aware Feedback

<details>

<summary>Abstract</summary>

The graph-based recommendation has achieved great success in recent years. However, most existing graph-based recommendations focus on capturing user preference based on positive edges/feedback, while ignoring negative edges/feedback (e.g., dislike, low rating) that widely exist in real-world recommender systems. How to utilize negative feedback in graph-based recommendations still remains underexplored. In this study, we first conducted a comprehensive experimental analysis and found that (1) existing graph neural networks are not well-suited for modeling negative feedback, which acts as a high-frequency signal in a user-item graph. (2) The graph-based recommendation suffers from the representation degeneration problem. Based on the two observations, we propose a novel model that models positive and negative feedback from a frequency filter perspective called Dual-frequency Graph Neural Network for Sign-aware Recommendation (DFGNN). Specifically, in DFGNN, the designed dual-frequency graph filter (DGF) captures both low-frequency and high-frequency signals that contain positive and negative feedback. Furthermore, the proposed signed graph regularization is applied to maintain the user/item embedding uniform in the embedding space to alleviate the representation degeneration problem. Additionally, we conduct extensive experiments on real-world datasets and demonstrate the effectiveness of the proposed model. Codes of our model will be released upon acceptance.

</details>

## 116. Cost-Efficient Fraud Risk Optimization with Submodularity in Insurance Claim

<details>

<summary>Abstract</summary>

The fraudulent insurance claim is critical for the insurance industry. Insurance companies or agency platforms aim to confidently estimate the fraud risk of claims by gathering data from various sources. Although more data sources can improve the estimation accuracy, they inevitably lead to increased costs. Therefore, a great challenge of fraud risk verification lies in well balancing these two aspects. To this end, this paper proposes a framework named cost-efficient fraud risk optimization with submodularity (CEROS) to optimize the process of fraud risk verification. CEROS efficiently allocates investigation resources across multiple information sources, balancing the trade-off between accuracy and cost. CEROS consists of two parts that we propose: a submodular set-wise classification model called SSCM to estimate the submodular objective function, and a primal-dual algorithm with segmentation point called PDA-SP to solve the objective function. Specifically, SSCM models the fraud probability associated with multiple information sources and ensures the properties of submodularity of fraud risk without making independence assumption. The submodularity in SSCM enables PDA-SP to significantly speed up dual optimization. Theoretically, we disclose that when PDA-SP optimizes this dual optimization problem, the process is monotonicity. Finally, the trade-off coefficients output by PDA-SP that balance accuracy and cost in fraud risk verification are applied to online insurance claim decision-making. We conduct experiments on offline trials and online A/B tests in two business areas at Alipay: healthcare insurance recommendation and claim verification. The extensive results indicate that, compared with other methods, CEROS achieves acceleration of 66.9\% in convergence speed and meanwhile 18.8\% in cost reduction. Currently, CEROS has been successfully deployed in Alipay.

</details>

## 117. A Deep Prediction Framework for Multi-Source Information via Heterogeneous GNN

<details>

<summary>Abstract</summary>

Predicting information diffusion is a fundamental task in online social networks (OSNs). Recent studies mainly focus on the popularity prediction of specific content but ignore the correlation between multiple pieces of information. The topic is often used to correlate such information and can correspond to multi-source information. The popularity of a topic relies not only on information diffusion time but also on users' followership. Current solutions concentrate on hard time partition, lacking versatility. Meanwhile, the hop-based sampling adopted in state-of-the-art (SOTA) methods encounters redundant user followership. Moreover, many SOTA methods are not designed with good modularity and lack evaluation for each functional module and enlightening discussion. This paper presents a novel extensible framework, coined as HIF, for effective popularity prediction in OSNs with four original contributions. First, HIF adopts a soft partition of users and time intervals to better learn users' behavioral preferences over time. Second, HIF utilizes weighted sampling to optimize the construction of heterogeneous graphs and reduce redundancy. Furthermore, HIF supports multi-task collaborative optimization to improve its learning capability. Finally, as an extensible framework, HIF provides generic module slots to combine different submodules (e.g., RNNs, Transformer encoders). Experiments show that HIF significantly improves performance and interpretability compared to SOTAs.

</details>

## 118. Predicting Cascading Failures with a Hyperparametric Diffusion Model

<details>

<summary>Abstract</summary>

In this paper, we study cascading failures in power grids through the lens of information diffusion models. Similar to the spread of rumors or influence in an online social network, it has been observed that failures (outages) in a power grid can spread contagiously, driven by viral spread mechanisms. We employ a stochastic diffusion model that is Markovian (memoryless) and local (the activation of one node, i.e., transmission line, can only be caused by its neighbors). Our model integrates viral diffusion principles with physics-based concepts, by correlating the diffusion weights (contagion probabilities between transmission lines) with the hyperparametric Information Cascades (IC) model. We show that this diffusion model can be learned from traces of cascading failures, enabling accurate modeling and prediction of failure propagation. This approach facilitates actionable information through well-understood and efficient graph analysis methods and graph diffusion simulations. Furthermore, by leveraging the hyperparametric model, we can predict diffusion and mitigate the risks of cascading failures even in unseen grid configurations, whereas existing methods falter due to a lack of training data. Extensive experiments based on a benchmark power grid and simulations therein show that our approach effectively captures the failure diffusion phenomena and guides decisions to strengthen the grid, reducing the risk of large-scale cascading failures. Additionally, we characterize our model's sample complexity, improving upon the existing bound.

</details>

## 119. Performative Debias with Fair-exposure Optimization Driven by Strategic Agents in Recommender Systems

<details>

<summary>Abstract</summary>

Data bias, e.g., popularity impairs the dynamics of two-sided markets within recommender systems. This overshadows the less visible but potentially intriguing long-tail items that could capture user interest. Despite the abundance of research surrounding this issue, it still poses challenges and remains a hot topic in academic circles. Along this line, in this paper, we developed a re-ranking approach in dynamic settings with fair-exposure optimization driven by strategic agents. Designed for the producer side, the execution of agents assumes content creators can modify item features based on strategic incentives to maximize their exposure. This iterative process entails an end-to-end optimization, employing differentiable ranking operators that simultaneously target accuracy and fairness. Joint objectives ensure the performance of recommendations while enhancing the visibility of tail items. We also leveraged the performativity nature of predictions to illustrate how strategic learning influences content creators to shift towards fairness efficiently, thereby incentivizing features of tail items. Through comprehensive experiments on both public and industrial datasets, we have substantiated the effectiveness and dominance of the proposed method especially on unveiling the potential of tail items.

</details>

## 120. How to Avoid Jumping to Conclusions: Measuring the Robustness of Outstanding Facts in Knowledge Graphs

<details>

<summary>Abstract</summary>

An outstanding fact (OF) is a striking claim by which some entities stand out from their peers on some attribute. OFs serve data journalism, fact checking, and recommendation. However, one could jump to conclusions by selecting truthful OFs while intentionally or inadvertently ignoring lateral contexts and data that render them less striking. This jumping conclusion bias from unstable OFs may disorient the public, including voters and consumers, raising concerns about fairness and transparency in political and business competition. It is thus ethically imperative for several stakeholders to measure the robustness of OFs with respect to lateral contexts and data. Unfortunately, a capacity for such inspection of OFs mined from knowledge graphs (KGs) is missing. In this paper, we propose a methodology that inspects the robustness of OFs in KGs by perturbation analysis. We define (1) entity perturbation, which detects outlying contexts by perturbing context entities in the OF; and (2) data perturbation, which considers plausible data that render an OF less striking. We compute the expected strikingness scores of OFs over perturbation relevance distributions and assess an OF as robust if its measured strikingness does not deviate significantly from the expected. We devise a suite of exact and sampling algorithms for perturbation analysis on large KGs. Extensive experiments reveal that our methodology accurately and efficiently detects frail OFs generated by existing mining approaches on KGs. We also show the effectiveness of our approaches through case and user studies.

</details>

## 121. Make Your Home Safe: Time-aware Unsupervised User Behavior Anomaly Detection in Smart Homes via Loss-guided Mask

<details>

<summary>Abstract</summary>

Smart homes, powered by the Internet of Things, offer great convenience but also pose security concerns due to abnormal behaviors, such as improper operations of users and potential attacks from malicious attackers. Several behavior modeling methods have been proposed to identify abnormal behaviors and mitigate potential risks. However, their performance often falls short because they do not effectively learn less frequent behaviors, consider temporal context, or account for the impact of noise in human behaviors. In this paper, we propose SmartGuard, an autoencoder-based unsupervised user behavior anomaly detection framework. First, we design a Loss-guided Dynamic Mask Strategy (LDMS) to encourage the model to learn less frequent behaviors, which are often overlooked during learning. Second, we propose a Three-level Time-aware Position Embedding (TTPE) to incorporate temporal information into positional embedding to detect temporal context anomaly. Third, we propose a Noise-aware Weighted Reconstruction Loss (NWRL) that assigns different weights for routine behaviors and noise behaviors to mitigate the interference of noise behaviors during inference. Comprehensive experiments demonstrate that SmartGuard consistently outperforms state-of-the-art baselines and also offers highly interpretable results.

</details>

## 122. Towards Lightweight Graph Neural Network Search with Curriculum Graph Sparsification 🌟

<details>

<summary>Abstract</summary>

Graph Neural Architecture Search (GNAS) has achieved superior performance on various graph-structured tasks. However, existing GNAS studies overlook the applications of GNAS in resource-constrained scenarios. This paper proposes to design a joint graph data and architecture mechanism, which identifies important sub-architectures via the valuable graph data. To search for optimal lightweight Graph Neural Networks (GNNs), we propose a Lightweight &lt;u&gt;G&lt;/u&gt;raph &lt;u&gt;N&lt;/u&gt;eural &lt;u&gt;A&lt;/u&gt;rchitecture &lt;u&gt;S&lt;/u&gt;earch with Graph &lt;u&gt;S&lt;/u&gt;pars&lt;u&gt;I&lt;/u&gt;fication and Network &lt;u&gt;P&lt;/u&gt;runing (GASSIP) method. In particular, GASSIP comprises an operation-pruned architecture search module to enable efficient lightweight GNN search. Meanwhile, we design a novel curriculum graph data sparsification module with an architecture-aware edge-removing difficulty measurement to help select optimal sub-architectures. With the aid of two differentiable masks, we iteratively optimize these two modules to efficiently search for the optimal lightweight architecture. Extensive experiments on five benchmarks demonstrate the effectiveness of GASSIP. Particularly, our method achieves on-par or even higher node classification performance with half or fewer model parameters of searched GNNs and a sparser graph.

</details>

## 123. Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback 🌟

<details>

<summary>Abstract</summary>

Multi-modal recommender systems (MRSs) are pivotal in diverse online web platforms and have garnered considerable attention in recent years. However, previous studies overlook the challenges of (1)noisy multi-modal content, (2) noisy user feedback, and (3) aligning multi-modal content and user feedback. To tackle these challenges, we propose Denoising and Aligning Multi-modal Recommender System (DA-MRS). To mitigate noise in multi-modal content, DA-MRS first constructs item-item graphs determined by consistent content similarity across modalities. To denoise user feedback, DA-MRS associates the probability of observed feedback with multi-modal content and devises a denoised BPR loss. Furthermore, DA-MRS implements Alignment guided by User preference to enhance task-specific item representation and Alignment guided by graded Item relations to provide finer-grained alignment. Extensive experiments verify that DA-MRS is a plug-and-play framework and achieves significant and consistent improvements across various datasets, backbone models, and noisy scenarios.

</details>

## 124. An Efficient Subgraph GNN with Provable Substructure Counting Power 🌟

<details>

<summary>Abstract</summary>

We investigate the enhancement of graph neural networks' (GNNs) representation power through their ability in substructure counting. Recent advances have seen the adoption of subgraph GNNs, which partition an input graph into numerous subgraphs, subsequently applying GNNs to each to augment the graph's overall representation. Despite their ability to identify various substructures, subgraph GNNs are hindered by significant computational and memory costs. In this paper, we tackle a critical question: Is it possible for GNNs to count substructures both efficiently and provably? Our approach begins with a theoretical demonstration that the distance to rooted nodes in subgraphs is key to boosting the counting power of subgraph GNNs. To avoid the need for repetitively applying GNN across all subgraphs, we introduce precomputed structural embeddings that encapsulate this crucial distance information. Experiments validate that our proposed model retains the counting power of subgraph GNNs while achieving significantly faster performance.

</details>

## 125. Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method

<details>

<summary>Abstract</summary>

Reciprocal recommender systems~(RRS), conducting bilateral recommendations between two involved parties, have gained increasing attention for enhancing matching efficiency. However, the majority of existing methods in the literature still reuse conventional ranking metrics to separately assess the performance on each side of the recommendation process. These methods overlook the fact that the ranking outcomes of both sides collectively influence the effectiveness of the RRS, neglecting the necessity of a more holistic evaluation and a capable systemic solution.In this paper, we systemically revisit the task of reciprocal recommendation, by introducing the new metrics, formulation, and method. Firstly, we propose five new evaluation metrics that comprehensively and accurately assess the performance of RRS from three distinct perspectives: overall coverage, bilateral stability, and balanced ranking. These metrics provide a more holistic understanding of the system's effectiveness and enable a comprehensive evaluation. Furthermore, we formulate the RRS from a causal perspective, formulating recommendations as bilateral interventions, which can better model the decoupled effects of potential influencing factors. By utilizing the potential outcome framework, we further develop a model-agnostic causal reciprocal recommendation method that considers the causal effects of recommendations. Additionally, we introduce a reranking strategy to maximize matching outcomes, as measured by the proposed metrics. Extensive experiments on two real-world datasets from recruitment and dating scenarios demonstrate the effectiveness of our proposed metrics and approach. The code and dataset are available at: https://github.com/RUCAIBox/CRRS.

</details>

## 126. Noisy Label Removal for Partial Multi-Label Learning

<details>

<summary>Abstract</summary>

This paper addresses the problem of partial multi-label learning (PML), a challenging weakly supervised learning framework, where each sample is associated with a candidate label set comprising both ground-true labels and noisy labels. We theoretically reveal that an increased number of noisy labels in the candidate label set leads to an enlarged generalization error bound, consequently degrading the classification performance. Accordingly, the key to solving PML lies in accurately removing the noisy labels within the candidate label set. To achieve this objective, we leverage prior knowledge about the noisy labels in PML, which suggests that they only exist within the candidate label set and possess binary values. Specifically, we propose a constrained regression model to learn a PML classifier and select the noisy labels. The constraints of the model strictly enforce the location and value of the noisy labels. Simultaneously, the supervision information provided by the candidate label set is unreliable due to the presence of noisy labels. In contrast, the non-candidate labels of a sample precisely indicate the classes to which the sample does not belong. To aid in the selection of noisy labels, we construct a competitive classifier based on the non-candidate labels. The PML classifier and the competitive classifier form a competitive relationship, encouraging mutual learning. We formulate the proposed model as a discrete optimization problem to effectively remove the noisy labels, and we solve it using an alternative algorithm. Extensive experiments conducted on 6 real-world partial multi-label data sets and 7 synthetic data sets, employing various evaluation metrics, demonstrate that our method significantly outperforms state-of-the-art PML methods. The code implementation is publicly available at https://github.com/Yangfc-ML/NLR.

</details>

## 127. Towards Test Time Adaptation via Calibrated Entropy Minimization

<details>

<summary>Abstract</summary>

Robust models must demonstrate strong generalizability, even amid environmental changes. However, the complex variability and noise in real-world data often lead to a pronounced performance gap between the training and testing phases. Researchers have recently introduced test-time-domain adaptation (TTA) to address this challenge. TTA methods primarily adapt source-pretrained models to a target domain using only unlabeled test data. This study found that existing TTA methods consider only the largest logit as a pseudo-label and aim to minimize the entropy of test time predictions. This maximizes the predictive confidence of the model. However, this corresponds to the model being overconfident in the local test scenarios. In response, we introduce a novel confidence-calibration loss function called Calibrated Entropy Test-Time Adaptation (CETA), which considers the model's largest logit and the next-highest-ranked one, aiming to strike a balance between overconfidence and underconfidence. This was achieved by incorporating a sample-wise regularization term. We also provide a theoretical foundation for the proposed loss function. Experimentally, our method outperformed existing strategies on benchmark corruption datasets across multiple models, underscoring the efficacy of our approach.

</details>

## 128. Balanced Confidence Calibration for Graph Neural Networks

<details>

<summary>Abstract</summary>

This paper delves into the confidence calibration in prediction when using Graph Neural Networks (GNNs), which has emerged as a notable challenge in the field. Despite their remarkable capabilities in processing graph-structured data, GNNs are prone to exhibit lower confidence in their predictions than what the actual accuracy warrants. Recent advances attempt to address this by minimizing prediction entropy to enhance confidence levels. However, this method inadvertently risks leading to over-confidence in model predictions. Our investigation in this work reveals that most existing GNN calibration methods predominantly focus on the highest logit, thereby neglecting the entire spectrum of prediction probabilities. To alleviate this limitation, we introduce a novel framework called Balanced Calibrated Graph Neural Network (BCGNN), specifically designed to establish a balanced calibration between over-confidence and under-confidence in GNNs' prediction. To theoretically support our proposed method, we further demonstrate the mechanism of the BCGNN framework in effective confidence calibration and significant trustworthiness improvement in prediction. We conduct extensive experiments to examine the developed framework. The empirical results show our method's superior performance in predictive confidence and trustworthiness, affirming its practical applicability and effectiveness in real-world scenarios.

</details>

## 129. Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space

<details>

<summary>Abstract</summary>

Hyperbolic geometry have shown significant potential in modeling complex structured data, particularly those with underlying tree-like and hierarchical structures. Despite the impressive performance of various hyperbolic neural networks across numerous domains, research on adapting the Transformer to hyperbolic space remains limited. Previous attempts have mainly focused on modifying self-attention modules in the Transformer. However, these efforts have fallen short of developing a complete hyperbolic Transformer. This stems primarily from: (i) the absence of well-defined modules in hyperbolic space, including linear transformation layers, LayerNorm layers, activation functions, dropout operations, etc. (ii) the quadratic time complexity of the existing hyperbolic self-attention module w.r.t the number of input tokens, which hinders its scalability. To address these challenges, we propose, Hypformer, a novel hyperbolic Transformer based on the Lorentz model of hyperbolic geometry. In Hypformer, we introduce two foundational blocks that define the essential modules of the Transformer in hyperbolic space. Furthermore, we develop a linear self-attention mechanism in hyperbolic space, enabling hyperbolic Transformer to process billion-scale graph data and long-sequence inputs for the first time. Our experimental results confirm the effectiveness and efficiency of method across various datasets, demonstrating its potential as an effective and scalable solution for large-scale data representation and large models.

</details>

## 130. Effective Clustering on Large Attributed Bipartite Graphs

<details>

<summary>Abstract</summary>

Attributed bipartite graphs (ABGs) are an expressive data model for describing the interactions between two sets of heterogeneous nodes that are associated with rich attributes, such as customer-product purchase networks and author-paper authorship graphs. Partitioning the target node set in such graphs into k disjoint clusters (referred to as k-ABGC) finds widespread use in various domains, including social network analysis, recommendation systems, information retrieval, and bioinformatics. However, the majority of existing solutions towards k-ABGC either overlook attribute information or fail to capture bipartite graph structures accurately, engendering severely compromised result quality. The severity of these issues is accentuated in real ABGs, which often encompass millions of nodes and a sheer volume of attribute data, rendering effective k-ABGC over such graphs highly challenging.In this paper, we propose TPO, an effective and efficient approach to k-ABGC that achieves superb clustering performance on multiple real datasets. TPO obtains high clustering quality through two major contributions: (i) a novel formulation and transformation of the k-ABGC problem based on multi-scale attribute affinity specialized for capturing attribute affinities between nodes with the consideration of their multi-hop connections in ABGs, and (ii) a highly efficient solver that includes a suite of carefully-crafted optimizations for sidestepping explicit affinity matrix construction and facilitating faster convergence. Extensive experiments, comparing TPO against 19 baselines over 5 real ABGs, showcase the superior clustering quality of TPO measured against ground-truth labels. Moreover, compared to the state of the arts, TPO is often more than 40x faster over both small and large ABGs.

</details>

## 131. Practical Single Domain Generalization via Training-time and Test-time Learning

<details>

<summary>Abstract</summary>

Single domain generalization aims to learn a model that generalizes well to unseen target domains by using a related source domain. However, most existing methods only focus on improving the generalization performance of the model during training, making it difficult to achieve satisfactory performance when deployed in the target domain with large domain shifts. In this paper, we propose a Practical Single Domain Generalization (PSDG) method, which first leverages the knowledge in a source domain to establish a model with good generalization ability in the training phase, and subsequently updates the model to adapt to target domain data using knowledge in the unlabeled target domain during the testing phase. Specifically, during training, PSDG leverages a newly proposed style (e.g., background features) generator named StyIN to generate novel domain data. Moreover, PSDG introduces style-diversity regularization to constantly synthesize distinct styles to expand the coverage of training data, and introduces object-consistency regularization to capture consistency between the currently generated data and the original data, making the model filter style knowledge during training. During testing, PSDG uses a sample-aware and sharpness-aware minimization method to seek for a flat entropy minimum surface for further model optimization by using the knowledge in the unlabeled target domain. Using three real-world datasets the experiments have demonstrated the effectiveness of PSDG, in comparison with several state-of-the-art methods.

</details>

## 132. ReCDA: Concept Drift Adaptation with Representation Enhancement for Network Intrusion Detection

<details>

<summary>Abstract</summary>

The deployment of learning-based models to detect malicious activities in network traffic flows is significantly challenged by concept drift. With evolving attack technology and dynamic attack behaviors, the underlying data distribution of recently arrived traffic flows deviates from historical empirical distributions over time. Existing approaches depend on a significant amount of labeled drifting samples to facilitate the deep model to handle concept drift, which faces labor-intensive manual labeling and the risk of label noise. In this paper, we propose ReCDA, a Concept Drift Adaptation method with Representation enhancement, which consists of a self-supervised representation enhancement stage and a weakly-supervised classifier tuning stage. Specifically, in the initial stage, ReCDA introduces drift-aware perturbation and representation alignment to facilitate the model in acquiring robust representations from drift-aware and drift-invariant perspectives. Moreover, in the subsequent stage, a meticulously crafted instructive sampling strategy and a robust representation constraint encourage the model to learn discriminative knowledge about benign and malicious activities during fine-tuning, thereby enhancing performance further. We conduct comprehensive evaluations on several benchmark datasets under varying degrees of concept drift. The experiment results demonstrate the superior adaptability and robustness of the proposed method.

</details>

## 133. Your Neighbor Matters: Towards Fair Decisions Under Networked Interference

<details>

<summary>Abstract</summary>

In the era of big data, decision-making in social networks may introduce bias due to interconnected individuals. For instance, in peer-to-peer loan platforms on the Web, considering an individual's attributes along with those of their interconnected neighbors, including sensitive attributes, is vital for loan approval or rejection downstream. Unfortunately, conventional fairness approaches often assume independent individuals, overlooking the impact of one person's sensitive attribute on others' decisions. To fill this gap, we introduce "Interference-aware Fairness" (IAF) by defining two forms of discrimination as Self-Fairness (SF) and Peer-Fairness (PF), leveraging advances in interference analysis within causal inference. Specifically, SF and PF causally capture and distinguish discrimination stemming from an individual's sensitive attributes (with fixed neighbors' sensitive attributes) and from neighbors' sensitive attributes (with fixed self's sensitive attributes), separately. Hence, a network-informed decision model is fair only when SF and PF are satisfied simultaneously, as interventions in individuals' sensitive attributes or those of their peers both yield equivalent outcomes. To achieve IAF, we develop a deep doubly robust framework to estimate and regularize SF and PF metrics for decision models. Extensive experiments on synthetic and real-world datasets validate our proposed concepts and methods.

</details>

## 134. SEBot: Structural Entropy Guided Multi-View Contrastive learning for Social Bot Detection 🌟

<details>

<summary>Abstract</summary>

Recent advancements in social bot detection have been driven by the adoption of Graph Neural Networks. The social graph, constructed from social network interactions, contains benign and bot accounts that influence each other. However, previous graph-based detection methods that follow the transductive message-passing paradigm may not fully utilize hidden graph information and are vulnerable to adversarial bot behavior. The indiscriminate message passing between nodes from different categories and communities results in excessively homogeneous node representations, ultimately reducing the effectiveness of social bot detectors. In this paper, we propose SEBot, a novel multi-view graph-based contrastive learning-enabled social bot detector. In particular, we use structural entropy as an uncertainty metric to optimize the entire graph's structure and subgraph-level granularity, revealing the implicitly existing hierarchical community structure. And we design an encoder to enable message passing beyond the homophily assumption, enhancing robustness to adversarial behaviors of social bots. Finally, we employ multi-view contrastive learning to maximize mutual information between different views and enhance the detection performance through multi-task learning. Experimental results demonstrate that our approach significantly improves the performance of social bot detection compared with SOTA methods.

</details>

## 135. Graph Bottlenecked Social Recommendation 🌟

<details>

<summary>Abstract</summary>

With the emergence of social networks, social recommendation has become an essential technique for personalized services. Recently, graph-based social recommendations have shown promising results by capturing the high-order social influence. Most empirical studies of graph-based social recommendations directly take the observed social networks into formulation, and produce user preferences based on social homogeneity. Despite the effectiveness, we argue that social networks in the real-world are inevitably noisy~(existing redundant social relations), which may obstruct precise user preference characterization. Nevertheless, identifying and removing redundant social relations is challenging due to a lack of labels. In this paper, we focus on learning the denoised social structure to facilitate recommendation tasks from an information bottleneck perspective. Specifically, we propose a novel Graph Bottlenecked Social Recommendation (GBSR) framework to tackle the social noise issue. GBSR is a model-agnostic social denoising framework, that aims to maximize the mutual information between the denoised social graph and recommendation labels, meanwhile minimizing it between the denoised social graph and the original one. This enables GBSR to learn the minimal yet sufficient social structure, effectively reducing redundant social relations and enhancing social recommendations. Technically, GBSR consists of two elaborate components, preference-guided social graph refinement, and HSIC-based bottleneck learning. Extensive experimental results demonstrate the superiority of the proposed GBSR, including high performances and good generality combined with various backbones. Our code is available at: https://github.com/yimutianyang/KDD24-GBSR.

</details>

## 136. AdaRD: An Adaptive Response Denoising Framework for Robust Learner Modeling

<details>

<summary>Abstract</summary>

Learner modeling is a crucial task in online learning environments, where Cognitive Diagnosis Models (CDMs) are employed to assess learners' knowledge mastery levels based on recorded response logs. However, the prevalence of noise in recorded response data poses significant challenges, including various behaviors such as guess and slip, casual answers, and system-induced errors. The existence of noise degrades the accuracy of diagnosis results and learner performance predictions. In this work, we propose a general framework, Adaptive Response Denoising (AdaRD), designed to salvage CDMs from the influence of noisy learner-exercise responses. AdaRD extends existing CDMs, incorporating primary training for denoised CDMs and auxiliary training for additional denoising support. The primary training employs binary Generalized Cross Entropy (GCE) loss to slow down the large update of learner knowledge states caused by noisy responses. Simultaneously, we utilize the variance of diagnosed knowledge mastery levels between primary and auxiliary diagnosis modules as a criterion to downweight high-variance responses that are likely to be noisy. In this manner, the proposed framework can prune noisy response learning during training, thereby enhancing the accuracy and robustness of CDMs. Extensive experiments on both real-world and synthetic datasets validate AdaRD's effectiveness in mitigating the impact of noisy learner-exercise responses.

</details>

## 137. Approximate Matrix Multiplication over Sliding Windows

<details>

<summary>Abstract</summary>

Large-scale streaming matrix multiplication is very common in various applications, sparking significant interest in develop efficient algorithms for approximate matrix multiplication (AMM) over streams. In addition, many practical scenarios require to process time-sensitive data and aim to compute matrix multiplication for most recent columns of the data matrices rather than the entire matrices, which motivated us to study efficient AMM algorithms over sliding windows. In this paper, we present two novel deterministic algorithms for this problem and provide corresponding error guarantees. We further reduce the space and time costs of our methods for sparse matrices by performing an approximate singular value decomposition which can utilize the sparsity of matrices. Extensive experimental results on both synthetic and real-world datasets validate our theoretical analysis and highlight the efficiency of our methods.

</details>

## 138. Efficient and Effective Anchored Densest Subgraph Search: A Convex-programming based Approach

<details>

<summary>Abstract</summary>

The quest to identify local dense communities closely connected to predetermined seed nodes is vital across numerous applications. Given the seed nodes R, the R-subgraph density of a subgraph S is defined as traditional graph density of S with penalties on the nodes in S / R. The state-of-the-art (SOTA) anchored densest subgraph model, which is based on R-subgraph density, is designed to address the community search problem. However, it often struggles to efficiently uncover truly dense communities. To eliminate this issue, we propose a novel NR-subgraph density metric, a nuanced measure that identifies communities intimately linked to seed nodes and also exhibiting overall high graph density. We redefine the anchored densest subgraph search problem through the lens of NR-subgraph density and cast it as a Linear Programming (LP) problem. This allows us to transition into a dual problem, tapping into the efficiency and effectiveness of convex programming-based iterative algorithm. To solve this redefined problem, we propose two algorithms: FDP, an iterative method that swiftly attains near-optimal solutions, and FDPE, an exact approach that ensures full convergence. We perform extensive experiments on 12 real-world networks. The results show that our proposed algorithms not only outperform the SOTA methods by 3.6~14.1 times in terms of running time, but also produce subgraphs with superior internal quality.

</details>

## 139. Using Self-supervised Learning Can Improve Model Fairness 🌟

<details>

<summary>Abstract</summary>

Self-supervised learning (SSL) has become the de facto training paradigm of large models, where pre-training is followed by supervised fine-tuning using domain-specific data and labels. Despite demonstrating comparable performance with supervised methods, comprehensive efforts to assess SSL's impact on machine learning fairness (i.e., performing equally on different demographic breakdowns) are lacking. Hypothesizing that SSL models would learn more generic, hence less biased representations, this study explores the impact of pre-training and fine-tuning strategies on fairness. We introduce a fairness assessment framework for SSL, comprising five stages: defining dataset requirements, pre-training, fine-tuning with gradual unfreezing, assessing representation similarity conditioned on demographics, and establishing domain-specific evaluation processes. We evaluate our method's generalizability on three real-world human-centric datasets (i.e., MIMIC, MESA, and GLOBEM) by systematically comparing hundreds of SSL and fine-tuned models on various dimensions spanning from the intermediate representations to appropriate evaluation metrics. Our findings demonstrate that SSL can significantly improve model fairness, while maintaining performance on par with supervised methods-exhibiting up to a 30\% increase in fairness with minimal loss in performance through self-supervision. We posit that such differences can be attributed to representation dissimilarities found between the best- and the worst-performing demographics across models-up to x13 greater for protected attributes with larger performance discrepancies between segments. Code: https://github.com/Nokia-Bell-Labs/SSLfairness

</details>

## 140. Dataset Regeneration for Sequential Recommendation 🌟

<details>

<summary>Abstract</summary>

The sequential recommender (SR) system is a crucial component of modern recommender systems, as it aims to capture the evolving preferences of users. Significant efforts have been made to enhance the capabilities of SR systems. These methods typically follow the model-centric paradigm, which involves developing effective models based on fixed datasets. However, this approach often overlooks potential quality issues and flaws inherent in the data. Driven by the potential of data-centric AI, we propose a novel data-centric paradigm for developing an ideal training dataset using a model-agnostic dataset regeneration framework called DR4SR. This framework enables the regeneration of a dataset with exceptional cross-architecture generalizability. Additionally, we introduce the DR4SR+ framework, which incorporates a model-aware dataset personalizer to tailor the regenerated dataset specifically for a target model. To demonstrate the effectiveness of the data-centric paradigm, we integrate our framework with various model-centric methods and observe significant performance improvements across four widely adopted datasets. Furthermore, we conduct in-depth analyses to explore the potential of the data-centric paradigm and provide valuable insights. The code can be found at https://github.com/USTC-StarTeam/DR4SR.

</details>

## 141. Unsupervised Generative Feature Transformation via Graph Contrastive Pre-training and Multi-objective Fine-tuning 🌟

<details>

<summary>Abstract</summary>

Feature transformation is to derive a new feature set from original features to augment the AI power of data. In many science domains such as material performance screening, while feature transformation can model material formula interactions and compositions and discover performance drivers, supervised labels are collected from expensive and lengthy experiments. This issue motivates an Unsupervised Feature Transformation Learning (UFTL) problem. Prior literature, such as manual transformation, supervised feedback guided search, and PCA, either relies on domain knowledge or expensive supervised feedback, or suffers from large search space, or overlooks non-linear feature-feature interactions. UFTL imposes a major challenge on existing methods: how to design a new unsupervised paradigm that captures complex feature interactions and avoids large search space? To fill this gap, we connect graph, contrastive, and generative learning to develop a measurement-pretrain-finetune paradigm for UFTL. For unsupervised feature set utility measurement, we propose a feature value consistency preservation perspective and develop a mean discounted cumulative gain like unsupervised metric to evaluate feature set utility. For unsupervised feature set representation pretraining, we regard a feature set as a feature-feature interaction graph, and develop an unsupervised graph contrastive learning encoder to embed feature sets into vectors. For generative transformation finetuning, we regard a feature set as a feature cross sequence and feature transformation as sequential generation. We develop a deep generative feature transformation model that coordinates the pretrained feature set encoder and the gradient information extracted from a feature set utility evaluator to optimize a transformed feature generator. Finally, we conduct extensive experiments to demonstrate the effectiveness, efficiency, traceability, and explicitness of our framework.

</details>

## 142. Top-Down Bayesian Posterior Sampling for Sum-Product Networks

<details>

<summary>Abstract</summary>

Sum-product networks (SPNs) are probabilistic models characterized by exact and fast evaluation of fundamental probabilistic operations. Its superior computational tractability has led to applications in many fields, such as machine learning with time constraints or accuracy requirements and real-time systems. The structural constraints of SPNs supporting fast inference, however, lead to increased learning-time complexity and can be an obstacle to building highly expressive SPNs. This study aimed to develop a Bayesian learning approach that can be efficiently implemented on large-scale SPNs. We derived a new full conditional probability of Gibbs sampling by marginalizing multiple random variables to expeditiously obtain the posterior distribution. The complexity analysis revealed that our sampling algorithm works efficiently even for the largest possible SPN. Furthermore, we proposed a hyperparameter tuning method that balances the diversity of the prior distribution and optimization efficiency in large-scale SPNs. Our method has improved learning-time complexity and demonstrated computational speed tens to more than one hundred times faster and superior predictive performance in numerical experiments on more than 20 datasets.

</details>

## 143. PolygonGNN: Representation Learning for Polygonal Geometries with Heterogeneous Visibility Graph

<details>

<summary>Abstract</summary>

Polygon representation learning is essential for diverse applications, encompassing tasks such as shape coding, building pattern classification, and geographic question answering. While recent years have seen considerable advancements in this field, much of the focus has been on single polygons, overlooking the intricate inner- and inter-polygonal relationships inherent in multipolygons. To address this gap, our study introduces a comprehensive framework specifically designed for learning representations of polygonal geometries, particularly multipolygons. Central to our approach is the incorporation of a heterogeneous visibility graph, which seamlessly integrates both inner- and inter-polygonal relationships. To enhance computational efficiency and minimize graph redundancy, we implement a heterogeneous spanning tree sampling method. Additionally, we devise a rotation-translation invariant geometric representation, ensuring broader applicability across diverse scenarios. Finally, we introduce Multipolygon-GNN, a novel model tailored to leverage the spatial and semantic heterogeneity inherent in the visibility graph. Experiments on five real-world and synthetic datasets demonstrate its ability to capture informative representations for polygonal geometries.

</details>

## 144. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data 🌟

<details>

<summary>Abstract</summary>

The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.

</details>

## 145. DipDNN: Preserving Inverse Consistency and Approximation Efficiency for Invertible Learning

<details>

<summary>Abstract</summary>

Consistent bi-directional inferences are the key for many machine learning applications. Without consistency, inverse learning-based inferences can cause fuzzy images, erroneous control signals, and cascading failure in SCADA systems. Since standard deep neural networks (DNNs) are not inherently invertible to offer consistency, some past methods reconstruct DNN architecture analytically for one-to-one correspondence but compromise key features such as universal approximation. Other work maintains the capability of universal approximation in DNNs via iterative numerical approximation. However, these methods limit their applications significantly due to Lipschitz conditions and issues of numerical convergence. The dilemma of the analytical and numerical methods is the incompatibility between nonlinear layer compositions and bijective function construction for inverse modeling. Based on the observation, we propose decomposed-invertible-pathway DNNs (DipDNN). It relaxes the redundant reconstruction of nested DNN in the former methods and eases the Lipschitz constraint. As a result, we strictly guarantee the consistency of global inverse modeling without harming DNN's capability for universal approximation. As numerical stability and generalizability are keys for controlling critical infrastructures, we integrate contractive property with a parallel structure for inductive biases, leading to stable performance. Numerical results show that DipDNN performs significantly better than past methods, thanks to its enforcement of inverse consistency, numerical stability, and physical regularization.

</details>

## 146. Graph Cross Supervised Learning via Generalized Knowledge

<details>

<summary>Abstract</summary>

The success of GNNs highly relies on the accurate labeling of data. Existing methods of ensuring accurate labels, such as weakly-supervised learning, mainly focus on the existing nodes in the graphs. However, in reality, new nodes always continuously emerge on dynamic graphs, with different categories and even label noises. To this end, we formulate a new problem, Graph Cross-Supervised Learning, or Graph Weak-Shot Learning, that describes the challenges of modeling new nodes with novel classes and potential label noises. To solve this problem, we propose Lipshitz-regularized Mixture-of-Experts similarity network (LIME), a novel framework to encode new nodes and handle label noises. Specifically, we first design a node similarity network to capture the knowledge from the original classes, aiming to obtain insights for the emerging novel classes. Then, to enhance the similarity network's generalization to new nodes that could have a distribution shift, we employ the Mixture-of-Experts technique to increase the generalization of knowledge learned by the similarity network. To further avoid losing generalization ability during training, we introduce the Lipschitz bound to stabilize model output and alleviate the distribution shift issue. Empirical experiments validate LIME's effectiveness: we observe a substantial enhancement of up to 11.34\% in node classification accuracy compared to the backbone model when subjected to the challenges of label noise on novel classes across five benchmark datasets. The code can be accessed through https://github.com/xiangchi-yuan/Graph-Cross-Supervised-Learning.

</details>

## 147. Conditional Logical Message Passing Transformer for Complex Query Answering

<details>

<summary>Abstract</summary>

Complex Query Answering (CQA) over Knowledge Graphs (KGs) is a challenging task. Given that KGs are usually incomplete, neural models are proposed to solve CQA by performing multi-hop logical reasoning. However, most of them cannot perform well on both one-hop and multi-hop queries simultaneously. Recent work proposes a logical message passing mechanism based on the pre-trained neural link predictors. While effective on both one-hop and multi-hop queries, it ignores the difference between the constant and variable nodes in a query graph. In addition, during the node embedding update stage, this mechanism cannot dynamically measure the importance of different messages, and whether it can capture the implicit logical dependencies related to a node and received messages remains unclear. In this paper, we propose Conditional Logical Message Passing Transformer (CLMPT), which considers the difference between constants and variables in the case of using pre-trained neural link predictors and performs message passing conditionally on the node type. We empirically verified that this approach can reduce computational costs without affecting performance. Furthermore, CLMPT uses the transformer to aggregate received messages and update the corresponding node embedding. Through the self-attention mechanism, CLMPT can assign adaptive weights to elements in an input set consisting of received messages and the corresponding node and explicitly model logical dependencies between various elements. Experimental results show that CLMPT is a new state-of-the-art neural CQA model. https://github.com/qianlima-lab/CLMPT.

</details>

## 148. Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction 🌟

<details>

<summary>Abstract</summary>

Link prediction is a fundamental task in graph learning, inherently shaped by the topology of the graph. While traditional heuristics are grounded in graph topology, they encounter challenges in generalizing across diverse graphs. Recent research efforts have aimed to leverage the potential of heuristics, yet a unified formulation accommodating both local and global heuristics remains undiscovered. Drawing insights from the fact that both local and global heuristics can be represented by adjacency matrix multiplications, we propose a unified matrix formulation to accommodate and generalize various heuristics. We further propose the Heuristic Learning Graph Neural Network (HL-GNN) to efficiently implement the formulation. HL-GNN adopts intra-layer propagation and inter-layer connections, allowing it to reach a depth of around 20 layers with lower time complexity than GCN. Extensive experiments on the Planetoid, Amazon, and OGB datasets underscore the effectiveness and efficiency of HL-GNN. It outperforms existing methods by a large margin in prediction performance. Additionally, HL-GNN is several orders of magnitude faster than heuristic-inspired methods while requiring only a few trainable parameters. The case study further demonstrates that the generalized heuristics and learned weights are highly interpretable.

</details>

## 149. Knowledge Distillation with Perturbed Loss: From a Vanilla Teacher to a Proxy Teacher

<details>

<summary>Abstract</summary>

Knowledge distillation is a popular technique to transfer knowledge from a large teacher model to a small student model. Typically, the student learns to imitate the teacher by minimizing the KL divergence of its output distribution with the teacher's output distribution. In this work, we argue that such a learning objective is sub-optimal because there exists a discrepancy between the teacher's output distribution and the ground truth label distribution. Therefore, forcing the student to blindly imitate the unreliable teacher output distribution leads to inferior performance. To this end, we propose a novel knowledge distillation objective PTLoss by first representing the vanilla KL-based distillation loss function via a Maclaurin series and then perturbing the leading-order terms in this series. This perturbed loss implicitly transforms the original teacher into a proxy teacher with a distribution closer to the ground truth distribution. We establish the theoretical connection between this "distribution closeness'' and the student model generalizability, which enables us to select the PTLoss's perturbation coefficients in a principled way. Extensive experiments on six public benchmark datasets demonstrate the effectiveness of PTLoss with teachers of different scales.

</details>

## 150. Towards Adaptive Neighborhood for Advancing Temporal Interaction Graph Modeling

<details>

<summary>Abstract</summary>

Temporal Graph Networks (TGNs) have demonstrated their remarkable performance in modeling temporal interaction graphs. These works can generate temporal node representations by encoding the surrounding neighborhoods for the target node. However, an inherent limitation of existing TGNs is their reliance onfixed, hand-crafted rules for neighborhood encoding, overlooking the necessity for an adaptive and learnable neighborhood that can accommodate both personalization and temporal evolution across different timestamps. In this paper, we aim to enhance existing TGNs by introducing anadaptive neighborhood encoding mechanism. We present SEAN (Selective Encoding for Adaptive Neighborhood), a flexible plug-and-play model that can be seamlessly integrated with existing TGNs, effectively boosting their performance. To achieve this, we decompose the adaptive neighborhood encoding process into two phases: (i) representative neighbor selection, and (ii) temporal-aware neighborhood information aggregation. Specifically, we propose the Representative Neighbor Selector component, which automatically pinpoints the most important neighbors for the target node. It offers a tailored understanding of each node's unique surrounding context, facilitating personalization. Subsequently, we propose a Temporal-aware Aggregator, which synthesizes neighborhood aggregation by selectively determining the utilization of aggregation routes and decaying the outdated information, allowing our model to adaptively leverage both the contextually significant and current information during aggregation. We conduct extensive experiments by integrating SEAN into three representative TGNs, evaluating their performance on four public datasets and one financial benchmark dataset introduced in this paper. The results demonstrate that SEAN consistently leads to performance improvements across all models, achieving SOTA performance and exceptional robustness.

</details>

## 151. Topology-aware Embedding Memory for Continual Learning on Expanding Networks

<details>

<summary>Abstract</summary>

Memory replay based techniques have shown great success for continual learning with incrementally accumulated Euclidean data. Directly applying them to continually expanding networks, however, leads to the potential memory explosion problem due to the need to buffer representative nodes and their associated topological neighborhood structures. To this end, we systematically analyze the key challenges in the memory explosion problem, and present a general framework,i.e., Parameter Decoupled Graph Neural Networks (PDGNNs) with Topology-aware Embedding Memory (TEM), to tackle this issue. The proposed framework not only reduces the memory space complexity from O (ndL) to O (n)1: memory budget, d: average node degree, L: the radius of the GNN receptive field, but also fully utilizes the topological information for memory replay. Specifically, PDGNNs decouple trainable parameters from the computation ego-subnetwork viaTopology-aware Embeddings (TEs), which compress ego-subnetworks into compact vectors (i.e., TEs) to reduce the memory consumption. Based on this framework, we discover a unique pseudo-training effect in continual learning on expanding networks and this effect motivates us to develop a novel coverage maximization sampling strategy that can enhance the performance with a tight memory budget. Thorough empirical studies demonstrate that, by tackling the memory explosion problem and incorporating topological information into memory replay, PDGNNs with TEM significantly outperform state-of-the-art techniques, especially in the challenging class-incremental setting.

</details>

## 152. Geometric View of Soft Decorrelation in Self-Supervised Learning

<details>

<summary>Abstract</summary>

Contrastive learning, a form of Self-Supervised Learning (SSL), typically consists of an alignment term and a regularization term. The alignment term minimizes the distance between the embeddings of a positive pair, while the regularization term prevents trivial solutions and expresses prior beliefs about the embeddings. As a widely used regularization technique, soft decorrelation has been employed by several non-contrastive SSL methods to avoid trivial solutions. While the decorrelation term is designed to address the issue of dimensional collapse, we find that it fails to achieve this goal theoretically and experimentally. Based on such a finding, we extend the soft decorrelation regularization to minimize the distance between the covariance matrix and an identity matrix. We provide a new perspective on the geometric distance between positive definite matrices to investigate why the soft decorrelation cannot efficiently solve the dimensional collapse. Furthermore, we construct a family of loss functions utilizing the Bregman Matrix Divergence (BMD), with the soft decorrelation representing a specific instance within this family. We prove that a loss function (LogDet) in this family can solve the issue of dimensional collapse. Our novel loss functions based on BMD exhibit superior performance compared to the soft decorrelation and other baseline techniques, as demonstrated by experimental results on graph and image datasets.

</details>

## 153. Multi-source Unsupervised Domain Adaptation on Graphs with Transferability Modeling 🌟

<details>

<summary>Abstract</summary>

In this paper, we tackle a new problem ofmulti-source unsupervised domain adaptation (MSUDA) for graphs, where models trained on annotated source domains need to be transferred to the unsupervised target graph for node classification. Due to the discrepancy in distribution across domains, the key challenge is how to select good source instances and how to adapt the model. Diverse graph structures further complicate this problem, rendering previous MSUDA approaches less effective. In this work, we present the framework Selective Multi-source Adaptation for Graph (SelMAG ), with a graph-modeling-based domain selector, a sub-graph node selector, and a bi-level alignment objective for the adaptation. Concretely, to facilitate the identification of informative source data, the similarity across graphs is disentangled and measured with the transferability of a graph-modeling task set, and we use it as evidence for source domain selection. A node selector is further incorporated to capture the variation in transferability of nodes within the same source domain. To learn invariant features for adaptation, we align the target domain to selected source data both at the embedding space by minimizing the optimal transport distance and at the classification level by distilling the label function. Modules are explicitly learned to select informative source data and conduct the alignment in virtual training splits with a meta-learning strategy. Experimental results on five graph datasets show the effectiveness of the proposed method.

</details>

## 154. Conformalized Link Prediction on Graph Neural Networks

<details>

<summary>Abstract</summary>

Graph Neural Networks (GNNs) excel in diverse tasks, yet their applications in high-stakes domains are often hampered by unreliable predictions. Although numerous uncertainty quantification methods have been proposed to address this limitation, they often lackrigorous uncertainty estimates. This work makes the first attempt to introduce a distribution-free and model-agnostic uncertainty quantification approach to construct a predictive interval with a statistical guarantee for GNN-based link prediction. We term it asconformalized link prediction. Our approach builds upon conformal prediction (CP), a framework that promises to construct statistically robust prediction sets or intervals. There are two primary challenges: first, given dependent data like graphs, it is unclear whether the critical assumption in CP --- exchangeability --- still holds when applied to link prediction. Second, even if the exchangeability assumption is valid for conformalized link prediction, we need to ensure high efficiency, i.e., the resulting prediction set or the interval length is small enough to provide useful information. To tackle these challenges, we first theoretically and empirically establish a permutation invariance condition for the application of CP in link prediction tasks, along with an exact test-time coverage. Leveraging the important structural information in graphs, we then identify a novel and crucial connection between a graph's adherence to the power law distribution and the efficiency of CP. This insight leads to the development of a simple yet effective sampling-based method to align the graph structure with a power law distribution prior to the standard CP procedure. Extensive experiments demonstrate that for conformalized link prediction, our approach achieves the desired marginal coverage while significantly improving the efficiency of CP compared to baseline methods.

</details>

## 155. GeoMix: Towards Geometry-Aware Data Augmentation

<details>

<summary>Abstract</summary>

Mixup has shown considerable success in mitigating the challenges posed by limited labeled data in image classification. By synthesizing samples through the interpolation of features and labels, Mixup effectively addresses the issue of data scarcity. However, it has rarely been explored in graph learning tasks due to the irregularity and connectivity of graph data. Specifically, in node classification tasks, Mixup presents a challenge in creating connections for synthetic data. In this paper, we propose Geometric Mixup (GeoMix), a simple and interpretable Mixup approach leveraging in-place graph editing. It effectively utilizes geometry information to interpolate features and labels with those from the nearby neighborhood, generating synthetic nodes and establishing connections for them. We conduct theoretical analysis to elucidate the rationale behind employing geometry information for node Mixup, emphasizing the significance of locality enhancement-a critical aspect of our method's design. Extensive experiments demonstrate that our lightweight Geometric Mixup achieves state-of-the-art results on a wide variety of standard datasets with limited labeled data. Furthermore, it significantly improves the generalization capability of underlying GNNs across various challenging out-of-distribution generalization tasks. Our code is available at https://github.com/WtaoZhao/geomix.

</details>

## 156. Bridging and Compressing Feature and Semantic Spaces for Robust Graph Neural Networks: An Information Theory Perspective 🌟

<details>

<summary>Abstract</summary>

The emerging Graph Convolutional Networks (GCNs) have attracted widespread attention in graph learning, due to their good ability of aggregating the information between higher-order neighbors. However, real-world graph data contains high noise and redundancy, making it hard for GCNs to accurately depict the complete relationships between nodes, which seriously degrades the quality of graph representations. Moreover, existing studies commonly ignore the distribution difference between feature and semantic spaces in graphs, causing inferior model generalization. To address these challenges, we propose DIB-RGCN, a novel robust GCN framework, to explore the optimal graph representation with the guidance of the well-designed dual information bottleneck principle. First, we analyze the reasons for distribution differences and theoretically prove that minimal sufficient representations in specific spaces cannot promise optimal performance for downstream tasks. Next, we design new dual channels to regularize feature and semantic spaces, eliminating the sharing of task-irrelevant information between spaces. Different from existing denoising algorithms that adopt a random dropping manner, we innovatively replace potential noisy features and edges with local neighboring representations. This design lowers edge-specific coefficient assignment, alleviating the interference of original representations while retaining graph structures. Further, we maximize the sharing of task-relevant information between feature and semantic spaces to alleviate the difference between them. Using real-world datasets, extensive experiments demonstrate the robustness of the proposed DIB-RGCN, which outperforms state-of-the-art methods on classification tasks.

</details>

## 157. Efficient and Effective Implicit Dynamic Graph Neural Network

<details>

<summary>Abstract</summary>

Implicit graph neural networks have gained popularity in recent years as they capture long-range dependencies while improving predictive performance in static graphs. Despite the tussle between performance degradation due to the oversmoothing of learned embeddings and long-range dependency being more pronounced in dynamic graphs, as features are aggregated both across neighborhood and time, no prior work has proposed an implicit graph neural model in a dynamic setting.In this paper, we present Implicit Dynamic Graph Neural Network (IDGNN) a novel implicit neural network for dynamic graphs which is the first of its kind. A key characteristic of IDGNN is that it demonstrably is well-posed, i.e., it is theoretically guaranteed to have a fixed-point representation. We then demonstrate that the standard iterative algorithm often used to train implicit models is computationally expensive in our dynamic setting as it involves computing gradients, which themselves have to be estimated in an iterative manner. To overcome this, we pose an equivalent bilevel optimization problem and propose an efficient single-loop training algorithm that avoids iterative computation by maintaining moving averages of key components of the gradients. We conduct extensive experiments on real-world datasets on both classification and regression tasks to demonstrate the superiority of our approach over state-of-the-art baselines. We also demonstrate that our bi-level optimization framework maintains the performance of the expensive iterative algorithm while obtaining up to 1600x speed-up.

</details>

## 158. Propagation Structure-Aware Graph Transformer for Robust and Interpretable Fake News Detection

<details>

<summary>Abstract</summary>

The rise of social media has intensified fake news risks, prompting a growing focus on leveraging graph learning methods such as graph neural networks (GNNs) to understand post-spread patterns of news. However, existing methods often produce less robust and interpretable results as they assume that all information within the propagation graph is relevant to the news item, without adequately eliminating noise from engaged users. Furthermore, they inadequately capture intricate patterns inherent in long-sequence dependencies of news propagation due to their use of shallow GNNs aimed at avoiding the over-smoothing issue, consequently diminishing their overall accuracy. In this paper, we address these issues by proposing the Propagation Structure-aware Graph Transformer (PSGT). Specifically, to filter out noise from users within propagation graphs, PSGT first designs a noise-reduction self-attention mechanism based on the information bottleneck principle, aiming to minimize or completely remove the noise attention links among task-irrelevant users. Moreover, to capture multi-scale propagation structures while considering long-sequence features, we present a novel relational propagation graph as a position encoding for the graph Transformer, enabling the model to capture both propagation depth and distance relationships of users. Extensive experiments demonstrate the effectiveness, interpretability, and robustness of our PSGT.

</details>

## 159. One Fits All: Learning Fair Graph Neural Networks for Various Sensitive Attributes 🌟

<details>

<summary>Abstract</summary>

Recent studies have highlighted fairness issues in Graph Neural Networks (GNNs), where they produce discriminatory predictions against specific protected groups categorized by sensitive attributes such as race and age. While various efforts to enhance GNN fairness have made significant progress, these approaches are often tailored to specific sensitive attributes. Consequently, they necessitate retraining the model from scratch to accommodate changes in the sensitive attribute requirement, resulting in high computational costs. To gain deeper insights into this issue, we approach the graph fairness problem from a causal modeling perspective, where we identify the confounding effect induced by the sensitive attribute as the underlying reason. Motivated by this observation, we formulate the fairness problem in graphs from an invariant learning perspective, which aims to learn invariant representations across environments. Accordingly, we propose a graph fairness framework based on invariant learning, namely FairINV, which enables the training of fair GNNs to accommodate various sensitive attributes within a single training session. Specifically, FairINV incorporates sensitive attribute partition and trains fair GNNs by eliminating spurious correlations between the label and various sensitive attributes. Experimental results on several real-world datasets demonstrate that FairINV significantly outperforms state-of-the-art fairness approaches, underscoring its effectiveness. Our code is available via: https://github.com/ZzoomD/FairINV/.

</details>

## 160. Topology-monitorable Contrastive Learning on Dynamic Graphs 🌟

<details>

<summary>Abstract</summary>

Graph contrastive learning is a representative self-supervised graph learning that has demonstrated excellent performance in learning node representations. Despite the extensive studies on graph contrastive learning models, most existing models are tailored to static graphs, hindering their application to real-world graphs which are often dynamically evolving. Directly applying these models to dynamic graphs brings in severe efficiency issues in repetitively updating the learned embeddings. To address this challenge, we propose IDOL, a novel contrastive learning framework for dynamic graph representation learning. IDOL conducts the graph propagation process based on a specially designed Personalized PageRank algorithm which can capture the topological changes incrementally. This effectively eliminates heavy recomputation while maintaining high learning quality. Our another main design is a topology-monitorable sampling strategy which lays the foundation of graph contrastive learning. We further show that the design in IDOL achieves a desired performance guarantee. Our experimental results on multiple dynamic graphs show that IDOL outperforms the strongest baselines on node classification tasks in various performance metrics.

</details>

## 161. Scalable Graph Learning for your Enterprise

<details>

<summary>Abstract</summary>

Much of the world's most valued data is stored in relational databases and data warehouses, where the data is organized into many tables connected by primary-foreign key relations. However, building machine learning models using this data is both challenging and time consuming. The core problem is that no machine learning method is capable of learning on multiple tables interconnected by primary-foreign key relations. Current methods can only learn from a single table, so the data must first be manually joined and aggregated into a single training table, the process known as feature engineering. Feature engineering is slow, error prone and leads to suboptimal models. At Kumo.ai we have worked with researchers worldwide to develop an end-to-end deep representation learning approach to directly learn on data laid out across multiple tables [1]. We name our approach Relational Deep Learning (RDL). The core idea is to view relational databases as a temporal, heterogeneous graph, with a node for each row in each table, and edges specified by primary-foreign key links. Message Passing Graph Neural Networks can then automatically learn across the graph to extract representations that leverage all input data, without any manual feature engineering. Our relational deep learning method to encode graph structure into low-dimensional embeddings brings several benefits: (1) automatic learning from the entire data spread across multiple relational tables (2) no manual feature engineering as the system learns optimal embeddings for a target problem; (3) state-of-the-art predictive performance.

</details>

## 162. DDCDR: A Disentangle-based Distillation Framework for Cross-Domain Recommendation

<details>

<summary>Abstract</summary>

Modern recommendation platforms frequently encompass multiple domains to cater to the varied preferences of users. Recently, cross-domain learning has gained traction as a significant paradigm within the context of recommendation systems, enabling the leveraging of rich information from a well-endowed source domain to enhance a target domain, often limited by inadequate data resources. A primary concern in cross-domain recommendation is the mitigation of negative transfer-ensuring the selective transference of pertinent knowledge from the source (domain-shared knowledge) while maintaining the integrity of domain-unique insights within the target domain (domain-specific knowledge). In this paper, we propose a novel Disentangle-based Distillation Framework for Cross-Domain Recommendation (DDCDR), designed to operate at the representational level and rooted in the established teacher-student knowledge distillation paradigm. Our methodology begins with the development of a cross-domain teacher model, trained adversarially alongside a domain discriminator. This is followed by the creation of a target domain-specific student model. By employing the trained domain discriminator, we successfully segregate domain-shared from domain-specific representations. The teacher model guides the learning of domain-shared features, while domain-specific features are enhanced via contrastive learning methods. Experiments conducted on both public datasets and an industrial dataset demonstrate DDCDR achieves a new state-of-the-art performance. The implementation within Ant Group's platform further confirms its online efficacy, manifesting relative improvements of 0.33\% and 0.45\% in Unique Visitor Click-Through Rate (UVCTR) across two distinct recommendation scenarios, compared to baseline performances.

</details>

## 163. GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting

<details>

<summary>Abstract</summary>

Recommender systems require the simultaneous optimization of multiple objectives to accurately model user interests, necessitating the application of multi-task learning methods. However, existing multi-task learning methods in recommendations overlook the specific characteristics of recommendation scenarios, falling short in achieving proper gradient balance. To address this challenge, we set the target of multi-task learning as attaining the appropriate magnitude balance and the global direction balance, and propose an innovative methodology named GradCraft in response. GradCraft dynamically adjusts gradient magnitudes to align with the maximum gradient norm, mitigating interference from gradient magnitudes for subsequent manipulation. It then employs projections to eliminate gradient conflicts in directions while considering all conflicting tasks simultaneously, theoretically guaranteeing the global resolution of direction conflicts. GradCraft ensures the concurrent achievement of appropriate magnitude balance and global direction balance, aligning with the inherent characteristics of recommendation scenarios. Both offline and online experiments attest to the efficacy of GradCraft in enhancing multi-task performance in recommendations. The source code for GradCraft can be accessed at https://github.com/baiyimeng/GradCraft.

</details>

## 164. CompanyKG: A Large-Scale Heterogeneous Graph for Company Similarity Quantification

<details>

<summary>Abstract</summary>

This paper presents CompanyKG (version 2), a large-scale heterogeneous graph developed for fine-grained company similarity quantification and relationship prediction, crucial for applications in the investment industry such as market mapping, competitor analysis, and mergers and acquisitions. CompanyKG comprises 1.17 million companies represented as graph nodes, enriched with company description embeddings, and 51.06 million weighted edges denoting 15 distinct inter-company relations. To facilitate a thorough evaluation of methods for company similarity quantification and relationship prediction, we have created four annotated evaluation tasks: similarity prediction, competitor retrieval, similarity ranking, and edge prediction. We offer extensive benchmarking results for 11 reproducible predictive methods, categorized into three groups: node-only, edge-only, and node+edge. To our knowledge, CompanyKG is the first large-scale heterogeneous graph dataset derived from a real-world investment platform, specifically tailored for quantifying inter-company similarity and relationships.

</details>

## 165. Diffusion Model-based Mobile Traffic Generation with Open Data for Network Planning and Optimization

<details>

<summary>Abstract</summary>

With the rapid development of the Fifth Generation Mobile Communication Technology (5G) networks, network planning and optimization have become increasingly crucial. Generating high-fidelity network traffic data can preemptively estimate the network demands of mobile users, which holds potential for network operators to improve network performance. However, the data required by existing generation methods is predominantly inaccessible to the public, resulting in a lack of reproducibility for the models and high deployment costs in practice. In this article, we propose an Open data-based Diffusion model for mobile traffic generation (OpenDiff), where a multi-positive contrastive learning algorithm is designed to construct conditional information for the diffusion model using entirely publicly available satellite remote sensing images, Point of Interest (POI), and population data. The conditional information contains relevant human activities in geographical areas, which can effectively guide the generation of network traffic data. We further design an attention-based fusion mechanism to capture the implicit correlations between network traffic and human activity features, enhancing the model's controllable generation capability. We conduct evaluations on three different cities with varying scales, where experimental results verify that our proposed model outperforms existing methods by 14.36\% and 13.05\% in terms of generation fidelity and controllability. To further validate the effectiveness of the model, we leverage the generated traffic data to assist the operators with network planning on a real-world network optimization platform of China Mobile Communications Corporation. The source code is available online:https://github.com/impchai/OpenDiff-diffusion-model-with-open-data.

</details>

## 166. MMBee: Live Streaming Gift-Sending Recommendations via Multi-Modal Fusion and Behaviour Expansion

<details>

<summary>Abstract</summary>

Live streaming services are becoming increasingly popular due to real-time interactions and entertainment. Viewers can chat and send comments or virtual gifts to express their preferences for the streamers. Accurately modeling the gifting interaction not only enhances users' experience but also increases streamers' revenue. Previous studies on live streaming gifting prediction treat this task as a conventional recommendation problem, and model users' preferences using categorical data and observed historical behaviors. However, it is challenging to precisely describe the real-time content changes in live streaming using limited categorical information. Moreover, due to the sparsity of gifting behaviors, capturing the preferences and intentions of users is quite difficult. In this work, we propose MMBee based on real-time &lt;u&gt;M&lt;/u&gt;ulti-&lt;u&gt;M&lt;/u&gt;odal Fusion and &lt;u&gt;Be&lt;/u&gt;haviour &lt;u&gt;E&lt;/u&gt;xpansion to address these issues. Specifically, we first present a Multi-modal Fusion Module with Learnable Query (MFQ) to perceive the dynamic content of streaming segments and process complex multi-modal interactions, including images, text comments and speech. To alleviate the sparsity issue of gifting behaviors, we present a novel Graph-guided Interest Expansion (GIE) approach that learns both user and streamer representations on large-scale gifting graphs with multi-modal attributes. It consists of two main parts: graph node representations pre-training and metapath-based behavior expansion, all of which help model jump out of the specific historical gifting behaviors for exploration and largely enrich the behavior representations. Comprehensive experiment results show that MMBee achieves significant performance improvements on both public datasets and Kuaishou real-world streaming datasets and the effectiveness has been further validated through online A/B experiments. MMBee has been deployed and is serving hundreds of millions of users at Kuaishou.

</details>

## 167. Enhancing E-commerce Spelling Correction with Fine-Tuned Transformer Models

<details>

<summary>Abstract</summary>

In the realm of e-commerce, the process of search stands as the primary point of interaction for users, wielding a profound influence on the platform's revenue generation. Notably, spelling correction assumes a pivotal role in shaping the user's search experience by rectifying erroneous query inputs, thus facilitating more accurate retrieval outcomes. Within the scope of this research paper, our aim is to enhance the existing state-of-the-art discriminative model performance with generative modelling strategies while concurrently addressing the engineering concerns associated with real-time online latency, inherent to models of this category. We endeavor to refine LSTM-based classification models for spelling correction through a generative fine-tuning approach hinged upon pre-trained language models. Our comprehensive offline assessments have yielded compelling results, showcasing that transformer-based architectures, such as BART (developed by Facebook) and T5 (a product of Google), have achieved a 4\% enhancement in F1 score compared to baseline models for the English language sites. Furthermore, to mitigate the challenges posed by latency, we have incorporated model pruning techniques like no-teacher distillation. We have undertaken the deployment of our model (English only) as an A/B test candidate for real-time e-commerce traffic, encompassing customers from the US and the UK. The model attest to a 100\% successful request service rate within real-time scenarios, with median, 90th percentile, and 99th percentile (p90/p99) latencies comfortably falling below production service level agreements. Notably, these achievements are further reinforced by positive customer engagement, transactional and search page metrics, including a significant reduction in instances of search results page with low or almost zero recall. Moreover, we have also extended our efforts into fine-tuning a multilingual model, which, notably, exhibits substantial accuracy enhancements, amounting to a minimum of 16\%, across four distinct European languages and English.

</details>

## 168. Achieving a Better Tradeoff in Multi-stage Recommender Systems through Personalization

<details>

<summary>Abstract</summary>

Recommender systems in social media websites provide value to their communities by recommending engaging content and meaningful connections. Scaling high-quality recommendations to billions of users in real-time requires sophisticated ranking models operating on a vast number of potential items to recommend, becoming prohibitively expensive computationally. A common technique "funnels'' these items through progressively complex models ("multi-stage''), each ranking fewer items but at higher computational cost for greater accuracy. This architecture introduces a trade-off between the cost of ranking items and providing users with the best recommendations. A key observation we make in this paper is that, all else equal, ranking more items indeed improves the overall objective but has diminishing returns. Following this observation, we provide a rigorous formulation through the framework of DR-submodularity, and argue that for a certain class of objectives (reward functions), it is possible to improve the trade-off between performance and computational cost in multi-stage ranking systems with strong theoretical guarantees. We show that this class of reward functions that provide this guarantee is large and robust to various noise models. Finally, we describe extensive experimentation of our method on three real-world recommender systems in Facebook, achieving 8.8\% reduction in overall compute resources with no significant impact on recommendation quality, compared to a 0.8\% quality loss in a non-personalized budget allocation.

</details>

## 169. Residual Multi-Task Learner for Applied Ranking

<details>

<summary>Abstract</summary>

Modern e-commerce platforms rely heavily on modeling diverse user feedback to provide personalized services. Consequently, multi-task learning has become an integral part of their ranking systems. However, existing multi-task learning methods encounter two main challenges: some lack explicit modeling of task relationships, resulting in inferior performance, while others have limited applicability due to being computationally intensive, having scalability issues, or relying on strong assumptions. To address these limitations and better fit our real-world scenario, pre-rank in Shopee Search, we introduce in this paper ResFlow, a lightweight multi-task learning framework that enables efficient cross-task information sharing via residual connections between corresponding layers of task networks. Extensive experiments on datasets from various scenarios and modalities demonstrate its superior performance and adaptability over state-of-the-art methods. The online A/B tests in Shopee Search showcase its practical value in large-scale industrial applications, evidenced by a 1.29\% increase in OPU (order-per-user) without additional system latency. ResFlow is now fully deployed in the pre-rank module of Shopee Search. To facilitate efficient online deployment, we propose a novel offline metric Weighted Recall@K, which aligns well with our online metric OPU, addressing the longstanding online-offline metric misalignment issue. Besides, we propose to fuse scores from the multiple tasks additively when ranking items, which outperforms traditional multiplicative fusion.

</details>

## 170. Controllable Multi-Behavior Recommendation for In-Game Skins with Large Sequential Model

<details>

<summary>Abstract</summary>

Online games often house virtual shops where players can acquire character skins. Our task is centered on tailoring skin recommendations across diverse scenarios by analyzing historical interactions such as clicks, usage, and purchases. Traditional multi-behavior recommendation models employed for this task are limited. They either only predict skins based on a single type of behavior or merely recommend skins for target behavior type/task. These models lack the ability to control predictions of skins that are associated with different scenarios and behaviors. To overcome these limitations, we utilize the pretraining capabilities of Large Sequential Models (LSMs) coupled with a novel stimulus prompt mechanism and build a controllable multi-behavior recommendation (CMBR) model. In our approach, the pretraining ability is used to encapsulate users' multi-behavioral sequences into the representation of users' general interests. Subsequently, our designed stimulus prompt mechanism stimulates the model to extract scenario-related interests, thus generating potential skin purchases (or clicks and other interactions) for users. To the best of our knowledge, this is the first work to provide controlled multi-behavior recommendations, and also the first to apply the pretraining capabilities of LSMs in game domain. Through offline experiments and online A/B tests, we validate our method significantly outperforms baseline models, exhibiting about a tenfold improvement on various metrics during the offline test.

</details>

## 171. Generative Auto-bidding via Conditional Diffusion Modeling

<details>

<summary>Abstract</summary>

Auto-bidding plays a crucial role in facilitating online advertising by automatically providing bids for advertisers. Reinforcement learning (RL) has gained popularity for auto-bidding. However, most current RL auto-bidding methods are modeled through the Markovian Decision Process (MDP), which assumes the Markovian state transition. This assumption restricts the ability to perform in long horizon scenarios and makes the model unstable when dealing with highly random online advertising environments. To tackle this issue, this paper introduces AI-Generated Bidding (AIGB), a novel paradigm for auto-bidding through generative modeling. In this paradigm, we propose DiffBid, a conditional diffusion modeling approach for bid generation. DiffBid directly models the correlation between the return and the entire trajectory, effectively avoiding error propagation across time steps in long horizons. Additionally, DiffBid offers a versatile approach for generating trajectories that maximize given targets while adhering to specific constraints. Extensive experiments conducted on the real-world dataset and online A/B test on Alibaba advertising platform demonstrate the effectiveness of DiffBid, achieving 2.81\% increase in GMV and 3.36\% increase in ROI.

</details>

## 172. Paths2Pair: Meta-path Based Link Prediction in Billion-Scale Commercial Heterogeneous Graphs

<details>

<summary>Abstract</summary>

Link prediction, determining if a relation exists between two entities, is an essential task in the analysis of heterogeneous graphs with diverse entities and relations. Despite extensive research in link prediction, most existing works focus on predicting the relation type between given pairs of entities. However, it is almost impractical to check every entity pair when trying to find most hidden relations in a billion-scale heterogeneous graph due to the billion squared number of possible pairs. Meanwhile, most methods aggregate information at the node level, potentially leading to the loss of direct connection information between the two nodes. In this paper, we introduce Paths2Pair, a novel framework to address these limitations for link prediction in billion-scale commercial heterogeneous graphs. (i) First, it selects a subset of reliable entity pairs for prediction based on relevant meta-paths. (ii) Then, it utilizes various types of content information from the meta-paths between each selected entity pair to predict whether a target relation exists. We first evaluate our Paths2Pair based on a large-scale dataset, and results show Paths2Pair outperforms state-of-the-art baselines significantly. We then deploy our Paths2Pair on JD Logistics, one of the largest logistics companies in the world, for business expansion. The uncovered relations by Paths2Pair have helped JD Logistics identify 108,709 contacts to attract new company customers, resulting in an 84\% increase in the success rate compared to the state-of-the-practice solution, demonstrating the practical value of our framework. We have released the code of our framework at https://github.com/JQHang/Paths2Pair.

</details>

## 173. Cross-Domain LifeLong Sequential Modeling for Online Click-Through Rate Prediction

<details>

<summary>Abstract</summary>

Lifelong sequential modeling (LSM) has significantly advanced recommendation systems on social media platforms. Diverging from single-domain LSM, cross-domain LSM involves modeling lifelong behavior sequences from a source domain to a different target domain. In this paper, we propose the Lifelong Cross Network (LCN), a novel approach for cross-domain LSM. LCN features a Cross Representation Production (CRP) module that utilizes contrastive loss to improve the learning of item embeddings, effectively bridging items across domains. This is important for enhancing the retrieval of relevant items in cross-domain lifelong sequences. Furthermore, we propose the Lifelong Attention Pyramid (LAP) module, which contains three cascading attention levels. By adding an intermediate level and integrating the results from all three levels, the LAP module can capture a broad spectrum of user interests and ensure gradient propagation throughout the sequence. The proposed LAP can also achieve remarkable consistency across attention levels, making it possible to further narrow the candidate item pool of the top level. This allows for the use of advanced attention techniques to effectively mitigate the impact of the noise in cross-domain sequences and improve the non-linearity of the representation, all while maintaining computational efficiency. Extensive experiments conducted on both a public dataset and an industrial dataset from the WeChat Channels platform reveal that the LCN outperforms current methods in terms of prediction accuracy and online performance metrics.

</details>

## 174. ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems

<details>

<summary>Abstract</summary>

Deep Recommender Systems (DRS) are increasingly dependent on a large number of feature fields for more precise recommendations. Effective feature selection methods are consequently becoming critical for further enhancing the accuracy and optimizing storage efficiencies to align with the deployment demands. This research area, particularly in the context of DRS, is nascent and faces three core challenges. Firstly, variant experimental setups across research papers often yield unfair comparisons, obscuring practical insights. Secondly, the existing literature's lack of detailed analysis on selection attributes, based on large-scale datasets and a thorough comparison among selection techniques and DRS backbones, restricts the generalizability of findings and impedes deployment on DRS. Lastly, research often focuses on comparing the peak performance achievable by feature selection methods. This approach is typically computationally infeasible for identifying the optimal hyperparameters and overlooks evaluating the robustness and stability of these methods. To bridge these gaps, this paper presents ERASE, a comprehensive bEnchmaRk for feAture SElection for DRS. ERASE comprises a thorough evaluation of eleven feature selection methods, covering both traditional and deep learning approaches, across four public datasets, private industrial datasets, and a real-world commercial platform, achieving significant enhancement. Our code is available online for ease of reproduction.

</details>

## 175. Contextual Distillation Model for Diversified Recommendation

<details>

<summary>Abstract</summary>

The diversity of recommendation is equally crucial as accuracy in improving user experience. Existing studies, e.g., Determinantal Point Process (DPP) and Maximal Marginal Relevance (MMR), employ a greedy paradigm to iteratively select items that optimize both accuracy and diversity. However, prior methods typically exhibit quadratic complexity, limiting their applications to the re-ranking stage and are not applicable to other recommendation stages with a larger pool of candidate items, such as the pre-ranking and ranking stages. In this paper, we propose Contextual Distillation Model (CDM), an efficient recommendation model that addresses diversification, suitable for the deployment in all stages of industrial recommendation pipelines. Specifically, CDM utilizes the candidate items in the same user request as context to enhance the diversification of the results. We propose a contrastive context encoder that employs attention mechanisms to model both positive and negative contexts. For the training of CDM, we compare each target item with its context embedding and utilize the knowledge distillation framework to learn the win probability of each target item under the MMR algorithm, where the teacher is derived from MMR outputs. During inference, ranking is performed through a linear combination of the recommendation and student model scores, ensuring both diversity and efficiency. We perform offline evaluations on two industrial datasets and conduct online A/B test of CDM on the short-video platform KuaiShou. The considerable enhancements observed in both recommendation quality and diversity, as shown by metrics, provide strong superiority for the effectiveness of CDM.

</details>

## 176. Chromosomal Structural Abnormality Diagnosis by Homologous Similarity

<details>

<summary>Abstract</summary>

Pathogenic chromosome abnormalities are very common among the general population. While numerical chromosome abnormalities can be quickly and precisely detected, structural chromosome abnormalities are far more complex and typically require considerable efforts by human experts for identification. This paper focuses on investigating the modeling of chromosome features and the identification of chromosomes with structural abnormalities. Most existing data-driven methods concentrate on a single chromosome and consider each chromosome independently, overlooking the crucial aspect of homologous chromosomes. In normal cases, homologous chromosomes share identical structures, with the exception that one of them is abnormal. Therefore, we propose an adaptive method to align homologous chromosomes and diagnose structural abnormalities through homologous similarity. Inspired by the process of human expert diagnosis, we incorporate information from multiple pairs of homologous chromosomes simultaneously, aiming to reduce noise disturbance and improve prediction performance. Extensive experiments on real-world datasets validate the effectiveness of our model compared to baselines.

</details>

## 177. Text Matching Indexers in Taobao Search

<details>

<summary>Abstract</summary>

Product search is an important service on Taobao, the largest e-commerce platform in China. Through this service, users can easily find products relevant to their specific needs. Coping with billion-size query loads, Taobao product search has traditionally relied on classical term-based retrieval models due to their powerful and interpretable indexes. In essence, efficient retrieval hinges on the proper storage of the inverted index. Recent successes involve reducing the size (pruning) of the inverted index but the construction and deployment of lossless static index pruning in practical product search still pose non-trivial challenges.In this work, we introduce a novel SM art INDexing (SMIND) solution in Taobao product search. SMIND is designed to reduce information loss during the static pruning process by incorporating user search preferences. Specifically, we first construct "user-query-item'' hypergraphs for four different search preferences, namely purchase, click, exposure, and relevance. Then, we develop an efficient TermRank algorithm applied to these hypergraphs, to preserve relevant items based on specific user preferences during the pruning of the inverted indexer. Our approach offers fresh insights into the field of product search, emphasizing that term dependencies in user search preferences go beyond mere text relevance. Moreover, to address the vocabulary mismatch problem inherent in term-based models, we also incorporate an multi-granularity semantic retrieval model to facilitate semantic matching. Empirical results from both offline evaluation and online A/B tests showcase the superiority of SMIND over state-of-the-art methods, especially in commerce metrics with significant improvements of 1.34\% in Pay Order Count and 1.50\% in Gross Merchandise Value. Besides, SMIND effectively mitigates the Matthew effect of user queries and has been in service for hundreds of millions of daily users since November 2022.

</details>

## 178. An Open and Large-Scale Dataset for Multi-Modal Climate Change-aware Crop Yield Predictions

<details>

<summary>Abstract</summary>

Precise crop yield predictions are of national importance for ensuring food security and sustainable agricultural practices. While AI-for-science approaches have exhibited promising achievements in solving many scientific problems such as drug discovery, precipitation nowcasting, etc., the development of deep learning models for predicting crop yields is constantly hindered by the lack of an open and large-scale deep learning-ready dataset with multiple modalities to accommodate sufficient information. To remedy this, we introduce the CropNet dataset, the first terabyte-sized, publicly available, and multi-modal dataset specifically targeting climate change-aware crop yield predictions for the contiguous United States (U.S.) continent at the county level. Our CropNet dataset is composed of three modalities of data, i.e., Sentinel-2 Imagery, WRF-HRRR Computed Dataset, and USDA Crop Dataset, for over 2200 U.S. counties spanning 6 years (2017-2022), expected to facilitate researchers in developing versatile deep learning models for timely and precisely predicting crop yields at the county-level, by accounting for the effects of both short-term growing season weather variations and long-term climate change on crop yields. Besides, we develop the CropNet package, offering three types of APIs, for facilitating researchers in downloading the CropNet data on the fly over the time and region of interest, and flexibly building their deep learning models for accurate crop yield predictions. Extensive experiments have been conducted on our CropNet dataset via employing various types of deep learning solutions, with the results validating the general applicability and the efficacy of the CropNet dataset in climate change-aware crop yield predictions. We have officially released our CropNet dataset on Hugging Face Datasets https://huggingface.co/datasets/CropNet/CropNet and our CropNet package on the Python Package Index (PyPI) https://pypi.org/project/cropnet. Code and tutorials are available at https://github.com/fudong03/CropNet.

</details>

## 179. Understanding the Ranking Loss for Recommendation with Sparse User Feedback

<details>

<summary>Abstract</summary>

Click-through rate (CTR) prediction is a crucial area of research in online advertising. While binary cross entropy (BCE) has been widely used as the optimization objective for treating CTR prediction as a binary classification problem, recent advancements have shown that combining BCE loss with an auxiliary ranking loss can significantly improve performance. However, the full effectiveness of this combination loss is not yet fully understood. In this paper, we uncover a new challenge associated with the BCE loss in scenarios where positive feedback is sparse: the issue of gradient vanishing for negative samples. We introduce a novel perspective on the effectiveness of the auxiliary ranking loss in CTR prediction: it generates larger gradients on negative samples, thereby mitigating the optimization difficulties when using the BCE loss only and resulting in improved classification ability. To validate our perspective, we conduct theoretical analysis and extensive empirical evaluations on public datasets. Additionally, we successfully integrate the ranking loss into Tencent's online advertising system, achieving notable lifts of 0.70\% and 1.26\% in Gross Merchandise Value (GMV) for two main scenarios. The code is openly accessible at: https://github.com/SkylerLinn/Understanding-the-Ranking-Loss.

</details>

## 180. Beyond Binary Preference: Leveraging Bayesian Approaches for Joint Optimization of Ranking and Calibration

<details>

<summary>Abstract</summary>

Predicting click-through rate (CTR) is a critical task in recommendation systems, where the models are optimized with pointwise loss to infer the probability of items being clicked. In industrial practice, applications also require ranking items based on these probabilities. Existing solutions primarily combine the ranking-based loss, i.e., pairwise and listwise loss, with CTR prediction. However, they can hardly calibrate or generalize well in CTR scenarios where the clicks reflect the binary preference. This is because the binary click feedback leads to a large number of ties, which renders high data sparsity. In this paper, we propose an effective data augmentation strategy, named Beyond Binary Preference (BBP) training framework, to address this problem. Our key idea is to break the ties by leveraging Bayesian approaches, where the beta distribution models click behavior as probability distributions in the training data that naturally break ties. Therefore, we can obtain an auxiliary training label that generates more comparable pairs and improves the ranking performance. Besides, BBP formulates ranking and calibration as a multi-task framework to optimize both objectives simultaneously. Through extensive offline experiments and online tests on various datasets, we demonstrate that BBP significantly outperforms state-of-the-art methods in both ranking and calibration capabilities, showcasing its effectiveness in addressing the limitations of existing methods. Our code is available at https://github.com/AlvinIsonomia/BBP.

</details>

## 181. Modeling User Retention through Generative Flow Networks

<details>

<summary>Abstract</summary>

Recommender systems aim to fulfill the user's daily demands. While most existing research focuses on maximizing the user's engagement with the system, it has recently been pointed out that how frequently the users come back for the service also reflects the quality and stability of recommendations. However, optimizing this user retention behavior is non-trivial and poses several challenges including the intractable leave-and-return user activities, the sparse and delayed signal, and the uncertain relations between users' retention and their immediate feedback towards each item in the recommendation list. In this work, we regard the retention signal as an overall estimation of the user's end-of-session satisfaction and propose to estimate this signal through a probabilistic flow. This flow-based modeling technique can back-propagate the retention reward towards each recommended item in the user session, and we show that the flow combined with traditional learning-to-rank objectives eventually optimizes a non-discounted cumulative reward for both immediate user feedback and user retention. We verify the effectiveness of our method through both offline empirical studies on two public datasets and online A/B tests in an industrial platform.

</details>

## 182. Ads Recommendation in a Collapsed and Entangled World

<details>

<summary>Abstract</summary>

We present Tencent's ads recommendation system and examine the challenges and practices of learning appropriate recommendation representations. Our study begins by showcasing our approaches to preserving prior knowledge when encoding features of diverse types into embedding representations. We specifically address sequence features, numeric features, and pre-trained embedding features. Subsequently, we delve into two crucial challenges related to feature representation: the dimensional collapse of embeddings and the interest entanglement across different tasks or scenarios. We propose several practical approaches to address these challenges that result in robust and disentangled recommendation representations. We then explore several training techniques to facilitate model optimization, reduce bias, and enhance exploration. Additionally, we introduce three analysis tools that enable us to study feature correlation, dimensional collapse, and interest entanglement. This work builds upon the continuous efforts of Tencent's ads recommendation team over the past decade. It summarizes general design principles and presents a series of readily applicable solutions and analysis tools. The reported performance is based on our online advertising platform, which handles hundreds of billions of requests daily and serves millions of ads to billions of users.

</details>

## 183. Addressing Shortcomings in Fair Graph Learning Datasets: Towards a New Benchmark 🌟

<details>

<summary>Abstract</summary>

Fair graph learning plays a pivotal role in numerous practical applications. Recently, many fair graph learning methods have been proposed; however, their evaluation often relies on poorly constructed semi-synthetic datasets or substandard real-world datasets. In such cases, even a basic Multilayer Perceptron (MLP) can outperform Graph Neural Networks (GNNs) in both utility and fairness. In this work, we illustrate that many datasets fail to provide meaningful information in the edges, which may challenge the necessity of using graph structures in these problems. To address these issues, we develop and introduce a collection of synthetic, semi-synthetic, and real-world datasets that fulfill a broad spectrum of requirements. These datasets are thoughtfully designed to include relevant graph structures and bias information crucial for the fair evaluation of models. The proposed synthetic and semi-synthetic datasets offer the flexibility to create data with controllable bias parameters, thereby enabling the generation of desired datasets with user-defined bias values with ease. Moreover, we conduct systematic evaluations of these proposed datasets and establish a unified evaluation approach for fair graph learning models. Our extensive experimental results with fair graph learning methods across our datasets demonstrate their effectiveness in benchmarking the performance of these methods. Our datasets and the code for reproducing our experiments are available at https://github.com/XweiQ/Benchmark-GraphFairness.

</details>

## 184. Leveraging Exposure Networks for Detecting Fake News Sources

<details>

<summary>Abstract</summary>

The scale and dynamic nature of the Web makes real-time detection of misinformation an extremely difficult task. Prior research mostly focused on offline (retrospective) detection of stories or claims using linguistic features of the content, flagging by users, and crowdsourced labels. Here, we develop a novel machine-learning methodology for detecting fake news sources using active learning, and examine the contribution of network, audience, and text features to the model accuracy. Importantly, we evaluate performance in both offline and online settings, mimicking the strategic choices fact-checkers have to make in practice as news sources emerge over time. We find that exposure networks provide information on considerably more sources than sharing networks (+49.6\%), and that the inclusion of exposure features greatly improves classification PR-AUC in both offline (+33\%) and online (+69.2\%) settings. Textual features perform best in offline settings, but their performance deteriorates by 12.0-18.7\% in online settings. Finally, the results show that a few iterations of active learning are sufficient for our model to attain predictive performance to comparable exhaustive labeling while incurring only 24.7\% of the labeling costs. These results stress the importance of exposure networks as a source of valuable information for the investigation of information dissemination in social networks and question the robustness of textual features.

</details>

## 185. Hierarchical Knowledge Guided Fault Intensity Diagnosis of Complex Industrial Systems

<details>

<summary>Abstract</summary>

Fault intensity diagnosis (FID) plays a pivotal role in monitoring and maintaining mechanical devices within complex industrial systems. As current FID methods are based on chain of thought without considering dependencies among target classes. To capture and explore dependencies, we propose a &lt;u&gt;h&lt;/u&gt;ierarchical &lt;u&gt;k&lt;/u&gt;nowledge &lt;u&gt;g&lt;/u&gt;uided fault intensity diagnosis framework (HKG) inspired by the tree of thought, which is amenable to any representation learning methods. The HKG uses graph convolutional networks to map the hierarchical topological graph of class representations into a set of interdependent global hierarchical classifiers, where each node is denoted by word embeddings of a class. These global hierarchical classifiers are applied to learned deep features extracted by representation learning, allowing the entire model to be end-to-end learnable. In addition, we develop a re-weighted hierarchical knowledge correlation matrix (Re-HKCM) scheme by embedding inter-class hierarchical knowledge into a data-driven statistical correlation matrix (SCM) which effectively guides the information sharing of nodes in graphical convolutional neural networks and avoids over-smoothing issues. The Re-HKCM is derived from the SCM through a series of mathematical transformations. Extensive experiments are performed on four real-world datasets from different industrial domains (three cavitation datasets from SAMSON AG and one existing publicly) for FID, all showing superior results and outperform recent state-of-the-art FID methods.

</details>

## 186. From Variability to Stability: Advancing RecSys Benchmarking Practices 🌟

<details>

<summary>Abstract</summary>

In the rapidly evolving domain of Recommender Systems (RecSys), new algorithms frequently claim state-of-the-art performance based on evaluations over a limited set of arbitrarily selected datasets. However, this approach may fail to holistically reflect their effectiveness due to the significant impact of dataset characteristics on algorithm performance. Addressing this deficiency, this paper introduces a novel benchmarking methodology to facilitate a fair and robust comparison of RecSys algorithms, thereby advancing evaluation practices. By utilizing a diverse set of 30 open datasets, including two introduced in this work, and evaluating 11 collaborative filtering algorithms across 9 metrics, we critically examine the influence of dataset characteristics on algorithm performance. We further investigate the feasibility of aggregating outcomes from multiple datasets into a unified ranking. Through rigorous experimental analysis, we validate the reliability of our methodology under the variability of datasets, offering a benchmarking strategy that balances quality and computational demands. This methodology enables a fair yet effective means of evaluating RecSys algorithms, providing valuable guidance for future research endeavors.

</details>

## 187. MGMatch: Fast Matchmaking with Nonlinear Objective and Constraints via Multimodal Deep Graph Learning

<details>

<summary>Abstract</summary>

As a core problem of online games, matchmaking is to assign players into multiple teams to maximize their gaming experience. With the rapid development of game industry, it is increasingly difficulty to explicitly model players' experiences as linear functions. Instead, it is often modeled in a data-driven way by training a neural network. Meanwhile, complex rules must be satisfied to ensure the robustness of matchmaking, which are often described using logical operators. Therefore, matchmaking in practical scenarios is a challenging combinatorial optimization problem with nonlinear objective, linear constraints and logical constraints, which receives much less attention in previous research. In this paper, we propose a novel deep learning method for high-quality matchmaking in real-time. We first cast the problem as standard mixed-integer programming (MIP) by linearizing ReLU networks and logical constraints. Then, based on supervised learning, we design and train a multi-modal graph learning architecture to predict optimal solutions end-to-end from instance data, and solve a surrogate problem to efficiently obtain feasible solutions. Evaluation results on real industry datasets show that our method can deliver near-optimal solutions within 100ms.

</details>

## 188. PEMBOT: Pareto-Ensembled Multi-task Boosted Trees

<details>

<summary>Abstract</summary>

Multi-task problems frequently arise in machine learning when there are multiple target variables, which share a common synergy while being sufficiently different that optimizing on any of the task does not necessarily imply an optimum for the others. In this work, we develop PEMBOT, a novel Pareto-based multi-task classification framework using a gradient boosted tree architecture. The proposed methodology involves a) generating multiple instances of Pareto optimal trees, b) diverse subset selection using a determinantal point process (DPP) model, and c) ensembling of diverse Pareto optimal trees to yield the final output. We tested our framework on a problem from an e-commerce domain wherein the task is to predict at order placement time the different adverse scenarios in the order shipment journey such as the package getting lost or damaged during shipment. This model enables us to take preemptive measures to prevent these scenarios from happening resulting in significant operational cost savings. Further, to show the generality of our approach, we demonstrate the performance of our algorithm on a publicly available wine quality prediction dataset and compare against state-of-the-art baselines.

</details>

## 189. Multi-objective Learning to Rank by Model Distillation

<details>

<summary>Abstract</summary>

In online marketplaces, search ranking's objective is not only to purchase or conversion (primary objective), but to also the purchase outcomes(secondary objectives), e.g. order cancellation(or return), review rating, customer service inquiries, platform long term growth. Multi-objective learning to rank has been widely studied to balance primary and secondary objectives. But traditional approaches in industry face some challenges including expensive parameter tuning leads to sub-optimal solution, suffering from imbalanced data sparsity issue, and being not compatible with ad-hoc objective. In this paper, we propose a distillation-based ranking solution for multi-objective ranking, which optimizes the end-to-end ranking system at Airbnb across multiple ranking models on different objectives along with various considerations to optimize training and serving efficiency to meet industry standards. We found it performs much better than traditional approaches, it doesn't only significantly increases primary objective by a large margin but also meet secondary objectives constraints and improve model stability. We also demonstrated the proposed system could be further simplified by model self-distillation. Besides this, we did additional simulations to show that this approach could also help us efficiently inject ad-hoc non-differentiable business objective into the ranking system while enabling us to balance our optimization objectives.

</details>

## 190. Choosing a Proxy Metric from Past Experiments

<details>

<summary>Abstract</summary>

In many randomized experiments, the treatment effect of the long-term metric (i.e. the primary outcome of interest) is often difficult or infeasible to measure. Such long-term metrics are often slow to react to changes and sufficiently noisy they are challenging to faithfully estimate in short-horizon experiments. A common alternative is to measure several short-term proxy metrics in the hope they closely track the long-term metric -- so they can be used to effectively guide decision-making in the near-term. We introduce a new statistical framework to both define and construct an optimal proxy metric for use in a homogeneous population of randomized experiments. Our procedure first reduces the construction of an optimal proxy metric in a given experiment to a portfolio optimization problem which depends on the true latent treatment effects and noise level of experiment under consideration. We then denoise the observed treatment effects of the long-term metric and a set of proxies in a historical corpus of randomized experiments to extract estimates of the latent treatment effects for use in the optimization problem. One key insight derived from our approach is that the optimal proxy metric for a given experiment is not apriori fixed; rather it should depend on the sample size (or effective noise level) of the randomized experiment for which it is deployed. To instantiate and evaluate our framework, we employ our methodology in a large corpus of randomized experiments from an industrial recommendation system and construct proxy metrics that perform favorably relative to several baselines.

</details>

## 191. ADSNet: Cross-Domain LTV Prediction with an Adaptive Siamese Network in Advertising

<details>

<summary>Abstract</summary>

Advertising platforms have evolved in estimating Lifetime Value (LTV) to better align with advertisers' true performance metric which considers cumulative sum of purchases a customer contributes over a period. Accurate LTV estimation is crucial for the precision of the advertising system and the effectiveness of advertisements. However, the sparsity of real-world LTV data presents a significant challenge to LTV predictive model(i.e., pLTV), severely limiting the their capabilities. Therefore, we propose to utilize external data, in addition to the internal data of advertising platform, to expand the size of purchase samples and enhance the LTV prediction model of the advertising platform. To tackle the issue of data distribution shift between internal and external platforms, we introduce an Adaptive Difference Siamese Network (ADSNet), which employs cross-domain transfer learning to prevent negative transfer. Specifically, ADSNet is designed to learn information that is beneficial to the target domain. We introduce a gain evaluation strategy to calculate information gain, aiding the model in learning helpful information for the target domain and providing the ability to reject noisy samples, thus avoiding negative transfer. Additionally, we also design a Domain Adaptation Module as a bridge to connect different domains, reduce the distribution distance between them, and enhance the consistency of representation space distribution. We conduct extensive offline experiments and online A/B tests on a real advertising platform. Our proposed ADSNet method outperforms other methods, improving GINI by 2\%. The ablation study highlights the importance of the gain evaluation strategy in negative gain sample rejection and improving model performance. Additionally, ADSNet significantly improves long-tail prediction. The online A/B tests confirm ADSNet's efficacy, increasing online LTV by 3.47\% and GMV by 3.89\%.

</details>

## 192. LiMAML: Personalization of Deep Recommender Models via Meta Learning

<details>

<summary>Abstract</summary>

In the realm of recommender systems, the ubiquitous adoption of deep neural networks has emerged as a dominant paradigm for modeling diverse business objectives. As user bases continue to expand, the necessity of personalization and frequent model updates have assumed paramount significance to ensure the delivery of relevant and refreshed experiences to a diverse array of members. In this work, we introduce an innovative meta-learning solution tailored to the personalization of models for individual members and other entities, coupled with the frequent updates based on the latest user interaction signals. Specifically, we leverage the Model-Agnostic Meta Learning (MAML) algorithm to adapt per-task sub-networks using recent user interaction data. Given the near infeasibility of productionizing original MAML-based models in online recommendation systems, we propose an efficient strategy to operationalize meta-learned sub-networks in production, which involves transforming them into fixed-sized vectors, termed meta embeddings, thereby enabling the seamless deployment of models with hundreds of billions of parameters for online serving. Through extensive experimentation on production data drawn from various applications at LinkedIn, we demonstrate that the proposed solution consistently outperforms the best performing baseline models of those applications, including strong baselines such as using wide-and-deep ID based personalization approach. Our approach has enabled the deployment of a range of highly personalized AI models across diverse LinkedIn applications, leading to substantial improvements in business metrics as well as refreshed experience for our members.

</details>

## 193. Enhancing Pre-Ranking Performance: Tackling Intermediary Challenges in Multi-Stage Cascading Recommendation Systems

<details>

<summary>Abstract</summary>

Large-scale search engines and recommendation systems utilize a three-stage cascading architecture-recall, pre-ranking, and ranking-to deliver relevant results within stringent latency limits. The pre-ranking stage is crucial for filtering a large number of recalled items into a manageable set for the ranking stage, greatly affecting the system's performance. Pre-ranking faces two intermediary challenges: Sample Selection Bias (SSB) arises when training is based on ranking stage feedback but the evaluation is on a broader recall dataset. Also, compared to the ranking stage, simpler pre-rank models may perform worse and less consistently. Traditional methods to tackle SSB issues include using all recall results and treating unexposed portions as negatives for training, which can be costly and noisy. To boost performance and consistency, some pre-ranking feature interaction enhancers don't fully fix consistency issues, while methods like knowledge distillation in ranking models ignore exposure bias. Our proposed framework targets these issues with three integral modules: Sample Selection, Domain Adaptation, and Unbiased Distillation. Sample Selection filters recall results to mitigate SSB and compute costs. Domain Adaptation enhances model robustness by assigning pseudo-labels to unexposed samples. Unbiased Distillation uses exposure-independent scores from Domain Adaptation to implement unbiased distillation for the pre-ranking model. The framework focuses on optimizing pre-ranking while maintaining training efficiency. We introduce new metrics for pre-ranking evaluation, while experiments confirm the effectiveness of our framework. Our framework is also deployed in real industrial systems.

</details>

## 194. On Finding Bi-objective Pareto-optimal Fraud Prevention Rule Sets for Fintech Applications

<details>

<summary>Abstract</summary>

Rules are widely used in Fintech institutions to make fraud prevention decisions, since rules are highly interpretable thanks to their intuitive if-then structure. In practice, a two-stage framework of fraud prevention decision rule set mining is usually employed in large Fintech institutions; Stage 1 generates a potentially large pool of rules and Stage 2 aims to produce a refined rule subset according to some criteria (typically based on precision and recall). This paper focuses on improving the flexibility and efficacy of this two-stage framework, and is concerned with finding high-quality rule subsets in a bi-objective space (such as precision and recall). To this end, we first introduce a novel algorithm called SpectralRules that directly generates a compact pool of rules in Stage 1 with high diversity. We empirically find such diversity improves the quality of the final rule subset. In addition, we introduce an intermediate stage between Stage 1 and 2 that adopts the concept of Pareto optimality and aims to find a set of non-dominated rule subsets, which constitutes a Pareto front. This intermediate stage greatly simplifies the selection criteria and increases the flexibility of Stage 2. For this intermediate stage, we propose a heuristic-based framework called PORS and we identify that the core of PORS is the problem of solution selection on the front (SSF). We provide a systematic categorization of the SSF problem and a thorough empirical evaluation of various SSF methods on both public and proprietary datasets. On two real application scenarios within Alipay, we demonstrate the advantages of our proposed methodology over existing work.

</details>

## 195. Nested Fusion: A Method for Learning High Resolution Latent Structure of Multi-Scale Measurement Data on Mars

<details>

<summary>Abstract</summary>

The Mars Perseverance Rover represents a generational change in the scale of measurements that can be taken on Mars, however this increased resolution introduces new challenges for techniques in exploratory data analysis. The multiple different instruments on the rover each measures specific properties of interest to scientists, so analyzing how underlying phenomena affect multiple different instruments together is important to understand the full picture. However each instrument has a unique resolution, making the mapping between overlapping layers of data non-trivial. In this work, we introduce Nested Fusion, a method to combine arbitrarily layered datasets of different resolutions and produce a latent distribution at the highest possible resolution, encoding complex interrelationships between different measurements and scales. Our method is efficient for large datasets, can perform inference even on unseen data, and outperforms existing methods of dimensionality reduction and latent analysis on real-world Mars rover data. We have deployed our method Nested Fusion within a Mars science team at NASA Jet Propulsion Laboratory (JPL) and through multiple rounds of participatory design enabled greatly enhanced exploratory analysis workflows for real scientists. To ensure the reproducibility of our work we have open sourced our code on GitHub at https://github.com/pixlise/NestedFusion.

</details>

## 196. Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks

<details>

<summary>Abstract</summary>

Representing information of multiple behaviors in the single graph collaborative filtering (CF) vector has been a long-standing challenge. This is because different behaviors naturally form separate behavior graphs and learn separate CF embeddings. Existing models merge the separate embeddings by appointing the CF embeddings for some behaviors as the primary embedding and utilizing other auxiliaries to enhance the primary embedding. However, this approach often results in the joint embedding performing well on the main tasks but poorly on the auxiliary ones. To address the problem arising from the separate behavior graphs, we propose the concept of &lt;u&gt;P&lt;/u&gt;artial &lt;u&gt;O&lt;/u&gt;rder Recommendation &lt;u&gt;G&lt;/u&gt;raphs (POG). POG defines the partial order relation of multiple behaviors and models behavior combinations as weighted edges to merge separate behavior graphs into a joint POG. Theoretical proof verifies that POG can be generalized to any given set of multiple behaviors. Based on POG, we propose the tailored &lt;u&gt;P&lt;/u&gt;artial &lt;u&gt;O&lt;/u&gt;rder &lt;u&gt;G&lt;/u&gt;raph &lt;u&gt;C&lt;/u&gt;onvolutional &lt;u&gt;N&lt;/u&gt;etworks (POGCN) that convolute neighbors' information while considering the behavior relations between users and items. POGCN also introduces a partial-order BPR sampling strategy for efficient and effective multiple-behavior CF training. POGCN has been successfully deployed on the homepage of Alibaba for two months, providing recommendation services for over one billion users. Extensive offline experiments conducted on three public benchmark datasets demonstrate that POGCN outperforms state-of-the-art multi-behavior baselines across all types of behaviors. Furthermore, online A/B tests confirm the superiority of POGCN in billion-scale recommender systems.

</details>

## 197. Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation

<details>

<summary>Abstract</summary>

Recommendation systems, which assist users in discovering their preferred items among numerous options, have served billions of users across various online platforms. Intuitively, users' interactions with items are highly driven by their unchanging inherent intents (e.g., always preferring high-quality items) and changing demand intents (e.g., wanting a T-shirt in summer but a down jacket in winter). However, both types of intents are implicitly expressed in recommendation scenario, posing challenges in leveraging them for accurate intent-aware recommendations. Fortunately, in search scenario, often found alongside recommendation on the same online platform, users express their demand intents explicitly through their query words. Intuitively, in both scenarios, a user shares the same inherent intent and his/her interactions may be influenced by the same demand intent. It is therefore feasible to utilize the interaction data from both scenarios to reinforce the dual intents for joint intent-aware modeling. But the joint modeling should deal with two problems: (1) accurately modeling users' implicit demand intents in recommendation; (2) modeling the relation between the dual intents and the interactive items. To address these problems, we propose a novel model named Unified Dual-Intents Translation for joint modeling of Search and Recommendation (UDITSR). To accurately simulate users' demand intents in recommendation, we utilize real queries from search data as supervision information to guide its generation. To explicitly model the relation among the triplet &lt;inherent intent, demand intent, interactive item&gt;, we propose a dual-intent translation propagation mechanism to learn the triplet in the same semantic space via embedding translations. Extensive experiments demonstrate that UDITSR outperforms SOTA baselines both in search and recommendation tasks. Moreover, our model has been deployed online on Meituan Waimai platform, leading to an average improvement in GMV (Gross Merchandise Value) of 1.46\% and CTR(Click-Through Rate) of 0.77\% over one month.

</details>

## 198. Inductive Modeling for Realtime Cold Start Recommendations 🌟

<details>

<summary>Abstract</summary>

In recommendation systems, the timely delivery of new content to their relevant audiences is critical for generating a growing and high quality collection of content for all users. The nature of this problem requires retrieval models to be able to make inferences in real time and with high relevance. There are two specific challenges for cold start contents. First, the information loss problem in a standard Two Tower model, due to the limited feature interactions between the user and item towers, is exacerbated for cold start items due to training data sparsity. Second, the huge volume of user-generated content in industry applications today poses a big bottleneck in the end-to-end latency of recommending new content. To overcome the two challenges, we propose a novel architecture, the Item History Model (IHM). IHM directly injects user-interaction information into the item tower to overcome information loss. In addition, IHM incorporates an inductive structure using attention-based pooling to eliminate the need for recurring training, a key bottleneck for the real-timeness. On both public and industry datasets, we demonstrate that IHM can not only outperform baselines in recommending cold start contents, but also achieves SoTA real-timeness in industry applications.

</details>

## 199. Recent and Upcoming Developments in Randomized Numerical Linear Algebra for Machine Learning

<details>

<summary>Abstract</summary>

Large matrices arise in many machine learning and data analysis applications, including as representations of datasets, graphs, model weights, and first and second-order derivatives. Randomized Numerical Linear Algebra (RandNLA) is an area which uses randomness to develop improved algorithms for ubiquitous matrix problems. The area has reached a certain level of maturity; but recent hardware trends, efforts to incorporate RandNLA algorithms into core numerical libraries, and advances in machine learning, statistics, and random matrix theory, have lead to new theoretical and practical challenges. This article provides a self-contained overview of RandNLA, in light of these developments.

</details>

## 200. Graph Machine Learning Meets Multi-Table Relational Data

<details>

<summary>Abstract</summary>

While graph machine learning, and notably graph neural networks (GNNs), have gained immense traction in recent years, application is predicated on access to a known input graph upon which predictive models can be trained. And indeed, within the most widely-studied public evaluation benchmarks such graphs are provided, with performance comparisons conditioned on curated data explicitly adhering to this graph. However, in real-world industrial applications, the situation is often quite different. Instead of a known graph, data are originally collected and stored across multiple tables in a repository, at times with ambiguous or incomplete relational structure. As such, to leverage the latest GNN architectures it is then up to a skilled data scientist to first manually construct a graph using intuition and domain knowledge, a laborious process that may discourage adoption in the first place. To narrow this gap and broaden the applicability of graph ML, we survey existing tools and strategies that can be combined to address the more fundamental problem of predictive tabular modeling over data native to multiple tables, with no explicit relational structure assumed a priori. This involves tracing a comprehensive path through related table join discovery and fuzzy table joining, column alignment, automated relational database (RDB) construction, extracting graphs from RDBs, graph sampling, and finally, graph-centric trainable predictive architectures. Although efforts to build deployable systems that integrate all of these components while minimizing manual effort remain in their infancy, this survey will nonetheless reduce barriers to entry and help steer the graph ML community towards promising research directions and wider real-world impact.

</details>

## 201. A Review of Graph Neural Networks in Epidemic Modeling

<details>

<summary>Abstract</summary>

Since the onset of the COVID-19 pandemic, there has been a growing interest in studying epidemiological models. Traditional mechanistic models mathematically describe the transmission mechanisms of infectious diseases. However, they often fall short when confronted with the growing challenges of today. Consequently, Graph Neural Networks (GNNs) have emerged as a progressively popular tool in epidemic research. In this paper, we endeavor to furnish a comprehensive review of GNNs in epidemic tasks and highlight potential future directions. To accomplish this objective, we introduce hierarchical taxonomies for both epidemic tasks and methodologies, offering a trajectory of development within this domain. For epidemic tasks, we establish a taxonomy akin to those typically employed within the epidemic domain. For methodology, we categorize existing work into Neural Models and Hybrid Models. Following this, we perform an exhaustive and systematic examination of the methodologies, encompassing both the tasks and their technical details. Furthermore, we discuss the limitations of existing methods from diverse perspectives and systematically propose future research directions. This survey aims to bridge literature gaps and promote the progression of this promising field. We hope that it will facilitate synergies between the communities of GNNs and epidemiology, and contribute to their collective progress.

</details>

## 202. Systems for Scalable Graph Analytics and Machine Learning: Trends and Methods

<details>

<summary>Abstract</summary>

Graph-theoretic algorithms and graph machine learning models are essential tools for addressing many real-life problems, such as social network analysis and bioinformatics. To support large-scale graph analytics, graph-parallel systems have been actively developed for over one decade, such as Google's Pregel and Spark's GraphX, which (i) promote a think-like-a-vertex computing model and target (ii) iterative algorithms and (iii) those problems that output a value for each vertex. However, this model is too restricted for supporting the rich set of heterogeneous operations for graph analytics and machine learning that many real applications demand. In recent years, two new trends emerge in graph-parallel systems research: (1) a novel think-like-a-task computing model that can efficiently support the various computationally expensive problems of subgraph search; and (2) scalable systems for learning graph neural networks. These systems effectively complement the diversity needs of graph-parallel tools that can flexibly work together in a comprehensive graph processing pipeline for real applications, with the capability of capturing structural features. This tutorial will provide an effective categorization of the recent systems in these two directions based on their computing models and adopted techniques, and will review the key design ideas of these systems.

</details>

## 203. Urban Foundation Models: A Survey

<details>

<summary>Abstract</summary>

Machine learning techniques are now integral to the advancement of intelligent urban services, playing a crucial role in elevating the efficiency, sustainability, and livability of urban environments. The recent emergence of foundation models such as ChatGPT marks a revolutionary shift in the fields of machine learning and artificial intelligence. Their unparalleled capabilities in contextual understanding, problem solving, and adaptability across a wide range of tasks suggest that integrating these models into urban domains could have a transformative impact on the development of smart cities. Despite growing interest in Urban Foundation Models (UFMs), this burgeoning field faces challenges such as a lack of clear definitions and systematic reviews. To this end, this paper first introduces the concept of UFMs and discusses the unique challenges involved in building them. We then propose a data-centric taxonomy that categorizes and clarifies current UFM-related works, based on urban data modalities and types. Furthermore, we explore the application landscape of UFMs, detailing their potential impact in various urban contexts. Relevant papers and open-source resources have been collated and are continuously updated at: https://github.com/usail-hkust/Awesome-Urban-Foundation-Models.

</details>

## 204. A Survey on Safe Multi-Modal Learning Systems

<details>

<summary>Abstract</summary>

In the rapidly evolving landscape of artificial intelligence, multimodal learning systems (MMLS) have gained traction for their ability to process and integrate information from diverse modality inputs. Their expanding use in vital sectors such as healthcare has made safety assurance a critical concern. However, the absence of systematic research into their safety is a significant barrier to progress in this field. To bridge the gap, we present the first taxonomy that systematically categorizes and assesses MMLS safety. This taxonomy is structured around four fundamental pillars that are critical to ensuring the safety of MMLS: robustness, alignment, monitoring, and controllability. Leveraging this taxonomy, we review existing methodologies, benchmarks, and the current state of research, while also pinpointing the principal limitations and gaps in knowledge. Finally, we discuss unique challenges in MMLS safety. In illuminating these challenges, we aim to pave the way for future research, proposing potential directions that could lead to significant advancements in the safety protocols of MMLS.

</details>

## 205. Heterogeneous Contrastive Learning for Foundation Models and Beyond 🌟

<details>

<summary>Abstract</summary>

In the era of big data and Artificial Intelligence, an emerging paradigm is to utilize contrastive self-supervised learning to model large-scale heterogeneous data. Many existing foundation models benefit from the generalization capability of contrastive self-supervised learning by learning compact and high-quality representations without relying on any label information. Amidst the explosive advancements in foundation models across multiple domains, including natural language processing and computer vision, a thorough survey on heterogeneous contrastive learning for the foundation model is urgently needed. In response, this survey critically evaluates the current landscape of heterogeneous contrastive learning for foundation models, highlighting the open challenges and future trends of contrastive learning. In particular, we first present how the recent advanced contrastive learning-based methods deal with view heterogeneity and how contrastive learning is applied to train and fine-tune the multi-view foundation models. Then, we move to contrastive learning methods for task heterogeneity, including pretraining tasks and downstream tasks, and show how different tasks are combined with contrastive learning loss for different purposes. Finally, we conclude this survey by discussing the open challenges and shedding light on the future directions of contrastive learning.

</details>
