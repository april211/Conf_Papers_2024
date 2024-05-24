import pandas as pd


def load_data():
    data = "The incredible success of transformers in sequence modeling tasks can be largely attributed to the self-attention mechanism, which allows information to be transferred between different parts of a sequence. Self-attention allows transformers to encode causal structure which makes them particularly suitable for sequence modeling. However, the process by which transformers learn such causal structure via gradient-based training algorithms remains poorly understood. To better understand this process, we introduce an in-context learning task that requires learning latent causal structure. We prove that gradient descent on a simplified two-layer transformer learns to solve this task by encoding the latent causal graph in the first attention layer. The key insight of our proof is that the gradient of the attention matrix encodes the mutual information between tokens. As a consequence of the data processing inequality, the largest entries of this gradient correspond to edges in the latent causal graph. As a special case, when the sequences are generated from in-context Markov chains, we prove that transformers learn an induction head (Olsson et al., 2022). We confirm our theoretical findings by showing that transformers trained on our in-context learning task are able to recover a wide variety of causal structures."
    return data


def load_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f:
            keywords.append(line.strip())
    return keywords




if __name__ == "__main__":

    import re

    data = load_data()

    # lower case
    data = data.lower()

    # remove punctuation
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)

    # remove stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    data = " ".join([word for word in data.split() if word not in stop_words])

    # load keywords
    pos_keywords_path = "data/pos_keywords.txt"
    neg_keywords_path = "data/neg_keywords.txt"
    aug_keywords_path = "data/aug_keywords.txt"
    pos_keywords = load_keywords(pos_keywords_path)
    neg_keywords = load_keywords(neg_keywords_path)
    aug_keywords = load_keywords(aug_keywords_path)
    print(neg_keywords)

    occ_neg_keywords = [kw for kw in neg_keywords if kw in data]
    print("occ_neg_keywords: ", occ_neg_keywords)

    occ_pos_keywords = [kw for kw in pos_keywords if kw in data]
    print("occ_pos_keywords: ", occ_pos_keywords)

    occ_aug_keywords = [kw for kw in aug_keywords if kw in data]
    print("occ_aug_keywords: ", occ_aug_keywords)

