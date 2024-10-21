

def load_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f:
            keywords.append(line.strip())
    return keywords




if __name__ == "__main__":

    # load keywords
    pos_keywords_path = "data/pos_keywords.txt"
    neg_keywords_path = "data/neg_keywords.txt"
    aug_keywords_path = "data/aug_keywords.txt"
    pos_keywords = load_keywords(pos_keywords_path)
    neg_keywords = load_keywords(neg_keywords_path)
    aug_keywords = load_keywords(aug_keywords_path)

    # sort the keywords
    pos_keywords.sort()
    neg_keywords.sort()
    aug_keywords.sort()

    # save the sorted keywords
    with open(pos_keywords_path, "w") as f:
        for kw in pos_keywords:
            f.write(kw + "\n")
    
    with open(neg_keywords_path, "w") as f:
        for kw in neg_keywords:
            f.write(kw + "\n")

    with open(aug_keywords_path, "w") as f:
        for kw in aug_keywords:
            f.write(kw + "\n")

