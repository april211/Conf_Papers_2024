import pandas as pd


def load_data():
    data = pd.read_csv("data/ICML_2024_Events.csv", 
                                dtype={"name": "str", "abstract": "str"})
    return data


def load_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f:
            keywords.append(line.strip())
    return keywords




if __name__ == "__main__":

    data = load_data()

    print("Original data: ", "len: ", len(data))
    print(data.head(), end="\n\n")
    
    # add a new column based on the first column and the second column
    data["text"] = data["name"] + " " + data["abstract"]

    print("Data with new column: ")
    print(data.head(), end="\n\n")

    # detect nan values, show the rows & fill them with empty string
    print("Rows with nan values: ")
    print(data[data["text"].isna()], end="\n\n")
    data["text"] = data["text"].fillna("")

    # lower case
    data["text"] = data["text"].str.lower()

    # remove punctuation
    data["text"] = data["text"].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    # remove stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    data["text"] = data["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

    # show results
    print("Data with text column processed: ")
    print(data.head(), end="\n\n")

    # load keywords
    pos_keywords_path = "data/pos_keywords.txt"
    neg_keywords_path = "data/neg_keywords.txt"
    pos_keywords = load_keywords(pos_keywords_path)
    neg_keywords = load_keywords(neg_keywords_path)

    # if `text` contains any of the keywords, then save the title
    cnt = 0
    with open("res/raw/icml_results.md", "w") as f:
        for row in data.itertuples():
            if any(p_kw in row.text for p_kw in pos_keywords) \
                    and not any(n_kw in row.text for n_kw in neg_keywords):
                cnt += 1
                f.write(data.loc[row.Index]['name'] + "\n")
                f.write(data.loc[row.Index]['abstract'] + "\n\n")

    # print current length
    print("Current length: ", cnt, ", reduced by: ", len(data) - cnt, " rows.")

