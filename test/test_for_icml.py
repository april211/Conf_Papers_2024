



if __name__ == "__main__":

    # read the test data
    test_lst = []
    with open("test/test_data_for_icml.txt", "r") as f:
        for line in f:
            test_lst.append(line.strip())

    # read the predicted data
    pred = ""
    with open("res/raw/icml_results.md", "r") as f:
        pred = f.read()
    
    # check if the test data is included in the predicted data
    for test_data in test_lst:
        if test_data not in pred:
            print(f"Test data: `{test_data}` not in the predicted data.")

