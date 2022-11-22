import pandas as pd
import random

def feature_search(data_path):
    df = pd.read_table(data_path, delim_whitespace=True, header=None)
    for i in range(1, len(df.columns)):
        print(f'I am on the {i}th level of the search tree')
        for j in range(1, len(df.columns)):
            print(f'--Considering adding the {j} feature')

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.random()

def main():
    feature_search("./test_data.txt")

if __name__ == "__main__":
    main()