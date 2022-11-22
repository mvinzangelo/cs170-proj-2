import pandas as pd
import random

data_path = "./test_data.txt"
data = pd.read_table(data_path, header=None)

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.random()

def main():
    print(data)

if __name__ == "__main__":
    main()