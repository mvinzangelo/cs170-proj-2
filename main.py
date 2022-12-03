import pandas as pd
import random

def feature_search(data):
    print(data)
    current_set_of_features = []
    for i in range(1, len(data.columns)):
        print(f'I am on the {i}th level of the search tree')
        feature_to_add_this_level = 0
        best_accuracy_so_far = 0
        for j in range(1, len(data.columns)):
            if j not in current_set_of_features:
                print(f'--Considering adding the {j} feature')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f'One level {i}, I added feature {feature_to_add_at_this_level} to current set')

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.random()

def main():
    df = pd.read_table("./test_data.txt", delim_whitespace=True, header=None)
    feature_search(df)

if __name__ == "__main__":
    main()