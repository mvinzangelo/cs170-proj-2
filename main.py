import pandas as pd
import numpy as np
import copy
import warnings

warnings.filterwarnings("ignore")

def feature_search(data):
    current_set_of_features = []
    accuracies = []
    for i in range(1, len(data.columns)):
        # print(f'I am on the {i}th level of the search tree')
        feature_to_add_this_level = 0
        best_accuracy_so_far = 0
        for j in range(1, len(data.columns)):
            if not (j in current_set_of_features):
                # print(f'--Considering adding the {j} feature')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)
                # print(accuracy)
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        current_set_of_features.append(feature_to_add_at_this_level)
        accuracies.append(best_accuracy_so_far)
        # print(f'On level {i}, I added feature {feature_to_add_at_this_level} to current set')
    print(current_set_of_features)
    print(accuracies)

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    tmp_set = copy.deepcopy(current_set)
    tmp_set.append(feature_to_add)
    columns_to_drop = list(range(1,data.shape[1]))
    # from https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
    columns_to_drop = [x for x in columns_to_drop if x not in tmp_set]
    for i in columns_to_drop:
        data = data.drop(i, axis = 1)
    for i, row_i in data.iterrows():
        object_to_classify = row_i[1:]
        label_object_to_classify = row_i[0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf') 
        nearest_neighbor_label = float('inf')
        for j, row_j in data.iterrows():
            if j == i:
                continue
            # print(f'Ask if {i} is nearest neighbor with {j}')
            # from https://www.geeksforgeeks.org/pandas-compute-the-euclidean-distance-between-two-series/
            distance = np.sqrt(np.sum([(a-b)*(a-b) for a, b in zip(object_to_classify, row_j[1:])]))
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = j
                nearest_neighbor_label = row_j[0]
        # print(f'--Object {i} has class of {label_object_to_classify}')
        # print(f'--Object {i} nearest neighbor is object {nearest_neighbor_location} which has a class of {nearest_neighbor_label}')
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / (data.shape[0])
    return accuracy

def main():
    df = pd.read_table("/home/vinz/repos/cs170/cs170-proj-2/CS170_Large_Data__21.txt", delim_whitespace=True, header=None)
    # print(leave_one_out_cross_validation(df, [1,2,6,4], 3))
    feature_search(df)

if __name__ == "__main__":
    main()