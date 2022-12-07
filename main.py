import pandas as pd
import numpy as np
import copy

def feature_search(data):
    data_dict = data.to_dict('records')
    current_set_of_features = []
    accuracies = []
    for i in range(1, len(data.columns)):
        print(f'I am on the {i}th level of the search tree')
        feature_to_add_this_level = 0
        best_accuracy_so_far = 0
        for j in range(1, len(data.columns)):
            if not (j in current_set_of_features):
                print(f'-Considering adding the {j} feature')
                accuracy = leave_one_out_cross_validation(data_dict, current_set_of_features, j)
                print(f'---Accuracy of {accuracy}')
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        current_set_of_features.append(feature_to_add_at_this_level)
        accuracies.append(best_accuracy_so_far)
        print(f'*On level {i}, I added feature {feature_to_add_at_this_level} to current set')
    print(current_set_of_features)
    print(accuracies)

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    tmp_set = copy.deepcopy(current_set)
    tmp_set.append(feature_to_add)
    columns_to_drop = list(range(1,len(data[0])))
    # from https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
    columns_to_drop = [x for x in columns_to_drop if x not in tmp_set]
    for i, row_i in enumerate(data):
        label_object_to_classify = row_i[0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf') 
        nearest_neighbor_label = float('inf')
        for j, row_j in enumerate(data):
            if j == i:
                continue
            # print(f'Ask if {i} is nearest neighbor with {j}')
            distance = euclidean_distance(row_i, row_j, columns_to_drop)
            # print(f'----Distance: {distance}')
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = j
                nearest_neighbor_label = row_j[0]
        # print(f'--Object {i} has class of {label_object_to_classify}')
        # print(f'--Object {i} nearest neighbor is object {nearest_neighbor_location} which has a class of {nearest_neighbor_label}')
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / (len(data))
    return accuracy

def euclidean_distance(row_a, row_b, zeroed_rows):
    distance = 0
    for i in range(1, len(row_a)):
        if i in zeroed_rows:
            continue
        distance += (row_a[i] - row_b[i]) ** 2
    return np.sqrt(distance)

def main():
    file_path_to_test = input("Type in name of file to test: ")
    type_of_algorithm = input("What type of algorithm do you want to run? (1: Forward Selection 2: Backward Elimination): ")
    df = pd.read_table(file_path_to_test, delim_whitespace=True, header=None)
    print(f'{file_path_to_test} has {df.shape[1] - 1} features (not including the class attribute), with {df.shape[0]} instances.')
    feature_search(df)

if __name__ == "__main__":
    main()