import pandas as pd
import numpy as np
import copy

def feature_search(data, algorithm):
    current_set_of_features = []
    accuracies = []
    best_accuracy = 0
    best_set = None
    found_max = False
    # loop through outer level
    for i in range(1, len(data[0])):
        print()
        feature_to_add_this_level = 0
        best_accuracy_so_far = 0
        # loop through all features
        for j in range(1, len(data[0])):
            # check that feature isn't already being accounted for
            if not (j in current_set_of_features):
                # do cross validation and get accuracy with additional feature added
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j, algorithm)
                if algorithm == '1':
                    print(f'---Using features {current_set_of_features} and adding {j} has an accuracy of {accuracy * 100}%')
                elif algorithm == '2':
                    print(f'---Removing features {current_set_of_features} and also {j} has an accuracy of {accuracy * 100}%')
                # compare the best accuracy with current accuracy to get max for level
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        # add accuracy to list to beck iterated on
        current_set_of_features.append(feature_to_add_at_this_level)
        accuracies.append(best_accuracy_so_far)
        # compare to see the best accuracy for the overall search
        if best_accuracy_so_far < best_accuracy and not found_max:
            found_max = True
            print('(Warning, accuracy has decreased! Continuing search in case of local maxima)')
        elif best_accuracy_so_far > best_accuracy:
            best_accuracy = best_accuracy_so_far
            best_set = copy.deepcopy(current_set_of_features)
            found_max = False
        # print best accuracy at this level
        if algorithm == '1':
            print(f'Feature set {current_set_of_features} was the best, accuracy is {best_accuracy_so_far * 100}%')
        elif algorithm == '2':
            print(f'Removing feature set {current_set_of_features} was the best, accuracy is {best_accuracy_so_far * 100}%')
    # print best features and accuracy for the search
    if algorithm == '1':
        print(f'\nFinished search! The best feature subset is {best_set}, which has an accuracy of {best_accuracy * 100}%')
    elif algorithm == '2':
        print(f'\nFinished search! The best feature subset to remove is {best_set}, which has an accuracy of {best_accuracy * 100}%')
    

def leave_one_out_cross_validation(data, current_set, feature_to_add, algorithm):
    number_correctly_classified = 0
    tmp_set = copy.deepcopy(current_set)
    tmp_set.append(feature_to_add)
    columns_to_drop = None
    if algorithm == '1':
        columns_to_drop = list(range(1, len(data[0])))
        # from https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
        columns_to_drop = [x for x in columns_to_drop if x not in tmp_set]
    elif algorithm == '2':
        columns_to_drop = tmp_set
    for i, row_i in enumerate(data):
        label_object_to_classify = row_i[0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf') 
        nearest_neighbor_label = float('inf')
        for j, row_j in enumerate(data):
            if j == i:
                continue
            distance = euclidean_distance(row_i, row_j, columns_to_drop)
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = j
                nearest_neighbor_label = row_j[0]
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
    type_of_algorithm = input("\nWhat type of algorithm do you want to run? (1: Forward Selection 2: Backward Elimination): ")

    df = pd.read_table(file_path_to_test, delim_whitespace=True, header=None)
    print(f'\n{file_path_to_test} has {df.shape[1] - 1} features (not including the class attribute), with {df.shape[0]} instances.')
    data_dict = df.to_dict('records')
    starting_accuracy = leave_one_out_cross_validation(data_dict, [x for x in range(1, len(data_dict[0]) - 1)], len(data_dict[0]), '1')
    print(f'\nRunning nearest neighbor with all {df.shape[1] - 1} features, using "leave-one-out" evaluation, I get an accuracy of {starting_accuracy * 100}%')
    print('\nBeginning search.')
    feature_search(data_dict, type_of_algorithm)

if __name__ == "__main__":
    main()