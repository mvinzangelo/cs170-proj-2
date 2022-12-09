import pandas as pd
import numpy as np
import copy
import time

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
                    print(f'---Using features {current_set_of_features} and adding {j} has an accuracy of {accuracy}')
                elif algorithm == '2':
                    print(f'---Removing features {current_set_of_features} and also {j} has an accuracy of {accuracy}')
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
            print(f'Feature set {current_set_of_features} was the best, accuracy is {best_accuracy_so_far0}')
        elif algorithm == '2':
            print(f'Removing feature set {current_set_of_features} was the best, accuracy is {best_accuracy_so_far}')
    # print best features and accuracy for the search
    if algorithm == '1':
        print(f'\nFinished search! The best feature subset is {best_set}, which has an accuracy of {best_accuracy}')
    elif algorithm == '2':
        print(f'\nFinished search! The best feature subset to remove is {best_set}, which has an accuracy of {best_accuracy}')
    

def leave_one_out_cross_validation(data, current_set, feature_to_add, algorithm):
    number_correctly_classified = 0
    # create list of all features included in this iteration
    tmp_set = copy.deepcopy(current_set)
    tmp_set.append(feature_to_add)
    columns_to_drop = None
    # based on algorithm, find the column that will be zeroed out
    if algorithm == '1':
        columns_to_drop = list(range(1, len(data[0])))
        # from https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
        columns_to_drop = [x for x in columns_to_drop if x not in tmp_set]
    elif algorithm == '2':
        columns_to_drop = tmp_set
    # loop through all rows
    for i, row_i in enumerate(data):
        label_object_to_classify = row_i[0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf') 
        nearest_neighbor_label = float('inf')
        # loop through all rows
        for j, row_j in enumerate(data):
            # make sure to remove itself from nearest neighbor classification
            if j == i:
                continue
            # compute euclidean distance
            distance = euclidean_distance(row_i, row_j, columns_to_drop)
            # find shortest distance and keep track of its index and label
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = j
                nearest_neighbor_label = row_j[0]
        # see if nearest neighbor classified the removed row correctly
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    # calculate accuracy
    accuracy = number_correctly_classified / (len(data))
    return round(accuracy, 3)

def euclidean_distance(row_a, row_b, zeroed_rows):
    distance = 0
    for i in range(1, len(row_a)):
        if i in zeroed_rows:
            continue
        # distance formula calculation
        distance += (row_a[i] - row_b[i]) ** 2
    return np.sqrt(distance)

def main():
    file_path_to_test = input("Type in name of file to test: ")
    type_of_algorithm = input("\nWhat type of algorithm do you want to run? (1: Forward Selection 2: Backward Elimination): ")
    df = pd.read_table(file_path_to_test, delim_whitespace=True, header=None)
    print(f'\n{file_path_to_test} has {df.shape[1] - 1} features (not including the class attribute), with {df.shape[0]} instances.')
    data_dict = df.to_dict('records')
    starting_accuracy = leave_one_out_cross_validation(data_dict, [], [], '1')
    print(f'\nRunning nearest neighbor with all {df.shape[1] - 1} features, using "leave-one-out" evaluation, I get an accuracy of {starting_accuracy }')
    print('\nBeginning search.')
    start_time = time.time()
    feature_search(data_dict, type_of_algorithm)
    print("Time: %s seconds" % round(time.time() - start_time, 2))

if __name__ == "__main__":
    main()