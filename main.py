# used to calculate euclidean distance
import math
# used to record the elapsed time of the algorithms
from time import time

def nearest_neighbor_classifier(data, object_to_classify, current_set):
    nearest_neighbor_distance, nearest_neighbor_location = math.inf, math.inf
    for i in range(len(data)):
        if i == object_to_classify:
            continue

        distance = 0
        for j in range(len(current_set)):
            distance += pow((data[i][current_set[j]] - data[object_to_classify][current_set[j]]), 2)
        distance = math.sqrt(distance)

        if distance < nearest_neighbor_distance:
            nearest_neighbor_distance = distance
            nearest_neighbor_location = i
            nearest_neighbor_label = data[nearest_neighbor_location][0]

    return nearest_neighbor_label


def leave_one_out_cross_validation(data, current_set):
    """
    Cross Validation. Leave an object out of the rest of the set. Use the rest of the set
    to build a model. Then, use the "one out" with the model built with the other data and determine its accuracy.

    Accuracy = # of correct classifications / # of instances in our database.
    """
    number_correctly_classified = 0
    for i in range(len(data)):
        label_object_to_classify = data[i][0]
        nearest_neighbor_label = nearest_neighbor_classifier(data, i, current_set)
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / len(data)
    return accuracy

def forward_selection(data):
    """
    Starts with no features then adds them one by one and returns the best set of features.
    """
    num_features = len(data[0]) - 1
    current_set_of_features, best_features = [], []

    best_so_far_accuracy = 0
    for i in range(num_features):
        print(f'\nOn the {i+1} level of the search tree')

        # flag to tell a better accuracy was found at each level
        better_accuracy_found = False
        best_lvl_accuracy = 0

        for j in range(1, num_features+1):
            if j not in current_set_of_features:
                print(f'-- Considering adding the {j} feature')
                
                accuracy = leave_one_out_cross_validation(data, current_set_of_features+[j])
                print(f'---- accuracy with {current_set_of_features+[j]} : {accuracy}')

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    best_feature_to_add_at_this_level = j
                    better_accuracy_found = True
                elif accuracy > best_lvl_accuracy:
                    best_lvl_accuracy = accuracy
                    lvl_feature = j

        if better_accuracy_found: 
            best_features.append(best_feature_to_add_at_this_level)
            current_set_of_features.append(best_feature_to_add_at_this_level)
            print(f'\nfeature subset {best_features} had the highest accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I added feature {best_feature_to_add_at_this_level} to current set')
        else:
            # add feature to current_set_of_features regardless of better_accuracy_found for potential higher accuracy late on
            current_set_of_features.append(lvl_feature)
            print(f'\nThe best accuracy at this level, {best_lvl_accuracy} was less than the best so far accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I added feature {lvl_feature} to current set')

        print('-----------------------------------------------')

    print(f'\nThe best features are {best_features} with accuracy {best_so_far_accuracy}')

def backward_elimination(data):
    """
    Starts with all the features then eliminates them until the best set of features is found.
    """
    num_features = len(data[0]) - 1
    current_set_of_features, best_features = [i+1 for i in range(num_features)], [i+1 for i in range(num_features)]

    best_so_far_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    for i in range(num_features):
        print(f'\nOn the {i+1} level of the search tree')

        # flag to tell a better accuracy was found at each level
        better_accuracy_found = False
        best_lvl_accuracy = 0

        for j in range(1, num_features+1):
            if j in current_set_of_features:
                print(f'-- Considering removing the {j} feature')
            
                copy_set = [feature for feature in current_set_of_features if feature != j]
                accuracy = leave_one_out_cross_validation(data, copy_set)
                print(f'---- accuracy with {copy_set} : {accuracy}')

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    best_feature_to_remove_at_this_level = j
                    better_accuracy_found = True
                elif accuracy > best_lvl_accuracy:
                    best_lvl_accuracy = accuracy
                    lvl_feature_to_remove = j

        if better_accuracy_found: 
            current_set_of_features.remove(best_feature_to_remove_at_this_level)
            best_features = [feature for feature in current_set_of_features]
            print(f'\nfeature subset {best_features} had the highest accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I removed feature {best_feature_to_remove_at_this_level} from the current set')
        else:
            # add feature to current_set_of_features regardless of better_accuracy_found for potential higher accuracy late on
            current_set_of_features.remove(lvl_feature_to_remove)
            print('\nAccuracy Decreased! :(')
            print(f'\nThe best accuracy at this level, {best_lvl_accuracy} was less than the best so far accuracy of {best_so_far_accuracy}')
            print(f'\nThe current set is now {current_set_of_features}')
            print(f'On level {i+1}, I removed feature {lvl_feature_to_remove} from the current set')

        print('-----------------------------------------------')

    print(f'\nThe best features are {best_features} with accuracy {best_so_far_accuracy}')

def read_data(file):
    f = open(file, 'r')
    lines = f.readlines()
    num_instances = sum(1 for _ in lines)

    f.seek(0)
    data = [[] for _ in range(num_instances)]
    for i in range(num_instances):
        data[i] = [float(j) for j in f.readline().split()]

    f.close()
    return data
    
def main():
    file_name = input('Enter the input file name: ')
    data = read_data(file_name)
    # data = read_data('CS170_Large_Data__78.txt')
    
    option = str(input('\n1. forward selection\n2. backward_elimination\n'))

    # time to measure the elapsed time of the algorithm to be run
    start_time = time()

    if option == '1': forward_selection(data)
    elif option == '2': backward_elimination(data)
    else: 
        print('please enter a valid option')
        return

    end_time = time()
    # take the difference between start_time and end_time to find the total elapsed time
    print(f"\nElapsed time: {end_time - start_time} seconds\n")
main()

