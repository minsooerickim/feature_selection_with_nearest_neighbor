# used to calculate euclidean distance
import math
# used to record the elapsed time of the algorithms
from time import time
# helper graph plotting function
from utility import graph

# lists to keep track of accuracies and feature sets for matlab plotting
matlab_feature_set, matlab_accuracy = [], []

#TODO: Add more comments and typings
def nearest_neighbor_classifier(data, object_to_classify, current_set):
    """
    Use Euclidean distance as our distance function in our nearest neighbor classifier.
    """
    # initialize the distance and location to infinity so they can be replaced with our actual calculations
    nearest_neighbor_distance, nearest_neighbor_location = math.inf, math.inf

    # loop through our entire dataset and ONLY consider the current features in our feature subset
    for i in range(len(data)):
        # skip the object we are trying to classify since that will obvious be the closest distance and mess up our accuracy
        if i == object_to_classify:
            continue

        distance = 0
        # use euclidean distance for our distance function
        for j in range(len(current_set)):
            distance += pow((data[i][current_set[j]] - data[object_to_classify][current_set[j]]), 2)
        distance = math.sqrt(distance)

        # update the nearest neighbor properties accordingly when a closer neighbor is found
        if distance < nearest_neighbor_distance:
            nearest_neighbor_distance = distance
            nearest_neighbor_location = i
            nearest_neighbor_label = data[nearest_neighbor_location][0]

    # return the label of the nereast neighbor
    return nearest_neighbor_label


def leave_one_out_cross_validation(data, current_set):
    """
    Cross Validation. Leave an object out of the rest of the set. Use the rest of the set
    to build a model. Then, use the "one out" with the model built with the other data and determine its accuracy.

    Accuracy = # of correct classifications / # of instances in our database.
    """
    # count to keep track of number of correctly classified objects
    number_correctly_classified = 0
    for i in range(len(data)):
        label_object_to_classify = data[i][0]
        nearest_neighbor_label = nearest_neighbor_classifier(data, i, current_set)
        
        # increment counter if the object was correctly classified
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

    # initial accuracy and feature set for matlab plotting
    matlab_accuracy.append(leave_one_out_cross_validation(data, []))
    matlab_feature_set.append([])

    print(f'\naccuracy with 0 features, {leave_one_out_cross_validation(data, [])}')
    best_so_far_accuracy = 0
    for i in range(num_features):
        print(f'\nOn the {i+1} level of the search tree')

        # flag to tell a better accuracy was found at each level
        better_accuracy_found = False
        best_lvl_accuracy = 0

        for j in range(1, num_features+1):
            if j not in current_set_of_features:
                print(f'-- Considering adding the {j} feature')
                
                # use leave one out cross validation to get the accuracy of the current feature considering the j'th feature
                accuracy = leave_one_out_cross_validation(data, current_set_of_features+[j])
                print(f'---- accuracy with {current_set_of_features+[j]} : {accuracy}')

                # update the accuracy, feature if a best so far accuracy is found
                # if best so far accuracy is not found, we still want to add the best accuracy from that level in order to continue searching
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    best_feature_to_add_at_this_level = j
                    # by setting a flag, we can tell whether best_so_far_accuracy was found at this level or not
                    better_accuracy_found = True
                elif accuracy > best_lvl_accuracy:
                    best_lvl_accuracy = accuracy
                    lvl_feature = j
        
        # using the better_accuracy_found flag set in line 94, we can determine if a best_so_far_accuracy was found at the i+1'th level of the search tree
        # we update our current_set_of_features accordingly
        if better_accuracy_found: 
            current_set_of_features.append(best_feature_to_add_at_this_level)
            best_features = [feature for feature in current_set_of_features]
            print(f'\nfeature subset {best_features} had the highest accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I added feature {best_feature_to_add_at_this_level} to current set')
            
            # store accuracy for plotting bar chart with matlab
            matlab_accuracy.append(best_so_far_accuracy)
            matlab_feature_set.append(current_set_of_features.copy())
        else:
            # add feature to current_set_of_features regardless of better_accuracy_found for potential higher accuracy late on
            current_set_of_features.append(lvl_feature)
            print(f'\nThe best accuracy at this level, {current_set_of_features} : {best_lvl_accuracy} was less than the best so far accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I added feature {lvl_feature} to current set')
            
            # store accuracy for plotting bar chart with matlab
            matlab_accuracy.append(best_lvl_accuracy)
            matlab_feature_set.append(current_set_of_features.copy())

        print('-----------------------------------------------')

    print(f'\nThe best features are {best_features} with accuracy {best_so_far_accuracy}')

def backward_elimination(data):
    """
    Starts with all the features then eliminates them until the best set of features is found.
    """
    num_features = len(data[0]) - 1
    current_set_of_features, best_features = [i+1 for i in range(num_features)], [i+1 for i in range(num_features)]

    # initial accuracy and feature set for matlab plotting
    matlab_accuracy.append(leave_one_out_cross_validation(data, current_set_of_features))
    matlab_feature_set.append(current_set_of_features.copy())
    
    best_so_far_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    for i in range(num_features):
        print(f'\nOn the {i+1} level of the search tree')

        # flag to tell a better accuracy was found at each level
        better_accuracy_found = False
        best_lvl_accuracy = 0

        for j in range(1, num_features+1):
            if j in current_set_of_features:
                print(f'-- Considering removing the {j} feature')
            
                # create a set excluding the j'th feature so we can calculate the accuracy of it
                copy_set = [feature for feature in current_set_of_features if feature != j]
                # use leave one out cross validation to get the accuracy of the current feature considering removing the j'th feature
                accuracy = leave_one_out_cross_validation(data, copy_set)
                print(f'---- accuracy with {copy_set} : {accuracy}')

                # update the accuracy, feature if a best so far accuracy is found
                # if best so far accuracy is not found, we still want to add the best accuracy from that level in order to continue searching
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    best_feature_to_remove_at_this_level = j
                    # by setting a flag, we can tell whether best_so_far_accuracy was found at this level or not
                    better_accuracy_found = True
                elif accuracy >= best_lvl_accuracy:
                    best_lvl_accuracy = accuracy
                    lvl_feature_to_remove = j
        print(matlab_feature_set)

        # using the better_accuracy_found flag set in line 94, we can determine if a best_so_far_accuracy was found at the i+1'th level of the search tree
        # we update our current_set_of_features accordingly
        if better_accuracy_found: 
            current_set_of_features.remove(best_feature_to_remove_at_this_level)
            best_features = [feature for feature in current_set_of_features]
            print(f'\nfeature subset {best_features} had the highest accuracy of {best_so_far_accuracy}')
            print(f'On level {i+1}, I removed feature {best_feature_to_remove_at_this_level} from the current set')

            # store accuracy for plotting bar chart with matlab
            matlab_accuracy.append(best_so_far_accuracy)
            matlab_feature_set.append(current_set_of_features.copy())
        else:
            # add feature to current_set_of_features regardless of better_accuracy_found for potential higher accuracy late on
            current_set_of_features.remove(lvl_feature_to_remove)
            print('\nAccuracy Decreased! :(')
            print(f'\nThe best accuracy at this level, {best_lvl_accuracy} was less than the best so far accuracy of {best_so_far_accuracy}')
            print(f'\nThe current set is now {current_set_of_features}')
            print(f'On level {i+1}, I removed feature {lvl_feature_to_remove} from the current set')

            # store accuracy for plotting bar chart with matlab
            matlab_accuracy.append(best_lvl_accuracy)
            matlab_feature_set.append(current_set_of_features.copy())

        print('-----------------------------------------------')

    print(f'\nThe best features are {best_features} with accuracy {best_so_far_accuracy}')

def read_data(file):
    """
    Read the data set and store the data in a list of lists.
    """
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

    # figure out number of features and instances using the data array we created with the read_data fn.
    num_features, num_instances = len(data[0]) - 1, len(data)
    all_feature_set = [i+1 for i in range(num_features)]
    print(all_feature_set)
    all_accuracy = leave_one_out_cross_validation(data, all_feature_set)
    print(f'\nThis data set has {num_features} features with {num_instances} instances.')
    print(f'\nRunning nearest neighbor with all {num_features}, using "leaving-one-out" evaluation, I get an accuracy of {all_accuracy}')
    # data = read_data('CS170_Large_Data__78.txt')
    
    option = str(input('\n1. forward selection\n2. backward_elimination\n'))

    # time to measure the elapsed time of the algorithm to be run
    start_time = time()

    # call the according function based on user input
    if option == '1': forward_selection(data)
    elif option == '2': backward_elimination(data)
    else: 
        print('please enter a valid option')
        return

    end_time = time()
    # take the difference between start_time and end_time to find the total elapsed time
    print(f"\nElapsed time: {(end_time - start_time):.3f} seconds\n")

    # call the matlab graph utility function I created
    graph(matlab_accuracy, matlab_feature_set, file_name)
main()

