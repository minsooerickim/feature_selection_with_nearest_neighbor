import math

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


def leave_one_out_cross_validation(data, current_set, feature_to_add):
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

def read_data(file):
    f = open(file, 'r')
    lines = f.readlines()
    num_instances = sum(1 for _ in lines)

    f.seek(0)
    data = [[] for _ in range(num_instances)]
    for i in range(num_instances):
        data[i] = [j for j in f.readline().split()]

    f.close()
    return data
    
def main():
    data = read_data('CS170_Small_Data__125.txt')
    # read_data('CS170_Large_Data__78.txt')

    current_set_of_features = []

    for i in range(len(data)):
        print(f'\nOn the {i+1} level of the search tree')
        
        feature_to_add_at_this_level = -1
        best_so_far_accuracy = 0

        for j in range(1, len(data)+1):
            if j not in current_set_of_features:
                print(f'-- Considering adding the {j} feature')

                #TODO: Stub function for NOW
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, j)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j
            current_set_of_features.append(feature_to_add_at_this_level)
            print(f'On level {i+1}, I added feature {feature_to_add_at_this_level} to current set')

    print(f'\nThe best features are {current_set_of_features}')

main()

