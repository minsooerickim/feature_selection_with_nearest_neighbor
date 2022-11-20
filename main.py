import random

def leave_one_out_cross_validation():
    return random.random()

def read_data(file):
    f = open(file, 'r')
    lines = f.readlines()
    columns = {}
    for i in range(len(lines)):
        row = lines[i].split()
        for j in range(1, len(row)):
            if j not in columns:
                columns[j] = []
            columns[j].append(row[j])
    f.close()
    return columns
    
def main():
    columns = read_data('CS170_Small_Data__125.txt')


    for i in range(len(columns)):
        print(f'\nOn the {i+1} level of the search tree')
        
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for j in range(len(columns)):
            print(f'-- Considering adding the {j+1} feature')

            #TODO: Stub function for NOW
            accuracy = leave_one_out_cross_validation(columns, j+1)

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = j
    # read_data('CS170_Large_Data__78.txt')

main()

