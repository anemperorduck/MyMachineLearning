"""
| Cap Color | Stalk Shape | Solitary | Edible |
|:---------:|:-----------:|:--------:|:------:|
|   Brown   |   Tapering  |    Yes   |    1   |
|   Brown   |  Enlarging  |    Yes   |    1   |
|   Brown   |  Enlarging  |    No    |    0   |
|   Brown   |  Enlarging  |    No    |    0   |
|   Brown   |   Tapering  |    Yes   |    1   |
|    Red    |   Tapering  |    Yes   |    0   |
|    Red    |  Enlarging  |    No    |    0   |
|   Brown   |  Enlarging  |    Yes   |    1   |
|    Red    |   Tapering  |    No    |    1   |
|   Brown   |  Enlarging  |    No    |    0   |

one-hot encoding
belowing

| Brown Cap | Tapering Stalk Shape | Solitary | Edible |
|:---------:|:--------------------:|:--------:|:------:|
|     1     |           1          |     1    |    1   |
|     1     |           0          |     1    |    1   |
|     1     |           0          |     0    |    0   |
|     1     |           0          |     0    |    0   |
|     1     |           1          |     1    |    1   |
|     0     |           1          |     1    |    0   |
|     0     |           0          |     0    |    0   |
|     1     |           0          |     1    |    1   |
|     0     |           1          |     0    |    1   |
|     1     |           0          |     0    |    0   |
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    if len(y) == 0:
        return 0.

    entropy = 0.
    p1 = len(y[y==1])/len(y)
    
    if p1 != 0 and p1 != 1:
        entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)

    return entropy


def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list): Indices with feature value == 1
        right_indices (list): Indices with feature value == 0
    """
    left_indices = []
    right_indices =[]
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    
    return left_indices, right_indices


def compute_information_gain(X, Y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        information_gain (float):        Cost computed
    
    """    
    # split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, Y_node = X[node_indices], Y[node_indices]
    X_left, Y_left = X[left_indices], Y[left_indices]
    X_right, Y_right = X[right_indices], Y[right_indices]

    # weight
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right)/len(X_node)

    # compute
    node_entropy = compute_entropy(Y_node)
    left_entropy = compute_entropy(Y_left)
    right_entropy = compute_entropy(Y_right) 

    weight_entropy = w_left*left_entropy + w_right*right_entropy
    information_gain = node_entropy - weight_entropy

    return information_gain


def get_best_split(X, Y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    num_features = X.shape[1]
    best_feature = -1

    max_info_gain = -1
    for i in range(num_features):
        info_gain = compute_information_gain(X, Y, node_indices, i)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = i
    
    return best_feature

def build_tree(X, Y, node_indices, branch_name, max_depth, current_depth):
    tree = []
    def build_tree_recursive(X, Y, node_indices, branch_name, max_depth, current_depth):
        """
        Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
        This function just prints the tree.
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
            max_depth (int):        Max depth of the resulting tree. 
            current_depth (int):    Current depth. Parameter used during recursive call.
    
        """
        if current_depth == max_depth:
            formatting = " "*current_depth + "-"*current_depth
            print(formatting, "%s leaf node with indices" % branch_name, node_indices)
            tree.append((current_depth, branch_name, -1, node_indices))
            return

        best_feature = get_best_split(X, Y, node_indices)
        tree.append((current_depth, branch_name, best_feature, node_indices))

        formatting = "-"*current_depth
        print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
        
        left_indices, right_indices = split_dataset(X, node_indices, best_feature)
        build_tree_recursive(X, Y, left_indices, "Left", max_depth, current_depth+1)
        build_tree_recursive(X, Y, right_indices, "Right", max_depth, current_depth+1)
    
    build_tree_recursive(X, Y, node_indices, branch_name, max_depth, current_depth)
    return tree


if __name__ == '__main__':
    X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
    Y_train = np.array([1,1,0,0,1,0,0,1,1,0])

    # print("First few elements of X_train:\n", X_train[:5])
    # print("Type of X_train:",type(X_train))
    # print ('The shape of X_train is:', X_train.shape)       # (10, 3)
    # print ('The shape of y_train is: ', Y_train.shape)      # (10, )
    # print ('Number of training examples (m):', len(X_train))

    # 1. compute entropy
    # print("Entropy at root node: ", compute_entropy(Y_train)) 

    # 2. Split dataset
    root_indices = np.arange(len(X_train)).tolist()
    feature = 0
    left_indices, right_indices = split_dataset(X_train, root_indices, feature)
    # print("Left indices: ", left_indices)
    # print("Right indices: ", right_indices)


    # 3. information gain
    # info_gain0 = compute_information_gain(X_train, Y_train, root_indices, feature=0)
    # print("Information Gain from splitting the root on brown cap: ", info_gain0)
        
    # info_gain1 = compute_information_gain(X_train, Y_train, root_indices, feature=1)
    # print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

    # info_gain2 = compute_information_gain(X_train, Y_train, root_indices, feature=2)
    # print("Information Gain from splitting the root on solitary: ", info_gain2)


    # 4. best split
    best_feature = get_best_split(X_train, Y_train, root_indices)
    # print("Best feature to split on: %d" % best_feature)


    # 5. build the tree
    tree = []
    tree = build_tree(X_train, Y_train, root_indices, "Root", max_depth=2, current_depth=0)
    print(tree)