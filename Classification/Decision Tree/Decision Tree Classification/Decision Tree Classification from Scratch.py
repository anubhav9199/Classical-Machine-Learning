# -----------------------------------------------------------------------------
# Decision Tree Classification from Scratch over Breast Cancer Data(Bi-Class Classification)
"""
Decision Tree algorithms are used for both predictions as well as classification 
in machine learning. Using the decision tree with a given set of inputs, 
one can map the various outcomes that are a result of the consequences or 
decisions.
Decision Tree Analysis is a general, predictive modelling tool that has 
applications spanning a number of different areas. In general, decision trees 
are constructed via an algorithmic approach that identifies ways to split a 
data set based on different conditions. It is one of the most widely used and 
practical methods for supervised learning.
Decision Trees are a non-parametric supervised learning method used for both 
classification and regression tasks. The goal is to create a model that 
predicts the value of a target variable by learning simple decision rules 
inferred from the data features.
The decision rules are generally in form of if-then-else statements. The 
deeper the tree, the more complex the rules and fitter the model.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
## Node Class
"""
Node Class is the class for designing the tree. Tree is a recursive data 
structure. hence, we can create it by calling again and again the same functon 
again and again till the function gets a false value for any condition.
Node Class Contain its constructor for carring all the basic data-objects. 
this structure is fully coded with the approach of learning of a person. 
It also contains a function for visualising of the tree.
"""
# for Similar code : https://stackoverflow.com/a/54074933/1143396
class Node:
    """A decision tree node."""
    
    # -------------------------------------------------------------------------
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def debug(self, feature_names, class_names, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def _debug_aux(self, feature_names, class_names, show_details, root=False):
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.predicted_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "gini = {:.2f}".format(self.gini),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1 :]
            return lines, width, height, middle
    
        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
            top_lines
            + [middle_line, bottom_line]
            + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1 :]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------    
# Classifier Class (Decision Tree)
"""
Decision trees are constructed via an algorithmic approach that identifies 
ways to split a data set based on different conditions. It is one of the most 
widely used and practical methods for supervised learning. Decision Trees are 
a non-parametric supervised learning method used for both classification and 
regression tasks.
Classifier Class work to fit anf learn the training data by creating split 
diversion in the data also it divides the data accordingly."""
# -----------------------------------------------------------------------------
# Gini Impurity
"""
Decision trees use the concept of Gini impurity to describe how homogeneous or 
“pure” a node is. A node is pure (G = 0) if all its samples belong to the same 
class, while a node with many samples from many different classes will have a 
Gini closer to 1.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Formula : Refer to Python Notebook
"""
but we use another formula because for big dataset thi hypothatical formula 
doesn't work well. The resulting Gini is a simple weighted average.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main Formula (Use here) : Refer to Python Notebook
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Grow Tree
"""
This function is use to create a decision tree. The function use Node class to 
create the node.
"""
class DecisionTreeClassifier:
    
    # -------------------------------------------------------------------------
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.
        Gini impurity is defined as Σ p(1-p) over all classes, with p the 
        frequency of a class within the node. Since Σ p = 1, this is equivalent 
        to 1 - Σ p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted 
        by their population, is the smallest possible. Additionally it must be 
        less than the impurity of the current node.
        To find the best split, we loop through all the features, and consider 
        all the midpoints between adjacent training samples as possible 
        thresholds. We compute the Gini impurity of the split generated by 
        that particular feature/threshold pair, and return the pair with 
        smallest impurity.
        
        Returns:
        best_idx: Index of the feature for best split, or None if no split is found.
        best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
        return best_idx, best_thr
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        """Build decision tree classifier."""
        print("\nTraining Start.....\n")
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        print("\nTraining Finish...!!!\nEnjoy....!!!")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        print("Training......")
        clear_output(wait=True)
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Prediction Function
"""
Prediction function is use to get the accuracy of the fitted model over 
training and testing data so that we can get to know how much accurate over 
model is trained over training data to predict right output.
"""
def accuracy_test(model,testing_data,testing_label):
    predict = model.predict(testing_data)
    accuracy = (np.count_nonzero(np.equal(testing_label,predict))/testing_data.shape[0])*100
    return accuracy
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main Function
def main():
    
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on Kaggle
    (www.kaggle.com).
    """
    data = pd.read_csv('./Dataset/Breast Cancer/Breast_Cancer_Data.csv')
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    """
    Removing unnecessary columns from the dataset so that you won't face any 
    trouble regarding the dataset. I use Breast Cancer Dataset to train model 
    and predict whether the person is having cancer or not.
    """
    data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
    train_len = int(0.7*data.shape[0])
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Separating Label Column
    data.replace(['B','M'],[0,1],inplace=True)
    labels = data['diagnosis']
    training_label = labels.iloc[0:train_len]
    testing_label = labels.iloc[train_len:]
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Splitting Training and Testing Data
    new_data = data.iloc[:,1:3]
    training_data = new_data.iloc[0:train_len,:].values
    testing_data = new_data.iloc[train_len:,:].values
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Fittiong Model
    regressor = DecisionTreeClassifier()
    regressor.fit(training_data, training_label)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Predicting result on both the datasets
    # Training Accuracy
    train_accuracy = accuracy_test(regressor,training_data,training_label)
    print("Accuracy over Training Data : {}".format(train_accuracy))
    
    # Testing Accuracy
    test_accuracy = accuracy_test(regressor,testing_data,testing_label)
    print("Accuracy over Testing Data : {}".format(test_accuracy))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Visualising Tree
    # Training Data Tree
    regressor.debug(
        feature_names=["radius_mean {}".format(i) for i in training_data]
        ,class_names=["class {}".format(i) for i in training_label]
        ,show_details=False
    )
    
    # Testing Data Tree
    regressor.debug(
        feature_names=["radius_mean {}".format(i) for i in testing_data]
        ,class_names=["class {}".format(i) for i in testing_label]
        ,show_details=False
    )
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Calling Main Function
if __name__ == "__main__" :
    
    # Calling Main Function
    main()
# -----------------------------------------------------------------------------