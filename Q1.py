import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

def load_data(file_path, percentage):
    try:
        data = pd.read_csv(file_path)
        sample_size = int(len(data) * (percentage / 100))
        return data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

def split_data(data, train_percentage):
    train_size = int(len(data) * (train_percentage / 100))
    train_data, test_data = np.split(data.sample(frac=1, random_state=42), [train_size])
    return train_data, test_data

def encode_categorical_columns(data):
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    return data

# Naive Bayes Classifier function
def naive_bayes_classifier(train_data, test_data, epsilon=1e-9):
    # Extract features and target variable from training data
    features = train_data.columns[:-1]
    target = train_data.columns[-1]
    
    # Calculate prior probabilities for each class
    priors = train_data[target].value_counts(normalize=True)
    
    # Calculate means and standard deviations for each feature and class
    means = train_data.groupby(target)[features].mean(numeric_only=True)
    stds = train_data.groupby(target)[features].std(numeric_only=True)

    # Initialize list to store predictions for test data
    predictions = []
    
    # Iterate over each row in the test data
    for i, row in test_data.iterrows():
        # Dictionary to store log probabilities for each class
        class_probabilities = {}
        
        # Iterate over each class
        for cls in priors.index:
            # Initialize log probability for the current class
            class_probabilities[cls] = np.log(priors[cls])
            
            # Iterate over each feature
            for feature in features:
                # Check if feature value is not missing
                if pd.notna(row[feature]):
                    x = row[feature]
                    mean = means.at[cls, feature]
                    std = stds.at[cls, feature]
                    
                    # Calculate Gaussian probability function with Laplace smoothing
                    if std > 0:
                        p = (1 / (np.sqrt(2 * np.pi) * std)) * \
                            np.exp(-((x - mean) ** 2 / (2 * std ** 2) + epsilon))
                        class_probabilities[cls] += np.log(p)
        
        # Append the class with the highest log probability as the prediction
        predictions.append(max(class_probabilities, key=class_probabilities.get))
    
    # Compute accuracy by comparing predicted classes with actual classes
    accuracy = (test_data[target] == predictions).mean()
    
    # Return predictions and accuracy
    return predictions, accuracy



def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, label=None, is_categorical=False):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.is_categorical = is_categorical

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria for tree growth
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(label=leaf_value)

        # Randomly select a subset of features without replacement
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        # Greedily select the best split based on information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        # Split the data based on the best split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        # Recursively grow the left and right children trees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        

        return Node(best_feat, best_thresh, left, right)


    def _best_criteria(self, X, y, feat_idxs):
        # Initialize the best gain 
        best_gain = -1
        # Initialize variables to store the index and threshold of the best split
        split_idx, split_thresh = None, None
        
        # Loop over each feature index provided
        for feat_idx in feat_idxs:
            # Extract the column corresponding to the current feature index
            X_column = X[:, feat_idx]
            # Find unique values in the current feature column
            thresholds = np.unique(X_column)
            
            # Loop over each unique value as a potential threshold
            for threshold in thresholds:
                # Calculate the information gain using the current feature and threshold
                gain = self._information_gain(y, X_column, threshold)

                # If the calculated gain is better than the current best gain, update the best gain,
                # best split index, and best split threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        # Return the index and threshold of the best split found
        return split_idx, split_thresh


    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        # If the current node is a leaf node, return its label
        if node.label is not None:
            return node.label

        # If the current node is categorical
        if node.is_categorical:
            if x[node.feature_index] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        # If the current node is not categorical
        else:
            if x[node.feature_index] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

def encode_categorical_columns(data):
    categorical_cols = data.select_dtypes(
        include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    return data    

def process_data():
    file_path = file_path_var.get()
    percentage = float(percentage_var.get()) / 100  # Convert to fraction
    train_percentage = float(train_percentage_var.get())
    data = load_data(file_path, percentage)
    if data is not None:
        # Encode all categorical columns
        data = encode_categorical_columns(data)
        train_data, test_data = split_data(data, train_percentage)

        nb_predictions, nb_accuracy = naive_bayes_classifier(train_data, test_data)
        
        # Initialize and fit decision tree classifier
        dt_classifier = DecisionTree()
        dt_classifier.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
        dt_predictions = dt_classifier.predict(test_data.iloc[:, :-1].values)
        dt_accuracy = accuracy_score(test_data.iloc[:, -1], dt_predictions)

        # Calculate additional metrics
        nb_precision = precision_score(test_data.iloc[:, -1], nb_predictions)
        nb_recall = recall_score(test_data.iloc[:, -1], nb_predictions)
        nb_f1 = f1_score(test_data.iloc[:, -1], nb_predictions)

        dt_precision = precision_score(test_data.iloc[:, -1], dt_predictions)
        dt_recall = recall_score(test_data.iloc[:, -1], dt_predictions)
        dt_f1 = f1_score(test_data.iloc[:, -1], dt_predictions)

        # Format accuracies as percentages
        nb_accuracy_percent = nb_accuracy * 100
        dt_accuracy_percent = dt_accuracy * 100

        # Display comparison results
        display_text = "Comparison of the two classifiers:\n"
        display_text += f"Naive Bayes Accuracy: {nb_accuracy_percent:.2f}%\n"
        display_text += f"Decision Tree Accuracy: {dt_accuracy_percent:.2f}%\n\n"

        display_text += "Test Data Predictions:\n"
        row_num = 0
        for _, row in test_data.iterrows():
            actual_label = int(row.iloc[-1])  # Convert to integer
            nb_predicted_label = nb_predictions[row_num]
            dt_predicted_label = dt_predictions[row_num]
            display_text += f"Row {row_num}: Actual Label: {actual_label}, Naive Bayes Predicted Label: {nb_predicted_label}, Decision Tree Predicted Label: {dt_predicted_label}\n"
            row_num += 1

        display_text += "\n\n"

        display_text += f"Naive Bayes Precision: {nb_precision:.2f}\n"
        display_text += f"Naive Bayes Recall: {nb_recall:.2f}\n"
        display_text += f"Naive Bayes F1-score: {nb_f1:.2f}\n\n"

        display_text += f"Decision Tree Precision: {dt_precision:.2f}\n"
        display_text += f"Decision Tree Recall: {dt_recall:.2f}\n"
        display_text += f"Decision Tree F1-score: {dt_f1:.2f}\n"

        result_text_widget.delete(1.0, tk.END)
        result_text_widget.insert(tk.END, display_text)

# Setup the main window
root = tk.Tk()
root.title("Data Classifier")

# Variables
file_path_var = tk.StringVar()
percentage_var = tk.StringVar()
train_percentage_var = tk.StringVar()

# Layout
tk.Label(root, text="File Path:").pack()
entry_file_path = tk.Entry(root, textvariable=file_path_var, width=100)
entry_file_path.pack()
tk.Button(root, text="Browse", command=lambda: file_path_var.set(filedialog.askopenfilename())).pack()
tk.Label(root, text="Percentage of Data:").pack()
entry_percentage = tk.Entry(root, textvariable=percentage_var, width=50)
entry_percentage.pack()
tk.Label(root, text="Training Set Percentage:").pack()
entry_train_percentage = tk.Entry(root, textvariable=train_percentage_var, width=50)
entry_train_percentage.pack()
tk.Button(root, text="Run Classifiers", command=process_data).pack()

output_frame = tk.Frame(root)
output_frame.pack(expand=True, fill='both')
result_text_widget = tk.Text(output_frame, height=20, width=120)
result_text_widget.pack(side=tk.LEFT, expand=True, fill='both')
scrollbar = tk.Scrollbar(output_frame, command=result_text_widget.yview)
scrollbar.pack(side=tk.RIGHT, fill='y')
result_text_widget['yscrollcommand'] = scrollbar.set

root.mainloop()
