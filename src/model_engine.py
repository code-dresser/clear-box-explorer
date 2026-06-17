from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np


def train_decision_tree(X, y, max_depth=None, min_samples_split = 2, random_state = 42):
    """
    Trains a single Decision Tree Classifier.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split= min_samples_split, random_state=random_state)
    model.fit(X,y)
    return model


def extract_rules(tree_model, feature_names, target_mapping=None):
    """
    Traverses the Scikit-learn tree structure to extract human-readable rules.
    Returns a list of dictionaries containing the rule string, prediction, and confidence.
    """
    # Access the underlying Cython tree structure
    tree_ = tree_model.tree_
    
    # Map node feature indices to actual feature names
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    extracted_rules = []

    def recurse(node, current_path):
        # If the current node is not a leaf
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Traverse Left Child (Condition <= Threshold)
            left_path = current_path + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_path)
            
            # Traverse Right Child (Condition > Threshold)
            right_path = current_path + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_path)
        else:
            # We reached a leaf node
            samples = tree_.n_node_samples[node]
            values = tree_.value[node][0] # Array of sample counts per class
            
            predicted_class_idx = np.argmax(values)
            confidence = values[predicted_class_idx] / np.sum(values)
            
            # Map the integer index back to the human-readable class name
            if target_mapping:
                predicted_class = target_mapping.get(predicted_class_idx, predicted_class_idx)
            else:
                predicted_class = predicted_class_idx
                
            # Combine the path array into a single readable string
            rule_string = " AND ".join(current_path)
            
            extracted_rules.append({
                "rule": rule_string,
                "prediction": predicted_class,
                "confidence": confidence,
                "samples": samples
            })

    # Start the recursion from the root node (0) with an empty path
    recurse(0, [])
    
    # Sort the rules: Highest confidence and largest sample size first
    extracted_rules = sorted(
        extracted_rules, 
        key=lambda x: (x['confidence'], x['samples']), 
        reverse=True
    )
    
    return extracted_rules


def predict_single_instance(model, instance_df):
    """
    Takes a trained model and a 1-row DataFrame (from the UI "What-If" simulator)
    and returns the prediction and class probabilities.
    """
    prediction_idx = model.predict(instance_df)[0]
    probabilities = model.predict_proba(instance_df)[0]
    
    return prediction_idx, probabilities