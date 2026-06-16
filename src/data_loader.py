import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_openml, load_wine

def load_builtin_iris():
    """
    Loads the Iris dataset, perfectly clean and ready for multi-class tree modeling.
    Returns: X (DataFrame), y (Series), target_mapping (dict)
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name = "Species")
    
    # Map target integers to actual names so our extracted rules make sense later
    target_mapping = {i: name for i, name in enumerate(iris.target_names)}
    
    return X, y, target_mapping


def load_builtin_wine():
    """
    Loads the Wine dataset, ready for multi-class modeling.
    Returns: X (DataFrame), y (Series), target_mapping (dict)
    """
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="Class")

    # Map target integers to their meaningful names for rule extraction
    target_mapping = {i: name for i, name in enumerate(wine.target_names)}

    return X, y, target_mapping



def load_builtin_titanic():
    """
    Fetches the Titanic dataset and handles mixed data types and NaNs.
    Returns: X (DataFrame), y (Series), target_mapping (dict)
    """
    
    #Fetch from OpenML
    titanic = fetch_openml("titanic", version=1, as_frame=True, parser='auto')
    df = titanic.frame
    
    #Drop columns that are too noisy or irrelevant for simple rule extraction
    cols_to_drop = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
    df = df.drop(columns=cols_to_drop)
    
    #Handle missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Binary encoding for categorical variables to keep tree splits clean
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    
    # One-hot encode multi-class categoricals (like Port of Embarkation)
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=['survived'])
    y = df['survived'].astype(int)
    
    target_mapping = {0: "Perished", 1: "Survived"}
    
    return X, y, target_mapping

def process_custom_data(df, target_column):
    """
    A robust generic loader for user-uploaded CSVs.
    Returns: X (DataFrame), y (Series), target_mapping (dict)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
    # Drop rows where the target itself is missing
    df = df.dropna(subset=[target_column])
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Quick, brute-force preprocessing to prevent scikit-learn from crashing
    # 1. Fill numeric NaNs with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # 2. Fill categorical NaNs with mode, then one-hot encode
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # If target is categorical, map it
    if y.dtype == 'object' or y.dtype.name == 'category':
        unique_targets = y.unique()
        target_mapping = {i: val for i, val in enumerate(unique_targets)}
        y = y.map({val: i for i, val in target_mapping.items()})
    else:
        target_mapping = None # Continuous target (Regression) or already numeric
        
    return X, y, target_mapping
