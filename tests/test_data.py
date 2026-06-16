import pytest
import pandas as pd
from src.data_loader import load_builtin_iris, load_builtin_titanic, process_custom_data

def test_load_iris():
    X, y, target_mapping = load_builtin_iris()
    
    # Assertions on structure
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(target_mapping, dict)
    
    # Assertions on data integrity
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert len(target_mapping) == 3
    assert "Species" in y.name
    
    
def test_load_titanic():
    X, y, target_mapping = load_builtin_titanic()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Ensure dropped columns are gone
    dropped_cols = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
    for col in dropped_cols:
        assert col not in X.columns
        
    # Ensure there are no missing values left after preprocessing
    assert X.isna().sum().sum() == 0
    
    # Ensure binary classification targets match mapping
    assert len(target_mapping) == 2
    assert set(y.unique()).issubset({0, 1})
    
    
def test_process_custom_data():
    # Create a dummy dataframe with missing values and a categorical column
    mock_df = pd.DataFrame({
        "feature_numeric": [1.0, 2.0, None, 4.0],
        "feature_cat": ["A", "B", "A", None],
        "target": ["Yes", "No", "Yes", "No"]
    })
    
    X, y, target_mapping = process_custom_data(mock_df, target_column="target")
    
    # Verify NaNs were handled
    assert X.isna().sum().sum() == 0
    
    # Verify one-hot encoding happened for category (drop_first=True leaves 1 column for 2 cats)
    assert "feature_cat_B" in X.columns
    
    # Verify target string mapping to numeric indices
    assert y.dtype in ['int64', 'int32']
    assert len(target_mapping) == 2