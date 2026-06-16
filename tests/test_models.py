import pytest
from sklearn.tree import DecisionTreeClassifier
from src.data_loader import load_builtin_iris
from src.model_engine import train_decision_tree, extract_rules, predict_single_instance


@pytest.fixture
def sample_iris_data():
    """Provides data to tests without reloading it multiple times."""
    return load_builtin_iris()


def test_train_decision_tree(sample_iris_data):
    X, y, _ = sample_iris_data
    model = train_decision_tree(X, y, max_depth=3)
    
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 3
    # Check that model is actually fitted
    assert hasattr(model, "classes_")
    
    
def test_extract_rules(sample_iris_data):
    X, y, mapping = sample_iris_data
    model = train_decision_tree(X, y, max_depth=2)
    
    rules = extract_rules(model, feature_names=X.columns, target_mapping=mapping)
    
    assert isinstance(rules, list)
    assert len(rules) > 0
    
    # Check structure of an individual rule dictionary
    first_rule = rules[0]
    required_keys = {"rule", "prediction", "confidence", "samples"}
    assert required_keys.issubset(first_rule.keys())
    
    # Confidence must be a valid probability boundary
    assert 0.0 <= first_rule["confidence"] <= 1.0
    
    # Verify target mapping applied correctly (should be 'setosa', 'versicolor', or 'virginica')
    assert first_rule["prediction"] in mapping.values()
    
    
def test_predict_single_instance(sample_iris_data):
    X, y, _ = sample_iris_data
    model = train_decision_tree(X, y, max_depth=2)
    
    # Take a single row as a new dataframe instance
    single_instance = X.iloc[[0]] 
    
    pred_idx, probabilities = predict_single_instance(model, single_instance)
    
    assert pred_idx in model.classes_
    assert len(probabilities) == len(model.classes_)
    assert abs(sum(probabilities) - 1.0) < 1e-6 # Must sum up to 1 roughly