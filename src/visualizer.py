import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

def plot_feature_importance(model, feature_names):
    """
    Generates a horizontal bar chart of feature importances.
    """
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig = go.Figure(go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        marker_color='#636EFA'
    ))
    
    fig.update_layout(
        title="What drove the model's decisions?",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=0, r=0, t=40, b=0),
        height=350
    )
    return fig


def plot_decision_boundary(X, y, feature_x, feature_y, target_mapping=None):
    """
    Trains a lightweight 2D tree on the fly to visualize classification boundaries 
    for any two features selected by the user.
    """
    # Isolate the two features
    X_slice = X[[feature_x, feature_y]]
    
    # Train a 2D model specifically for the background contour
    clf2d = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf2d.fit(X_slice, y)
    
    # Create a mesh grid
    x_min, x_max = X_slice[feature_x].min() - 1, X_slice[feature_x].max() + 1
    y_min, y_max = X_slice[feature_y].min() - 1, X_slice[feature_y].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict the background grid
    Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()

    # Add the decision boundary contour
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.1),
        y=np.arange(y_min, y_max, 0.1),
        z=Z,
        colorscale='RdBu',
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    ))

    # Map colors for the actual data points
    colors = y.astype(str)
    if target_mapping:
        colors = y.map(target_mapping)
        
    # Add the actual data points
    fig.add_trace(go.Scatter(
        x=X_slice[feature_x],
        y=X_slice[feature_y],
        mode='markers',
        marker=dict(
            color=y,
            colorscale='RdBu',
            line=dict(color='black', width=1),
            size=8
        ),
        text=colors,
        hovertemplate=f"{feature_x}: %{{x}}<br>{feature_y}: %{{y}}<br>Class: %{{text}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Decision Boundary: {feature_x} vs {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def plot_prediction_gauge(confidence, prediction_label):
    """
    Creates a real-time speedometer gauge for the What-If simulator.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Predicted Class:<br><b>{prediction_label}</b>"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00CC96"},
            'steps': [
                {'range': [0, 50], 'color': "#FF9999"},
                {'range': [50, 80], 'color': "#FFCC99"},
                {'range': [80, 100], 'color': "#E6F2E6"}
            ],
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_tree_structure(model, feature_names, target_mapping=None):
    """
    Generates a Graphviz object representing the decision tree topology.
    """
    # If we have a target mapping, extract the class names in the correct order
    class_names = None
    if target_mapping:
        # Sort by key to ensure the class names match the model's internal indexing
        class_names = [target_mapping[k] for k in sorted(target_mapping.keys())]
        # Convert all class names to strings to prevent Graphviz errors
        class_names = [str(name) for name in class_names]

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,         # Colors the nodes based on the dominant class
        rounded=True,        # Visually softer, more modern nodes
        special_characters=True,
        proportion=True,     # Shows percentages instead of raw sample counts for a cleaner UI
        impurity=False       # Hides Gini/Entropy scores to keep it readable for non-technical users
    )
    
    # Create and return the graph object
    graph = graphviz.Source(dot_data)
    
    return graph