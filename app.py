import streamlit as st
import pandas as pd
import numpy as np

# Import our custom modules
from src.data_loader import load_builtin_iris, load_builtin_titanic, process_custom_data
from src.model_engine import train_decision_tree, extract_rules, predict_single_instance
from src.visualizer import plot_feature_importance, plot_decision_boundary, plot_prediction_gauge, plot_tree_structure


# ==========================================
# 1. Page Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="The Clear Box | XAI Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧊 The Clear Box: Explainable AI Explorer")
st.markdown("""
Welcome to **The Clear Box**. Most machine learning models act as "black boxes"—taking data and spitting out predictions with no explanation. 
This tool trains a model in real-time, opens the box, and translates its mathematical splits into human-readable logic.
""")

# ==========================================
# 2. Sidebar: Controls & Data Loading
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Room")
    
    st.subheader("1. Choose Dataset")
    dataset_name = st.selectbox(
        "Select a dataset to explore:",
        ["Iris (Multi-class)", "Titanic (Binary)", "Upload Custom CSV"]
    )
    
    if dataset_name == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload a clean CSV file", type=["csv"])
        if uploaded_file:
            custom_target = st.text_input("Enter the exact name of the Target Column:")
    
    st.markdown("---")
    
    st.subheader("2. Model Architecture")
    max_depth = st.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3, 
                          help="Deeper trees are more accurate but harder to explain (overfitting).")
    min_samples = st.slider("Min Samples to Split", min_value=2, max_value=20, value=2)

# Load the data based on selection
X, y, target_mapping = None, None, None

try:
    if dataset_name == "Iris (Multi-class)":
        X, y, target_mapping = load_builtin_iris()
    elif dataset_name == "Titanic (Binary)":
        X, y, target_mapping = load_builtin_titanic()
    elif dataset_name == "Upload Custom CSV" and uploaded_file and custom_target:
        df = pd.read_csv(uploaded_file)
        X, y, target_mapping = process_custom_data(df, custom_target)
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    
    
# ==========================================
# 3. Core Application Logic
# ==========================================
if X is not None and y is not None:
    
    # Optional Data Preview Expander
    with st.expander("🔍 Preview Raw Dataset"):
        st.dataframe(X.head(10), use_container_width=True)
        
    # Train the Model
    model = train_decision_tree(X, y, max_depth=max_depth, min_samples_split=min_samples)
    
    # Extract the Rules
    rules = extract_rules(model, feature_names=X.columns, target_mapping=target_mapping)
    
    # Layout using Tabs for a clean UI
    tab1, tab2, tab3, tab4 = st.tabs([
        "Human-Readable Rules", 
        "Tree Topology", 
        "Visual Analytics", 
        "What-If Simulator"
    ])
    
    # --- TAB 1: EXTRACTED RULES ---
    with tab1:
        st.subheader("Extracted Decision Rules")
        st.markdown("The algorithm's internal logic, translated into plain English. Sorted by confidence.")
        
        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Rules Extracted", len(rules))
        col2.metric("Max Model Depth", max_depth)
        
        st.markdown("---")
        for i, rule in enumerate(rules[:10]): # Show top 10 to avoid clutter
            confidence_pct = rule['confidence'] * 100
            st.info(
                f"**Rule #{i+1}:** IF `{rule['rule']}` ➔ **Predict {rule['prediction']}** "
                f"*(Confidence: {confidence_pct:.1f}% based on {rule['samples']} samples)*"
            )

    # --- TAB 2: TREE TOPOLOGY ---
    with tab2:
        st.subheader("Algorithmic Blueprint")
        st.markdown("The mathematical structure of the trained decision tree.")
        
        graph = plot_tree_structure(model, X.columns, target_mapping)
        st.graphviz_chart(graph.source)

    # --- TAB 3: VISUAL ANALYTICS ---
    with tab3:
        colA, colB = st.columns(2)
        
        with colA:
            st.plotly_chart(plot_feature_importance(model, X.columns), use_container_width=True)
            
        with colB:
            st.markdown("### 🗺️ Decision Boundary Explorer")
            st.markdown("Select two features to see where the algorithm draws its lines.")
            # Let user pick axes
            fx = st.selectbox("X-Axis Feature", X.columns, index=0)
            fy = st.selectbox("Y-Axis Feature", X.columns, index=1 if len(X.columns)>1 else 0)
            
            if fx != fy:
                st.plotly_chart(plot_decision_boundary(X, y, fx, fy, target_mapping), use_container_width=True)
            else:
                st.warning("Please select two different features to visualize the boundary.")

    # --- TAB 4: WHAT-IF SIMULATOR ---
    with tab4:
        st.subheader("Real-Time Prediction Simulator")
        st.markdown("Tweak the parameters below to see how the model reacts to new data instantly.")
        
        # Dynamically create input widgets based on the dataset features
        sim_col1, sim_col2 = st.columns([1, 1])
        user_inputs = {}
        
        with sim_col1:
            for col in X.columns:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                mean_val = float(X[col].mean())
                
                # Create a slider for every feature
                user_inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
        
        with sim_col2:
            # Convert user inputs to a 1-row DataFrame
            instance_df = pd.DataFrame([user_inputs])
            
            # Run prediction
            pred_idx, probabilities = predict_single_instance(model, instance_df)
            
            # Map the prediction to human-readable string
            if target_mapping:
                predicted_label = target_mapping.get(pred_idx, pred_idx)
            else:
                predicted_label = str(pred_idx)
                
            # Grab the confidence score of the winning class
            winning_confidence = probabilities[np.argmax(probabilities)]
            
            st.markdown("### Live Prediction")
            st.plotly_chart(plot_prediction_gauge(winning_confidence, predicted_label), use_container_width=True)
            
            # Show the raw probabilities for all classes
            with st.expander("View Raw Class Probabilities"):
                prob_dict = {
                    target_mapping[i] if target_mapping else f"Class {i}": round(prob * 100, 2) 
                    for i, prob in enumerate(probabilities)
                }
                st.json(prob_dict)

else:
    # State when no data is loaded yet (e.g., waiting for custom CSV)
    st.info("👈 Please select a dataset or upload a CSV in the sidebar to begin.")