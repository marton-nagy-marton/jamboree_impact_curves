import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pygam import LinearGAM, s, ExpectileGAM

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Impact Curves Demo")

st.title("Fast and Accurate Feature Impact Curves in Dynamic Dashboards")
st.markdown("""
This demo simulates the challenge of visualizing **feature impact curves** in a dynamic dashboard.
We use a simple **random forest** model to predict **house prices** on the California Housing Dataset.
            
You can interactively filter the dataset using the sidebar controls. Press the "Apply Filters & Update Curves" button to see how different methods adapt to the selected subgroup.
""")

# --- Data & Model Loading (Cached) ---
@st.cache_data
def load_data_and_model():
    # Load Data
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    X = X[
        (X['AveRooms'] <= 15) &
        (X['AveBedrms'] <= 10) &
        (X['Population'] <= 15000) &
        (X['AveOccup'] <= 15)
    ]
    y = y[X.index]

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)


    x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    # Train a simple RF (Simulating a complex production model)
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(x_train, y_train)
    
    # Pre-calculate Global SHAP (The "Naive" Optimization)
    explainer = shap.TreeExplainer(model, data=x_test, feature_perturbation="interventional")
    shap_values = explainer.shap_values(x_test, check_additivity=False)
    
    return x_test, y_test, model, shap_values, explainer

X, y, model, global_shap_values, explainer = load_data_and_model()

def silverman_bw(x):
    """Calculate Silverman's bandwidth for KDE."""
    std_dev = np.std(x)
    n = len(x)
    bandwidth = 1.06 * std_dev * n ** (-1 / 5)
    return bandwidth

# --- Jacobian Pre-calculation (Cached) ---
@st.cache_data
def precompute_jacobian(feature_name, _X_data, _model):
    """
    Simulates the 'Scientific' approach:
    1. Vary the feature over a grid.
    2. Get predictions -> Smooth -> Differentiate (Jacobian).
    """
    
    # 1. Create Grid
    feat_min = _X_data[feature_name].min()
    feat_max = _X_data[feature_name].max()
    grid_size = 200
    grid = np.linspace(feat_min, feat_max, grid_size)
    
    # 2. Calculate Derivatives
    predictions = np.zeros((len(_X_data), grid_size))
    
    X_temp = _X_data.copy()
    for i, val in enumerate(grid):
        X_temp[feature_name] = val
        predictions[:, i] = _model.predict(X_temp)
    
    # 4. Smooth & Differentiate
    smoothed_preds = gaussian_filter1d(predictions, sigma=silverman_bw(predictions), axis=1, mode='nearest')
    dx = grid[1] - grid[0]
    gradients = np.gradient(smoothed_preds, axis=1) / dx
    
    return _X_data, _X_data.index, grid, gradients

target_feature = st.selectbox("Select Feature for Impact Analysis", options=X.columns.tolist(), index=X.columns.get_loc("MedInc"), key="feature_select")
X_jac_sample, jac_indices, jac_grid, jac_gradients = precompute_jacobian(target_feature, X, model)

feat_to_name_dict = {
    "MedInc": "Median Income ($10k)",
    "HouseAge": "House Age",
    "AveRooms": "Average Rooms",
    "AveBedrms": "Average Bedrooms",
    "Population": "Population",
    "AveOccup": "Average Occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}


# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
st.sidebar.write("Refine the dataset to see how methods adapt to specific subgroups.")

with st.sidebar.expander("Geographic Filters", expanded=False):
    lat_min, lat_max = float(X['Latitude'].min()), float(X['Latitude'].max())
    selected_lat = st.slider("Latitude Range", lat_min, lat_max, (32.0, 42.0))

    long_min, long_max = float(X['Longitude'].min()), float(X['Longitude'].max())
    selected_long = st.slider("Longitude Range", long_min, long_max, (long_min, long_max))

with st.sidebar.expander("Building Specifications", expanded=False):
    age_min, age_max = float(X['HouseAge'].min()), float(X['HouseAge'].max())
    selected_age = st.slider("House Age", age_min, age_max, (age_min, age_max))
    
    rooms_min, rooms_max = float(X['AveRooms'].min()), float(X['AveRooms'].max())
    selected_rooms = st.slider("Average Rooms", rooms_min, rooms_max, (rooms_min, rooms_max))

    bedrms_min, bedrms_max = float(X['AveBedrms'].min()), float(X['AveBedrms'].max())
    selected_bedrms = st.slider("Average Bedrooms", bedrms_min, bedrms_max, (bedrms_min, bedrms_max))

with st.sidebar.expander("Demographics", expanded=False):
    pop_min, pop_max = float(X['Population'].min()), float(X['Population'].max())
    selected_pop = st.slider("Population (Block Group)", pop_min, pop_max, (pop_min, pop_max))

    occup_min, occup_max = float(X['AveOccup'].min()), float(X['AveOccup'].max())
    selected_occup = st.slider("Average Occupancy", occup_min, occup_max, (occup_min, occup_max))

if st.sidebar.button("Apply Filters & Update Curves"):
    # Apply Filters
    mask = (
        (X['Latitude'] >= selected_lat[0]) & (X['Latitude'] <= selected_lat[1]) &
        (X['Longitude'] >= selected_long[0]) & (X['Longitude'] <= selected_long[1]) &
        (X['HouseAge'] >= selected_age[0]) & (X['HouseAge'] <= selected_age[1]) &
        (X['AveRooms'] >= selected_rooms[0]) & (X['AveRooms'] <= selected_rooms[1]) &
        (X['AveBedrms'] >= selected_bedrms[0]) & (X['AveBedrms'] <= selected_bedrms[1]) &
        (X['Population'] >= selected_pop[0]) & (X['Population'] <= selected_pop[1]) &
        (X['AveOccup'] >= selected_occup[0]) & (X['AveOccup'] <= selected_occup[1])
    )
    X_filtered = X[mask]
    y_filtered = y[mask]
    global_shap_filtered = global_shap_values[mask]

    if len(X_filtered) < 50:
        st.error("Too few samples selected (<50)! Please widen your filters for reliable calculations.")
        st.stop()

    # --- Analysis Section ---
    st.divider()
    feature_idx = X.columns.get_loc(target_feature)

    st.subheader(f"Analyzing Feature: **{target_feature}**")

    # Display Filter Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(X))
    col2.metric("Filtered Samples", len(X_filtered))
    col3.metric("Global Avg Price", f"${y.mean():.2f}k")
    col4.metric("Filtered Avg Price", f"${y_filtered.mean():.2f}k")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Global SHAP", 
        "2. Dynamic SHAP", 
        "3. Rebased SHAP",
        "4. ALE (standard)",
        "5. ALE (Jacobian)"
    ])

    # Common styling
    X_LABEL = feat_to_name_dict.get(target_feature, target_feature)
    Y_LABEL = "Impact on Price ($k)"

    # --- Helper Plot Function ---
    def plot_gam_curve(x_data, y_data, title, color):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Fit GAM for smooth trend line
        n_splines = min(12, len(x_data) // 5)
        if n_splines < 4: n_splines = 4
        
        try:
            gam = ExpectileGAM(s(0, n_splines=n_splines), expectile=0.5).fit(x_data.values.reshape(-1, 1), y_data)
            
            x_grid = np.linspace(x_data.min(), x_data.max(), 200)
            y_smooth = gam.predict(x_grid)
            
            ax.plot(x_grid, y_smooth, color=color, linewidth=3, alpha=0.9, label="Impact Curve")
            
            ax.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Aesthetic touches
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            return fig, None
            
        except Exception as e:
            return None, f"Could not fit GAM: {str(e)}"

    # --- TAB 1: Naive Static SHAP ---
    with tab1:

        t1c1, t1c2 = st.columns([2,3])
        with t1c1:
            st.markdown("### SHAP Dependency Plot (Pre-calculated)")
            st.write("Pre-calculated SHAP values filtered to the selected subgroup. Pre-calculated SHAP-values may not be accurate for the filtered distribution.")
        
        fig1, err = plot_gam_curve(
            X_filtered[target_feature], 
            global_shap_filtered[:, feature_idx], 
            "SHAP Dependency Plot (Pre-calculated)", 
            "#1f77b4", # Blue
        )
        if fig1:
            with t1c2:
                st.pyplot(fig1)
        else:
            with t1c2:
                st.error(err)

    # --- TAB 2: Dynamic Recalculation ---
    with tab2:
        t2c1, t2c2 = st.columns([2,3])
        with t2c1:
            st.markdown("### Dynamic SHAP Dependency Plot")
            st.write("SHAP values are recalculated on a random sample of 500 filtered observations.")
        
        # 1. Create Background Sample (max 500)
        subset_size = min(500, len(X_filtered))
        background_data = X_filtered.sample(n=subset_size, random_state=42)
        
        # 2. New Explainer with Background Data (Interventional)
        dyn_explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")
        
        # 3. Calculate SHAP Values
        dynamic_shap = dyn_explainer.shap_values(background_data, approximate=False, check_additivity=False)
        
        fig2, err = plot_gam_curve(
            background_data[target_feature],
            dynamic_shap[:, feature_idx],
            "Dynamic SHAP Dependency Plot",
            "#1f77b4" # Blue
        )       
        if fig2:
            with t2c2:
                st.pyplot(fig2)
        else:
            with t2c2:
                st.error(err)

    # --- TAB 3: Rebasing (Pragmatic) ---
    with tab3:
        t3c1, t3c2 = st.columns([2,3])
        with t3c1:
            st.markdown("### Rebased SHAP Dependency Plot")
            st.write("Global SHAP values shifted to align with the filtered base value (mean). The baseline difference is distributed evenly across features.")

        raw_impact = global_shap_filtered[:, feature_idx]
        rebased_impact = raw_impact - np.mean(raw_impact)
        
        fig3, err = plot_gam_curve(
            X_filtered[target_feature],
            rebased_impact,
            "Rebased SHAP Dependency Plot",
            "#1f77b4" # Blue
        )
        
        if fig3:
            with t3c2:
                st.pyplot(fig3)
        else:
            with t3c2:
                st.error(err)

    # --- TAB 4: Vanilla ALE ---
    with tab4:
        t4c1, t4c2 = st.columns([2,3])
        with t4c1:
            st.markdown("### Accumulated Local Effects (standard approach with fixed bins)")
            st.write("Standard ALE with 10 fixed-width bins. As the plot is built up of linear segments, the plot may look rugged, hindering interpratation for non-expert users.")
        
        try:
            feat_vals = X_filtered[target_feature].values
            
            # Fixed Width Bins (10 bins = 11 edges)
            grid = np.linspace(feat_vals.min(), feat_vals.max(), 11)
            
            ale_effects = []
            valid_grid_points = [] # Keep track of bins that actually had data
            
            for i in range(len(grid) - 1):
                z_lower, z_upper = grid[i], grid[i+1]
                in_bin = (feat_vals >= z_lower) & (feat_vals <= z_upper)
                
                if not np.any(in_bin):
                    # If bin is empty, skip it to avoid graph artifacts
                    continue
                    
                # Center point for plotting
                valid_grid_points.append((z_lower + z_upper) / 2)
                
                X_bin = X_filtered[in_bin].copy()
                X_bin[target_feature] = z_lower
                pred_lower = model.predict(X_bin)
                X_bin[target_feature] = z_upper
                pred_upper = model.predict(X_bin)
                
                ale_effects.append(np.mean(pred_upper - pred_lower))
                
            # Accumulate
            accumulated = np.cumsum([0] + ale_effects)
            # Adjust X axis to match accumulated length
            x_plot = [grid[0]] + valid_grid_points
            if len(x_plot) > len(accumulated): x_plot = x_plot[:len(accumulated)]
            
            # Center
            centered_ale = accumulated - np.mean(accumulated)
            
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.plot(x_plot, centered_ale, marker=None, markersize=5, color='#1f77b4', linewidth=2, label="ALE Curve")
            
            ax4.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax4.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax4.set_title("Standard ALE: 10 Fixed Bins", fontsize=12)
            ax4.grid(True, alpha=0.2, linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            with t4c2:
                st.pyplot(fig4)

        except Exception as e:
            with t4c2:
                st.error(f"Calculation Error: {e}")

    # --- TAB 5: ALE (Jacobian Integration) ---
    with tab5:
        t5c1, t5c2 = st.columns([2,3])
        with t5c1:
            st.markdown("### Accumulated Local Effects (Jacobian smoothing & integration)")
            st.write("Smooth curve derived by integrating the partial derivatives (Jacobian) of the model.")
        
        try:
            valid_indices = np.intersect1d(X_filtered.index, jac_indices)
            idx_map = {idx: i for i, idx in enumerate(jac_indices)}
            array_positions = [idx_map[idx] for idx in valid_indices]
            subset_gradients = jac_gradients[array_positions, :]
            
            # Flatten for GAM fitting
            X_flat = np.tile(jac_grid, len(subset_gradients))
            y_flat = subset_gradients.flatten()
            # Determine number of splines
            n_splines = min(12, len(X_filtered) // 5)
            if n_splines < 4: n_splines = 4
            # Fit GAM to Gradients
            gam_deriv = ExpectileGAM(s(0, n_splines=n_splines), expectile=0.5).fit(X_flat.reshape(-1, 1), y_flat)
            
            # Integrate
            smooth_derivs = gam_deriv.predict(jac_grid)
            dx = jac_grid[1] - jac_grid[0]
            integrated_curve = np.cumsum(smooth_derivs) * dx
            integrated_curve = integrated_curve - integrated_curve.mean()
            
            # Plot
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(jac_grid, integrated_curve, color='#1f77b4', linewidth=3, label="Jacobian ALE")
            
            ax5.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax5.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax5.set_title("Jacobian ALE: Smoothed & Integrated", fontsize=12)
            ax5.grid(True, alpha=0.2, linestyle='--')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            
            with t5c2:
                st.pyplot(fig5)
                
        except Exception as e:
            with t5c2:
                st.error(f"Calculation Error: {e}")