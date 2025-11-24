import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from pygam import s, LinearGAM

import dpbinning as dp

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Impact Curves Demo")

st.title("Fast and Accurate Feature Impact Curves in Dynamic Dashboards")

st.write("*Márton Nagy (Gabors Data and AI Lab) | 2 December 2025 | 2nd CEU Vienna Data Analytics Jamboree*")

st.markdown("""
This demo simulates the challenge of visualizing **feature impact curves** in a dynamic dashboard.
We use a simple **random forest** model to predict **house prices** on the California Housing Dataset.
The model was trained on a 80% random sample of the data, and we analyze feature impacts on the remaining 20%.
            
You can interactively filter the dataset using the sidebar controls. Press the *Update Results* button to see how different methods adapt to the selected subgroup.
""")

def fit_piecewise_linear_spline(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray
):
    """
    Fits a piecewise linear spline model to data.

    The model uses the truncated linear power basis:
    y = beta_0 + beta_1*x + sum_{k=1}^K beta_{k+1}*max(0, x - tau_k)

    Args:
        x: A 1D numpy array of predictor (independent) variable values.
        y: A 1D numpy array of response (dependent) variable values.
        knots: A 1D numpy array of internal knot locations (tau_k).

    Returns:
        A tuple containing:
            1. The fitted coefficients (beta_0, beta_1, beta_2, ...).
            2. A function that can be used to predict y values given new x values.
    """
    if x.shape != y.shape:
        raise ValueError("x and y arrays must have the same shape.")
    
    n = len(x)
    k = len(knots)

    # --- 1. Construct the Design Matrix (Basis Matrix) X ---
    # The matrix X will have dimensions (n, k + 2)
    # Column 0: Intercept (beta_0). All ones.
    # Column 1: Linear term (beta_1 * x).
    # Columns 2 to k+1: Truncated linear terms (beta_{k+1} * max(0, x - tau_k)).

    # Initialize the matrix with the first two columns (Intercept and Linear term)
    X = np.ones((n, k + 2))
    X[:, 1] = x

    # Add the truncated linear terms (basis functions)
    for j, tau in enumerate(knots):
        # max(0, x - tau) is the truncated linear power basis function
        # np.maximum(0, x - tau) is a vectorized way to compute (x - tau)_+
        X[:, j + 2] = np.maximum(0, x - tau)

    # --- 2. Solve the Least Squares Problem ---
    # We solve X * beta = y for beta using numpy's least squares solver
    # result[0] contains the coefficients (beta)
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    
    # --- 3. Create the Prediction Function ---
    def predict_spline(x_new: np.ndarray) -> np.ndarray:
        """Predicts y values for new x values using the fitted coefficients."""
        x_new = np.asarray(x_new) # Ensure it's a numpy array for consistent operation
        n_new = len(x_new)
        
        # Construct the new design matrix X_new
        X_new = np.ones((n_new, k + 2))
        X_new[:, 1] = x_new
        
        # Add the truncated terms based on the *original* knots
        for j, tau in enumerate(knots):
            X_new[:, j + 2] = np.maximum(0, x_new - tau)
            
        # The prediction is X_new @ coefficients
        return X_new @ coefficients

    return coefficients, predict_spline

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

    x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
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

    gradients = np.gradient(smoothed_preds, dx, axis=1)

    # get the gradients at the original value of each sample
    gradients = np.array([np.interp(_X_data[feature_name].iloc[i], grid, gradients[i]) for i in range(len(_X_data))])
    
    return grid, gradients

@st.cache_data
def get_feature_importance(_model, _X_data, _y_data):
    result = permutation_importance(_model, _X_data, _y_data, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=_X_data.columns).sort_values(ascending=False)
    return importances


st.subheader('Model Overview')
# RMSE on test set and
model_overview_col1, model_overview_col2 = st.columns([2,3])
with model_overview_col1:
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    st.markdown(f"- **Model Type:** Random Forest Regressor")
    st.markdown(f"- **Test RMSE:** ${rmse:.2f}k")
    # R^2 Score on test set
    r2 = r2_score(y, y_pred)
    st.markdown(f"- **Test R-Squared:** {r2*100:.2f}%")
with model_overview_col2:
    # Feature Importance plot based on permuation importance
    importances = get_feature_importance(model, X, y)
    fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(8, 2))
    importances.plot(kind='bar', ax=ax_feat_imp, color='#3a5e8c', edgecolor='black')
    ax_feat_imp.set_title("Permutation Feature Importance", fontsize=12)
    ax_feat_imp.set_ylabel("Mean Importance", fontdict={'fontsize':10, 'fontweight':'bold'})
    ax_feat_imp.spines['top'].set_visible(False)
    ax_feat_imp.spines['right'].set_visible(False)
    st.pyplot(fig_feat_imp)

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
st.sidebar.header("Dashboard Settings")

target_feature = st.sidebar.selectbox("Select Feature for Impact Analysis", options=X.columns.tolist(), index=X.columns.get_loc("MedInc"), key="feature_select")
jac_grid, jac_gradients = precompute_jacobian(target_feature, X, model)

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
    medinc_min, medinc_max = float(X['MedInc'].min()), float(X['MedInc'].max())
    selected_medinc = st.slider("Median Income ($10k)", medinc_min, medinc_max, (medinc_min, medinc_max))

    pop_min, pop_max = float(X['Population'].min()), float(X['Population'].max())
    selected_pop = st.slider("Population (Block Group)", pop_min, pop_max, (pop_min, pop_max))

    occup_min, occup_max = float(X['AveOccup'].min()), float(X['AveOccup'].max())
    selected_occup = st.slider("Average Occupancy", occup_min, occup_max, (occup_min, occup_max))

# Apply Filters
mask = (
    (X['Latitude'] >= selected_lat[0]) & (X['Latitude'] <= selected_lat[1]) &
    (X['Longitude'] >= selected_long[0]) & (X['Longitude'] <= selected_long[1]) &
    (X['HouseAge'] >= selected_age[0]) & (X['HouseAge'] <= selected_age[1]) &
    (X['AveRooms'] >= selected_rooms[0]) & (X['AveRooms'] <= selected_rooms[1]) &
    (X['AveBedrms'] >= selected_bedrms[0]) & (X['AveBedrms'] <= selected_bedrms[1]) &
    (X['Population'] >= selected_pop[0]) & (X['Population'] <= selected_pop[1]) &
    (X['AveOccup'] >= selected_occup[0]) & (X['AveOccup'] <= selected_occup[1]) &
    (X['MedInc'] >= selected_medinc[0]) & (X['MedInc'] <= selected_medinc[1])
)
X_filtered = X[mask]
y_filtered = y[mask]
y_pred_filtered = model.predict(X_filtered)
global_shap_filtered = global_shap_values[mask]
jac_filtered = jac_gradients[mask.values]

if len(X_filtered) < 50:
    st.sidebar.error("Too few samples selected (<50) - please widen your filters!")
    disable_button = True
else:
    disable_button = False
if st.sidebar.button("Update Results", disabled=disable_button):
    # --- Analysis Section ---
    st.divider()
    feature_idx = X.columns.get_loc(target_feature)

    st.subheader("Filtered Dataset Feature Distributions")
    # Show histograms of features in filtered dataset
    dist_cols = st.columns(4)
    for i, col in enumerate(X.columns):
        with dist_cols[i % 4]:
            fig_dist, ax_dist = plt.subplots(figsize=(4, 3))
            ax_dist.hist(X_filtered[col], bins=20, color='#3a5e8c', edgecolor='black')
            ax_dist.set_title(feat_to_name_dict.get(col, col), fontsize=10)
            ax_dist.set_xlabel("", fontsize=8)
            ax_dist.set_ylabel("Count", fontsize=8)
            ax_dist.spines['top'].set_visible(False)
            ax_dist.spines['right'].set_visible(False)
            st.pyplot(fig_dist)

    st.subheader("Filtered Dataset Correlation Matrix")
    st.dataframe(X_filtered.corr().round(2).style.background_gradient(cmap='viridis'), use_container_width=True)

    st.subheader(f"Analyzing Feature: **{target_feature}**")

    # Display Filter Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(X))
    col2.metric("Filtered Samples", len(X_filtered))
    col3.metric("Global Avg Price", f"${y.mean():.2f}k")
    col4.metric("Filtered Avg Price", f"${y_filtered.mean():.2f}k")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Global SHAP", 
        "2. Dynamic SHAP", 
        "3. Rebased SHAP",
        "4. PDP",
        "5. ALE (standard)",
        "6. ALE (Jacobian)"
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
            gam = LinearGAM(s(0, n_splines=n_splines)).fit(x_data.values.reshape(-1, 1), y_data)

            x_grid = np.linspace(x_data.min(), x_data.max(), 200)
            y_grid = gam.predict(x_grid)
            ax.plot(x_grid, y_grid, color=color, linewidth=3, alpha=0.9, label="Impact Curve")
            
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
            "#3a5e8c", # Blue
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
            "#3a5e8c" # Blue
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
            "#3a5e8c" # Blue
        )
        
        if fig3:
            with t3c2:
                st.pyplot(fig3)
        else:
            with t3c2:
                st.error(err)

    # --- TAB 4: PDP
    with tab4:
        t4c1, t4c2 = st.columns([2,3])
        with t4c1:
            st.markdown("### Partial Dependence Plot (PDP)")
            st.write("PDP calculated on the filtered dataset. PDPs can be misleading when features are correlated.")
        
        try:
            pd_results = partial_dependence(
                model, 
                X_filtered, 
                features=target_feature, 
                grid_resolution=100,
                percentiles=(0,1)
            )
            pd_x = pd_results['grid_values'][0]
            pd_y = pd_results['average'][0]
            
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.plot(pd_x, pd_y - np.mean(pd_y), color='#3a5e8c', linewidth=3, label="PDP Curve")
            
            ax4.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax4.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax4.set_title("Partial Dependence Plot", fontsize=12)
            ax4.grid(True, alpha=0.2, linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            with t4c2:
                st.pyplot(fig4)

        except Exception as e:
            with t4c2:
                st.error(f"Calculation Error: {e}")
    # --- TAB 5: Vanilla ALE ---
    with tab5:
        t5c1, t5c2 = st.columns([2,3])
        with t5c1:
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
            
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(x_plot, centered_ale, marker=None, markersize=5, color='#3a5e8c', linewidth=3, label="ALE Curve")
            
            ax5.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax5.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax5.set_title("Standard ALE", fontsize=12)
            ax5.grid(True, alpha=0.2, linestyle='--')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            
            with t5c2:
                st.pyplot(fig5)

        except Exception as e:
            with t5c2:
                st.error(f"Calculation Error: {e}")

    # --- TAB 6: ALE (Jacobian Integration) ---
    with tab6:
        t6c1, t6c2 = st.columns([2,3])
        with t6c1:
            st.markdown("### Accumulated Local Effects (Jacobian smoothing & integration)")
            st.write("Smooth curve derived by integrating the partial derivatives (Jacobian) of the model.")
        
        try:            
            # dummy col as data and data effect is expeceted to be 2D array
            data_2d = X_filtered[target_feature].values.reshape(-1, 1)
            data_effect_2d = jac_filtered.reshape(-1, 1)
            binning = dp.DP(
                data=data_2d,
                data_effect=data_effect_2d,
                feature=0
            )

            binning.find(discount=0.15)

            limits = binning.limits

            limits = limits[:-1]

            _, predict_spline = fit_piecewise_linear_spline(
                x=X_filtered[target_feature].values,
                y=jac_filtered,
                knots=limits
            )

            # Integrate
            x_grid = np.linspace(X_filtered[target_feature].min(), X_filtered[target_feature].max(), 200)
            smooth_derivs = predict_spline(x_grid)
            dx = x_grid[1] - x_grid[0]
            integrated_curve = np.cumsum(smooth_derivs) * dx
            integrated_curve = integrated_curve - integrated_curve.mean()
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            ax6.plot(x_grid, integrated_curve, color='#3a5e8c', linewidth=3, label="ALE (Jacobian)")
            ax6.set_xlabel(X_LABEL, fontsize=10, fontweight='bold')
            ax6.set_ylabel(Y_LABEL, fontsize=10, fontweight='bold')
            ax6.set_title("ALE (Jacobian)", fontsize=12)
            ax6.grid(True, alpha=0.2, linestyle='--')
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            
            with t6c2:
                st.pyplot(fig6)
                
        except Exception as e:
            with t6c2:
                st.error(f"Calculation Error: {e}")