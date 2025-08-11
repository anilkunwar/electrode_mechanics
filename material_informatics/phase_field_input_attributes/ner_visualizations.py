import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import logging
import pickle
import torch
import io
import os
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go

# Set Matplotlib to non-interactive backend for Streamlit
matplotlib.use('Agg')

# Initialize logging
logging.basicConfig(filename='ner_visualizations.log', level=logging.DEBUG)

# Initialize Streamlit app
st.set_page_config(page_title="NER Visualizations for Phase Field Parameters", layout="wide")
st.title("Publication-Quality Visualizations for Phase Field NER Data")
st.markdown("""
This tool visualizes Named Entity Recognition (NER) results for phase field model parameters (interface energy in J/m², interface width in nm, temperature in K, and materials) from `.h5`, `.pkl`, or `.pt` files. It generates histograms with customizable legends, scatter plots, heatmaps, and interactive ternary plots (normalized and inverse scaled) with customizable styling. Download data as CSV/JSON and plots as PNG/SVG.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `pandas`, `streamlit`, `matplotlib`, `seaborn`, `numpy`, `plotly`, `h5py`, `torch`, `scipy`
- Install with: `pip install pandas streamlit matplotlib seaborn numpy plotly h5py torch scipy`
""")

# Define entity types and default colors
param_types = ["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"]
default_colors = {
    "MATERIAL": '#9467bd',         # Purple
    "INTERFACE_ENERGY": '#1f77b4', # Blue
    "INTERFACE_WIDTH": '#ff7f0e',  # Orange
    "TEMPERATURE_K": '#2ca02c'     # Green
}
logging.info(f"Default colors: {default_colors}")

# Get all available Matplotlib colormaps
colormaps = sorted(matplotlib.colormaps)  # Includes jet, rainbow, and over 50 others
default_colormap = 'tab20'

# Load NER data
def load_ner_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".h5"):
            df = pd.read_hdf(uploaded_file, key="ner_results")
        elif uploaded_file.name.endswith(".pkl"):
            df = pd.read_pickle(uploaded_file)
        elif uploaded_file.name.endswith(".pt"):
            data = torch.load(uploaded_file)
            df = pd.DataFrame(data)
        else:
            st.error("Unsupported file format. Please upload .h5, .pkl, or .pt.")
            return None, None
        # Attempt to load PMI data if available
        pmi_results = {}
        if uploaded_file.name.endswith(".pkl"):
            try:
                with open(uploaded_file.name, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and "pmi_results" in data:
                        pmi_results = data["pmi_results"]
            except:
                logging.debug("No PMI results found in .pkl file.")
        st.success(f"Loaded **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
        return df, pmi_results
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logging.error(f"Error loading file: {str(e)}")
        return None, None

# Visualize NER results
def visualize_ner_results(df, pmi_results):
    # Set Seaborn style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': False,  # Disabled for Streamlit compatibility
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    # Sidebar for plot customization
    st.sidebar.subheader("Visualization Customization")
    font_size = st.sidebar.slider("Font Size", 8, 24, 12, step=1)
    axes_thickness = st.sidebar.slider("Axes Thickness", 0.5, 2.0, 1.0, step=0.1)
    axes_line_color = st.sidebar.color_picker("Axes Line Color", "#808080")
    marker_size = st.sidebar.slider("Scatter Marker Size", 20, 500, 100, step=10)
    alpha = st.sidebar.slider("Marker Transparency", 0.1, 1.0, 0.6, step=0.1)
    font_family = st.sidebar.selectbox("Font Family", ["Arial", "Helvetica", "Times New Roman"], index=0)
    colormap = st.sidebar.selectbox("Colormap", colormaps, index=colormaps.index(default_colormap))
    hist_edge_width = st.sidebar.slider("Histogram Edge Line Width", 0.5, 2.0, 1.0, step=0.1)
    xlabel_color = st.sidebar.color_picker("X-Label Color", "#000000")
    ylabel_color = st.sidebar.color_picker("Y-Label Color", "#000000")
    title_color = st.sidebar.color_picker("Title Color", "#000000")
    use_log_heatmap = st.sidebar.checkbox("Use Logarithmic Scale for Heatmap", value=False)
    
    # Legend customization for histograms
    st.sidebar.subheader("Histogram Legend Customization")
    legend_position = st.sidebar.selectbox("Legend Position", ["upper right", "upper left", "lower right", "lower left", "best", "none"], index=4)
    legend_title = st.sidebar.text_input("Legend Title", value="Parameter Type")
    legend_fontsize = st.sidebar.slider("Legend Font Size", 8, 20, 10, step=1)
    legend_framealpha = st.sidebar.slider("Legend Frame Transparency", 0.0, 1.0, 0.8, step=0.1)
    legend_borderpad = st.sidebar.slider("Legend Border Padding", 0.1, 1.0, 0.4, step=0.1)
    
    # Ternary plot customization
    st.sidebar.subheader("Ternary Plot Customization")
    energy_label = st.sidebar.text_input("Energy Axis Label", value="Interface Energy (J/m²)")
    width_label = st.sidebar.text_input("Width Axis Label", value="Interface Width (nm)")
    temp_label = st.sidebar.text_input("Temperature Axis Label", value="Temperature (K)")
    ternary_title = st.sidebar.text_input("Normalized Ternary Plot Title", value="Ternary Distribution of Normalized Parameters by Paper")
    ternary_inverse_title = st.sidebar.text_input("Inverse Scaled Ternary Plot Title", value="Ternary Distribution of Inverse Scaled Parameters by Paper")
    ternary_marker_size = st.sidebar.slider("Ternary Marker Size", 5, 50, 10, step=1)
    ternary_colormap = st.sidebar.selectbox("Ternary Colormap", colormaps, index=colormaps.index(default_colormap))
    ternary_grid_thickness = st.sidebar.slider("Ternary Grid Line Thickness", 0.5, 2.0, 1.0, step=0.1)
    ternary_grid_color = st.sidebar.color_picker("Ternary Grid Line Color", "#808080")
    label_font_size = st.sidebar.slider("Axis Label Font Size", 10, 24, 14, step=1)
    marker_scale = st.sidebar.slider("Marker Size Scaling", 0.5, 3.0, 1.5, step=0.1)
    show_original_scale = st.sidebar.checkbox("Show Original Scale Indicators", True)

    # Filters
    st.sidebar.subheader("Data Filters")
    selected_papers = st.sidebar.multiselect("Select Papers", df['paper_id'].unique(), default=df['paper_id'].unique())
    entity_types = st.sidebar.multiselect("Parameter Types", param_types, default=param_types)
    temp_range = st.sidebar.slider("Temperature Range (K)", 200, 2000, (200, 350), step=10)
    energy_range = st.sidebar.slider("Interface Energy Range (J/m²)", 0.001, 10.0, (0.001, 2.0), step=0.001)
    width_range = st.sidebar.slider("Interface Width Range (nm)", 0.1, 1000.0, (0.1, 100.0), step=0.1)
    sort_by = st.sidebar.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")

    # Update Matplotlib parameters
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'axes.linewidth': axes_thickness,
        'xtick.major.width': axes_thickness,
        'ytick.major.width': axes_thickness,
    })

    # Update colors
    cmap = plt.get_cmap(colormap)
    param_colors = {param: cmap(i / len(param_types)) for i, param in enumerate(param_types)}
    if colormap == "tab20":
        param_colors = default_colors

    # Filter DataFrame
    filtered_df = df[df['paper_id'].isin(selected_papers) & 
                    df['entity_label'].isin(entity_types) &
                    (df['value'].between(energy_range[0], energy_range[1]) | ~df['entity_label'].isin(['INTERFACE_ENERGY'])) &
                    (df['value'].between(width_range[0], width_range[1]) | ~df['entity_label'].isin(['INTERFACE_WIDTH'])) &
                    (df['value'].between(temp_range[0], temp_range[1]) | ~df['entity_label'].isin(['TEMPERATURE_K']))]
    
    if sort_by == "entity_label":
        filtered_df = filtered_df.sort_values(["entity_label", "value"])
    else:
        filtered_df = filtered_df.sort_values(["value", "entity_label"], na_position="last")

    # Display filtered data
    st.subheader("Extracted Parameters")
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
            use_container_width=True,
            column_config={
                "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
                "value": st.column_config.NumberColumn("Value", help="Numerical value (J/m², nm, K)."),
                "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., capacity).")
            }
        )
    else:
        st.warning("No data available after filtering.")
        logging.debug("Filtered DataFrame is empty.")

    # Download filtered data
    st.subheader("Download Filtered Data")
    if not filtered_df.empty:
        st.download_button(
            label="Download Filtered Parameters CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_ner_params.csv",
            mime="text/csv"
        )
        with open("filtered_ner_params.json", "w") as f:
            filtered_df.to_json(f, orient="records", lines=True)
        with open("filtered_ner_params.json", "rb") as f:
            st.download_button(
                label="Download Filtered Parameters JSON",
                data=f,
                file_name="filtered_ner_params.json",
                mime="application/json"
            )

    # PMI results
    if pmi_results:
        st.subheader("PMI Scores for Phase Field Context Phrases")
        pmi_data = []
        for paper_id, phrases in pmi_results.items():
            if paper_id in selected_papers:
                for phrase, score in phrases.items():
                    pmi_data.append({"paper_id": paper_id, "phrase": phrase, "PMI Score": round(score, 3)})
        pmi_df = pd.DataFrame(pmi_data)
        if not pmi_df.empty:
            st.dataframe(pmi_df, use_container_width=True)
            st.download_button(
                label="Download PMI Scores CSV",
                data=pmi_df.to_csv(index=False),
                file_name="pmi_scores.csv",
                mime="text/csv"
            )
        else:
            st.info("No PMI scores available for the selected papers.")

    # Histograms
    st.subheader("Parameter Distribution Analysis")
    for param_type in entity_types:
        if param_type in param_types:
            param_df = filtered_df[filtered_df["entity_label"] == param_type]
            if not param_df.empty and param_df['value'].notnull().any():
                fig, ax = plt.subplots(figsize=(8, 5))
                for spine in ax.spines.values():
                    spine.set_color(axes_line_color)
                values = param_df["value"].dropna()
                unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                bins = 30 if param_type == "INTERFACE_ENERGY" else 20
                counts, bins, _ = ax.hist(values, bins=bins, edgecolor='black', linewidth=hist_edge_width,
                                         color=param_colors[param_type], alpha=alpha,
                                         label=param_type.replace('_', ' ').title())
                ax.set_xlabel(f"{param_type.replace('_', ' ').title()} ({unit})", fontweight='bold', color=xlabel_color)
                ax.set_ylabel("Count", fontweight='bold', color=ylabel_color)
                ax.set_title(f"Distribution of {param_type.replace('_', ' ').title()}", fontweight='bold', pad=15, color=title_color)
                ax.grid(True, alpha=0.3)
                if legend_position != "none":
                    ax.legend(title=legend_title, loc=legend_position, fontsize=legend_fontsize, framealpha=legend_framealpha, borderpad=legend_borderpad)
                plt.tight_layout()
                
                # Display plot
                st.pyplot(fig)
                
                # Download plot as PNG and SVG
                buf_png = io.BytesIO()
                buf_svg = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
                fig.savefig(buf_svg, format="svg", bbox_inches='tight')
                st.download_button(
                    label=f"Download {param_type} Histogram PNG",
                    data=buf_png.getvalue(),
                    file_name=f"{param_type.lower()}_histogram.png",
                    mime="image/png"
                )
                st.download_button(
                    label=f"Download {param_type} Histogram SVG",
                    data=buf_svg.getvalue(),
                    file_name=f"{param_type.lower()}_histogram.svg",
                    mime="image/svg+xml"
                )
                
                # Download histogram data
                hist_data = pd.DataFrame({
                    'Bin_Lower': bins[:-1],
                    'Bin_Upper': bins[1:],
                    'Count': counts
                })
                st.download_button(
                    label=f"Download {param_type} Histogram Counts CSV",
                    data=hist_data.to_csv(index=False),
                    file_name=f"{param_type.lower()}_histogram_counts.csv",
                    mime="text/csv"
                )
                plt.close(fig)

    # Scatter Plot: Interface Energy vs Temperature
    energy_df = filtered_df[filtered_df["entity_label"] == "INTERFACE_ENERGY"]
    temp_df = filtered_df[filtered_df["entity_label"] == "TEMPERATURE_K"]
    if not energy_df.empty and not temp_df.empty:
        st.subheader("Interface Energy vs Temperature")
        scatter_data = []
        for paper_id in energy_df["paper_id"].unique():
            energy_entries = energy_df[energy_df["paper_id"] == paper_id]
            temp_entries = temp_df[temp_df["paper_id"] == paper_id]
            for _, energy_row in energy_entries.iterrows():
                energy_value = energy_row["value"]
                for _, temp_row in temp_entries.iterrows():
                    temp_value = temp_row["value"]
                    if temp_range[0] <= temp_value <= temp_range[1] and energy_range[0] <= energy_value <= energy_range[1]:
                        scatter_data.append({
                            "paper_id": paper_id,
                            "title": energy_row["title"],
                            "Interface Energy (J/m²)": energy_value,
                            "Temperature (K)": temp_value
                        })
        energy_scatter_df = pd.DataFrame(scatter_data) if scatter_data else pd.DataFrame()
        if not energy_scatter_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for spine in ax.spines.values():
                spine.set_color(axes_line_color)
            paper_ids = energy_scatter_df["paper_id"].unique()
            for i, paper_id in enumerate(paper_ids):
                paper_data = energy_scatter_df[energy_scatter_df["paper_id"] == paper_id]
                ax.scatter(paper_data["Interface Energy (J/m²)"], paper_data["Temperature (K)"],
                           c=[cmap(i / len(paper_ids))], s=marker_size, alpha=alpha, label=paper_id[:10])
            ax.set_xlabel(r"Interface Energy (J/m²)", fontweight='bold', color=xlabel_color)
            ax.set_ylabel(r"Temperature (K)", fontweight='bold', color=ylabel_color)
            ax.set_title("Interface Energy vs Temperature by Paper", fontweight='bold', pad=15, color=title_color)
            ax.grid(True, alpha=0.3)
            if legend_position != "none":
                ax.legend(title=legend_title, loc=legend_position, fontsize=legend_fontsize, framealpha=legend_framealpha, borderpad=legend_borderpad)
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
            
            # Download plot as PNG and SVG
            buf_png = io.BytesIO()
            buf_svg = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
            fig.savefig(buf_svg, format="svg", bbox_inches='tight')
            st.download_button(
                label="Download Energy vs Temperature Scatter PNG",
                data=buf_png.getvalue(),
                file_name="energy_vs_temp_scatter.png",
                mime="image/png"
            )
            st.download_button(
                label="Download Energy vs Temperature Scatter SVG",
                data=buf_svg.getvalue(),
                file_name="energy_vs_temp_scatter.svg",
                mime="image/svg+xml"
            )
            
            # Download scatter data
            st.download_button(
                label="Download Interface Energy vs Temperature Scatter CSV",
                data=energy_scatter_df.to_csv(index=False),
                file_name="energy_vs_temp_scatter.csv",
                mime="text/csv"
            )
            plt.close(fig)
        else:
            st.warning("No valid interface energy-temperature pairs found within the selected ranges.")
            logging.debug("No valid energy-temperature pairs after filtering.")

    # Scatter Plot: Interface Width vs Temperature
    width_df = filtered_df[filtered_df["entity_label"] == "INTERFACE_WIDTH"]
    if not width_df.empty and not temp_df.empty:
        st.subheader("Interface Width vs Temperature")
        scatter_data = []
        for paper_id in width_df["paper_id"].unique():
            width_entries = width_df[width_df["paper_id"] == paper_id]
            temp_entries = temp_df[temp_df["paper_id"] == paper_id]
            for _, width_row in width_entries.iterrows():
                width_value = width_row["value"]
                for _, temp_row in temp_entries.iterrows():
                    temp_value = temp_row["value"]
                    if temp_range[0] <= temp_value <= temp_range[1] and width_range[0] <= width_value <= width_range[1]:
                        scatter_data.append({
                            "paper_id": paper_id,
                            "title": width_row["title"],
                            "Interface Width (nm)": width_value,
                            "Temperature (K)": temp_value
                        })
        width_scatter_df = pd.DataFrame(scatter_data) if scatter_data else pd.DataFrame()
        if not width_scatter_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for spine in ax.spines.values():
                spine.set_color(axes_line_color)
            paper_ids = width_scatter_df["paper_id"].unique()
            for i, paper_id in enumerate(paper_ids):
                paper_data = width_scatter_df[width_scatter_df["paper_id"] == paper_id]
                ax.scatter(paper_data["Interface Width (nm)"], paper_data["Temperature (K)"],
                           c=[cmap(i / len(paper_ids))], s=marker_size, alpha=alpha, label=paper_id[:10])
            ax.set_xlabel(r"Interface Width (nm)", fontweight='bold', color=xlabel_color)
            ax.set_ylabel(r"Temperature (K)", fontweight='bold', color=ylabel_color)
            ax.set_title("Interface Width vs Temperature by Paper", fontweight='bold', pad=15, color=title_color)
            ax.grid(True, alpha=0.3)
            if legend_position != "none":
                ax.legend(title=legend_title, loc=legend_position, fontsize=legend_fontsize, framealpha=legend_framealpha, borderpad=legend_borderpad)
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
            
            # Download plot as PNG and SVG
            buf_png = io.BytesIO()
            buf_svg = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
            fig.savefig(buf_svg, format="svg", bbox_inches='tight')
            st.download_button(
                label="Download Width vs Temperature Scatter PNG",
                data=buf_png.getvalue(),
                file_name="width_vs_temp_scatter.png",
                mime="image/png"
            )
            st.download_button(
                label="Download Width vs Temperature Scatter SVG",
                data=buf_svg.getvalue(),
                file_name="width_vs_temp_scatter.svg",
                mime="image/svg+xml"
            )
            
            # Download scatter data
            st.download_button(
                label="Download Interface Width vs Temperature Scatter CSV",
                data=width_scatter_df.to_csv(index=False),
                file_name="width_vs_temp_scatter.csv",
                mime="text/csv"
            )
            plt.close(fig)
        else:
            st.warning("No valid interface width-temperature pairs found within the selected ranges.")
            logging.debug("No valid width-temperature pairs after filtering.")

    # Heatmap: Interface Energy vs Temperature
    if not energy_df.empty and not temp_df.empty:
        st.subheader("Interface Energy vs Temperature Heatmap")
        if 'energy_scatter_df' not in locals() or energy_scatter_df is None or energy_scatter_df.empty:
            scatter_data = []
            for paper_id in energy_df["paper_id"].unique():
                energy_entries = energy_df[energy_df["paper_id"] == paper_id]
                temp_entries = temp_df[temp_df["paper_id"] == paper_id]
                for _, energy_row in energy_entries.iterrows():
                    energy_value = energy_row["value"]
                    for _, temp_row in temp_entries.iterrows():
                        temp_value = temp_row["value"]
                        if temp_range[0] <= temp_value <= temp_range[1] and energy_range[0] <= energy_value <= energy_range[1]:
                            scatter_data.append({
                                "paper_id": paper_id,
                                "title": energy_row["title"],
                                "Interface Energy (J/m²)": energy_value,
                                "Temperature (K)": temp_value
                            })
            energy_scatter_df = pd.DataFrame(scatter_data) if scatter_data else pd.DataFrame()
        
        if not energy_scatter_df.empty:
            required_cols = ["Temperature (K)", "Interface Energy (J/m²)"]
            if all(col in energy_scatter_df.columns for col in required_cols):
                heatmap_data = energy_scatter_df[required_cols].dropna()
                logging.debug(f"Heatmap data shape: {heatmap_data.shape}, columns: {heatmap_data.columns.tolist()}")
                
                if not heatmap_data.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for spine in ax.spines.values():
                        spine.set_color(axes_line_color)
                    
                    # Create bins
                    temp_bins = np.linspace(temp_range[0], temp_range[1], 21)  # 20 bins
                    energy_bins = np.linspace(energy_range[0], energy_range[1], 31)  # 30 bins
                    
                    heatmap, xedges, yedges = np.histogram2d(
                        heatmap_data["Temperature (K)"],
                        heatmap_data["Interface Energy (J/m²)"],
                        bins=[temp_bins, energy_bins]
                    )
                    
                    # Apply log scale if selected
                    if use_log_heatmap:
                        heatmap = np.log1p(heatmap)
                    
                    # Create heatmap plot
                    im = ax.imshow(
                        heatmap.T,
                        origin='lower',
                        aspect='auto',
                        cmap=cmap,
                        interpolation='nearest',
                        extent=[temp_range[0], temp_range[1], energy_range[0], energy_range[1]]
                    )
                    
                    ax.set_xlabel(r"Interface Energy (J/m²)", fontweight='bold', color=xlabel_color)
                    ax.set_ylabel(r"Temperature (K)", fontweight='bold', color=ylabel_color)
                    ax.set_title("Density of Interface Energy vs Temperature", fontweight='bold', pad=15, color=title_color)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Count" + (" (log scale)" if use_log_heatmap else ""), rotation=270, labelpad=15)
                    
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig)
                    
                    # Download plot as PNG and SVG
                    buf_png = io.BytesIO()
                    buf_svg = io.BytesIO()
                    fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
                    fig.savefig(buf_svg, format="svg", bbox_inches='tight')
                    st.download_button(
                        label="Download Energy vs Temperature Heatmap PNG",
                        data=buf_png.getvalue(),
                        file_name="energy_vs_temp_heatmap.png",
                        mime="image/png"
                    )
                    st.download_button(
                        label="Download Energy vs Temperature Heatmap SVG",
                        data=buf_svg.getvalue(),
                        file_name="energy_vs_temp_heatmap.svg",
                        mime="image/svg+xml"
                    )
                    
                    # Prepare heatmap data for download
                    temp_centers = (xedges[:-1] + xedges[1:]) / 2
                    energy_centers = (yedges[:-1] + yedges[1:]) / 2
                    temp_grid, energy_grid = np.meshgrid(temp_centers, energy_centers)
                    
                    heatmap_df = pd.DataFrame({
                        'Temperature (K)': temp_grid.flatten(),
                        'Interface Energy (J/m²)': energy_grid.flatten(),
                        'Count': heatmap.T.flatten()
                    })
                    
                    st.download_button(
                        label="Download Interface Energy vs Temperature Heatmap CSV",
                        data=heatmap_df.to_csv(index=False),
                        file_name="energy_vs_temp_heatmap.csv",
                        mime="text/csv"
                    )
                    plt.close(fig)
                else:
                    st.warning("No valid data available for the Interface Energy vs Temperature heatmap after filtering.")
                    logging.debug("Heatmap data empty after dropna.")
            else:
                st.error(f"Required columns {required_cols} not found in energy_scatter_df. Available columns: {energy_scatter_df.columns.tolist()}")
                logging.error(f"Heatmap failed: Required columns {required_cols} not in energy_scatter_df.")
        else:
            st.warning("No valid Interface Energy vs Temperature data available for heatmap.")
            logging.debug("energy_scatter_df is None or empty.")

    # Ternary Distribution (Normalized)
    st.subheader("Enhanced Ternary Distribution of Normalized Parameters")
    ternary_df = filtered_df[filtered_df['entity_label'].isin(['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K'])]
    if not ternary_df.empty:
        # Normalize values
        df_norm = ternary_df.copy()
        min_max_dict = {}
        
        # Calculate min/max for each parameter
        for param in ['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K']:
            param_values = df_norm[df_norm['entity_label'] == param]['value']
            if not param_values.empty:
                min_val = param_values.min()
                max_val = param_values.max()
                min_max_dict[param] = {'min': min_val, 'max': max_val}
                
                # Apply min-max normalization
                mask = df_norm['entity_label'] == param
                df_norm.loc[mask, 'value'] = (df_norm.loc[mask, 'value'] - min_val) / (max_val - min_val + 1e-6)
        
        # Pivot data for normalized ternary plot
        df_pivot = df_norm.pivot_table(
            index='paper_id',
            columns='entity_label',
            values='value',
            aggfunc='mean'
        ).reset_index().fillna(0)
        
        # Rename columns
        df_pivot = df_pivot.rename(columns={
            'INTERFACE_ENERGY': 'Energy',
            'INTERFACE_WIDTH': 'Width',
            'TEMPERATURE_K': 'Temperature'
        })
        
        # Normalize to sum to 1
        df_pivot['sum'] = df_pivot['Energy'] + df_pivot['Width'] + df_pivot['Temperature']
        df_pivot['Energy'] /= df_pivot['sum'].replace(0, 1)
        df_pivot['Width'] /= df_pivot['sum'].replace(0, 1)
        df_pivot['Temperature'] /= df_pivot['sum'].replace(0, 1)
        
        # Create enhanced ternary plot
        fig = go.Figure()
        
        # Create hover text with original values
        hover_text = []
        for _, row in df_pivot.iterrows():
            paper_id = row['paper_id']
            orig_values = []
            for param in ['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K']:
                orig_val = ternary_df[(ternary_df['paper_id'] == paper_id) & 
                                     (ternary_df['entity_label'] == param)]['value'].mean()
                unit = "J/m²" if param == "INTERFACE_ENERGY" else "nm" if param == "INTERFACE_WIDTH" else "K"
                orig_values.append(f"{param.split('_')[-1]}: {orig_val:.4f} {unit}")
            hover_text.append("<br>".join(orig_values))
        
        # Convert Matplotlib colormap to Plotly colorscale for categorical colormaps
        if ternary_colormap in ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2']:
            cmap_ternary = plt.get_cmap(ternary_colormap)
            colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b, _ in cmap_ternary(np.linspace(0, 1, cmap_ternary.N))]
            colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
        else:
            colorscale = ternary_colormap
        
        # Add scatter trace with enhanced styling
        fig.add_trace(go.Scatterternary({
            'a': df_pivot['Energy'],
            'b': df_pivot['Width'],
            'c': df_pivot['Temperature'],
            'mode': 'markers',
            'marker': {
                'size': ternary_marker_size * marker_scale,
                'color': df_pivot.index,
                'colorscale': colorscale,
                'opacity': alpha,
                'line': {
                    'width': ternary_grid_thickness/2,
                    'color': ternary_grid_color
                }
            },
            'text': hover_text,
            'hoverinfo': 'text+a+b+c',
            'hovertemplate': (
                "<b>Paper: %{text}</b><br><br>" +
                "Normalized Values:<br>" +
                "Energy: %{a:.3f}<br>" +
                "Width: %{b:.3f}<br>" +
                "Temperature: %{c:.3f}<extra></extra>"
            )
        }))
        
        # Enhanced axis configuration
        axis_config = {
            'aaxis': {
                'title': {
                    'text': f'<b>{energy_label}</b>',
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01
            },
            'baxis': {
                'title': {
                    'text': f'<b>{width_label}</b>',
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01
            },
            'caxis': {
                'title': {
                    'text': f'<b>{temp_label}</b>',
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01
            }
        }
        
        # Add original scale indicators if requested
        if show_original_scale:
            axis_config['aaxis']['title']['text'] += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['INTERFACE_ENERGY']['min']:.2f}-{min_max_dict['INTERFACE_ENERGY']['max']:.2f} J/m²)</span>"
            )
            axis_config['baxis']['title']['text'] += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['INTERFACE_WIDTH']['min']:.2f}-{min_max_dict['INTERFACE_WIDTH']['max']:.2f} nm)</span>"
            )
            axis_config['caxis']['title']['text'] += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['TEMPERATURE_K']['min']:.0f}-{min_max_dict['TEMPERATURE_K']['max']:.0f} K)</span>"
            )
        
        # Update layout with enhanced styling
        fig.update_layout({
            'title': {
                'text': f'<b>{ternary_title}</b>',
                'font': {'size': font_size+4},
                'x': 0.5,
                'xanchor': 'center'
            },
            'ternary': axis_config,
            'showlegend': False,
            'hoverlabel': {
                'font': {'size': font_size},
                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                'bordercolor': '#AAA'
            },
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80}
        })
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Inverse scale legend
        st.subheader("Inverse Scale Legend for Normalized Ternary Plot")
        inverse_scale_data = {
            'Normalized Value': [0.0, 0.5, 1.0]
        }
        for param, label in zip(['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K'], [energy_label, width_label, temp_label]):
            min_val = min_max_dict[param]['min']
            max_val = min_max_dict[param]['max']
            inverse_scale_data[label] = [
                round(min_val, 4),
                round((min_val + max_val) / 2, 4),
                round(max_val, 4)
            ]
        inverse_scale_df = pd.DataFrame(inverse_scale_data)
        st.dataframe(inverse_scale_df, use_container_width=True)
        
        # Download ternary data
        st.download_button(
            label="Download Normalized Ternary Plot Data (CSV)",
            data=df_pivot.to_csv(index=False),
            file_name="ternary_plot_normalized.csv",
            mime="text/csv"
        )

    else:
        st.warning("Insufficient data for ternary plot. Requires interface energy, width, and temperature parameters.")

    # Ternary Distribution (Inverse Scaled)
    st.subheader("Enhanced Ternary Distribution of Inverse Scaled Parameters")
    if not ternary_df.empty:
        # Use original values for inverse scaled ternary plot
        df_inverse = ternary_df.copy()
        df_inverse['value'] = df_inverse['value'].fillna(0)
        
        # Pivot data for inverse scaled ternary plot
        df_inverse_pivot = df_inverse.pivot_table(
            index='paper_id',
            columns='entity_label',
            values='value',
            aggfunc='mean'
        ).reset_index().fillna(0)
        
        # Rename columns for ternary plot
        df_inverse_pivot = df_inverse_pivot.rename(columns={
            'INTERFACE_ENERGY': 'Energy',
            'INTERFACE_WIDTH': 'Width',
            'TEMPERATURE_K': 'Temperature'
        })
        
        # Normalize to sum to 1 for ternary plot
        df_inverse_pivot['sum'] = df_inverse_pivot['Energy'] + df_inverse_pivot['Width'] + df_inverse_pivot['Temperature']
        df_inverse_pivot['Energy'] = df_inverse_pivot['Energy'] / df_inverse_pivot['sum'].replace(0, 1)
        df_inverse_pivot['Width'] = df_inverse_pivot['Width'] / df_inverse_pivot['sum'].replace(0, 1)
        df_inverse_pivot['Temperature'] = df_inverse_pivot['Temperature'] / df_inverse_pivot['sum'].replace(0, 1)
        
        # Create enhanced ternary plot
        fig_inverse = go.Figure()
        
        # Create hover text with original values
        hover_text = []
        for _, row in df_inverse_pivot.iterrows():
            paper_id = row['paper_id']
            orig_values = []
            for param in ['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K']:
                orig_val = df_inverse[(df_inverse['paper_id'] == paper_id) & 
                                     (df_inverse['entity_label'] == param)]['value'].mean()
                unit = "J/m²" if param == "INTERFACE_ENERGY" else "nm" if param == "INTERFACE_WIDTH" else "K"
                orig_values.append(f"{param.split('_')[-1]}: {orig_val:.4f} {unit}")
            hover_text.append("<br>".join(orig_values))
        
        # Convert Matplotlib colormap to Plotly colorscale for categorical colormaps
        if ternary_colormap in ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2']:
            cmap_ternary = plt.get_cmap(ternary_colormap)
            colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b, _ in cmap_ternary(np.linspace(0, 1, cmap_ternary.N))]
            colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
        else:
            colorscale = ternary_colormap
        
        # Add scatter trace with enhanced styling
        fig_inverse.add_trace(go.Scatterternary({
            'a': df_inverse_pivot['Energy'],
            'b': df_inverse_pivot['Width'],
            'c': df_inverse_pivot['Temperature'],
            'mode': 'markers',
            'marker': {
                'size': ternary_marker_size * marker_scale,
                'color': df_inverse_pivot.index,
                'colorscale': colorscale,
                'opacity': alpha,
                'line': {
                    'width': ternary_grid_thickness/2,
                    'color': ternary_grid_color
                }
            },
            'text': hover_text,
            'hoverinfo': 'text+a+b+c',
            'hovertemplate': (
                "<b>Paper: %{text}</b><br><br>" +
                "Normalized Proportions:<br>" +
                "Energy: %{a:.3f}<br>" +
                "Width: %{b:.3f}<br>" +
                "Temperature: %{c:.3f}<extra></extra>"
            )
        }))
        
        # Create axis labels with original ranges
        energy_axis_label = f"<b>{energy_label}</b>"
        width_axis_label = f"<b>{width_label}</b>"
        temp_axis_label = f"<b>{temp_label}</b>"
        
        # Calculate tick values in physical units (approximation)
        tickvals = [0.0, 0.5, 1.0]
        energy_ticks = [
            min_max_dict['INTERFACE_ENERGY']['min'],
            (min_max_dict['INTERFACE_ENERGY']['min'] + min_max_dict['INTERFACE_ENERGY']['max']) / 2,
            min_max_dict['INTERFACE_ENERGY']['max']
        ]
        width_ticks = [
            min_max_dict['INTERFACE_WIDTH']['min'],
            (min_max_dict['INTERFACE_WIDTH']['min'] + min_max_dict['INTERFACE_WIDTH']['max']) / 2,
            min_max_dict['INTERFACE_WIDTH']['max']
        ]
        temp_ticks = [
            min_max_dict['TEMPERATURE_K']['min'],
            (min_max_dict['TEMPERATURE_K']['min'] + min_max_dict['TEMPERATURE_K']['max']) / 2,
            min_max_dict['TEMPERATURE_K']['max']
        ]
        
        # Format tick labels
        energy_ticktext = [f"{val:.3f} J/m²" for val in energy_ticks]
        width_ticktext = [f"{val:.3f} nm" for val in width_ticks]
        temp_ticktext = [f"{val:.0f} K" for val in temp_ticks]
        
        # Add original scale indicators if requested
        if show_original_scale:
            energy_axis_label += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['INTERFACE_ENERGY']['min']:.2f}-{min_max_dict['INTERFACE_ENERGY']['max']:.2f} J/m²)</span>"
            )
            width_axis_label += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['INTERFACE_WIDTH']['min']:.2f}-{min_max_dict['INTERFACE_WIDTH']['max']:.2f} nm)</span>"
            )
            temp_axis_label += (
                f"<br><span style='font-size:{font_size-2}px'>" +
                f"({min_max_dict['TEMPERATURE_K']['min']:.0f}-{min_max_dict['TEMPERATURE_K']['max']:.0f} K)</span>"
            )
        
        # Enhanced axis configuration
        axis_config = {
            'aaxis': {
                'title': {
                    'text': energy_axis_label,
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01,
                'tickvals': tickvals,
                'ticktext': energy_ticktext
            },
            'baxis': {
                'title': {
                    'text': width_axis_label,
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01,
                'tickvals': tickvals,
                'ticktext': width_ticktext
            },
            'caxis': {
                'title': {
                    'text': temp_axis_label,
                    'font': {'size': label_font_size}
                },
                'tickfont': {'size': font_size-2},
                'gridcolor': ternary_grid_color,
                'linewidth': ternary_grid_thickness,
                'min': 0.01,
                'tickvals': tickvals,
                'ticktext': temp_ticktext
            }
        }
        
        # Update layout with enhanced styling
        fig_inverse.update_layout({
            'title': {
                'text': f'<b>{ternary_inverse_title}</b>',
                'font': {'size': font_size+4},
                'x': 0.5,
                'xanchor': 'center'
            },
            'ternary': axis_config,
            'showlegend': False,
            'hoverlabel': {
                'font': {'size': font_size},
                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                'bordercolor': '#AAA'
            },
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80}
        })
        
        st.plotly_chart(fig_inverse, use_container_width=True)
        
        # Inverse scale legend
        st.subheader("Inverse Scale Legend for Inverse Scaled Ternary Plot")
        inverse_scale_data = {
            'Normalized Proportion': [0.0, 0.5, 1.0]
        }
        for param, label in zip(['INTERFACE_ENERGY', 'INTERFACE_WIDTH', 'TEMPERATURE_K'], [energy_label, width_label, temp_label]):
            min_val = min_max_dict[param]['min']
            max_val = min_max_dict[param]['max']
            inverse_scale_data[label] = [
                round(min_val, 4),
                round((min_val + max_val) / 2, 4),
                round(max_val, 4)
            ]
        inverse_scale_df = pd.DataFrame(inverse_scale_data)
        st.dataframe(inverse_scale_df, use_container_width=True)
        
        # Download inverse scaled ternary data
        st.download_button(
            label="Download Inverse Scaled Ternary Plot Data (CSV)",
            data=df_inverse_pivot.to_csv(index=False),
            file_name="ternary_plot_inverse_scaled.csv",
            mime="text/csv"
        )
    else:
        st.warning("Insufficient data for ternary plots. Requires data for interface energy, interface width, and temperature.")

    # Correlation Analysis
    if not energy_df.empty and not temp_df.empty and not width_df.empty:
        st.subheader("Correlation Analysis")
        if 'energy_scatter_df' in locals() and not energy_scatter_df.empty:
            energy_temp_data = energy_scatter_df[["Interface Energy (J/m²)", "Temperature (K)"]].dropna()
            if len(energy_temp_data) > 1:
                corr, p_value = pearsonr(energy_temp_data["Interface Energy (J/m²)"], energy_temp_data["Temperature (K)"])
                st.write(f"Pearson Correlation (Interface Energy vs Temperature): {corr:.3f} (p-value: {p_value:.3f})")
        if 'width_scatter_df' in locals() and not width_scatter_df.empty:
            width_temp_data = width_scatter_df[["Interface Width (nm)", "Temperature (K)"]].dropna()
            if len(width_temp_data) > 1:
                corr, p_value = pearsonr(width_temp_data["Interface Width (nm)"], width_temp_data["Temperature (K)"])
                st.write(f"Pearson Correlation (Interface Width vs Temperature): {corr:.3f} (p-value: {p_value:.3f})")

    # Summary
    st.write(f"**Summary**: Loaded {len(filtered_df)} parameters, including {len(filtered_df[filtered_df['entity_label'] == 'INTERFACE_ENERGY'])} interface energy, {len(filtered_df[filtered_df['entity_label'] == 'INTERFACE_WIDTH'])} interface width, and {len(filtered_df[filtered_df['entity_label'] == 'TEMPERATURE_K'])} temperature parameters.")

# Main app
st.header("Visualize NER Results")
st.markdown("Upload a `.h5`, `.pkl`, or `.pt` file containing NER results for phase field parameters. Customize visualizations and download data/plots.")

uploaded_file = st.file_uploader("Upload NER Results File (.h5, .pkl, or .pt)", type=["h5", "pkl", "pt"])

if uploaded_file:
    df, pmi_results = load_ner_data(uploaded_file)
    if df is not None and not df.empty:
        visualize_ner_results(df, pmi_results)
    else:
        st.error("Failed to load valid data. Check the file format and content.")

# Footer
st.markdown("---")
st.write("Developed for visualizing NER results for phase field model parameters in Li/Sn-based battery anode systems.")
st.markdown("""
**How to Run**:
1. Install dependencies: `pip install pandas streamlit matplotlib seaborn numpy plotly h5py torch scipy`
2. Save this code as `ner_visualizations.py`.
3. Run with: `streamlit run ner_visualizations.py --server.fileWatcherType none`
4. Upload a `.h5`, `.pkl`, or `.pt` file generated by an NER tool (e.g., phase_field_ner_analysis.py).
""")
