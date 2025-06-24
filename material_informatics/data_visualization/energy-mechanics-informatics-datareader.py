import os
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import logging
import tempfile
import pickle
import glob
import uuid
from wordcloud import WordCloud
from itertools import combinations

# Configure Matplotlib with default settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.5,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.transparent': True
})

# Configure logging
DB_DIR = tempfile.gettempdir()
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'streamlit_viz.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analysis Visualization Tool", layout="wide")
st.title("Visualization for Lithium-Ion Battery Mechanics Analysis")
st.markdown("""
Upload a .h5, .pkl, or .pt file or select an existing results file to visualize database inspection, common terms, and NER analysis results.
Adjust visualization parameters for publication-quality plots.
""")

# Session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logger.info(message)

def load_results(file_path=None, uploaded_file=None):
    try:
        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            temp_file_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_name = uploaded_file.name
        else:
            file_ext = os.path.splitext(file_path)[1].lower()
            temp_file_path = file_path
            file_name = os.path.basename(file_path)

        results = {}
        if file_ext == ".h5":
            with pd.HDFStore(temp_file_path, mode="r") as store:
                available_keys = store.keys()
                expected_keys = [
                    "inspection_tables", "inspection_schema", "inspection_sample_rows",
                    "inspection_total_papers", "inspection_term_counts", "common_terms", "ner_results"
                ]
                for key in expected_keys:
                    hdf_key = f"/{key}"
                    if hdf_key in available_keys:
                        results[key] = store.get(hdf_key)
                    else:
                        results[key] = pd.DataFrame()
                        update_log(f"Key {hdf_key} not found in .h5 file, returning empty DataFrame")
            update_log(f"Loaded results from {file_name} (.h5)")
        elif file_ext == ".pkl":
            with open(temp_file_path, "rb") as f:
                results = pickle.load(f)
            update_log(f"Loaded results from {file_name} (.pkl)")
        elif file_ext == ".pt":
            pt_data = torch.load(temp_file_path)
            for key, data in pt_data.items():
                if not data:
                    results[key] = pd.DataFrame()
                else:
                    df_data = {}
                    for col, values in data.items():
                        if isinstance(values, torch.Tensor) and values.dtype in [torch.int64, torch.float32]:
                            df_data[col] = values.numpy()
                        else:
                            df_data[col] = values
                    results[key] = pd.DataFrame(df_data)
            update_log(f"Loaded results from {file_name} (.pt)")
        else:
            raise ValueError("Unsupported file format. Use .h5, .pkl, or .pt")
        return results
    except Exception as e:
        update_log(f"Failed to load file {file_name}: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def plot_word_cloud(df, top_n, font_size, colormap, figure_width, figure_height, title_fontsize, background_color):
    try:
        term_dict = dict(zip(df["term"].head(top_n), df["frequency"].head(top_n)))
        wordcloud = WordCloud(
            width=int(figure_width * 100), height=int(figure_height * 100),
            background_color=background_color, min_font_size=8, max_font_size=font_size,
            colormap=colormap, max_words=top_n, prefer_horizontal=0.9
        ).generate_from_frequencies(term_dict)
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud of Top {top_n} Terms", fontsize=title_fontsize)
        plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception as e:
        update_log(f"Error generating word cloud: {str(e)}")
        plt.close()
        return None

@st.cache_data
def plot_term_histogram(df, top_n, figure_width, figure_height, title_fontsize, label_fontsize, edge_width, alpha):
    try:
        terms = df["term"].head(top_n).tolist()
        counts = df["frequency"].head(top_n).tolist()
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        ax.bar(terms, counts, color="skyblue", edgecolor="black", linewidth=edge_width, alpha=alpha)
        ax.set_xlabel("Terms/Phrases", fontsize=label_fontsize)
        ax.set_ylabel("Frequency", fontsize=label_fontsize)
        ax.set_title(f"Top {top_n} Terms", fontsize=title_fontsize)
        plt.xticks(rotation=45, ha="right", fontsize=label_fontsize-2)
        plt.tight_layout()
        term_df = pd.DataFrame({"term": terms, "frequency": counts})
        csv_filename = f"term_histogram_{uuid.uuid4().hex}.csv"
        csv_path = os.path.join(DB_DIR, csv_filename)
        term_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            csv_data = f.read()
        plt.close(fig)
        return fig, csv_data, csv_filename
    except Exception as e:
        update_log(f"Error generating term histogram: {str(e)}")
        plt.close()
        return None, None, None

@st.cache_data
def plot_term_co_occurrence(df, top_n, font_size, colormap, figure_width, figure_height, title_fontsize, node_size_scale, edge_width_scale):
    try:
        update_log(f"Building term co-occurrence network for top {top_n} terms")
        top_terms = df["term"].head(top_n).tolist()
        term_freqs = dict(zip(df["term"].head(top_n), df["frequency"].head(top_n)))
        G = nx.Graph()
        for term in top_terms:
            G.add_node(term, type="term", freq=term_freqs[term])
        for i in range(len(top_terms)):
            for j in range(i + 1, len(top_terms)):
                weight = 1 / (i + j + 1)
                G.add_edge(top_terms[i], top_terms[j], weight=weight)
        if G.edges():
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_sizes = [500 + node_size_scale * (G.nodes[term]["freq"] / max(term_freqs.values())) for term in G.nodes]
            node_colors = [cm.get_cmap(colormap)(i / len(top_terms)) for i in range(len(top_terms))]
            edge_widths = [edge_width_scale * G[u][v]["weight"] for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, 
                    width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"Term Co-occurrence Network (Top {top_n} Terms)", fontsize=title_fontsize)
            plt.tight_layout()
            nodes_df = pd.DataFrame([(n, d["type"]) for n, d in G.nodes(data=True)], columns=["node", "type"])
            edges_df = pd.DataFrame([(u, v, d["weight"]) for u, v, d in G.edges(data=True)], columns=["source", "target", "weight"])
            nodes_csv_filename = f"term_co_nodes_{uuid.uuid4().hex}.csv"
            edges_csv_filename = f"term_co_edges_{uuid.uuid4().hex}.csv"
            nodes_csv_path = os.path.join(DB_DIR, nodes_csv_filename)
            edges_csv_path = os.path.join(DB_DIR, edges_csv_filename)
            nodes_df.to_csv(nodes_csv_path, index=False)
            edges_df.to_csv(edges_csv_path, index=False)
            with open(nodes_csv_path, "rb") as f:
                nodes_csv_data = f.read()
            with open(edges_csv_path, "rb") as f:
                edges_csv_data = f.read()
            plt.close(fig)
            return fig, (nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename)
        update_log("No co-occurrences found")
        plt.close()
        return None, None
    except Exception as e:
        update_log(f"Error plotting term co-occurrence: {str(e)}")
        plt.close()
        return None, None

@st.cache_data
def plot_ner_histogram(df, top_n, figure_width, figure_height, title_fontsize, label_fontsize, edge_width, alpha):
    try:
        update_log(f"Building NER histogram for top {top_n} entities")
        if df.empty:
            update_log("Empty NER DataFrame")
            return None
        label_counts = df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        counts = label_counts.values
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=edge_width, alpha=alpha)
        ax.set_xlabel("Entity Labels", fontsize=label_fontsize)
        ax.set_ylabel("Frequency", fontsize=label_fontsize)
        ax.set_title(f"Histogram of Top {top_n} NER Entities", fontsize=title_fontsize)
        plt.xticks(rotation=45, ha="right", fontsize=label_fontsize-2)
        plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception as e:
        update_log(f"Error plotting NER histogram: {str(e)}")
        plt.close()
        return None

@st.cache_data
def plot_ner_co_occurrence(df, top_n, font_size, colormap, figure_width, figure_height, title_fontsize, node_size_scale, edge_width_scale):
    try:
        update_log(f"Building NER co-occurrence network for top {top_n} entities")
        G = nx.Graph()
        entity_labels = df["entity_label"].value_counts().head(top_n).index.tolist()
        for label in entity_labels:
            G.add_node(label, type="entity")
        for paper_id in df["paper_id"].unique():
            paper_df = df[df["paper_id"] == paper_id]
            terms = paper_df["entity_label"].values
            for term1, term2 in combinations(terms, 2):
                if term1 != term2:
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        if G.edges():
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_colors = [cm.get_cmap(colormap)(i / len(entity_labels)) for i in range(len(entity_labels))]
            edge_widths = [edge_width_scale * G[u][v]["weight"] for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, 
                    width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"NER Co-occurrence Network (Top {top_n} Entities)", fontsize=title_fontsize)
            plt.tight_layout()
            plt.close(fig)
            return fig
        update_log("No co-occurrences found")
        plt.close()
        return None
    except Exception as e:
        update_log(f"Error plotting NER co-occurrence: {str(e)}")
        plt.close()
        return None

@st.cache_data
def plot_ner_value_histogram(df, top_n, colormap, figure_width, figure_height, title_fontsize, label_fontsize, alpha):
    try:
        update_log(f"Building NER value histogram for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values in NER dataframe")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities with numerical values and units")
            return None
        label_counts = value_df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        for i, label in enumerate(labels):
            values = value_df[value_df["entity_label"] == label]["value"]
            unit = value_df[value_df["entity_label"] == label]["unit"].iloc[0] if not value_df[value_df["entity_label"] == label].empty else "Unknown"
            ax.hist(values, bins=10, alpha=alpha, label=f"{label} ({unit})", color=colors[i], edgecolor="black")
        ax.set_xlabel("Value", fontsize=label_fontsize)
        ax.set_ylabel("Frequency", fontsize=label_fontsize)
        ax.set_title(f"Combined Histogram of Numerical Values for Top {top_n} NER Entities", fontsize=title_fontsize)
        ax.legend(fontsize=label_fontsize-2)
        plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception as e:
        update_log(f"Error plotting NER value histogram: {str(e)}")
        plt.close()
        return None

@st.cache_data
def plot_individual_ner_value_histograms(df, colormap, figure_width, figure_height, title_fontsize, label_fontsize, alpha):
    try:
        update_log(f"Building individual NER value histograms")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values for individual histograms")
            return None, None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities with numerical values")
            return None, None
        labels = sorted(value_df["entity_label"].unique())
        figs = []
        csv_data = {}
        for label in labels:
            label_df = value_df[value_df["entity_label"] == label]
            if label_df.empty:
                update_log(f"No numerical values for label {label}")
            values = label_df["value"].values
            unit = label_df["unit"].iloc[0]
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            color = cm.get_cmap(colormap)(labels.index(label) / len(labels))
            ax.hist(values, bins=10, color=color, edgecolor="black", alpha=alpha)
            ax.set_xlabel(f"Value ({unit})", fontsize=label_fontsize)
            ax.set_ylabel("Frequency", fontsize=label_fontsize)
            ax.set_title(f"Histogram of Numerical Values for {label}", fontsize=title_fontsize)
            plt.tight_layout()
            figs.append(fig)
            hist_df = pd.DataFrame({"Value": values, "Unit": unit})
            csv_filename = f"ner_value_histogram_{label.lower()}_{uuid.uuid4().hex}.csv"
            csv_path = os.path.join(DB_DIR, csv_filename)
            hist_df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                csv_data[label] = (f.read(), csv_filename)
            update_log(f"Generated histogram for {label}")
            plt.close(fig)
        return figs, csv_data
    except Exception as e:
        update_log(f"Error plotting individual histograms: {str(e)}")
        plt.close()
        return None, None

@st.cache_data
def plot_ner_value_radial(df, top_n, colormap, figure_width, figure_height, title_fontsize, label_fontsize):
    try:
        update_log(f"Building NER value radial chart for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values for radial chart")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities for radial chart")
            return None
        label_means = value_df.groupby("entity_label").agg({"value": "mean", "unit": "first"}).reset_index()
        label_means = label_means.sort_values("value", ascending=False).head(top_n)
        labels = label_means["entity_label"].tolist()
        values = label_means["value"].tolist()
        units = label_means["unit"].tolist()
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        widths = np.array([2 * np.pi / len(labels)] * len(labels))
        fig = plt.figure(figsize=(figure_width, figure_height))
        ax = fig.add_subplot(111, projection='polar')
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        bars = ax.bar(theta, values, width=widths, color=colors, edgecolor="black")
        ax.set_xticks(theta)
        ax.set_xticklabels([f"{label} ({unit})" for label, unit in zip(labels, units)], fontsize=label_fontsize-2)
        ax.set_title(f"Radial Chart of Average Values for Top {top_n} NER Entities", fontsize=title_fontsize, pad=20)
        plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception as e:
        update_log(f"Error generating radial chart: {str(e)}")
        plt.close()
        return None

@st.cache_data
def plot_ner_value_boxplot(df, top_n, colormap, figure_width, figure_height, title_fontsize, label_fontsize):
    try:
        update_log(f"Building NER boxplot for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("No data for boxplot")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No valid entities for boxplot")
            return None
        label_counts = value_df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        data = [value_df[value_df["entity_label"] == label]["value"].values for label in labels]
        units = [value_df[value_df["entity_label"] == label]["unit"].iloc[0] if not value_df[value_df["entity_label"] == label].empty else "Unknown" for label in labels]
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        box = ax.boxplot(data, patch_artist=True, labels=[f"{label} ({unit})" for label, unit in zip(labels, units)])
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlabel("Entity Labels", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        ax.set_title(f"Box Plot of Numerical Values for Top {top_n} NER Entities", fontsize=title_fontsize)
        plt.xticks(rotation=45, ha="right", fontsize=label_fontsize-2)
        plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception as e:
        update_log(f"Error generating boxplot: {str(e)}")
        plt.close()
        return None

# Streamlit interface
st.header("Select or Upload Results File")
result_files = glob.glob(os.path.join(DB_DIR, "*.h5")) + glob.glob(os.path.join(DB_DIR, "*.pkl")) + glob.glob(os.path.join(DB_DIR, "*.pt"))
result_options = [os.path.basename(f) for f in result_files] + ["Upload a new results file"]
result_selection = st.selectbox("Select Results File", result_options, key="result_select")

uploaded_file = None
if result_selection == "Upload a new results file":
    uploaded_file = st.file_uploader("Upload Results (.h5, .pkl, or .pt)", type=["h5", "pkl", "pt"], key="result_upload")
    if uploaded_file:
        st.session_state.results = load_results(uploaded_file=uploaded_file)
        st.session_state.selected_file = uploaded_file.name
else:
    if result_selection:
        file_path = os.path.join(DB_DIR, result_selection)
        st.session_state.results = load_results(file_path=file_path)
        st.session_state.selected_file = result_selection

if st.session_state.results:
    st.success(f"Loaded results from {st.session_state.selected_file}")
    with st.sidebar:
        st.subheader("Visualization Parameters")
        top_n = st.slider("Number of Top Entities/Terms", min_value=5, max_value=30, value=10, key="top_n")
        font_size = st.slider("Word Cloud Max Font Size", min_value=20, max_value=80, value=40, key="font_size")
        network_font_size = st.slider("Network Font Size", min_value=6, max_value=12, value=8, key="network_font_size")
        figure_width = st.slider("Figure Width (inches)", min_value=4.0, max_value=12.0, value=8.0, step=0.5, key="figure_width")
        figure_height = st.slider("Figure Height (inches)", min_value=3.0, max_value=10.0, value=4.0, step=0.5, key="figure_height")
        title_fontsize = st.slider("Title Font Size", min_value=8.0, max_value=20.0, value=12.0, step=0.5, key="title_fontsize")
        label_fontsize = st.slider("Label Font Size", min_value=6.0, max_value=16.0, value=10.0, step=0.5, key="label_fontsize")
        edge_width = st.slider("Bar/Network Edge Width", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key="edge_width")
        node_size_scale = st.slider("Node Size Scale (Networks)", min_value=1000.0, max_value=5000.0, value=3000.0, step=100.0, key="node_size_scale")
        edge_width_scale = st.slider("Edge Width Scale (Networks)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key="edge_width_scale")
        alpha = st.slider("Plot Transparency (Alpha)", min_value=0.3, max_value=1.0, value=0.8, step=0.05, key="alpha")
        colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "hot", "cool", "rainbow"], key="colormap")
        background_color = st.selectbox("Word Cloud Background", ["white", "black", "lightgray"], key="background_color")

    tab1, tab2, tab3 = st.tabs(["Database Inspection", "Common Terms", "NER Analysis"])
    
    with tab1:
        st.header("Database Inspection")
        if "inspection_tables" in st.session_state.results:
            st.subheader("Tables")
            st.write(st.session_state.results["inspection_tables"]["table_name"].tolist())
        if "inspection_schema" in st.session_state.results:
            st.subheader("Schema of 'papers' Table")
            st.dataframe(st.session_state.results["inspection_schema"], use_container_width=True)
        if "inspection_sample_rows" in st.session_state.results:
            st.subheader("Sample Rows")
            st.dataframe(st.session_state.results["inspection_sample_rows"], use_container_width=True)
        if "inspection_total_papers" in st.session_state.results:
            st.subheader("Total Papers")
            st.write(st.session_state.results["inspection_total_papers"]["total_papers"].iloc[0])
        if "inspection_term_counts" in st.session_state.results:
            st.subheader("Term Frequencies")
            st.dataframe(st.session_state.results["inspection_term_counts"], use_container_width=True)
    
    with tab2:
        st.header("Common Terms")
        if "common_terms" in st.session_state.results and not st.session_state.results["common_terms"].empty:
            st.dataframe(st.session_state.results["common_terms"], use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Show Term Histogram"):
                    fig_hist, csv_data, csv_filename = plot_term_histogram(
                        st.session_state.results["common_terms"], top_n, figure_width, figure_height,
                        title_fontsize, label_fontsize, edge_width, alpha
                    )
                    if fig_hist:
                        st.pyplot(fig_hist)
                        st.download_button(
                            label="Download Term Histogram Data",
                            data=csv_data,
                            file_name=csv_filename,
                            mime="text/csv",
                            key="download_term_histogram"
                        )
            with col2:
                if st.checkbox("Show Word Cloud"):
                    fig_cloud = plot_word_cloud(
                        st.session_state.results["common_terms"], top_n, font_size, colormap,
                        figure_width, figure_height, title_fontsize, background_color
                    )
                    if fig_cloud:
                        st.pyplot(fig_cloud)
            if st.checkbox("Show Term Co-occurrence Network"):
                fig_net, net_csv = plot_term_co_occurrence(
                    st.session_state.results["common_terms"], top_n, network_font_size, colormap,
                    figure_width, figure_height, title_fontsize, node_size_scale, edge_width_scale
                )
                if fig_net:
                    st.pyplot(fig_net)
                    if net_csv:
                        nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename = net_csv
                        st.download_button(
                            label="Download Nodes",
                            data=nodes_csv_data,
                            file_name=nodes_csv_filename,
                            mime="text/csv",
                            key="download_term_co_nodes"
                        )
                        st.download_button(
                            label="Download Edges",
                            data=edges_csv_data,
                            file_name=edges_csv_filename,
                            mime="text/csv",
                            key="download_term_co_edges"
                        )
    
    with tab3:
        st.header("NER Analysis")
        if "ner_results" in st.session_state.results and not st.session_state.results["ner_results"].empty:
            st.dataframe(st.session_state.results["ner_results"].head(100), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Show NER Co-occurrence Network"):
                    fig_net = plot_ner_co_occurrence(
                        st.session_state.results["ner_results"], top_n, network_font_size, colormap,
                        figure_width, figure_height, title_fontsize, node_size_scale, edge_width_scale
                    )
                    if fig_net:
                        st.pyplot(fig_net)
            with col2:
                if st.checkbox("Show NER Frequency Histogram"):
                    fig_hist = plot_ner_histogram(
                        st.session_state.results["ner_results"], top_n, figure_width, figure_height,
                        title_fontsize, label_fontsize, edge_width, alpha
                    )
                    if fig_hist:
                        st.pyplot(fig_hist)
            if st.checkbox("Show Individual NER Value Histograms"):
                figs_hist, csv_hist = plot_individual_ner_value_histograms(
                    st.session_state.results["ner_results"], colormap, figure_width, figure_height,
                    title_fontsize, label_fontsize, alpha
                )
                if figs_hist:
                    for i, fig in enumerate(figs_hist):
                        st.pyplot(fig)
                        label = sorted(csv_hist.keys())[i]
                        csv_data, csv_filename = csv_hist[label]
                        st.download_button(
                            label=f"Download {label} Histogram Data",
                            data=csv_data,
                            file_name=csv_filename,
                            mime="text/csv",
                            key=f"download_ner_hist_{label}"
                        )
            col3, col4 = st.columns(2)
            with col3:
                if st.checkbox("Show Combined NER Value Histogram"):
                    fig_value_hist = plot_ner_value_histogram(
                        st.session_state.results["ner_results"], top_n, colormap, figure_width, figure_height,
                        title_fontsize, label_fontsize, alpha
                    )
                    if fig_value_hist:
                        st.pyplot(fig_value_hist)
            with col4:
                if st.checkbox("Show NER Radial Chart"):
                    fig_radial = plot_ner_value_radial(
                        st.session_state.results["ner_results"], top_n, colormap, figure_width, figure_height,
                        title_fontsize, label_fontsize
                    )
                    if fig_radial:
                        st.pyplot(fig_radial)
            if st.checkbox("Show NER Box Plot"):
                fig_box = plot_ner_value_boxplot(
                    st.session_state.results["ner_results"], top_n, colormap, figure_width, figure_height,
                    title_fontsize, label_fontsize
                )
                if fig_box:
                    st.pyplot(fig_box)
    
    st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150)
else:
    st.warning("Please select or upload a results file (.h5, .pkl, or .pt).")
