import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import pickle
import torch
import re
import os
from spacy.language import Language
from spacy.tokens import Doc
from collections import Counter
from math import log2
import spacy
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr

# Set Matplotlib to use a non-interactive backend for Streamlit
matplotlib.use('Agg')

# Initialize logging
logging.basicConfig(filename='phase_field_ner_analysis.log', level=logging.DEBUG)

# Initialize Streamlit app
st.set_page_config(page_title="Phase Field Model Parameter Analysis Tool", layout="wide")
st.title("Phase Field Model Parameter Analysis for Li/Sn-based Battery Anodes")
st.markdown("""
This tool extracts phase field model parameters (interface energy in J/m², interface width in nm, temperature in K) from scientific papers stored in `phase_field_knowledgeuniverse.db` or saved NER results in `.pkl` files. It uses regex-based NER with SpaCy for material detection and Pointwise Mutual Information (PMI) to identify significant phrases related to phase field modeling for Li, Sn, or Li-Sn systems. Use the **NER Analysis** tab to process the database or a `.pkl` file, and the **Visualize Results** tab to load and visualize results with customizable, publication-quality Matplotlib plots.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `sqlite3`, `pandas`, `streamlit`, `matplotlib`, `numpy`, `spacy`, `fuzzywuzzy`, `python-Levenshtein`, `h5py`, `torch`, `scipy`
- Install with: `pip install pandas streamlit matplotlib numpy spacy fuzzywuzzy python-Levenshtein h5py torch scipy`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Tabs for NER Analysis and Visualization
tab1, tab2 = st.tabs(["NER Analysis", "Visualize Results"])

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
        st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
    except Exception as e2:
        st.error(f"Failed to load SpaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
        st.stop()

# Define entity types and default colors
param_types = ["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"]
default_colors = {
    "MATERIAL": '#9467bd',              # Purple
    "INTERFACE_ENERGY": '#1f77b4',      # Blue
    "INTERFACE_WIDTH": '#ff7f0e',       # Orange
    "TEMPERATURE_K": '#2ca02c'          # Green
}
logging.info(f"Default colors: {default_colors}")

# Parameter validation ranges (in SI units)
valid_ranges = {
    "INTERFACE_ENERGY": (1e-3, 10, "J/m²"),  # 1 mJ/m² to 10 J/m²
    "INTERFACE_WIDTH": (0.1, 500, "nm"),    # 0.1 nm to 1 µm
    "TEMPERATURE_K": (200, 350, "K")        # Realistic range for battery anode modeling
}

# PMI calculation for phase field context
def calculate_pmi(text, window_size=5, min_count=2):
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        bigram_counts = Counter()
        total_words = len(words)
        total_bigrams = 0
        
        for i in range(len(words) - 1):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                bigram = (words[i], words[j])
                bigram_counts[bigram] += 1
                total_bigrams += 1
        
        pmi_scores = {}
        for (w1, w2), count in bigram_counts.items():
            if count >= min_count:
                p_w1 = word_counts[w1] / total_words
                p_w2 = word_counts[w2] / total_words
                p_w1_w2 = count / total_bigrams
                if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
                    pmi = log2(p_w1_w2 / (p_w1 * p_w2))
                    pmi_scores[f"{w1} {w2}"] = pmi
        
        phase_field_phrases = [
            "phase field", "interface energy", "interface width", "diffuse interface",
            "lithium anode", "tin anode", "li sn", "battery anode", "phase field model",
            "surface energy", "interfacial energy", "sigma", "gamma"
        ]
        relevant_phrases = {phrase: score for phrase, score in pmi_scores.items() if phrase in phase_field_phrases and score > 0}
        logging.info(f"PMI phrases: {relevant_phrases}")
        return relevant_phrases
    except Exception as e:
        logging.error(f"PMI calculation failed: {str(e)}")
        return {}

# Extract parameters
def extract_parameters(text, paper_id, title, year):
    try:
        pmi_phrases = calculate_pmi(text)
        context_terms = list(pmi_phrases.keys()) + [
            "phase field", "interface energy", "interface width", "diffuse interface",
            "lithium", "tin", "li sn", "battery anode", "surface energy", "interfacial energy",
            "sigma", "gamma", "γ", "σ"
        ]
        
        doc = nlp(text)
        entities = []
        
        # Material detection (Li, Sn, or Li-Sn systems)
        for ent in doc.ents:
            if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or any(fuzz.partial_ratio(term, ent.text.lower()) > 75 for term in ["li ", "sn ", "lithium", "tin", "li-sn", "li sn", "lithium tin"]):
                entities.append({
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "entity_text": ent.text,
                    "entity_label": "MATERIAL",
                    "value": None,
                    "unit": None,
                    "outcome": None,
                    "context": text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)].replace("\n", " ")
                })
        
        # Expanded regex patterns for phase field parameters
        patterns = [
            # Interface energy (expanded units and contexts: J/m², mJ/m², erg/cm², eV/Å², N/m, etc.)
            (r"(-?\d+\.?\d*)\s*(J/m2|J m2|J/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x),
            (r"(-?\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*(erg/cm2|erg cm2|erg/cm²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*(eV/Å2|eV Å2|eV/Å²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1.602e-19 / 1e-20),  # 1 eV/Å² ≈ 16 J/m²
            (r"(-?\d+\.?\d*)\s*(N/m|N m-1|N/m)", "INTERFACE_ENERGY", "J/m²", lambda x: x),  # 1 N/m = 1 J/m²
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(J/m2|J m2|J/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(erg/cm2|erg cm2|erg/cm²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(eV/Å2|eV Å2|eV/Å²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1.602e-19 / 1e-20),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(N/m|N m-1|N/m)", "INTERFACE_ENERGY", "J/m²", lambda x: x),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(J/m2|J m2|J/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(erg/cm2|erg cm2|erg/cm²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1e-3),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(eV/Å2|eV Å2|eV/Å²)", "INTERFACE_ENERGY", "J/m²", lambda x: x * 1.602e-19 / 1e-20),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(N/m|N m-1|N/m)", "INTERFACE_ENERGY", "J/m²", lambda x: x),
            # Interface width (expanded units: m, cm, mm, pm, etc.)
            (r"(\d+\.?\d*)\s*(nm|nanometer|nanometers)", "INTERFACE_WIDTH", "nm", lambda x: x),
            (r"(\d+\.?\d*)\s*(Å|angstrom|angstroms)", "INTERFACE_WIDTH", "nm", lambda x: x * 0.1),
            (r"(\d+\.?\d*)\s*(µm|micrometer|micrometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e3),
            (r"(\d+\.?\d*)\s*(m|meter|meters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e9),
            (r"(\d+\.?\d*)\s*(cm|centimeter|centimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e7),
            (r"(\d+\.?\d*)\s*(mm|millimeter|millimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e6),
            (r"(\d+\.?\d*)\s*(pm|picometer|picometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e-3),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(nm|nanometer|nanometers)", "INTERFACE_WIDTH", "nm", lambda x: x),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(Å|angstrom|angstroms)", "INTERFACE_WIDTH", "nm", lambda x: x * 0.1),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(µm|micrometer|micrometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e3),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(m|meter|meters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e9),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(cm|centimeter|centimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e7),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(mm|millimeter|millimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e6),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(pm|picometer|picometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e-3),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(nm|nanometer|nanometers)", "INTERFACE_WIDTH", "nm", lambda x: x),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(Å|angstrom|angstroms)", "INTERFACE_WIDTH", "nm", lambda x: x * 0.1),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(µm|micrometer|micrometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e3),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(m|meter|meters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e9),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(cm|centimeter|centimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e7),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(mm|millimeter|millimeters)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e6),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(pm|picometer|picometers)", "INTERFACE_WIDTH", "nm", lambda x: x * 1e-3),
            # Temperature (expanded units: °F, mK, etc.)
            (r"(\d+\.?\d*)\s*(K|kelvin)", "TEMPERATURE_K", "K", lambda x: x),
            (r"(\d+\.?\d*)\s*(°C|○C|celsius)", "TEMPERATURE_K", "K", lambda x: x + 273.15),
            (r"(\d+\.?\d*)\s*(°F|fahrenheit)", "TEMPERATURE_K", "K", lambda x: (x - 32) * 5/9 + 273.15),
            (r"(\d+\.?\d*)\s*(mK|millikelvin)", "TEMPERATURE_K", "K", lambda x: x * 1e-3),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(K|kelvin)", "TEMPERATURE_K", "K", lambda x: x),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(°C|○C|celsius)", "TEMPERATURE_K", "K", lambda x: x + 273.15),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(°F|fahrenheit)", "TEMPERATURE_K", "K", lambda x: (x - 32) * 5/9 + 273.15),
            (r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*(mK|millikelvin)", "TEMPERATURE_K", "K", lambda x: x * 1e-3),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(K|kelvin)", "TEMPERATURE_K", "K", lambda x: x),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(°C|○C|celsius)", "TEMPERATURE_K", "K", lambda x: x + 273.15),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(°F|fahrenheit)", "TEMPERATURE_K", "K", lambda x: (x - 32) * 5/9 + 273.15),
            (r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(mK|millikelvin)", "TEMPERATURE_K", "K", lambda x: x * 1e-3)
        ]
        
        for pattern, label, unit, convert in patterns:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                context_lower = context.lower()
                
                # Ensure context is relevant to phase field or Li/Sn
                if label in ["INTERFACE_ENERGY", "INTERFACE_WIDTH"] and not any(term in context_lower for term in context_terms):
                    logging.debug(f"Skipping entity {match.group(0)}: no relevant phase field context")
                    continue
                
                if "to" in pattern:
                    start_val = convert(float(match.group(1)))
                    end_val = convert(float(match.group(2)))
                    if valid_ranges[label][0] <= start_val <= valid_ranges[label][1] and valid_ranges[label][0] <= end_val <= valid_ranges[label][1]:
                        for val in np.linspace(start_val, end_val, 5):
                            entities.append({
                                "paper_id": paper_id,
                                "title": title,
                                "year": year,
                                "entity_text": f"{start_val} to {end_val}",
                                "entity_label": label,
                                "value": val,
                                "unit": unit,
                                "outcome": None,
                                "context": context.replace("\n", " ")
                            })
                elif "±" in pattern:
                    value = convert(float(match.group(1)))
                    uncertainty = convert(float(match.group(2)))
                    if valid_ranges[label][0] <= value <= valid_ranges[label][1]:
                        for val, val_type in [(value, "Central"), (value - uncertainty, "Lower"), (value + uncertainty, "Upper")]:
                            if valid_ranges[label][0] <= val <= valid_ranges[label][1]:
                                entities.append({
                                    "paper_id": paper_id,
                                    "title": title,
                                    "year": year,
                                    "entity_text": f"{value} ± {uncertainty}",
                                    "entity_label": label,
                                    "value": val,
                                    "unit": unit,
                                    "outcome": None,
                                    "context": context.replace("\n", " ")
                                })
                else:
                    value = convert(float(match.group(1)))
                    if valid_ranges[label][0] <= value <= valid_ranges[label][1]:
                        outcome = None
                        outcome_terms = ["capacity", "conductivity", "diffusion", "stability", "cycle life"]
                        for term in outcome_terms:
                            if term in context_lower:
                                outcome = term
                                break
                        entities.append({
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "entity_text": match.group(0),
                            "entity_label": label,
                            "value": value,
                            "unit": unit,
                            "outcome": outcome,
                            "context": context.replace("\n", " ")
                        })
                        logging.debug(f"Extracted entity: {match.group(0)}, label: {label}, value: {value}, unit: {unit}")
        
        return entities, pmi_phrases
    except Exception as e:
        logging.error(f"NER failed for paper {paper_id}: {str(e)}")
        return [{"paper_id": paper_id, "title": title, "year": year, "entity_text": f"Error: {str(e)}", "entity_label": "ERROR", "value": None, "unit": None, "outcome": None, "context": ""}], {}

# Process SQLite database
def process_sqlite(db_file):
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql("SELECT * FROM full_text WHERE full_text IS NOT NULL", conn)
        conn.close()
        
        results = []
        pmi_results = {}
        relevant_entries = 0
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            text = row["full_text"]
            if not any(term.lower() in text.lower() for term in ["lithium", "tin", "li ", "sn ", "li-sn", "battery anode", "phase field"]):
                logging.debug(f"Skipping paper {row['paper_id']}: no relevant terms in full text")
                continue
            relevant_entries += 1
            entities, pmi_phrases = extract_parameters(text, row["paper_id"], row["title"], row["year"])
            results.extend(entities)
            pmi_results[row["paper_id"]] = pmi_phrases
            progress_bar.progress((i + 1) / len(df))
        
        st.info(f"Processed {relevant_entries} relevant full-text entries from {db_file}.")
        return pd.DataFrame(results), pmi_results
    except Exception as e:
        st.error(f"Error processing {db_file}: {str(e)}")
        logging.error(f"Error processing {db_file}: {str(e)}")
        return None, {}

# Save NER results
def save_ner_results(df, base_name="phase_field_params"):
    try:
        h5_path = f"{base_name}.h5"
        df.to_hdf(h5_path, key="ner_results", mode="w")
        pkl_path = f"{base_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
        pt_path = f"{base_name}.pt"
        torch.save(df.to_dict(orient="records"), pt_path)
        logging.info(f"Saved NER results to {h5_path}, {pkl_path}, {pt_path}")
        return h5_path, pkl_path, pt_path
    except Exception as e:
        logging.error(f"Failed to save NER results: {str(e)}")
        return None, None, None

# Visualize results with enhanced Matplotlib plots
def visualize_results(df, entity_types, pmi_results):
    # Set Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sidebar for plot customization
    st.sidebar.subheader("Visualization Customization")
    font_size = st.sidebar.slider("Font Size", 8, 20, 12)
    axes_thickness = st.sidebar.slider("Axes Thickness", 0.5, 2.0, 0.8, step=0.1)
    axes_line_color = st.sidebar.color_picker("Axes Line Color", "#808080")
    marker_size = st.sidebar.slider("Scatter Marker Size", 20, 500, 100)
    alpha = st.sidebar.slider("Marker Transparency", 0.1, 1.0, 0.5, step=0.1)
    font_family = st.sidebar.selectbox("Font Family", ["Arial", "Times New Roman", "Helvetica"], index=0)
    
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'rainbow', 'jet', 'tab10', 'tab20',
        'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
        'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'hot_r', 'cool_r', 'rainbow_r', 'jet_r',
        'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'pink', 'ocean', 'terrain', 'gist_earth',
        'gist_rainbow', 'gist_heat', 'coolwarm', 'twilight', 'twilight_shifted', 'hsv', 'flag', 'prism',
        'nipy_spectral', 'gist_ncar', 'brg', 'cmr_map', 'cubehelix', 'gnuplot', 'gnuplot2', 'seismic'
    ]
    colormap = st.sidebar.selectbox("Colormap", colormaps, index=colormaps.index('tab20'))
    hist_edge_width = st.sidebar.slider("Histogram Edge Line Width", 0.5, 2.0, 1.0, step=0.1)
    xlabel_color = st.sidebar.color_picker("X-Label Color", "#000000")
    ylabel_color = st.sidebar.color_picker("Y-Label Color", "#000000")
    title_color = st.sidebar.color_picker("Title Color", "#000000")
    
    # Filters for scatter and heatmap
    selected_papers = st.sidebar.multiselect("Select Papers", df['paper_id'].unique(), default=df['paper_id'].unique())
    temp_range = st.sidebar.slider("Temperature Range (K)", 200, 2000, (200, 1000), step=10)
    energy_range = st.sidebar.slider("Interface Energy Range (J/m²)", 0.001, 10.0, (0.001, 2.0), step=0.001)
    width_range = st.sidebar.slider("Interface Width Range (nm)", 0.1, 1000.0, (0.1, 100.0), step=0.1)
    use_log_heatmap = st.sidebar.checkbox("Use Logarithmic Scale for Heatmap", value=False)
    
    # Update colors
    cmap = plt.get_cmap(colormap)
    param_colors = {param: cmap(i / len(param_types)) for i, param in enumerate(param_types)}
    if colormap == "tab20":
        param_colors = default_colors
    
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': axes_thickness,
        'xtick.major.width': axes_thickness,
        'ytick.major.width': axes_thickness,
        'text.usetex': False  # Disabled for Streamlit compatibility
    })
    
    st.subheader("Extracted Parameters")
    filtered_df = df[df['paper_id'].isin(selected_papers) & 
                    df['entity_label'].isin(entity_types) &
                    (df['value'].between(energy_range[0], energy_range[1]) | ~df['entity_label'].isin(['INTERFACE_ENERGY'])) &
                    (df['value'].between(width_range[0], width_range[1]) | ~df['entity_label'].isin(['INTERFACE_WIDTH'])) &
                    (df['value'].between(temp_range[0], temp_range[1]) | ~df['entity_label'].isin(['TEMPERATURE_K']))]
    st.dataframe(
        filtered_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
        use_container_width=True,
        column_config={
            "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
            "value": st.column_config.NumberColumn("Value", help="Numerical value (Interface Energy in J/m², Interface Width in nm, Temperature in K)."),
            "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., capacity).")
        }
    )
    
    # PMI results
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
    
    # Download filtered data
    st.subheader("Download Filtered Data")
    st.download_button(
        label="Download Filtered Parameters CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_phase_field_params.csv",
        mime="text/csv"
    )
    
    # Histograms
    st.subheader("Parameter Distribution Analysis")
    for param_type in entity_types:
        if param_type in param_types:
            param_df = filtered_df[filtered_df["entity_label"] == param_type]
            if not param_df.empty:
                values = param_df["value"].dropna()
                if not values.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for spine in ax.spines.values():
                        spine.set_color(axes_line_color)
                    bins = 30 if param_type == "INTERFACE_ENERGY" else 20
                    counts, bins, _ = ax.hist(values, bins=bins, edgecolor='black', linewidth=hist_edge_width,
                                             color=param_colors[param_type], alpha=alpha,
                                             label=param_type.replace('_', ' ').title())
                    unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                    ax.set_xlabel(f"{param_type.replace('_', ' ').title()} ({unit})", fontweight='bold', color=xlabel_color)
                    ax.set_ylabel("Count", fontweight='bold', color=ylabel_color)
                    ax.set_title(f"Distribution of {param_type.replace('_', ' ').title()}", fontweight='bold', pad=15, color=title_color)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                    plt.tight_layout()
                    st.pyplot(fig)
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
    energy_scatter_df = None  # Initialize to avoid undefined variable
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
        if scatter_data:
            energy_scatter_df = pd.DataFrame(scatter_data)
            logging.debug(f"Energy-Temperature scatter_df columns: {energy_scatter_df.columns.tolist()}")
            if len(energy_scatter_df) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                for spine in ax.spines.values():
                    spine.set_color(axes_line_color)
                paper_ids = energy_scatter_df["paper_id"].unique()
                for i, paper_id in enumerate(paper_ids):
                    paper_data = energy_scatter_df[energy_scatter_df["paper_id"] == paper_id]
                    ax.scatter(paper_data["Interface Energy (J/m²)"], paper_data["Temperature (K)"],
                               c=[cmap(i % 20)], s=marker_size, alpha=alpha, label=paper_id[:10])
                ax.set_xlabel(r"Interface Energy (J/m²)", fontweight='bold', color=xlabel_color)
                ax.set_ylabel(r"Temperature (K)", fontweight='bold', color=ylabel_color)
                ax.set_title("Interface Energy vs Temperature by Paper", fontweight='bold', pad=15, color=title_color)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=font_size - 2)
                plt.tight_layout()
                st.pyplot(fig)
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
        if scatter_data:
            width_scatter_df = pd.DataFrame(scatter_data)
            logging.debug(f"Width-Temperature scatter_df columns: {width_scatter_df.columns.tolist()}")
            if len(width_scatter_df) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                for spine in ax.spines.values():
                    spine.set_color(axes_line_color)
                paper_ids = width_scatter_df["paper_id"].unique()
                for i, paper_id in enumerate(paper_ids):
                    paper_data = width_scatter_df[width_scatter_df["paper_id"] == paper_id]
                    ax.scatter(paper_data["Interface Width (nm)"], paper_data["Temperature (K)"],
                               c=[cmap(i % 20)], s=marker_size, alpha=alpha, label=paper_id[:10])
                ax.set_xlabel(r"Interface Width (nm)", fontweight='bold', color=xlabel_color)
                ax.set_ylabel(r"Temperature (K)", fontweight='bold', color=ylabel_color)
                ax.set_title("Interface Width vs Temperature by Paper", fontweight='bold', pad=15, color=title_color)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=font_size - 2)
                plt.tight_layout()
                st.pyplot(fig)
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
        if 'energy_scatter_df' not in locals() or energy_scatter_df is None:
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
                    
                    # Create bins that span the entire selected range
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
                    
                    # Create the heatmap plot
                    im = ax.imshow(
                        heatmap.T,
                        origin='lower',
                        aspect='auto',
                        cmap=cmap,
                        interpolation='nearest',
                        extent=[temp_range[0], temp_range[1], energy_range[0], energy_range[1]]
                    )
                    
                    ax.set_xlabel(r"Temperature (K)", fontweight='bold', color=xlabel_color)
                    ax.set_ylabel(r"Interface Energy (J/m²)", fontweight='bold', color=ylabel_color)
                    ax.set_title("Density of Interface Energy vs Temperature", fontweight='bold', pad=15, color=title_color)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Count" + (" (log scale)" if use_log_heatmap else ""), rotation=270, labelpad=15)
                    
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
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
                logging.error(f"Heatmap failed: Required columns {required_cols} not in energy_scatter_df columns: {energy_scatter_df.columns.tolist()}")
        else:
            st.warning("No valid Interface Energy vs Temperature data available for heatmap.")
            logging.debug("energy_scatter_df is None or empty before heatmap.")
    
    # Correlation Analysis
    if not energy_df.empty and not temp_df.empty and not width_df.empty:
        st.subheader("Correlation Analysis")
        if energy_scatter_df is not None and not energy_scatter_df.empty:
            energy_temp_data = energy_scatter_df[["Interface Energy (J/m²)", "Temperature (K)"]].dropna()
            if len(energy_temp_data) > 1:
                energy_temp_corr, energy_temp_p = pearsonr(energy_temp_data["Interface Energy (J/m²)"], energy_temp_data["Temperature (K)"])
                st.write(f"Pearson Correlation (Interface Energy vs Temperature): {energy_temp_corr:.3f} (p-value: {energy_temp_p:.3f})")
        if 'width_scatter_df' in locals() and not width_scatter_df.empty:
            width_temp_data = width_scatter_df[["Interface Width (nm)", "Temperature (K)"]].dropna()
            if len(width_temp_data) > 1:
                width_temp_corr, width_temp_p = pearsonr(width_temp_data["Interface Width (nm)"], width_temp_data["Temperature (K)"])
                st.write(f"Pearson Correlation (Interface Width vs Temperature): {width_temp_corr:.3f} (p-value: {width_temp_p:.3f})")
    
    st.write(f"**Summary**: {len(filtered_df)} parameters loaded, including {len(filtered_df[filtered_df['entity_label'] == 'INTERFACE_ENERGY'])} interface energy, {len(filtered_df[filtered_df['entity_label'] == 'INTERFACE_WIDTH'])} interface width, and {len(filtered_df[filtered_df['entity_label'] == 'TEMPERATURE_K'])} temperature parameters.")

# --- NER Analysis Tab ---
with tab1:
    st.header("NER Analysis for Phase Field Model Parameters")
    st.markdown("Extract interface energy (J/m²), interface width (nm), and temperature (K) from `phase_field_knowledgeuniverse.db` or a `.pkl` file for Li/Sn-based battery anode systems. Results are saved as `.h5`, `.pkl`, and `.pt`.")

    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        source_type = st.selectbox(
            "Select Data Source",
            ["Full Text (phase_field_knowledgeuniverse.db)", "Saved Results (.pkl)"],
            help="Choose whether to analyze full text or load saved results."
        )
        entity_types = st.multiselect(
            "Parameter Types to Display",
            ["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"],
            default=["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"],
            help="Select parameter types to filter results."
        )
        sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
        analyze_button = st.button("Run NER Analysis")
        if source_type == "Saved Results (.pkl)":
            uploaded_file = st.file_uploader("Upload .pkl File", type=["pkl"], key="ner_pkl")

    if analyze_button:
        if source_type == "Full Text (phase_field_knowledgeuniverse.db)":
            db_file = "phase_field_knowledgeuniverse.db"
            if not os.path.exists(db_file):
                st.error(f"Database {db_file} not found. Ensure it exists in the working directory.")
            else:
                with st.spinner(f"Processing {db_file}..."):
                    df, pmi_results = process_sqlite(db_file)
        else:
            if not uploaded_file:
                st.error("Please upload a .pkl file.")
                st.stop()
            try:
                df = pd.read_pickle(uploaded_file)
                pmi_results = {}
                st.info(f"Loaded {len(df)} entities from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error loading .pkl file: {str(e)}")
                logging.error(f"Error loading .pkl file: {str(e)}")
                st.stop()
        
        if df is None or df.empty:
            st.warning(f"No parameters extracted. Ensure the source contains relevant papers mentioning Li, Sn, or phase field modeling.")
        else:
            if entity_types:
                df = df[df["entity_label"].isin(entity_types)]
            
            if sort_by == "entity_label":
                df = df.sort_values(["entity_label", "value"])
            else:
                df = df.sort_values(["value", "entity_label"], na_position="last")
            
            visualize_results(df, entity_types, pmi_results)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Phase Field Parameters CSV",
                csv,
                "phase_field_params.csv",
                "text/csv"
            )
            
            json_data = df.to_json("phase_field_params.json", orient="records", lines=True)
            with open("phase_field_params.json", "rb") as f:
                st.download_button(
                    "Download Phase Field Parameters JSON",
                    f,
                    "phase_field_params.json",
                    "application/json"
                )
            
            h5_path, pkl_path, pt_path = save_ner_results(df)
            if h5_path:
                st.info(f"Saved NER results to {h5_path}, {pkl_path}, and {pt_path}")
                for path, mime in [(h5_path, "application/x-hdf"), (pkl_path, "application/octet-stream"), (pt_path, "application/octet-stream")]:
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"Download {path}",
                            data=f,
                            file_name=path,
                            mime=mime
                        )

# --- Visualize Results Tab ---
with tab2:
    st.header("Visualize Existing NER Results")
    st.markdown("Load previously saved NER results from `.h5`, `.pkl`, or `.pt` files and visualize them with customizable, publication-quality Matplotlib plots. Adjust font size, axes thickness, axes line color, colormap, histogram edges, and label colors.")

    uploaded_file = st.file_uploader("Upload NER Results File (.h5, .pkl, or .pt)", type=["h5", "pkl", "pt"])
    
    if uploaded_file:
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
                st.stop()
            
            st.success(f"Loaded **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
            
            entity_types = st.multiselect(
                "Parameter Types to Display",
                ["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"],
                default=["MATERIAL", "INTERFACE_ENERGY", "INTERFACE_WIDTH", "TEMPERATURE_K"],
                key="viz_entity_types"
            )
            sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.", key="viz_sort_by")
            
            if entity_types:
                df = df[df["entity_label"].isin(entity_types)]
            
            if sort_by == "entity_label":
                df = df.sort_values(["entity_label", "value"])
            else:
                df = df.sort_values(["value", "entity_label"], na_position="last")
            
            visualize_results(df, entity_types, {})
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logging.error(f"Error loading visualization file: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed for phase field model parameter analysis in Li/Sn-based battery anode systems using regex-based NER and PMI.")
st.markdown("**How to Run**:")
st.markdown("""
1. Install dependencies: `pip install pandas streamlit matplotlib numpy spacy fuzzywuzzy python-Levenshtein h5py torch scipy`
2. Install SpaCy model: `python -m spacy download en_core_web_lg`
3. Save this code as `phase_field_ner_analysis.py`.
4. Run with: `streamlit run phase_field_ner_analysis.py --server.fileWatcherType none` to avoid PyTorch conflicts.
5. Place `phase_field_knowledgeuniverse.db` in the same directory or upload a `.pkl` file.
""")
