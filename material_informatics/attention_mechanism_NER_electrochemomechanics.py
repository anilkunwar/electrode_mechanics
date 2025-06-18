import arxiv
import fitz
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import sqlite3
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
from streamlit import session_state
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse

# Define database directory and file
DB_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(DB_DIR, "lithiation_knowledge.db")

# Initialize logging
logging.basicConfig(filename='lithiation_ner.log', level=logging.INFO)

# Initialize Streamlit app
st.set_page_config(page_title="Li-Sn Phase NER Tool with SciBERT", layout="wide")
st.title("Li-Sn Phase NER Tool with SciBERT for Lithium-Ion Battery Engineering")
st.markdown("""
This tool uses **SciBERT** to prioritize arXiv papers relevant to **Sn anodes**, **lithium-ion batteries**, **Li-Sn phases** (Li2Sn5, Li7Sn2, Li13Sn5), and **electro-chemo-mechanics**. It extracts parameters like voltage plateau, volumetric strain, and elastic modulus with an enhanced NER system. Features:

- **SciBERT Scoring**: Filters papers with >50% relevance probability.
- **Advanced NER**: Quantifies diverse parameters across multiple Li-Sn phases with TF-IDF-weighted patterns.
- **Interactive Visualizations**: Plotly-based histograms, pie charts, and scatter plots.
- **Multi-Phase Support**: Analyzes Li2Sn5, Li7Sn2, and Li13Sn5.

**Note**: SciBERT is not fine-tuned; fallback scoring may dominate. For optimal NER, train a custom spaCy model: [spaCy Training Guide](https://spacy.io/usage/training).
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `arxiv`, `pymupdf`, `spacy`, `pandas`, `streamlit`, `matplotlib`, `numpy`, `pyarrow`, `transformers`, `torch`, `scipy`, `plotly`
- Install with: `pip install arxiv pymupdf spacy pandas streamlit matplotlib numpy pyarrow transformers torch scipy plotly`
- For optimal NER: `python -m spacy download en_core_web_lg`
""")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
        st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
    except Exception as e2:
        st.error(f"Failed to load spaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
        st.stop()

# Load SciBERT model and tokenizer
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install with: `pip install transformers torch`")
    st.stop()

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Define units
units = {
    "VOLTAGE_PLATEAU": ["V", "volt", "volts"],
    "VOLUMETRIC_STRAIN": ["%", "dimensionless"],
    "VON_MISES_STRESS": ["MPa", "GPa", "megapascal", "gigapascal"],
    "ELASTIC_MODULUS": ["GPa", "MPa"],
    "HARDNESS": ["GPa", "MPa"],
    "DIFFUSION_COEFFICIENT": ["cm^2/s", "cm2/s", "m^2/s"],
    "FRACTURE_TOUGHNESS": ["MPa·m^0.5", "MPa m^0.5"],
    "CHEMO_MECHANICAL_STRESS": ["MPa", "GPa"],
    "DIFFUSION_BARRIER": ["eV"],
    "ELECTROCHEMICAL_CAPACITY": ["mAh/g", "mAh g^-1"],
    "LI2SN5_MATERIAL": [],
    "LI7SN2_MATERIAL": [],
    "LI13SN5_MATERIAL": [],
}

# Validation ranges
valid_ranges = {
    "VOLTAGE_PLATEAU": (0.01, 5.0, "V"),
    "VOLUMETRIC_STRAIN": (0, 500, "%"),
    "VON_MISES_STRESS": (0, 1000, "MPa"),
    "ELASTIC_MODULUS": (0, 100, "GPa"),
    "HARDNESS": (0, 10, "GPa"),
    "DIFFUSION_COEFFICIENT": (1e-10, 1e-5, "cm^2/s"),
    "FRACTURE_TOUGHNESS": (0, 10, "MPa·m^0.5"),
    "CHEMO_MECHANICAL_STRESS": (0, 1000, "MPa"),
    "DIFFUSION_BARRIER": (0, 10, "eV"),
    "ELECTROCHEMICAL_CAPACITY": (0, 2000, "mAh/g"),
}

# Expanded NER patterns
matcher = Matcher(nlp.vocab)
patterns = [
    # VOLTAGE_PLATEAU
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["v", "volt", "volts"]}}, {"LOWER": {"IN": ["plateau", "potential"]}, "OP": "?"}],
    [{"LOWER": {"IN": ["voltage", "potential"]}}, {"LOWER": "plateau"}],
    # VOLUMETRIC_STRAIN
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["%", "dimensionless"]}}, {"LOWER": {"IN": ["volumetric", "volume"]}}, {"LOWER": {"IN": ["strain", "deformation", "expansion", "swelling"]}}],
    [{"LOWER": {"IN": ["volumetric", "volume"]}}, {"LOWER": {"IN": ["strain", "expansion", "swelling"]}}],
    # VON_MISES_STRESS
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa", "megapascal", "gigapascal"]}}, {"LOWER": {"IN": ["von mises", "von-mises"]}}, {"LOWER": "stress", "OP": "?"}],
    [{"LOWER": {"IN": ["von mises", "equivalent"]}}, {"LOWER": "stress"}],
    # ELASTIC_MODULUS
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["gpa", "mpa"]}}, {"LOWER": {"IN": ["young's", "elastic"]}}, {"LOWER": "modulus"}],
    [{"LOWER": {"IN": ["young's", "elastic"]}}, {"LOWER": "modulus"}],
    # HARDNESS
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["gpa", "mpa"]}}, {"LOWER": "hardness"}],
    [{"LOWER": "hardness"}, {"LOWER": {"IN": ["vickers", "nanoindentation"]}, "OP": "?"}],
    # DIFFUSION_COEFFICIENT
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?(?:[eE][+-]?\d+)?$"}}, {"LOWER": {"IN": ["cm^2/s", "cm2/s", "m^2/s"]}}, {"LOWER": {"IN": ["diffusion", "lithium"]}}, {"LOWER": "coefficient"}],
    [{"LOWER": {"IN": ["lithium", "li"]}}, {"LOWER": "diffusion"}, {"LOWER": "coefficient"}],
    # FRACTURE_TOUGHNESS
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa·m^0.5", "mpa m^0.5"]}}, {"LOWER": {"IN": ["fracture", "toughness"]}}],
    [{"LOWER": {"IN": ["fracture", "crack"]}}, {"LOWER": "toughness"}],
    # CHEMO_MECHANICAL_STRESS
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa"]}}, {"LOWER": {"IN": ["chemo-mechanical", "chemomechanical"]}}, {"LOWER": "stress"}],
    [{"LOWER": {"IN": ["chemo-mechanical", "chemomechanical"]}}, {"LOWER": "stress"}],
    # DIFFUSION_BARRIER
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["ev"]}}, {"LOWER": {"IN": ["diffusion", "migration"]}}, {"LOWER": "barrier"}],
    [{"LOWER": {"IN": ["diffusion", "migration"]}}, {"LOWER": "barrier"}],
    # ELECTROCHEMICAL_CAPACITY
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mah/g", "mah g^-1"]}}],
    [{"LOWER": {"IN": ["electrochemical", "specific"]}}, {"LOWER": "capacity"}],
    # LI-SN PHASES
    [{"TEXT": {"REGEX": r"Li2Sn5"}}],
    [{"TEXT": {"REGEX": r"Li7Sn2"}}],
    [{"TEXT": {"REGEX": r"Li13Sn5"}}],
]

param_types = [
    "VOLTAGE_PLATEAU", "VOLUMETRIC_STRAIN", "VON_MISES_STRESS", "ELASTIC_MODULUS", "HARDNESS",
    "DIFFUSION_COEFFICIENT", "FRACTURE_TOUGHNESS", "CHEMO_MECHANICAL_STRESS", "DIFFUSION_BARRIER",
    "ELECTROCHEMICAL_CAPACITY", "LI2SN5_MATERIAL", "LI7SN2_MATERIAL", "LI13SN5_MATERIAL"
]

# Pattern weights
pattern_weights = {param: 0.7 for param in param_types}
pattern_weights.update({
    "VOLTAGE_PLATEAU": 1.0,
    "VOLUMETRIC_STRAIN": 0.9,
    "VON_MISES_STRESS": 0.8,
    "ELASTIC_MODULUS": 0.8,
    "LI2SN5_MATERIAL": 0.9,
    "ELECTROCHEMICAL_CAPACITY": 0.8,
})

for i, pattern in enumerate(patterns):
    matcher.add(f"LIXSNY_PARAM_{param_types[i % len(param_types)]}", [pattern])

# Color map for visualizations
param_colors = {param: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, param in enumerate(param_types)}

# SciBERT relevance scoring
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        attentions = outputs.attentions[-1][0, 0].numpy()
        probs = softmax(logits, axis=1)
        relevance_prob = probs[0][1]
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keywords = ["li2sn5", "lithiation", "sn anode", "li-sn", "lithium-ion", "battery", "volumetric strain", "mechanical stress", "solid mechanics"]
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw in token.lower() for kw in keywords)]
        if keyword_indices:
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1 and relevance_prob < 0.5:
                relevance_prob = min(relevance_prob + 0.2 * len(keyword_indices), 1.0)
        logging.info(f"SciBERT scored abstract: {relevance_prob:.3f} (attention boost: {len(keyword_indices)} keywords)")
        return relevance_prob
    except Exception as e:
        logging.error(f"SciBERT scoring failed: {str(e)}")
        keywords = {
            "li2sn5": 2.0, "li-sn phase": 2.0, "lithiation": 1.5, "sn anode": 1.5,
            "lithium-ion battery": 1.2, "volumetric strain": 1.2, "mechanical stress": 1.2,
            "solid mechanics": 1.0, "intermetallic compound": 1.0, "battery": 0.8
        }
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = 0.0
        matched_keywords = []
        for kw, weight in keywords.items():
            kw_lower = kw.lower()
            if kw_lower in abstract_lower:
                tf = sum(1 for _ in re.finditer(r'\b' + re.escape(kw_lower) + r'\b', abstract_lower))
                score += weight * tf / (total_words + 1e-6)
                matched_keywords.append(kw_lower)
        if matched_keywords:
            score = max(score, 0.1)
        max_possible_score = sum(keywords.values()) / 10
        relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        logging.info(f"Fallback scoring used: {relevance_prob:.3f} (matched: {', '.join(matched_keywords)})")
        return relevance_prob

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# SciBERT embedding for similarity-based matching
@st.cache_data
def get_scibert_embedding(text):
    try:
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        return embedding
    except Exception as e:
        logging.error(f"SciBERT embedding failed for '{text}': {str(e)}")
        return None

# Compute TF-IDF weights
@st.cache_data
def compute_tfidf_weights(texts, param_types):
    word_counts = Counter()
    doc_counts = Counter()
    for text in texts:
        doc = nlp(text)
        doc_words = set(token.text.lower() for token in doc if not token.is_stop)
        for param in param_types:
            param_words = param.lower().split('_')
            if any(word in doc_words for word in param_words):
                doc_counts[param] += 1
        word_counts.update(doc_words)
    total_docs = len(texts) + 1e-6
    tfidf = {param: doc_counts[param] / total_docs for param in param_types}
    max_score = max(tfidf.values()) if tfidf.values() else 1.0
    for param in tfidf:
        pattern_weights[param] = 0.5 + 0.5 * (tfidf[param] / max_score)
    return pattern_weights

# Extract value and unit
def extract_value_and_unit(doc, start, end, text, param_type):
    search_range = doc[max(0, start - 15):end + 15]
    for token in search_range:
        if token.like_num or re.match(r"^\d+\.\d+(?:[eE][+-]?\d+)?$", token.text):
            try:
                value = float(token.text)
                for i in range(1, 7):
                    try:
                        next_token = doc[token.i + i]
                        if next_token.text.lower() in units.get(param_type, []):
                            return value, next_token.text
                    except:
                        break
                for i in range(1, 7):
                    try:
                        prev_token = doc[token.i - i]
                        if prev_token.text.lower() in units.get(param_type, []):
                            return value, prev_token.text
                    except:
                        break
            except:
                continue
    context = text[max(0, doc[start].idx - 50):doc[end - 1].idx + 50]
    num_match = re.search(r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([a-zA-Z^2^3/·-]+)?", context, re.I)
    if num_match and num_match.group(2) and num_match.group(2).lower() in units.get(param_type, []):
        return float(num_match.group(1)), num_match.group(2)
    return None, None

# Contextual phase association
def is_phase_relevant(doc, start, end, phase):
    context = doc[max(0, start - 50):min(len(doc), end + 50)]
    phase_pattern = re.compile(rf'\b{re.escape(phase)}\b', re.IGNORECASE)
    return bool(phase_pattern.search(context.text))

# Enhanced NER function
def extract_lithiation_parameters(text, phase, relevance_score):
    try:
        doc = nlp(text)
        entities = []
        
        # Apply TF-IDF weights
        compute_tfidf_weights([text], param_types)
        
        # Detect material entities
        for ent in doc.ents:
            if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or any(p.lower() in ent.text.lower() for p in ["li2sn5", "li7sn2", "li13sn5"]):
                entities.append({
                    "text": ent.text,
                    "label": f"{phase.upper()}_MATERIAL",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "value": None,
                    "unit": None,
                    "context": None,
                    "phase": phase,
                    "score": relevance_score
                })
        
        # Apply custom matcher
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id].replace("LIXSNY_PARAM_", "")
            if not is_phase_relevant(doc, start, end, phase):
                continue
            value, unit = extract_value_and_unit(doc, start, end, text, label)
            
            # Unit conversions
            if unit and value is not None:
                if label == "VOLTAGE_PLATEAU" and unit.lower() in ["v", "volt", "volts"]:
                    unit = "V"
                elif label == "VOLUMETRIC_STRAIN" and unit.lower() in ["%", "dimensionless"]:
                    unit = "%"
                elif label == "ELECTROCHEMICAL_CAPACITY" and unit.lower() in ["mah/g", "mah g^-1"]:
                    unit = "mAh/g"
                elif label in ["VON_MISES_STRESS", "CHEMO_MECHANICAL_STRESS"]:
                    if unit.lower() in ["mpa", "megapascal"]:
                        unit = "MPa"
                    elif unit.lower() in ["gpa", "gigapascal"]:
                        unit = "MPa"
                        value *= 1000
                elif label == "ELASTIC_MODULUS":
                    if unit.lower() == "mpa":
                        unit = "GPa"
                        value /= 1000
                elif label == "HARDNESS" and unit.lower() == "mpa":
                    unit = "GPa"
                    value /= 1000
                elif label == "DIFFUSION_COEFFICIENT" and unit.lower() == "m^2/s":
                    unit = "cm^2/s"
                    value *= 1e4
                elif label == "FRACTURE_TOUGHNESS" and unit.lower() in ["mpa·m^0.5", "mpa m^0.5"]:
                    unit = "MPa·m^0.5"
                elif label == "DIFFUSION_BARRIER" and unit.lower() == "ev":
                    unit = "eV"
            
            # Validate ranges
            if label in valid_ranges and value is not None:
                min_val, max_val, expected_unit = valid_ranges[label]
                if not (min_val <= value <= max_val and (unit and unit.lower() in units.get(label, []) or unit is None)):
                    continue
            
            context_start = max(0, span.start_char - 100)
            context_end = min(len(text), span.end_char + 100)
            context_text = text[context_start:context_end].replace("\n", " ")
            
            entities.append({
                "text": span.text,
                "label": label,
                "start": span.start_char,
                "end": span.end_char,
                "value": value,
                "unit": unit,
                "context": context_text,
                "phase": phase,
                "score": pattern_weights.get(label, 0.7) * relevance_score
            })
        
        # Similarity-based matching
        reference_terms = {
            "VOLUMETRIC_STRAIN": ["volumetric strain", "volume strain", "lithiation-induced strain"],
            "VON_MISES_STRESS": ["von mises stress", "mechanical stress", "stress distribution"],
            "CHEMO_MECHANICAL_STRESS": ["chemo-mechanical stress", "chemomechanical stress"],
            "ELASTIC_MODULUS": ["elastic modulus", "young's modulus"],
            "DIFFUSION_COEFFICIENT": ["diffusion coefficient", "lithium diffusion"],
            "FRACTURE_TOUGHNESS": ["fracture toughness", "crack toughness"]
        }
        similarity_threshold = 0.85
        for span in doc.noun_chunks:
            if len(span.text.split()) > 5:
                continue
            span_embedding = get_scibert_embedding(span.text.lower())
            if span_embedding is None:
                continue
            for label, ref_terms in reference_terms.items():
                for ref_term in ref_terms:
                    ref_embedding = get_scibert_embedding(ref_term)
                    if ref_embedding is None:
                        continue
                    similarity = np.dot(span_embedding, ref_embedding) / (np.linalg.norm(span_embedding) * np.linalg.norm(ref_embedding))
                    if similarity > similarity_threshold and is_phase_relevant(doc, span.start, span.end, phase):
                        value, unit = extract_value_and_unit(doc, span.start, span.end, text, label)
                        if label in valid_ranges and value is not None:
                            min_val, max_val, expected_unit = valid_ranges[label]
                            if not (min_val <= value <= max_val and (unit and unit.lower() in units.get(label, []) or unit is None)):
                                continue
                        context_start = max(0, span.start_char - 100)
                        context_end = min(len(text), span.end_char + 100)
                        context_text = text[context_start:context_end].replace("\n", " ")
                        entities.append({
                            "text": span.text,
                            "label": label,
                            "start": span.start_char,
                            "end": span.end_char,
                            "value": value,
                            "unit": unit,
                            "context": context_text,
                            "phase": phase,
                            "score": similarity * relevance_score
                        })
                        logging.info(f"Similarity match: '{span.text}' as {label} (similarity: {similarity:.3f}, score: {similarity * relevance_score:.3f})")
        
        # Remove duplicates
        entities.sort(key=lambda x: x["score"], reverse=True)
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity["text"], entity["start"], entity["end"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        logging.info(f"Extracted {len(unique_entities)} entities for phase {phase}")
        return unique_entities
    except Exception as e:
        logging.error(f"NER failed: {str(e)}")
        return [{"text": f"Error: {str(e)}", "label": "ERROR", "start": 0, "end": 0, "value": None, "unit": None, "context": None, "phase": phase, "score": 0.0}]

# Save to SQLite database
def save_to_sqlite(papers_df, params_list, db_file):
    try:
        conn = sqlite3.connect(db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        return f"Saved metadata and parameters to {db_file}"
    except Exception as e:
        logging.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Save to Parquet
def save_to_parquet(df, parquet_file):
    try:
        df.to_parquet(parquet_file, index=False)
        return f"Saved metadata to {parquet_file}"
    except Exception as e:
        logging.error(f"Parquet save failed: {str(e)}")
        return f"Failed to save to Parquet: {str(e)}"

# Visualization functions
@st.cache_data
def plot_histogram(df, param_type, phase):
    param_df = df[(df["entity_label"] == param_type) & (df["phase"] == phase)]
    if not param_df.empty and not param_df["value"].dropna().empty:
        unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
        fig = px.histogram(param_df, x="value", nbins=20, title=f"Distribution of {param_type} for {phase}",
                           labels={"value": f"{param_type} ({unit})", "count": "Count"},
                           color_discrete_sequence=[param_colors[param_type]])
        return fig
    return None

@st.cache_data
def plot_pie_chart(df, phase):
    param_counts = df[df["phase"] == phase]["entity_label"].value_counts()
    if not param_counts.empty:
        fig = px.pie(names=param_counts.index, values=param_counts.values, title=f"Parameter Distribution for {phase}",
                     color=param_counts.index, color_discrete_map=param_colors)
        return fig
    return None

@st.cache_data
def plot_scatter(df, x_param, y_param, phase):
    param_df = df[(df["phase"] == phase) & (df["entity_label"].isin([x_param, y_param]))]
    if len(param_df["entity_label"].unique()) == 2:
        pivot_df = param_df.pivot_table(index="paper_id", columns="entity_label", values="value", aggfunc="first").dropna()
        if not pivot_df.empty:
            x_unit = param_df[param_df["entity_label"] == x_param]["unit"].iloc[0] if not param_df[param_df["entity_label"] == x_param]["unit"].empty else ""
            y_unit = param_df[param_df["entity_label"] == y_param]["unit"].iloc[0] if not param_df[param_df["entity_label"] == y_param]["unit"].empty else ""
            fig = px.scatter(pivot_df, x=x_param, y=y_param, title=f"{x_param} vs {y_param} for {phase}",
                             labels={x_param: f"{x_param} ({x_unit})", y_param: f"{y_param} ({y_unit})"})
            return fig
    return None

# Tabs for arXiv Query and NER Analysis
tab1, tab2 = st.tabs(["arXiv Query", "NER Analysis"])

# Initialize session state
if "vis_type" not in session_state:
    session_state.vis_type = "Histogram"

# --- arXiv Query Tab ---
with tab1:
    st.header("arXiv Query for Li2Sn5 Phase Papers with SciBERT")
    st.markdown("Search arXiv, score abstracts with SciBERT, download PDFs for papers with relevance probability > 0%, extract parameters, and save to `lithiation_knowledge.db`.")

    @st.cache_data
    def query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases):
        try:
            query_terms = set(query.strip().split())  # Deduplicate terms
            formatted_terms = []
            synonyms = {
                "li2sn5": ["li2sn5"],
                "lithiation": ["lithium alloying", "lithium insertion"],
                "tin anode": ["sn anode", "tin-based anode", "sn-based anode"],
                "lithium-tin alloy": ["li-sn alloy", "tin-based alloy"],
                "lithium-ion battery": ["li-ion battery", "lithium battery"],
                "li-sn phase": ["lithium-tin phase", "li-sn system"],
                "mechanical stress": ["mechanical strain", "stress gradient"],
                "volumetric strain": ["volume strain", "volumetric expansion"],
                "intermetallic compound": ["intermetallic phase", "intermetallic alloy"]
            }
            exact_phrases = list(set(exact_phrases))  # Deduplicate exact phrases
            
            # Build query terms
            for term in query_terms:
                if term.startswith('"') and term.endswith('"'):
                    formatted_terms.append(term.strip('"').replace(" ", "+"))
                else:
                    formatted_terms.append(term)
                    for key, syn_list in synonyms.items():
                        if term.lower() == key:
                            formatted_terms.extend(syn.replace(" ", "+") for syn in syn_list)
            
            # Remove duplicates and filter out terms already in exact phrases
            formatted_terms = list(set(formatted_terms) - set(phrase.replace(" ", "+") for phrase in exact_phrases))
            
            # Construct initial query
            api_query = "+".join(formatted_terms)
            for phrase in exact_phrases:
                api_query += f'+{urllib.parse.quote(f"{phrase}")}'
            
            # Ensure query length is under 1000 characters
            if len(api_query) > 1000:
                st.warning("Query too long; simplifying to core terms.")
                api_query = "Li2Sn5+tin+anode+lithium-ion+battery"
                exact_phrases = ["Li2Sn5"]
            
            logging.info(f"arXiv API query: {api_query}")
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=api_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            papers = []
            for result in client.results(search):
                if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                    abstract = result.summary.lower()
                    title = result.title.lower()
                    query_words = set(word.lower() for word in re.split(r'\s+|\".*?\"', query) if word and not word.startswith('"'))
                    for key, syn_list in synonyms.items():
                        if key in query_words:
                            query_words.update(syn_list)
                    matched_terms = [word for word in query_words if word in abstract or word in title]
                    if not matched_terms:
                        continue
                    relevance_prob = score_abstract_with_scibert(result.summary)
                    abstract_highlighted = abstract
                    for term in matched_terms:
                        abstract_highlighted = re.sub(r'\b{}\b'.format(re.escape(term)), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                    
                    papers.append({
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "year": result.published.year,
                        "categories": ", ".join(result.categories),
                        "abstract": abstract,
                        "abstract_highlighted": abstract_highlighted,
                        "pdf_url": result.pdf_url,
                        "download_status": "Not downloaded",
                        "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                        "relevance_prob": round(relevance_prob * 100, 2),
                        "pdf_path": None
                    })
                if len(papers) >= max_results:
                    break
            papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
            if all(p["relevance_prob"] == 0.0 for p in papers):
                st.warning("All papers scored 0.0 relevance. Check 'lithiation_ner.log' for errors or fine-tune SciBERT.")
            return papers
        except Exception as e:
            logging.error(f"arXiv query failed: {str(e)}")
            st.warning(f"Initial query failed: {str(e)}. Retrying with simpler query...")
            # Retry with a simpler query
            try:
                simple_query = "Li2Sn5+tin+anode"
                client = arxiv.Client()
                search = arxiv.Search(
                    query=simple_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending
                )
                papers = []
                for result in client.results(search):
                    if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                        abstract = result.summary.lower()
                        title = result.title.lower()
                        query_words = set(["li2sn5", "tin anode"])
                        matched_terms = [word for word in query_words if word in abstract or word in title]
                        if not matched_terms:
                            continue
                        relevance_prob = score_abstract_with_scibert(result.summary)
                        abstract_highlighted = abstract
                        for term in matched_terms:
                            abstract_highlighted = re.sub(r'\b{}\b'.format(re.escape(term)), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                        
                        papers.append({
                            "id": result.entry_id.split('/')[-1],
                            "title": result.title,
                            "year": result.published.year,
                            "categories": ", ".join(result.categories),
                            "abstract": abstract,
                            "abstract_highlighted": abstract_highlighted,
                            "pdf_url": result.pdf_url,
                            "download_status": "Not downloaded",
                            "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                            "relevance_prob": round(relevance_prob * 100, 2),
                            "pdf_path": None
                        })
                    if len(papers) >= max_results:
                        break
                papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
                if papers:
                    st.info("Retry with simplified query succeeded.")
                return papers
            except Exception as e2:
                logging.error(f"Retry query failed: {str(e2)}")
                st.error(f"Error querying arXiv with simplified query: {str(e2)}. Try manually simplifying the query.")
                return []

    @st.cache_data
    def download_pdf_and_extract(pdf_url, paper_id):
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        params = []
        try:
            urllib.request.urlretrieve(pdf_url, pdf_path)
            file_size = os.path.getsize(pdf_path) / 1024
            text = extract_text_from_pdf(pdf_path)
            if not text.startswith("Error"):
                for phase in ["Li2Sn5", "Li7Sn2", "Li13Sn5"]:
                    if phase.lower() in text.lower():
                        relevance_prob = score_abstract_with_scibert(text[:512])
                        entities = extract_lithiation_parameters(text, phase, relevance_prob)
                        for entity in entities:
                            params.append({
                                "paper_id": paper_id,
                                "entity_text": entity["text"],
                                "entity_label": entity["label"],
                                "value": entity["value"],
                                "unit": entity["unit"],
                                "context": entity["context"],
                                "phase": entity["phase"],
                                "score": entity["score"]
                            })
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, params
        except Exception as e:
            logging.error(f"PDF download failed for {paper_id}: {str(e)}")
            params.append({
                "paper_id": paper_id,
                "entity_text": f"Failed: {str(e)}",
                "entity_label": "ERROR",
                "value": None,
                "unit": None,
                "context": "",
                "phase": None,
                "score": 0.0
            })
            return f"Failed: {str(e)}", None, params

    with st.sidebar:
        st.subheader("arXiv Search Parameters")
        query_option = st.radio(
            "Select Query Type",
            ["Default Query", "Custom Query", "Suggested Queries"]
        )
        exact_phrases = []
        if query_option == "Default Query":
            query = 'Li2Sn5 OR "tin anode" OR "lithium-ion battery"'
            st.write("Using default query: **" + query + "**")
            exact_phrases = ["Li2Sn5", "tin anode"]
        elif query_option == "Custom Query":
            query = st.text_input("Enter Custom Query", value='Li2Sn5 OR "tin anode" OR "lithium-ion battery"')
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated)", value="Li2Sn5, tin anode")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
        else:
            suggested_queries = [
                'Li2Sn5 OR "tin anode" OR "lithium-ion battery"',
                '"lithium-ion battery" OR "Li-Sn alloy" OR "Li2Sn5"',
                '"electrochemical properties" OR "Li2Sn5" OR "intermetallic compound"'
            ]
            query = st.selectbox("Choose Suggested Query", suggested_queries)
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated)", value="Li2Sn5, tin anode")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
        
        default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.mes-hall", "physics.comp-ph"]
        categories = st.multiselect(
            "Select arXiv Categories",
            default_categories,
            default=default_categories
        )
        max_results = st.slider("Maximum Number of Papers", min_value=1, max_value=500, value=200)
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=1900, max_value=current_year, value=2010)
        with col2:
            end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
        output_formats = st.multiselect(
            "Select Output Formats",
            ["CSV", "SQLite (.db)", "Parquet", "JSON"],
            default=["CSV", "SQLite (.db)"]
        )
        search_button = st.button("Search arXiv")

    if search_button:
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not categories:
            st.error("Please select at least one category.")
        elif start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        else:
            with st.spinner("Querying arXiv and scoring abstracts with SciBERT..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases)
            
            if not papers:
                st.warning("No papers found. Try broadening the query or categories.")
            else:
                st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 50%...")
                relevant_papers = [p for p in papers if p["relevance_prob"] > 50.0]
                if not relevant_papers:
                    st.warning("No papers with relevance probability > 50%. Adjust query or check SciBERT setup.")
                else:
                    st.success(f"**{len(relevant_papers)}** papers have relevance > 50%. Downloading PDFs...")
                    progress_bar = st.progress(0)
                    all_params = []
                    for i, paper in enumerate(relevant_papers):
                        if paper["pdf_url"]:
                            status, pdf_path, params = download_pdf_and_extract(paper["pdf_url"], paper["id"])
                            paper["download_status"] = status
                            paper["pdf_path"] = pdf_path
                            all_params.extend(params)
                        progress_bar.progress((i + 1) / len(relevant_papers))
                        time.sleep(0.1)
                    
                    df = pd.DataFrame(relevant_papers)
                    st.subheader("Paper Details (Relevance > 50%)")
                    st.dataframe(
                        df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
                        use_container_width=True
                    )
                    
                    if "CSV" in output_formats:
                        csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                        st.download_button(
                            label="Download Paper Metadata CSV",
                            data=csv,
                            file_name="li2sn5_papers_metadata.csv",
                            mime="text/csv"
                        )
                    
                    if "SQLite (.db)" in output_formats:
                        sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), all_params, DB_FILE)
                        st.info(sqlite_status)
                    
                    if "Parquet" in output_formats:
                        parquet_status = save_to_parquet(df.drop(columns=["abstract_highlighted"]), "li2sn5_papers_metadata.parquet")
                        st.info(parquet_status)
                    
                    if "JSON" in output_formats:
                        json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                        st.download_button(
                            label="Download Paper Metadata JSON",
                            data=json_data,
                            file_name="li2sn5_papers_metadata.json",
                            mime="application/json"
                        )

# --- NER Analysis Tab ---
with tab2:
    st.header("NER Analysis for Li-Sn Phase Parameters")
    st.markdown("Analyze parameters stored in `lithiation_knowledge.db` with interactive visualizations.")

    @st.cache_data
    def validate_db(db_file):
        try:
            conn = sqlite3.connect(db_file)
            df_papers = pd.read_sql_query("SELECT * FROM papers LIMIT 1", conn)
            required_columns = ["id", "title", "year", "abstract"]
            missing_columns = [col for col in required_columns if col not in df_papers.columns]
            if missing_columns:
                conn.close()
                return False, f"Database 'papers' table missing columns: {', '.join(missing_columns)}"
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
            if not cursor.fetchone():
                conn.close()
                return False, "Database missing 'parameters' table."
            df_params = pd.read_sql_query("SELECT * FROM parameters LIMIT 1", conn)
            required_param_columns = ["paper_id", "entity_text", "entity_label", "phase", "score"]
            missing_param_cols = [col for col in required_param_columns if col not in df_params.columns]
            if missing_param_cols:
                conn.close()
                return False, f"Database 'parameters' table missing columns: {', '.join(missing_param_cols)}"
            conn.close()
            return True, "Database format is valid."
        except Exception as e:
            return False, f"Error reading database: {str(e)}"

    @st.cache_data
    def process_params_from_db(db_file):
        if not os.path.isabs(db_file):
            db_file = os.path.join(DB_DIR, db_file)
        
        if not os.path.exists(db_file):
            st.error(f"Database file {db_file} not found. Run the arXiv Query tab first.")
            return None
        
        is_valid, validation_message = validate_db(db_file)
        if not is_valid:
            st.error(validation_message)
            return None
        st.info(validation_message)
        
        try:
            conn = sqlite3.connect(db_file)
            params_df = pd.read_sql_query("SELECT * FROM parameters", conn)
            papers_df = pd.read_sql_query("SELECT id, title, year FROM papers", conn)
            conn.close()
            
            if params_df.empty:
                st.warning("No parameters found in the database.")
                return None
            
            df = pd.merge(params_df, papers_df, left_on="paper_id", right_on="id", how="inner")
            df = df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "phase", "score"]]
            return df
        except Exception as e:
            st.error(f"Error reading database: {str(e)}")
            return None

    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        db_file_input = st.text_input("SQLite Database Path", value=DB_FILE)
        phases = st.multiselect("Select Phases", ["Li2Sn5", "Li7Sn2", "Li13Sn5"], default=["Li2Sn5"])
        entity_types = st.multiselect(
            "Parameter Types to Display",
            param_types,
            default=["VOLTAGE_PLATEAU", "VOLUMETRIC_STRAIN", "VON_MISES_STRESS", "ELASTIC_MODULUS", "ELECTROCHEMICAL_CAPACITY"]
        )
        session_state.vis_type = st.selectbox("Visualization Type", ["Histogram", "Pie Chart", "Scatter Plot"], index=["Histogram", "Pie Chart", "Scatter Plot"].index(session_state.vis_type if "vis_type" in session_state else "Histogram"))
        if session_state.vis_type == "Scatter Plot":
            x_param = st.selectbox("X-Axis Parameter", entity_types)
            y_param = st.selectbox("Y-Axis Parameter", entity_types)
        sort_by = st.selectbox("Sort By", ["entity_label", "value", "score"])
        analyze_button = st.button("Run NER Analysis")

    if analyze_button:
        if not db_file_input:
            st.error("Please specify the SQLite database file.")
        else:
            with st.spinner("Processing parameters from database..."):
                df = process_params_from_db(db_file_input)
            
            if df is None or df.empty:
                st.warning("No parameters extracted. Run the arXiv Query tab first.")
            else:
                df = df[df["phase"].isin(phases)]
                if entity_types:
                    df = df[df["entity_label"].isin(entity_types)]
                
                if df.empty:
                    st.warning("No parameters found for selected phases. Try querying more papers.")
                else:
                    st.success(f"Extracted **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                    
                    if sort_by == "entity_label":
                        df = df.sort_values(["entity_label", "value"])
                    elif sort_by == "value":
                        df = df.sort_values(["value", "entity_label"], na_position="last")
                    else:
                        df = df.sort_values(["score", "entity_label"], na_position="last")
                    
                    st.subheader("Extracted Parameters")
                    st.dataframe(
                        df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "phase", "score"]],
                        use_container_width=True
                    )
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Parameters CSV",
                        csv,
                        "lisn_params.csv",
                        "text/csv"
                    )
                    
                    json_data = df.to_json(orient="records", lines=True)
                    st.download_button(
                        "Download Parameters JSON",
                        json_data,
                        "lisn_params.json",
                        "application/json"
                    )
                    
                    st.subheader("Parameter Distribution Analysis")
                    if session_state.vis_type == "Histogram":
                        for phase in phases:
                            for param_type in entity_types:
                                fig = plot_histogram(df, param_type, phase)
                                if fig:
                                    st.plotly_chart(fig)
                    
                    elif session_state.vis_type == "Pie Chart":
                        for phase in phases:
                            fig = plot_pie_chart(df, phase)
                            if fig:
                                st.plotly_chart(fig)
                    
                    elif session_state.vis_type == "Scatter Plot":
                        for phase in phases:
                            fig = plot_scatter(df, x_param, y_param, phase)
                            if fig:
                                st.plotly_chart(fig)