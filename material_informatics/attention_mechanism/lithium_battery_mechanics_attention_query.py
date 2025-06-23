import arxiv
import fitz  # PyMuPDF
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time
from transformers import AutoModel, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from tenacity import retry, stop_after_attempt, wait_fixed
from scipy.special import softmax

# Define database directory and files
DB_DIR = os.path.dirname(os.path.abspath(__file__))
LITHIATION_DB_FILE = os.path.join(DB_DIR, "lithiation_knowledge.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "knowledge_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'lithiation_ner.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Lithium-Ion Battery Mechanics NER Tool", layout="wide")
st.title("Lithium-Ion Battery Mechanics NER Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **mechanics of lithium-ion batteries**, using SciBERT to prioritize terms like **strain**, **energy**, **elastic**, **anode**, **Cauchy stress**, **C-rate**, and **Young’s modulus**. It extracts parameters (e.g., elastic strain energy in J/mol, stress in MPa, capacity in mA h cm⁻²) via NER and saves to `lithiation_knowledge.db`. A backup database `knowledge_universe.db` stores full PDF text for fallback searches with synonym mapping (e.g., "volume change" → "volume expansion"). The **NER Analysis** tab visualizes extracted entities.

**Note**: For better accuracy, consider fine-tuning SciBERT or training a custom spaCy model ([spaCy Training Guide](https://spacy.io/usage/training)).
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- See `requirements.txt` for pinned versions.
- Install: `pip install -r requirements.txt`
- For NER: `python -m spacy download en_core_web_lg`
""")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

# Load SciBERT model and tokenizer
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Synonym mapping for related phrases
SYNONYM_MAPPING = {
    "volume change": "VOLUME_EXPANSION_NUM",
    "volumetric change": "VOLUME_EXPANSION_NUM",
    "volume swelling": "VOLUME_EXPANSION_NUM",
    "swelling change": "VOLUME_EXPANSION_NUM"
}

# Custom NER patterns
matcher = Matcher(nlp.vocab)
patterns = [
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["j", "joule", "joules", "kj/mol", "j/mol"]}}, {"LOWER": {"IN": ["strain", "elastic", "mechanical"]}, "OP": "?"}, {"LOWER": "energy"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa"]}}, {"LOWER": {"IN": ["strain", "elastic", "mechanical"]}, "OP": "?"}, {"LOWER": "energy"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["%", "dimensionless"]}}, {"LOWER": {"IN": ["strain", "deformation"]}}],
    [{"LOWER": {"IN": ["elastic", "mechanical"]}}, {"LOWER": {"IN": ["strain", "deformation"]}}, {"LOWER": "energy"}],
    [{"LOWER": {"IN": ["volume", "volumetric"]}}, {"LOWER": {"IN": ["expansion", "swelling"]}}],
    [{"LOWER": {"IN": ["mechanical", "lithiation-induced"]}}, {"LOWER": {"IN": ["stress", "strain"]}}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["%", "dimensionless"]}}, {"LOWER": {"IN": ["volume", "volumetric"]}}, {"LOWER": {"IN": ["expansion", "swelling", "change"]}}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa", "kg", "m−1", "s−2"]}}, {"LOWER": {"IN": ["cauchy"]}, "OP": "?"}, {"LOWER": "stress"}, {"LOWER": {"IN": ["tensor"]}, "OP": "?"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["c"]}}, {"LOWER": {"IN": ["rate", "rates"]}}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["ma", "mah", "ma h", "ma h cm-2"]}}, {"LOWER": "capacity"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa"]}}, {"LOWER": {"IN": ["young", "young’s"]}}, {"LOWER": "modulus"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["%", "dimensionless"]}}, {"LOWER": {"IN": ["linear"]}, "OP": "?"}, {"LOWER": "strain"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["cm", "mm"]}}, {"LOWER": {"IN": ["film", "wafer"]}, "OP": "?"}, {"LOWER": "diameter"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["nm", "µm", "um"]}}, {"LOWER": {"IN": ["film"]}, "OP": "?"}, {"LOWER": {"IN": ["thickness", "height"]}}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["µm", "um", "mm"]}}, {"LOWER": "wafer"}, {"LOWER": "thickness"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["m"]}}, {"LOWER": {"IN": ["mirror"]}, "OP": "?"}, {"LOWER": "constant"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["dimensionless"]}, "OP": "?"}, {"LOWER": {"IN": ["poisson", "poisson’s"]}}, {"LOWER": "ratio"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["g/cm3", "g cm-3"]}}, {"LOWER": {"IN": ["film"]}, "OP": "?"}, {"LOWER": "density"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["ua/cm2", "µa/cm2", "ma/cm2"]}}, {"LOWER": {"IN": ["current"]}, "OP": "?"}, {"LOWER": "density"}],
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mpa", "gpa"]}}, {"LOWER": "of"}, {"LOWER": {"IN": ["stress", "cauchy stress"]}}]
]
param_types = [
    "ELASTIC_STRAIN_ENERGY_J", "ELASTIC_STRAIN_ENERGY_MPA", "STRAIN", "ELASTIC_STAIN_ENERGY",
    "VOLUME_EXPANSION", "MECHANICAL_STRESS", "VOLUME_EXPANSION_NUM", "CAUCHY_STRESS", "C_RATE", "CAPACITY",
    "YOUNGS_MODULUS", "LINEAR_STRAIN", "FILM_DIAMETER", "FILM_THICKNESS", "WAFER_THICKNESS",
    "MIRROR_CONSTANT", "POISSONS_RATIO", "FILM_DENSITY", "CURRENT_DENSITY", "CAUCHY_STRESS_ALTERNATE"
]

# Validate pattern lengths
if len(patterns) != len(param_types):
    raise ValueError(f"Mismatch between patterns ({len(patterns)}) and param_types ({len(param_types)})")

# Add patterns to matcher
for i, pattern in enumerate(patterns):
    matcher.add(f"MECHANICS_PARAM_{param_types[i]}", [pattern])

# Parameter weights and validation ranges
pattern_weights = {
    "ELASTIC_STRAIN_ENERGY_J": 1.2, "ELASTIC_STRAIN_ENERGY_MPA": 1.2, "STRAIN": 1.0, "ELASTIC_STAIN_ENERGY": 1.1,
    "VOLUME_EXPANSION": 1.0, "MECHANICAL_STRESS": 0.9, "VOLUME_EXPANSION_NUM": 1.0, "CAUCHY_STRESS": 1.0,
    "C_RATE": 0.8, "CAPACITY": 0.8, "YOUNGS_MODULUS": 1.0, "LINEAR_STRAIN": 1.0, "FILM_DIAMETER": 0.7,
    "FILM_THICKNESS": 0.7, "WAFER_THICKNESS": 0.7, "MIRROR_CONSTANT": 0.7, "POISSONS_RATIO": 0.8,
    "FILM_DENSITY": 0.7, "CURRENT_DENSITY": 0.8, "CAUCHY_STRESS_ALTERNATE": 1.0
}
valid_ranges = {
    "ELASTIC_STRAIN_ENERGY_J": (0, 100000, "J/mol"), "ELASTIC_STRAIN_ENERGY_MPA": (0, 1000, "MPa"),
    "STRAIN": (0, 100, "%"), "ELASTIC_STAIN_ENERGY": (0, 1000, "MPa"), "VOLUME_EXPANSION": (0, 500, "%"),
    "MECHANICAL_STRESS": (0, 1000, "MPa"), "VOLUME_EXPANSION_NUM": (0, 500, "%"), "CAUCHY_STRESS": (0, 1000, "MPa"),
    "C_RATE": (0, 100, "C"), "CAPACITY": (0, 1000, "mA h cm-2"), "YOUNGS_MODULUS": (0, 1000, "GPa"),
    "LINEAR_STRAIN": (0, 100, "%"), "FILM_DIAMETER": (0, 100, "cm"), "FILM_THICKNESS": (0, 10000, "nm"),
    "WAFER_THICKNESS": (0, 10000, "µm"), "MIRROR_CONSTANT": (0, 100, "m"), "POISSONS_RATIO": (0, 0.5, "dimensionless"),
    "FILM_DENSITY": (0, 10, "g/cm3"), "CURRENT_DENSITY": (0, 1000, "µA/cm2"), "CAUCHY_STRESS_ALTERNATE": (0, 1000, "MPa")
}

# Color map for visualizations
param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}
param_colors.update({
    "ELASTIC_STRAIN_ENERGY_J": "purple", "ELASTIC_STRAIN_ENERGY_MPA": "purple", "VOLUME_EXPANSION": "blue",
    "STRAIN": "green", "MECHANICAL_STRESS": "red", "VOLUME_EXPANSION_NUM": "blue", "CAUCHY_STRESS": "red",
    "C_RATE": "orange", "CAPACITY": "cyan", "YOUNGS_MODULUS": "brown", "LINEAR_STRAIN": "green",
    "FILM_DIAMETER": "gray", "FILM_THICKNESS": "gray", "WAFER_THICKNESS": "gray", "MIRROR_CONSTANT": "pink",
    "POISSONS_RATIO": "yellow", "FILM_DENSITY": "gray", "CURRENT_DENSITY": "teal", "CAUCHY_STRESS_ALTERNATE": "red"
})

# Create PDFs directory
pdf_dir = os.path.join(DB_DIR, "pdfs")
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir}")

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "vis_type" not in st.session_state:
    st.session_state.vis_type = "Histogram"

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# SciBERT scoring with attention mechanism (retained from original)
@st.cache_data
def score_abstract_with_scibert(abstract):
    prioritized_words = ["strain", "energy", "elastic", "mechanical", "stress", "deformation", "lithium", "anode", "electrode", "battery", "cauchy", "c-rate", "capacity", "young’s", "modulus", "linear", "film", "wafer", "thickness", "diameter", "poisson’s", "ratio", "density", "current"]
    secondary_words = ["lithiation", "li-ion", "volume", "expansion", "swelling"]
    anode_terms = ["anode", "electrode", "lithium-ion battery"]
    
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0].mean(dim=0).numpy()
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        relevance_prob = 0.5  # Base score
        keyword_indices = []
        prioritized_indices = []
        for i, token in enumerate(tokens):
            token_lower = token.lower().replace("##", "")
            for kw in prioritized_words:
                if re.search(rf'\b{kw}\b', token_lower):
                    prioritized_indices.append(i)
                    keyword_indices.append(i)
            for kw in secondary_words:
                if re.search(rf'\b{kw}\b', token_lower):
                    keyword_indices.append(i)
        
        if prioritized_indices:
            attn_scores = attentions[prioritized_indices].sum()
            avg_attn_score = attn_scores / len(prioritized_indices)
            relevance_prob = min(relevance_prob + 0.4 * len(prioritized_indices) * avg_attn_score, 1.0)
            logging.info(f"Attention boost: {len(prioritized_indices)} prioritized tokens, avg attention: {avg_attn_score:.3f}")
            update_log(f"Attention boost: {len(prioritized_indices)} prioritized tokens, avg attention: {avg_attn_score:.3f}")
        
        abstract_lower = abstract.lower()
        for word in ["strain", "energy", "elastic", "stress", "cauchy", "c-rate", "capacity", "young’s", "modulus"]:
            if word in abstract_lower:
                word_pos = abstract_lower.find(word)
                context_window = abstract_lower[max(0, word_pos - 50):word_pos + len(word) + 50]
                if any(anode_term in context_window for anode_term in anode_terms):
                    relevance_prob = min(relevance_prob + 0.25, 1.0)
                    logging.info(f"Contextual boost: {word} near {', '.join([t for t in anode_terms if t in context_window])}")
                    update_log(f"Contextual boost: {word} near {', '.join([t for t in anode_terms if t in context_window])}")
        
        logging.info(f"SciBERT scored abstract: {relevance_prob:.3f}")
        update_log(f"SciBERT scored abstract: {relevance_prob:.3f}")
        return relevance_prob
    except Exception as e:
        logging.error(f"SciBERT scoring failed: {str(e)}")
        update_log(f"SciBERT scoring failed: {str(e)}")
        keywords = {
            "strain": 2.5, "energy": 2.5, "elastic": 2.5, "mechanical": 2.0,
            "stress": 2.0, "deformation": 2.0, "lithium": 1.5, "anode": 1.5,
            "electrode": 1.5, "battery": 1.2, "lithiation": 1.0, "li-ion": 1.0,
            "volume": 1.0, "expansion": 2.5, "swelling": 1.0, "cauchy": 2.0,
            "c-rate": 1.5, "capacity": 1.5, "young’s": 2.0, "modulus": 2.0,
            "linear": 1.5, "film": 1.0, "wafer": 1.0, "thickness": 1.0,
            "diameter": 1.0, "poisson’s": 1.5, "ratio": 1.5, "density": 1.0,
            "current": 1.5
        }
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = 0.0
        matched_keywords = []
        for kw, weight in keywords.items():
            if kw in word_counts:
                score += weight * word_counts[kw] / (total_words + 1e-6)
                matched_keywords.append(kw)
        for word in ["strain", "energy", "elastic", "stress", "cauchy", "c-rate", "capacity", "young’s", "modulus"]:
            if word in abstract_lower:
                word_pos = abstract_lower.find(word)
                context_window = abstract_lower[max(0, word_pos - 50):word_pos + len(word) + 50]
                if any(anode_term in context_window for anode_term in anode_terms):
                    score += 1.0
                    logging.info(f"Fallback contextual boost: {word} near {', '.join([t for t in anode_terms if t in context_window])}")
                    update_log(f"Fallback contextual boost: {word} near {', '.join([t for t in anode_terms if t in context_window])}")
        if matched_keywords:
            score = max(score, 0.1)
        max_possible_score = sum(keywords.values()) / 10
        relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        logging.info(f"Fallback scoring: {relevance_prob:.3f} (matched: {', '.join(matched_keywords)})")
        update_log(f"Fallback scoring: {relevance_prob:.3f} (matched: {', '.join(matched_keywords)})")
        return relevance_prob

# Get SciBERT embedding for NER
@st.cache_data
def get_scibert_embedding(text):
    try:
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True, output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        attentions = outputs.attentions[-1][0].mean(dim=0).mean(dim=0).numpy()
        weighted_embedding = last_hidden_state * attentions[:len(last_hidden_state)]
        norm = np.linalg.norm(weighted_embedding)
        return weighted_embedding / norm if norm > 0 else weighted_embedding
    except Exception as e:
        logging.error(f"SciBERT embedding failed for '{text}': {str(e)}")
        update_log(f"SciBERT embedding failed for '{text}': {str(e)}")
        return None

# Initialize database
def initialize_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                download_status TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                pdf_path TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                paper_id TEXT,
                entity_text TEXT,
                entity_label TEXT,
                value REAL,
                unit TEXT,
                context TEXT,
                phase TEXT,
                score REAL,
                co_occurrence BOOLEAN,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
        conn.commit()
        conn.close()
        update_log(f"Initialized database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

# Create knowledge_universe.db incrementally
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper["id"],
            paper.get("title", ""),
            paper.get("authors", "Unknown"),
            paper.get("year", 0),
            paper.get("content", "No text extracted")
        ))
        conn.commit()
        conn.close()
        update_log(f"Updated {db_file} with paper {paper['id']}")
        st.info(f"Added paper {paper['id']} to {db_file}")
        return db_file
    except Exception as e:
        update_log(f"Error updating {db_file}: {str(e)}")
        st.error(f"Error updating {db_file}: {str(e)}")
        return None

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
        update_log(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# NER extraction with synonym mapping
def extract_lithiation_parameters(text, phase, relevance_score):
    try:
        doc = nlp(text)
        entities = []
        
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id].replace("MECHANICS_PARAM_", "")
            match_text = span.text
            value_match = re.match(r"(\d+\.?\d*)", match_text)
            value = float(value_match.group(1)) if value_match else None
            unit = match_text[value_match.end():].strip() if value_match else None
            
            if unit and value is not None:
                if label in ["ELASTIC_STRAIN_ENERGY_MPA", "MECHANICAL_STRESS", "CAUCHY_STRESS", "CAUCHY_STRESS_ALTERNATE", "YOUNGS_MODULUS"]:
                    if unit.lower() in ["mpa", "megapascal"]:
                        unit = "MPa"
                    elif unit.lower() in ["gpa", "gigapascal"]:
                        unit = "MPa" if label != "YOUNGS_MODULUS" else "GPa"
                        value *= 1000
                elif label == "ELASTIC_STRAIN_ENERGY_J":
                    if unit.lower() in ["j", "joule", "joules"]:
                        unit = "J/mol"
                    elif unit.lower() in ["kj/mol"]:
                        unit = "J/mol"
                        value *= 1000
                elif label in ["STRAIN", "VOLUME_EXPANSION", "VOLUME_EXPANSION_NUM", "LINEAR_STRAIN"]:
                    if unit.lower() in ["%", "dimensionless"]:
                        unit = "%"
                elif label == "C_RATE":
                    if unit.lower() in ["c"]:
                        unit = "C"
                elif label == "CAPACITY":
                    if unit.lower() in ["ma", "mah", "ma h", "ma h cm-2"]:
                        unit = "mA h cm-2"
                elif label == "FILM_DIAMETER":
                    if unit.lower() in ["cm", "mm"]:
                        unit = "cm"
                        if unit.lower() == "mm":
                            value /= 10
                elif label in ["FILM_THICKNESS", "WAFER_THICKNESS"]:
                    if unit.lower() in ["nm", "µm", "um", "mm"]:
                        unit = "nm" if label == "FILM_THICKNESS" else "µm"
                        if unit.lower() == "mm":
                            value *= 1e6 if label == "FILM_THICKNESS" else 1e3
                        elif unit.lower() in ["µm", "um"] and label == "FILM_THICKNESS":
                            value *= 1e3
                elif label == "MIRROR_CONSTANT":
                    if unit.lower() in ["m"]:
                        unit = "m"
                elif label == "POISSONS_RATIO":
                    if unit.lower() in ["dimensionless", ""]:
                        unit = "dimensionless"
                elif label == "FILM_DENSITY":
                    if unit.lower() in ["g/cm3", "g cm-3"]:
                        unit = "g/cm3"
                elif label == "CURRENT_DENSITY":
                    if unit.lower() in ["ua/cm2", "µa/cm2"]:
                        unit = "µA/cm2"
                    elif unit.lower() == "ma/cm2":
                        unit = "µA/cm2"
                        value *= 1e3
            
            if label in valid_ranges and value is not None:
                min_val, max_val, expected_unit = valid_ranges[label]
                if not (min_val <= value <= max_val and (unit == expected_unit or unit is None)):
                    continue
            
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context_text = text[context_start:context_end].replace("\n", " ")
            co_occurrence = any(t in context_text.lower() for t in ["lithium", "anode", "battery"])
            
            entities.append({
                "entity_text": span.text,
                "entity_label": label,
                "start": start,
                "end": end,
                "value": value,
                "unit": unit,
                "context": context_text,
                "phase": phase,
                "score": pattern_weights.get(label, 0.7) * relevance_score,
                "co_occurrence": co_occurrence
            })
        
        # Similarity-based matching with synonym mapping
        reference_terms = {
            "ELASTIC_STRAIN_ENERGY_J": ["elastic strain energy", "strain energy", "elastic energy", "mechanical energy"],
            "ELASTIC_STRAIN_ENERGY_MPA": ["elastic strain energy", "strain energy", "elastic energy", "mechanical energy"],
            "STRAIN": ["strain", "mechanical strain", "deformation"],
            "ELASTIC_STAIN_ENERGY": ["elastic strain energy", "strain energy", "elastic energy", "mechanical energy"],
            "VOLUME_EXPANSION": ["volume expansion", "volumetric expansion", "swelling"],
            "MECHANICAL_STRESS": ["mechanical stress", "stress"],
            "VOLUME_EXPANSION_NUM": ["volume expansion", "volumetric expansion", "swelling", "volume change", "volumetric change", "swelling change"],
            "CAUCHY_STRESS": ["cauchy stress", "stress tensor", "cauchy stress tensor"],
            "CAUCHY_STRESS_ALTERNATE": ["cauchy stress", "stress tensor", "cauchy stress tensor"],
            "C_RATE": ["c-rate", "c rate", "charge rate"],
            "CAPACITY": ["capacity", "battery capacity"],
            "YOUNGS_MODULUS": ["young’s modulus", "young modulus", "elastic modulus"],
            "LINEAR_STRAIN": ["linear strain", "strain"],
            "FILM_DIAMETER": ["film diameter", "wafer diameter", "diameter"],
            "FILM_THICKNESS": ["film thickness", "thickness", "film height"],
            "WAFER_THICKNESS": ["wafer thickness", "substrate thickness"],
            "MIRROR_CONSTANT": ["mirror constant", "optical constant"],
            "POISSONS_RATIO": ["poisson’s ratio", "poisson ratio"],
            "FILM_DENSITY": ["film density", "density"],
            "CURRENT_DENSITY": ["current density", "charge density"]
        }
        similarity_threshold = 0.65
        ref_embeddings = {label: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for label, terms in reference_terms.items()}
        
        for span in doc.noun_chunks:
            span_text = span.text.lower()
            if len(span_text.split()) > 5:
                continue
            span_embedding = get_scibert_embedding(span_text)
            if span_embedding is None:
                continue
            for label, ref_embeds in ref_embeddings.items():
                for ref_embed in ref_embeds:
                    similarity = np.dot(span_embedding, ref_embed) / (np.linalg.norm(span_embedding) * np.linalg.norm(ref_embed))
                    if similarity > similarity_threshold:
                        mapped_label = SYNONYM_MAPPING.get(span_text, label)
                        value_match = re.match(r"(\d+\.?\d*)", span_text)
                        value = float(value_match.group(1)) if value_match else None
                        unit = None
                        if value:
                            unit_match = re.search(r"(?:j|joule|kj/mol|mpa|gpa|c|ma h cm-2|cm|mm|nm|µm|um|m|dimensionless|g/cm3|g cm-3|ua/cm2|µa/cm2|ma/cm2|%)", span_text, re.IGNORECASE)
                            unit = unit_match.group(0).upper() if unit_match else None
                            if unit == "GPA" and mapped_label != "YOUNGS_MODULUS":
                                unit = "MPa"
                                value *= 1000
                            elif unit == "KJ/MOL":
                                unit = "J/mol"
                                value *= 1000
                            elif unit in ["MM"] and mapped_label == "FILM_DIAMETER":
                                unit = "cm"
                                value /= 10
                            elif unit in ["MM", "µM", "UM"] and mapped_label in ["FILM_THICKNESS", "WAFER_THICKNESS"]:
                                unit = "nm" if mapped_label == "FILM_THICKNESS" else "µm"
                                value *= 1e6 if unit == "MM" and mapped_label == "FILM_THICKNESS" else 1e3 if unit == "MM" else 1e3 if mapped_label == "FILM_THICKNESS" else 1
                            elif unit in ["G/CM3", "G CM-3"]:
                                unit = "g/cm3"
                            elif unit in ["UA/CM2", "µA/CM2"]:
                                unit = "µA/cm2"
                            elif unit == "MA/CM2":
                                unit = "µA/cm2"
                                value *= 1e3
                            elif unit == "%":
                                unit = "%"
                        if mapped_label in valid_ranges and value is not None:
                            min_val, max_val, expected_unit = valid_ranges[mapped_label]
                            if not (min_val <= value <= max_val and (unit == expected_unit or unit is None)):
                                continue
                        context_start = max(0, span.start_char - 100)
                        context_end = min(len(text), span.end_char + 100)
                        context_text = text[context_start:context_end].replace("\n", " ")
                        co_occurrence = any(t in context_text.lower() for t in ["lithium", "anode", "battery"])
                        entities.append({
                            "entity_text": span.text,
                            "entity_label": mapped_label,
                            "start": span.start_char,
                            "end": span.end_char,
                            "value": value,
                            "unit": unit,
                            "context": context_text,
                            "phase": phase,
                            "score": similarity * relevance_score * pattern_weights.get(mapped_label, 0.7),
                            "co_occurrence": co_occurrence
                        })
                        logging.info(f"Similarity match: '{span.text}' as {mapped_label} (similarity: {similarity:.3f})")
                        update_log(f"Similarity match: '{span.text}' as {mapped_label} (similarity: {similarity:.3f})")
        
        entities = sorted(entities, key=lambda x: x["score"], reverse=True)
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity["entity_text"], entity["start"], entity["end"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        logging.info(f"Extracted {len(unique_entities)} entities for phase {phase}")
        update_log(f"Extracted {len(unique_entities)} entities for phase {phase}")
        return unique_entities
    except Exception as e:
        logging.error(f"NER failed: {str(e)}")
        update_log(f"NER failed: {str(e)}")
        return [{"entity_text": f"Error: {str(e)}", "entity_label": "ERROR", "start": 0, "end": 0, "value": None, "unit": None, "context": None, "phase": phase, "score": 0.0, "co_occurrence": False}]

# Save to SQLite
def save_to_sqlite(papers_df, params_list, lithiation_db_file=LITHIATION_DB_FILE):
    try:
        initialize_db(lithiation_db_file)
        conn = sqlite3.connect(lithiation_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        
        update_log(f"Saved {len(papers_df)} papers and {len(params_list)} parameters to {lithiation_db_file}")
        return f"Saved to {lithiation_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        st.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Search knowledge_universe.db
def search_universe_db(db_file=UNIVERSE_DB_FILE, entity_types=None, phase="Lithium-Ion Battery Mechanics"):
    try:
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, authors, year, content FROM papers"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        parameters = []
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            text = row["content"]
            if not isinstance(text, str) or text.startswith("Error"):
                update_log(f"Skipping paper {row['id']}: Invalid content")
                continue
            relevance_score = score_abstract_with_scibert(text[:512])
            params = extract_lithiation_parameters(text, phase, relevance_score)
            if entity_types:
                params = [p for p in params if p["entity_label"] in entity_types or p["entity_text"].lower() in SYNONYM_MAPPING]
            for param in params:
                param["paper_id"] = row["id"]
                param["title"] = row["title"]
                param["year"] = row["year"]
                if param["entity_text"].lower() in SYNONYM_MAPPING:
                    update_log(f"Mapped '{param['entity_text']}' to {param['entity_label']} in paper {row['id']}")
            parameters.extend(params)
            update_log(f"Processed paper {i+1}/{len(df)} in {db_file}")
            progress_bar.progress((i + 1) / len(df))
        
        update_log(f"Found {len(parameters)} parameters in {db_file}")
        return parameters
    except Exception as e:
        update_log(f"Error searching {db_file}: {str(e)}")
        st.error(f"Error searching {db_file}: {str(e)}")
        return []

# Visualization functions
@st.cache_data
def plot_histogram(df, param_type, phase):
    param_df = df[(df["entity_label"] == param_type) & (df["phase"] == phase)]
    if param_df.empty:
        st.warning(f"No entities found for {param_type} in phase {phase}.")
        return None
    if param_df["value"].dropna().empty:
        st.warning(f"No numerical values available for {param_type} in phase {phase}.")
        return None
    values = param_df["value"].dropna()
    fig, ax = plt.subplots()
    ax.hist(values, bins=20, edgecolor="black", color=param_colors.get(param_type, "gray"))
    unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
    ax.set_xlabel(f"{param_type} ({unit})")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {param_type} for {phase}")
    return fig

@st.cache_data
def plot_pie_chart(df, phase):
    param_counts = df[df["phase"] == phase]["entity_label"].value_counts()
    if not param_counts.empty:
        fig, ax = plt.subplots()
        ax.pie(param_counts, labels=param_counts.index, autopct='%1.1f%%', colors=[param_colors.get(p, "gray") for p in param_counts.index])
        ax.set_title(f"Parameter Distribution for {phase}")
        return fig
    return None

@st.cache_data
def plot_co_occurrence_network(df, phase):
    G = nx.Graph()
    prioritized_terms = ["ELASTIC_STRAIN_ENERGY_J", "ELASTIC_STRAIN_ENERGY_MPA", "VOLUME_EXPANSION", "MECHANICAL_STRESS", "CAUCHY_STRESS", "CAUCHY_STRESS_ALTERNATE", "YOUNGS_MODULUS", "LINEAR_STRAIN"]
    for term in prioritized_terms:
        G.add_node(term, type="mechanical")
    for paper_id in df["paper_id"].unique():
        paper_df = df[df["paper_id"] == paper_id]
        terms_present = paper_df["entity_label"].unique()
        terms_present = [t for t in terms_present if t in prioritized_terms]
        for i, term1 in enumerate(terms_present):
            for term2 in terms_present[i+1:]:
                if G.has_edge(term1, term2):
                    G[term1][term2]["weight"] += 1
                else:
                    G.add_edge(term1, term2, weight=1)
    if G.edges():
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        node_colors = ["purple" for _ in G.nodes()]
        node_sizes = [1200 for _ in G.nodes()]
        edge_widths = [3 * G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        ax.set_title(f"Co-occurrence Network: Mechanical Terms")
        return fig
    return None

# Database diagnostics
@st.cache_data
def diagnose_database(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        papers_exists = cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
        params_exists = cursor.fetchone() is not None
        papers_count = 0
        params_count = 0
        prioritized_params = pd.DataFrame(columns=["entity_label", "count"])
        if papers_exists:
            papers_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM papers", conn)["count"].iloc[0]
        if params_exists:
            params_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM parameters", conn)["count"].iloc[0]
            prioritized_params = pd.read_sql_query(
                "SELECT entity_label, COUNT(*) AS count FROM parameters WHERE entity_label IN ('ELASTIC_STRAIN_ENERGY_J', 'ELASTIC_STRAIN_ENERGY_MPA', 'VOLUME_EXPANSION', 'MECHANICAL_STRESS', 'CAUCHY_STRESS', 'CAUCHY_STRESS_ALTERNATE', 'YOUNGS_MODULUS', 'LINEAR_STRAIN') GROUP BY entity_label",
                conn
            )
        conn.close()
        return {
            "papers_count": papers_count,
            "params_count": params_count,
            "prioritized_terms": prioritized_params.to_dict("records") if not prioritized_params.empty else []
        }
    except Exception as e:
        update_log(f"Database diagnostics failed: {str(e)}")
        return {
            "error": str(e),
            "papers_count": 0,
            "params_count": 0,
            "prioritized_terms": []
        }

# Database validation
@st.cache_data
def validate_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not cursor.fetchone():
            conn.close()
            return False, "Database missing 'papers' table."
        df_papers = pd.read_sql_query("SELECT * FROM papers LIMIT 1", conn)
        required_columns = ["id", "title", "year", "abstract"]
        missing_columns = [col for col in required_columns if col not in df_papers.columns]
        if missing_columns:
            conn.close()
            return False, f"Database 'papers' table missing columns: {', '.join(missing_columns)}"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
        if not cursor.fetchone():
            conn.close()
            return False, "Database missing 'parameters' table."
        df_params = pd.read_sql_query("SELECT * FROM parameters LIMIT 1", conn)
        required_param_columns = ["paper_id", "entity_text", "entity_label", "phase", "score", "co_occurrence"]
        missing_param_cols = [col for col in required_param_columns if col not in df_params.columns]
        conn.close()
        if missing_param_cols:
            return False, f"Database 'parameters' table missing columns: {', '.join(missing_param_cols)}"
        return True, "Database format is valid."
    except Exception as e:
        return False, f"Error reading database: {str(e)}"

# Process parameters from database
@st.cache_data
def process_params_from_db(db_file, entity_types=None, phase="Lithium-Ion Battery Mechanics", use_universe_db=False):
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
        
        df = pd.merge(params_df, papers_df, left_on="paper_id", right_on="id", how="inner")
        df = df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "phase", "score", "co_occurrence"]]
        if entity_types:
            df = df[df["entity_label"].isin(entity_types)]
        if df.empty and use_universe_db and os.path.exists(UNIVERSE_DB_FILE):
            st.warning(f"No parameters found in {db_file}. Searching {UNIVERSE_DB_FILE}...")
            with st.spinner("Searching knowledge_universe.db (this may take longer)..."):
                universe_params = search_universe_db(UNIVERSE_DB_FILE, entity_types, phase)
                if universe_params:
                    universe_df = pd.DataFrame(universe_params)
                    df = pd.concat([df, universe_df], ignore_index=True)
                    st.success(f"Found {len(universe_params)} additional parameters in {UNIVERSE_DB_FILE}")
                else:
                    st.warning(f"No parameters found in {UNIVERSE_DB_FILE}")
        
        return df
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        update_log(f"Error reading database: {str(e)}")
        return None

# Create tabs
tab1, tab2 = st.tabs(["arXiv Query", "NER Analysis"])

# arXiv Query Tab
with tab1:
    st.header("arXiv Query for Lithium-Ion Battery Mechanics")
    st.markdown("Search for abstracts on lithium-ion battery mechanics, prioritizing **strain**, **energy**, **elastic**, **anode**, **Cauchy stress**, and **Young’s modulus** using SciBERT's attention mechanism.")
    
    log_container = st.empty()
    def display_logs():
        log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)
    
    # Retry logic for PDF download
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def download_pdf(url, path):
        urllib.request.urlretrieve(url, path)
    
    # arXiv query (retained from original)
    @st.cache_data
    def query_arxiv(query, categories, max_results, start_year, end_year):
        try:
            query_terms = query.strip().split()
            formatted_terms = []
            synonyms = {
                "strain": ["strain", "deformation"],
                "energy": ["energy"],
                "elastic": ["elastic", "elasticity"],
                "mechanical": ["mechanical", "mechanics"],
                "stress": ["stress"],
                "lithium": ["lithium", "li-ion"],
                "anode": ["anode", "electrode"],
                "battery": ["battery", "lithium-ion battery", "li-ion"],
                "cauchy": ["cauchy", "cauchy stress"],
                "c-rate": ["c-rate", "charge rate"],
                "capacity": ["capacity"],
                "young’s": ["young’s", "young"],
                "modulus": ["modulus"],
                "volume": ["volume"],
                "expansion": ["expansion", "swelling", "change"],
                "current": ["current", "charge"],
                "density": ["density"],
                "film": ["film"],
                "thickness": ["thickness"],
                "poisson’s": ["poisson’s", "poisson"],
                "ratio": ["ratio"]
            }
            for term in query_terms:
                term_clean = term.strip('"')
                formatted_terms.append(term_clean.replace(" ", "+"))
                for key, syn_list in synonyms.items():
                    if term_clean.lower() == key:
                        formatted_terms.extend(syn.replace(" ", "+") for syn in syn_list)
            api_query = " ".join(formatted_terms)
            
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
                    query_words = set(word.lower().strip('"') for word in re.split(r'\s+', query) if word)
                    for key, syn_list in synonyms.items():
                        if key in query_words:
                            query_words.update(syn_list)
                    matched_terms = [word for word in query_words if word in abstract or word in title]
                    if not matched_terms:
                        continue
                    relevance_prob = score_abstract_with_scibert(result.summary)
                    abstract_highlighted = abstract
                    for term in matched_terms:
                        abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                    
                    papers.append({
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "authors": ", ".join([author.name for author in result.authors]),
                        "year": result.published.year,
                        "categories": ", ".join(result.categories),
                        "abstract": abstract,
                        "abstract_highlighted": abstract_highlighted,
                        "pdf_url": result.pdf_url,
                        "download_status": "Not downloaded",
                        "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                        "relevance_prob": round(relevance_prob * 100, 2),
                        "pdf_path": None,
                        "content": None
                    })
                if len(papers) >= max_results:
                    break
            papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
            return papers
        except Exception as e:
            logging.error(f"arXiv query failed: {str(e)}")
            update_log(f"arXiv query failed: {str(e)}")
            st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
            return []

    @st.cache_data
    def download_pdf_and_extract(pdf_url, paper_id, paper_metadata):
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        params = []
        try:
            download_pdf(pdf_url, pdf_path)
            file_size = os.path.getsize(pdf_path) / 1024
            text = extract_text_from_pdf(pdf_path)
            if not text.startswith("Error"):
                phase = "Lithium-Ion Battery Mechanics"
                if any(p in text.lower() for p in ["strain", "energy", "elastic", "lithium", "anode", "battery", "cauchy", "c-rate", "capacity", "young’s"]):
                    relevance_prob = score_abstract_with_scibert(text[:512])
                    params = extract_lithiation_parameters(text, phase, relevance_prob)
                    for param in params:
                        param["paper_id"] = paper_id
                # Incremental update to knowledge_universe.db
                paper_data = {
                    "id": paper_id,
                    "title": paper_metadata.get("title", ""),
                    "authors": paper_metadata.get("authors", "Unknown"),
                    "year": paper_metadata.get("year", 0),
                    "content": text
                }
                create_universe_db(paper_data)
                return f"Downloaded ({file_size:.2f} KB)", pdf_path, params, text
            else:
                params.append({
                    "paper_id": paper_id,
                    "entity_text": f"Failed: {text}",
                    "entity_label": "ERROR",
                    "value": None,
                    "unit": None,
                    "context": "",
                    "phase": None,
                    "score": 0.0,
                    "co_occurrence": False
                })
                return f"Failed: {text}", None, params, text
        except Exception as e:
            logging.error(f"PDF download failed for {paper_id}: {str(e)}")
            update_log(f"PDF download failed for {paper_id}: {str(e)}")
            params.append({
                "paper_id": paper_id,
                "entity_text": f"Failed: {str(e)}",
                "entity_label": "ERROR",
                "value": None,
                "unit": None,
                "context": "",
                "phase": None,
                "score": 0.0,
                "co_occurrence": False
            })
            return f"Failed: {str(e)}", None, params, f"Error: {str(e)}"

    with st.sidebar:
        st.subheader("Search Parameters")
        query = st.text_input("Query", value='mechanics "lithium-ion battery" strain energy elastic anode stress "Cauchy stress" deformation "C-rate" capacity "Young’s modulus" "volume expansion" "current density" "film thickness" "Poisson’s ratio" "film density"')
        default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.soft"]
        categories = st.multiselect("Categories", default_categories, default=default_categories)
        max_results = st.slider("Max Papers", min_value=1, max_value=200, value=10)
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2010)
        with col2:
            end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
        output_formats = st.multiselect("Output Formats", ["CSV", "SQLite (.db)", "JSON"], default=["SQLite (.db)"])
        search_button = st.button("Search arXiv")

    if search_button:
        if not query.strip():
            st.error("Enter a valid query.")
        elif not categories:
            st.error("Select at least one category.")
        elif start_year > end_year:
            st.error("Start year must be ≤ end year.")
        else:
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year)
            
            if not papers:
                st.warning("No papers found. Broaden query or categories.")
            else:
                st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 30%...")
                relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
                if not relevant_papers:
                    st.warning("No papers with relevance > 30%. Broaden query or check 'lithiation_ner.log'.")
                else:
                    st.success(f"**{len(relevant_papers)}** papers with relevance > 30%. Downloading PDFs...")
                    progress_bar = st.progress(0)
                    all_params = []
                    for i, paper in enumerate(relevant_papers):
                        if paper["pdf_url"]:
                            status, pdf_path, params, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
                            paper["download_status"] = status
                            paper["pdf_path"] = pdf_path
                            paper["content"] = content
                            all_params.extend(params)
                        progress_bar.progress((i + 1) / len(relevant_papers))
                        time.sleep(1)  # Avoid rate-limiting
                        update_log(f"Processed paper {i+1}/{len(relevant_papers)}: {paper['title']}")
                    
                    df = pd.DataFrame(relevant_papers)
                    st.subheader("Papers (Relevance > 30%)")
                    st.dataframe(
                        df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
                        use_container_width=True
                    )
                    
                    if "CSV" in output_formats:
                        csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                        st.download_button(
                            label="Download Paper Metadata CSV",
                            data=csv,
                            file_name="battery_mechanics_papers.csv",
                            mime="text/csv"
                        )
                    
                    if "SQLite (.db)" in output_formats:
                        sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), all_params)
                        st.info(sqlite_status)
                    
                    if "JSON" in output_formats:
                        json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                        st.download_button(
                            label="Download Paper Metadata JSON",
                            data=json_data,
                            file_name="battery_mechanics_papers.json",
                            mime="application/json"
                        )
                    
                    display_logs()

# NER Analysis Tab
with tab2:
    st.header("NER Analysis for Lithium-Ion Battery Mechanics")
    st.markdown("Analyze extracted parameters, focusing on **elastic strain energy**, **strain**, **volume expansion**, **Cauchy stress**, **Young’s modulus**, and **capacity** in lithium-ion battery contexts. Uses `knowledge_universe.db` as a fallback if parameters are not found in `lithiation_knowledge.db`.")
    
    log_container = st.empty()
    def display_logs():
        log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)
    
    st.subheader("Database Diagnostics")
    diagnostics = diagnose_database(LITHIATION_DB_FILE)
    if "error" in diagnostics:
        st.error(f"Diagnostics failed: {diagnostics['error']}")
    else:
        st.write(f"Total papers: {diagnostics['papers_count']}")
        st.write(f"Total parameters: {diagnostics['params_count']}")
        if diagnostics.get('prioritized_terms', []):
            st.write("Prioritized parameters:")
            st.table(pd.DataFrame(diagnostics['prioritized_terms']))
        else:
            st.info("No prioritized parameters found (e.g., ELASTIC_STRAIN_ENERGY_J, CAUCHY_STRESS).")
        if diagnostics['params_count'] == 0:
            st.warning("No parameters found. Run a query in the arXiv Query tab.")
    
    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        db_file_input = st.text_input("SQLite Database Path", value=LITHIATION_DB_FILE)
        use_universe_db = st.checkbox("Use knowledge_universe.db as fallback", value=True)
        entity_types = st.multiselect(
            "Parameter Types to Display",
            param_types,
            default=["ELASTIC_STRAIN_ENERGY_J", "ELASTIC_STRAIN_ENERGY_MPA", "VOLUME_EXPANSION", "MECHANICAL_STRESS", "CAUCHY_STRESS", "CAUCHY_STRESS_ALTERNATE", "YOUNGS_MODULUS", "LINEAR_STRAIN", "CAPACITY", "C_RATE", "VOLUME_EXPANSION_NUM"]
        )
        phase = "Lithium-Ion Battery Mechanics"
        st.write(f"Phase: {phase} (fixed)")
        sort_by = st.selectbox("Sort By", ["entity_label", "value", "score", "co_occurrence"])
        show_co_occurrences = st.checkbox("Show only co-occurrences with lithium/anode/battery", value=False)
        analyze_button = st.button("Run NER Analysis")
    
    if analyze_button:
        if not db_file_input:
            st.error("Specify the SQLite database file.")
        else:
            with st.spinner("Processing parameters from database..."):
                df = process_params_from_db(db_file_input, entity_types, phase, use_universe_db)
            
            if df is None or df.empty:
                st.warning("No parameters extracted. Run the arXiv Query tab first.")
            else:
                df = df[df["phase"] == phase]
                if entity_types:
                    df = df[df["entity_label"].isin(entity_types)]
                if show_co_occurrences:
                    df = df[df["co_occurrence"] == True]
                
                if df.empty:
                    st.warning("No parameters match the filters. Try broader entity types or disable co-occurrence filter.")
                else:
                    st.success(f"Extracted **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                    
                    if sort_by == "entity_label":
                        df = df.sort_values(["entity_label", "value"])
                    elif sort_by == "value":
                        df = df.sort_values(["value", "entity_label"], na_position="last")
                    elif sort_by == "score":
                        df = df.sort_values(["score", "entity_label"], na_position="last")
                    else:
                        df = df.sort_values(["co_occurrence", "score"], na_position="last")
                    
                    st.subheader("Extracted Parameters")
                    st.dataframe(
                        df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "phase", "score", "co_occurrence"]],
                        use_container_width=True
                    )
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Parameters CSV",
                        csv,
                        "battery_mechanics_params.csv",
                        "text/csv"
                    )
                    
                    json_data = df.to_json(orient="records", lines=True)
                    st.download_button(
                        "Download Parameters JSON",
                        json_data,
                        "battery_mechanics_params.json",
                        "application/json"
                    )
                    
                    st.subheader("Parameter Distribution Analysis")
                    st.session_state.vis_type = st.selectbox(
                        "Select Visualization Type",
                        ["Histogram", "Pie Chart", "Co-occurrence Network"],
                        index=["Histogram", "Pie Chart", "Co-occurrence Network"].index(st.session_state.vis_type)
                    )
                    
                    if st.session_state.vis_type == "Histogram":
                        for param_type in entity_types:
                            fig = plot_histogram(df, param_type, phase)
                            if fig:
                                st.pyplot(fig)
                    
                    elif st.session_state.vis_type == "Pie Chart":
                        fig = plot_pie_chart(df, phase)
                        if fig:
                            st.pyplot(fig)
                    
                    elif st.session_state.vis_type == "Co-occurrence Network":
                        fig = plot_co_occurrence_network(df, phase)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.warning("No co-occurrences found among prioritized terms.")
                    
                    display_logs()
