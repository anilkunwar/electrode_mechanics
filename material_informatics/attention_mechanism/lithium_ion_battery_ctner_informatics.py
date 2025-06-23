import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from collections import Counter
import re
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import networkx as nx
from wordcloud import WordCloud
from nltk import ngrams
from itertools import chain, combinations
import math
import glob
import uuid

# Define database directory
DB_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'common_term_ner.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Common Term and NER Analysis Tool", layout="wide")
st.title("Common Term and NER Analysis for Lithium-Ion Battery Mechanics")
st.markdown("""
This tool inspects SQLite databases, extracts common terms and phrases, and performs NER analysis for lithium-ion battery mechanics.
Select or upload a database, then use the tabs to inspect the database, analyze terms, or extract entities.
""")

# Load spaCy model with custom tokenizer
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

# Customize tokenizer
from spacy.language import Language
@Language.component("custom_tokenizer")
def custom_tokenizer(doc):
    hyphenated_phrases = ["lithium-ion", "Li-ion", "young’s modulus"]
    for phrase in hyphenated_phrases:
        if phrase.lower() in doc.text.lower():
            with doc.retokenize() as retokenizer:
                for match in re.finditer(rf'\b{re.escape(phrase)}\b', doc.text, re.IGNORECASE):
                    start_char, end_char = match.span()
                    start_token = None
                    for token in doc:
                        if token.idx >= start_char:
                            start_token = token.i
                            break
                    if start_token is not None:
                        retokenizer.merge(doc[start_token:start_token+len(phrase.split('-'))])
    return doc

nlp.add_pipe("custom_tokenizer", before="parser")
nlp.max_length = 1_000_000

# Load SciBERT model and tokenizer
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "ner_results" not in st.session_state:
    st.session_state.ner_results = None
if "raw_common_terms" not in st.session_state:
    st.session_state.raw_common_terms = None
if "common_terms" not in st.session_state:
    st.session_state.common_terms = None
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "term_counts" not in st.session_state:
    st.session_state.term_counts = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None

def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# Get SciBERT embedding
@st.cache_data
def get_scibert_embedding(text):
    try:
        if not text.strip():
            update_log(f"Skipping empty text for SciBERT embedding")
            return None
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=32, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        norm = np.linalg.norm(last_hidden_state)
        if norm == 0:
            update_log(f"Zero norm for embedding of '{text}'")
            return None
        return last_hidden_state / norm
    except Exception as e:
        update_log(f"SciBERT embedding failed for '{text}': {str(e)}")
        return None

# Inspect database
def inspect_database(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        st.subheader("Tables in Database")
        if tables:
            st.write([table[0] for table in tables])
        else:
            st.warning("No tables found in the database.")
            conn.close()
            return None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
        if not cursor.fetchone():
            st.warning("No 'papers' table found in the database.")
            conn.close()
            return None
        cursor.execute("PRAGMA table_info(papers);")
        schema = cursor.fetchall()
        st.subheader("Schema of 'papers' Table")
        schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)
        query = "SELECT id, title, year, substr(content, 1, 200) as sample_content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%' LIMIT 5"
        df = pd.read_sql_query(query, conn)
        st.subheader("Sample Rows from 'papers' Table (First 5 Papers)")
        if df.empty:
            st.warning("No valid papers found in the 'papers' table.")
        else:
            st.dataframe(df, use_container_width=True)
        cursor.execute("SELECT COUNT(*) as count FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'")
        total_papers = cursor.fetchone()[0]
        st.subheader("Total Valid Papers")
        st.write(f"{total_papers} papers")
        terms_to_search = ["lithium-ion battery", "stress", "strain", "young’s modulus", "volume expansion"]
        st.subheader("Term Frequency in 'content' Column")
        term_counts = {}
        for term in terms_to_search:
            cursor.execute(f"SELECT COUNT(*) FROM papers WHERE content LIKE '%{term}%' AND content IS NOT NULL AND content NOT LIKE 'Error%'")
            count = cursor.fetchone()[0]
            term_counts[term] = count
            st.write(f"'{term}': {count} papers")
        query = "SELECT id, title, year, substr(content, 1, 1000) as content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%' LIMIT 10"
        df_full = pd.read_sql_query(query, conn)
        csv_filename = f"database_sample_{uuid.uuid4().hex}.csv"
        csv_path = os.path.join(DB_DIR, csv_filename)
        df_full.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.session_state.csv_data = f.read()
        st.session_state.csv_filename = csv_filename
        st.subheader("Download Sample Content")
        st.download_button(
            label="Download Sample CSV",
            data=st.session_state.csv_data,
            file_name="database_sample.csv",
            mime="text/csv",
            key="download_csv"
        )
        conn.close()
        st.success(f"Database inspection completed for {os.path.basename(db_path)}")
        return term_counts
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        return None

# Extract common terms and phrases
@st.cache_data(hash_funcs={str: lambda x: x})
def extract_common_terms(db_file, min_freq=10, phrase_weight=1.5, pmi_threshold=2.0):
    try:
        update_log(f"Starting common term extraction from {os.path.basename(db_file)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid papers found in {os.path.basename(db_file)}")
            st.warning(f"No valid papers found in {os.path.basename(db_file)}.")
            return []
        update_log(f"Loaded {len(df)} papers")
        total_words = 0
        term_counts = Counter()
        word_counts = Counter()
        phrase_counts = Counter()
        prioritized_phrases = [
            "lithium-ion battery", "li-ion battery", "young’s modulus", "volume expansion",
            "mechanical stress", "linear strain"
        ]
        prioritized_single_terms = ["strain", "stress", "modulus", "lithium", "anode"]
        progress_bar = st.progress(0)
        for i, content in enumerate(df["content"].dropna()):
            if len(content) > nlp.max_length:
                content = content[:nlp.max_length]
                update_log(f"Truncated content for paper {i+1}")
            chunk_size = 100_000
            content_chunks = [content[j:j+chunk_size] for j in range(0, len(content), chunk_size)]
            for chunk_idx, chunk in enumerate(content_chunks):
                try:
                    doc = nlp(chunk.lower())
                    phrases = [span.text.strip() for span in doc.noun_chunks if 1 < len(span.text.split()) <= 3]
                    single_words = [token.text for token in doc if token.text.isalpha() and not token.is_stop and len(token.text) > 3]
                    words = [token.text for token in doc if token.text.isalpha() and not token.is_stop]
                    n_grams = list(chain(ngrams(words, 2), ngrams(words, 3)))
                    n_gram_phrases = [' '.join(gram) for gram in n_grams if 1 < len(gram) <= 3]
                    all_phrases = phrases + n_gram_phrases
                    merged_phrases = []
                    for p in all_phrases:
                        if p.replace(" ", "") in ["lithiumionbattery", "liionbattery"]:
                            merged_phrases.append("lithium-ion battery")
                        elif p == "lithium ion battery":
                            merged_phrases.append("lithium-ion battery")
                        else:
                            merged_phrases.append(p)
                    all_terms = merged_phrases + single_words
                    term_counts.update(all_terms)
                    word_counts.update(words)
                    phrase_counts.update([t for t in all_terms if len(t.split()) > 1])
                    total_words += len(words)
                except Exception as e:
                    update_log(f"Error processing chunk {chunk_idx+1} in paper {i+1}: {str(e)}")
            progress_bar.progress((i + 1) / len(df))
        weighted_terms = []
        for term, count in term_counts.most_common():
            if term in prioritized_phrases or term in prioritized_single_terms:
                weighted_terms.append((term, count))
            elif len(term.split()) > 1:
                pmi = calculate_pmi(term, word_counts, phrase_counts, total_words)
                if pmi >= pmi_threshold or count >= min_freq:
                    weighted_count = count * phrase_weight
                    weighted_terms.append((term, weighted_count))
            elif count >= min_freq:
                weighted_terms.append((term, count))
        common_terms = sorted(weighted_terms, key=lambda x: x[1], reverse=True)[:50]
        if not common_terms:
            update_log(f"No terms/phrases extracted from {os.path.basename(db_file)}")
            st.warning(f"No terms/phrases extracted. Adjust parameters.")
            return []
        update_log(f"Extracted {len(common_terms)} common terms")
        return common_terms
    except Exception as e:
        update_log(f"Error extracting terms: {str(e)}")
        st.error(f"Error extracting terms: {str(e)}")
        return []

# Calculate PMI
def calculate_pmi(phrase, word_counts, phrase_counts, total_words):
    words = phrase.split()
    if len(words) < 2:
        return 0.0
    joint_prob = phrase_counts[phrase] / total_words
    word_probs = [word_counts[word] / total_words for word in words]
    if any(p == 0 for p in word_probs) or joint_prob == 0:
        return 0.0
    pmi = math.log2(joint_prob / np.prod(word_probs))
    return pmi

# Perform NER with SciBERT
def perform_ner_on_terms(db_file, selected_terms):
    try:
        update_log(f"Starting NER for terms: {', '.join(selected_terms)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid papers found in {os.path.basename(db_file)}")
            st.error("No valid papers found.")
            return pd.DataFrame()
        update_log(f"Loaded {len(df)} papers for NER")
        entities = []
        progress_bar = st.progress(0)
        reference_terms = {
            "STRAIN": ["strain", "mechanical strain", "deformation"],
            "STRESS": ["stress", "mechanical stress"],
            "VOLUME_EXPANSION": ["volume expansion", "swelling"],
            "YOUNGS_MODULUS": ["young’s modulus", "elastic modulus"],
            "BATTERY": ["lithium-ion battery", "li-ion battery", "battery"]
        }
        valid_ranges = {
            "STRAIN": (0, 100, "%"),
            "STRESS": (0, 1000, "MPa"),
            "VOLUME_EXPANSION": (0, 500, "%"),
            "YOUNGS_MODULUS": (0, 1000, "GPa")
        }
        similarity_threshold = 0.5
        term_similarity_threshold = 0.3
        ref_embeddings = {label: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for label, terms in reference_terms.items()}
        batch_size = 2
        for batch_start in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_start:batch_start+batch_size]
            for i, row in batch_df.iterrows():
                try:
                    text = row["content"].lower()
                    if len(text) > nlp.max_length:
                        text = text[:nlp.max_length]
                        update_log(f"Truncated content for paper {row['id']}")
                    if not text.strip() or len(text) < 10:
                        update_log(f"Skipping paper {row['id']} due to empty/short content")
                        continue
                    doc = nlp(text)
                    spans = [span for span in doc.noun_chunks if len(span.text.split()) <= 5] + \
                            [token for token in doc if token.text.lower() in selected_terms] + \
                            [doc[start:end] for start, end in [(i, i+2) for i in range(len(doc)-1)] + [(i, i+3) for i in range(len(doc)-2)] if len(doc[start:end].text.split()) <= 5]
                    if not spans:
                        update_log(f"No valid spans in paper {row['id']}")
                        continue
                    for span in spans:
                        span_text = span.text.lower().strip()
                        if not span_text:
                            update_log(f"Skipping empty span '{span_text}' in paper {row['id']}")
                            continue
                        term_matched = False
                        if any(re.search(rf'\b{re.escape(term)}\b', span_text, re.IGNORECASE) for term in selected_terms):
                            term_matched = True
                        else:
                            span_embedding = get_scibert_embedding(span_text)
                            if span_embedding is None:
                                update_log(f"Skipping span '{span_text}' in paper {row['id']}: no embedding")
                                continue
                            term_embeddings = [get_scibert_embedding(term) for term in selected_terms if get_scibert_embedding(term) is not None]
                            if not term_embeddings:
                                update_log(f"No valid embeddings for selected terms in paper {row['id']}")
                                continue
                            similarities = []
                            for t_emb in term_embeddings:
                                norm_span = np.linalg.norm(span_embedding)
                                norm_term = np.linalg.norm(t_emb)
                                if norm_span == 0 or norm_term == 0:
                                    update_log(f"Zero norm for span '{span_text}' or term in paper {row['id']}")
                                    continue
                                sim = np.dot(span_embedding, t_emb) / (norm_span * norm_term)
                                similarities.append(sim)
                            if any(s > term_similarity_threshold for s in similarities):
                                term_matched = True
                            else:
                                update_log(f"No similarity match for span '{span_text}' in paper {row['id']}")
                        if not term_matched:
                            continue
                        span_embedding = get_scibert_embedding(span_text)
                        if span_embedding is None:
                            update_log(f"Skipping span '{span_text}' in paper {row['id']}: no embedding for label match")
                            continue
                        for label, ref_embeds in ref_embeddings.items():
                            for ref_embed in ref_embeds:
                                norm_span = np.linalg.norm(span_embedding)
                                norm_ref = np.linalg.norm(ref_embed)
                                if norm_span == 0 or norm_ref == 0:
                                    update_log(f"Zero norm for span '{span_text}' or reference '{label}' in paper {row['id']}")
                                    continue
                                similarity = np.dot(span_embedding, ref_embed) / (norm_span * norm_ref)
                                if similarity > similarity_threshold:
                                    value_match = re.match(r"(\d+\.?\d*[eE]?-?\d*)", span_text)
                                    value = float(value_match.group(1)) if value_match else None
                                    unit = None
                                    if value:
                                        unit_match = re.search(r"(?:mpa|gpa|kpa|pa|%)", span_text, re.IGNORECASE)
                                        unit = unit_match.group(0).upper() if unit_match else None
                                        if unit == "GPA" and label == "YOUNGS_MODULUS":
                                            unit = "GPa"
                                        elif unit == "GPA":
                                            unit = "MPa"
                                            value *= 1000
                                        elif unit == "KPA":
                                            unit = "MPa"
                                            value /= 1000
                                        elif unit == "PA":
                                            unit = "MPa"
                                            value /= 1_000_000
                                    if value is not None and label in valid_ranges:
                                        min_val, max_val, expected_unit = valid_ranges[label]
                                        if not (min_val <= value <= max_val and (unit == expected_unit or unit is None)):
                                            update_log(f"Skipping span '{span_text}' in paper {row['id']}: invalid value/unit")
                                            continue
                                    context_start = max(0, span.start_char - 100)
                                    context_end = min(len(text), span.end_char + 100)
                                    context_text = text[context_start:context_end].replace("\n", " ")
                                    entities.append({
                                        "paper_id": row["id"],
                                        "title": row["title"],
                                        "year": row["year"],
                                        "entity_text": span.text,
                                        "entity_label": label,
                                        "value": value,
                                        "unit": unit,
                                        "context": context_text,
                                        "score": similarity
                                    })
                                    update_log(f"Extracted entity: term='{span.text}', label={label}, value={value}, unit={unit}, paper_id={row['id']}")
                except MemoryError as e:
                    update_log(f"Memory error in paper {row['id']}: {str(e)}")
                    st.error("Memory exhausted. Try reducing text length or batch size.")
                    return pd.DataFrame()
                except Exception as e:
                    update_log(f"Error processing paper {row['id']}: {str(e)}")
                progress_bar.progress(min((batch_start + i + 1) / len(df), 1.0))
        update_log(f"Completed NER analysis: extracted {len(entities)} entities")
        if not entities:
            update_log("No entities extracted. Possible issues: terms not in spans, low similarity, or invalid content.")
        return pd.DataFrame(entities)
    except Exception as e:
        update_log(f"NER analysis failed: {str(e)}")
        st.error(f"NER analysis failed: {str(e)}")
        return pd.DataFrame()

# Plot word cloud
@st.cache_data
def plot_word_cloud(terms, top_n, font_size, font_type, colormap):
    term_dict = dict(terms[:top_n])
    font_path = None
    if font_type and font_type != "None":
        font_map = {'DejaVu Sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'}
        font_path = font_map.get(font_type, font_type)
        if not os.path.exists(font_path):
            update_log(f"Font path '{font_path}' not found")
            font_path = None
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=8, max_font_size=font_size,
        font_path=font_path, colormap=colormap, max_words=top_n, prefer_horizontal=0.9
    ).generate_from_frequencies(term_dict)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud of Top {top_n} Terms", fontsize=12)
    plt.tight_layout()
    return fig

# Plot term co-occurrence network
@st.cache_data
def plot_term_co_occurrence(terms, top_n, db_file, font_size, colormap):
    try:
        update_log(f"Building term co-occurrence network for top {top_n} terms")
        conn = sqlite3.connect(db_file)
        query = "SELECT content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        top_terms = [term for term, _ in terms[:top_n]]
        term_freqs = dict(terms[:top_n])
        G = nx.Graph()
        for term in top_terms:
            G.add_node(term, type="term", freq=term_freqs[term])
        for content in df["content"].dropna():
            content_lower = content.lower()
            terms_present = [term for term in top_terms if re.search(rf'\b{re.escape(term)}\b', content_lower)]
            for term1, term2 in combinations(terms_present, 2):
                if term1 != term2:  # Avoid self-loops
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        if G.edges():
            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_sizes = [500 + 3000 * (G.nodes[term]["freq"] / max(term_freqs.values())) for term in G.nodes]
            node_colors = [cm.get_cmap(colormap)(i / len(top_terms)) for i in range(len(top_terms))]
            edge_widths = [2 * G[u][v]["weight"] / max([d["weight"] for _, _, d in G.edges(data=True)]) for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"Term Co-occurrence Network (Top {top_n} Terms)", fontsize=12)
            plt.tight_layout()
            return fig
        update_log("No co-occurrences found for term network")
        return None
    except Exception as e:
        update_log(f"Error building term co-occurrence network: {str(e)}")
        return None

# Plot term frequency histogram
@st.cache_data
def plot_term_histogram(terms, top_n):
    terms, counts = zip(*terms[:top_n])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.bar(terms, counts, color="skyblue", edgecolor="black")
    ax.set_xlabel("Terms/Phrases", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"Top {top_n} Terms", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return fig

# Plot NER co-occurrence network
@st.cache_data
def plot_ner_co_occurrence(df, top_n, font_size, colormap):
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
                if term1 != term2:  # Avoid self-loops
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        if G.edges():
            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_colors = [cm.get_cmap(colormap)(i / len(entity_labels)) for i in range(len(entity_labels))]
            edge_widths = [2 * G[u][v]["weight"] for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"NER Co-occurrence Network (Top {top_n} Entities)", fontsize=12)
            plt.tight_layout()
            return fig
        update_log("No co-occurrences found for NER network")
        return None
    except Exception as e:
        update_log(f"Error plotting NER co-occurrence network: {str(e)}")
        return None

# Plot NER histogram
@st.cache_data
def plot_ner_histogram(df, top_n, colormap):
    try:
        update_log(f"Building NER histogram for top {top_n} entities")
        if df.empty:
            update_log("Empty NER dataframe for histogram")
            return None
        label_counts = df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        counts = label_counts.values
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        ax.bar(labels, counts, color=colors, edgecolor="black")
        ax.set_xlabel("Entity Labels", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Histogram of Top {top_n} NER Entities", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER histogram: {str(e)}")
        return None

# Plot NER radial chart
@st.cache_data
def plot_ner_radial(df, top_n, colormap):
    try:
        update_log(f"Building NER radial chart for top {top_n} entities")
        if df.empty:
            update_log("Empty NER dataframe for radial chart")
            return None
        label_counts = df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        counts = label_counts.values
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        widths = np.array([2 * np.pi / len(labels)] * len(labels))
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(111, projection='polar')
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        bars = ax.bar(theta, counts, width=widths, color=colors, edgecolor="black")
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"Radial Chart of Top {top_n} NER Entities", fontsize=12, pad=20)
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER radial chart: {str(e)}")
        return None

# Main application
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
db_selection = st.selectbox("Select Database", db_options, key="db_select")
uploaded_file = None
if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
        update_log(f"Uploaded database saved as {temp_db_path}")
else:
    if db_selection:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
        update_log(f"Selected database: {db_selection}")
if st.session_state.db_file:
    tab1, tab2, tab3 = st.tabs(["Database Inspection", "Common Terms Analysis", "NER Analysis"])
    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner(f"Inspecting {os.path.basename(st.session_state.db_file)}..."):
                st.session_state.term_counts = inspect_database(st.session_state.db_file)
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="inspection_logs")
    with tab2:
        st.header("Common Terms and Phrases")
        with st.sidebar:
            st.subheader("Term Analysis Parameters")
            exclude_words = [w.strip().lower() for w in st.text_input("Exclude Words/Phrases (comma-separated)", key="exclude_words").split(",") if w.strip()]
            top_n = st.slider("Number of Top Terms", min_value=5, max_value=30, value=10, key="top_n")
            min_freq = st.slider("Minimum Frequency", min_value=1, max_value=20, value=5, key="min_freq")
            phrase_weight = st.slider("Phrase Weight", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key="phrase_weight")
            pmi_threshold = st.slider("PMI Threshold", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="pmi_threshold")
            wordcloud_font_size = st.slider("Word Cloud Font Size", min_value=20, max_value=80, value=40, key="wordcloud_font_size")
            font_type = st.selectbox("Font Type", ["None", "DejaVu Sans"], index=0, key="font_type")
            colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "hot", "cool", "rainbow"], index=0, key="colormap")
            network_font_size = st.slider("Network Font Size", min_value=6, max_value=12, value=8, key="network_font_size")
            analyze_terms_button = st.button("Extract Common Terms", key="analyze_terms")
        if analyze_terms_button:
            with st.spinner(f"Extracting terms from {os.path.basename(st.session_state.db_file)}..."):
                st.session_state.raw_common_terms = extract_common_terms(st.session_state.db_file, min_freq, phrase_weight, pmi_threshold)
        if st.session_state.raw_common_terms:
            st.session_state.common_terms = [(term, freq) for term, freq in st.session_state.raw_common_terms if not any(w in term.lower() for w in exclude_words)]
            if not st.session_state.common_terms:
                st.warning("No terms remain after applying exclude words. Adjust exclusions.")
            else:
                st.success(f"Extracted **{len(st.session_state.common_terms)}** terms after filtering!")
                st.subheader("Visualizations")
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = plot_term_histogram(st.session_state.common_terms, top_n)
                    st.pyplot(fig_hist)
                with col2:
                    fig_cloud = plot_word_cloud(st.session_state.common_terms, top_n, wordcloud_font_size, font_type, colormap)
                    st.pyplot(fig_cloud)
                fig_net = plot_term_co_occurrence(st.session_state.common_terms, top_n, st.session_state.db_file, network_font_size, colormap)
                if fig_net:
                    st.pyplot(fig_net)
                else:
                    st.warning("No term co-occurrences found.")
                term_df = pd.DataFrame(st.session_state.common_terms, columns=["Term/Phrase", "Frequency"])
                st.subheader("Common Terms")
                st.dataframe(term_df, use_container_width=True)
                term_csv = term_df.to_csv(index=False)
                st.download_button("Download Term CSV", data=term_csv, file_name="terms.csv", mime="text/csv", key="download_terms")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="common_terms_logs")
    with tab3:
        st.header("NER Analysis")
        if st.session_state.term_counts or st.session_state.common_terms:
            available_terms = []
            if st.session_state.term_counts:
                available_terms += [term for term, count in st.session_state.term_counts.items() if count > 0]
            if st.session_state.common_terms:
                available_terms += [term for term, _ in st.session_state.common_terms]
            available_terms = sorted(list(set(available_terms)))
            default_terms = [term for term in ["lithium-ion battery", "stress", "strain", "young’s modulus", "volume expansion"] if term in available_terms]
            selected_terms = st.multiselect("Select Terms for NER", options=available_terms, default=default_terms, key="select_terms")
            if st.button("Run NER Analysis", key="analyze_ner"):
                if not selected_terms:
                    st.warning("Select at least one term for NER.")
                else:
                    with st.spinner(f"Performing NER analysis for {len(selected_terms)} terms..."):
                        ner_df = perform_ner_on_terms(st.session_state.db_file, selected_terms)
                        st.session_state.ner_results = ner_df
                    if ner_df.empty:
                        st.warning("No entities found. Check logs or try different terms.")
                        update_log("No entities extracted. Possible issues: terms not in spans, low similarity, or invalid content.")
                    else:
                        st.success(f"Extracted **{len(ner_df)}** entities!")
                        st.dataframe(
                            ner_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context", "score"]].head(100),
                            use_container_width=True
                        )
                        ner_csv = ner_df.to_csv(index=False)
                        st.download_button(
                            label="Download NER CSV",
                            data=ner_csv,
                            file_name="ner_data.csv",
                            mime="text/csv",
                            key="download_ner"
                        )
                        st.subheader("NER Visualizations")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("NER Co-occurrence Network")
                            fig_net = plot_ner_co_occurrence(ner_df, top_n, network_font_size, colormap)
                            if fig_net:
                                st.pyplot(fig_net)
                            else:
                                st.warning("No co-occurrences found.")
                        with col2:
                            st.subheader("NER Histogram")
                            fig_hist = plot_ner_histogram(ner_df, top_n, colormap)
                            if fig_hist:
                                st.pyplot(fig_hist)
                            else:
                                st.warning("No entities for histogram.")
                        st.subheader("NER Radial Chart")
                        fig_radial = plot_ner_radial(ner_df, top_n, colormap)
                        if fig_radial:
                            st.pyplot(fig_radial)
                        else:
                            st.warning("No entities for radial chart.")
        else:
            st.warning("Run Database Inspection or Common Terms Analysis first to load available terms.")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")
else:
    st.warning("Select or upload a valid database file.")