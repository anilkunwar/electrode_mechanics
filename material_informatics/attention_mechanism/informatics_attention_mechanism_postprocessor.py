import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
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
import seaborn as sns

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

DB_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(filename=os.path.join(DB_DIR, 'common_term_ner_scibert.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Common Term and NER Analysis Tool (SciBERT)", layout="wide")
st.title("Common Term and Rule-Based NER Analysis for Lithium-Ion Battery Mechanics (SciBERT)")
st.markdown("""
This tool inspects SQLite databases, extracts common terms and phrases, and performs rule-based NER analysis using SciBERT for lithium-ion battery mechanics.
Select or upload a database, then use the tabs to inspect the database, analyze terms, or extract entities associated with numerical values.
""")

try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

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
nlp.max_length = 500_000

try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

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

@st.cache_data
def get_scibert_embedding(text):
    try:
        if not text.strip():
            update_log(f"Skipping empty text for SciBERT embedding")
            return None
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
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

def perform_ner_on_terms(db_file, selected_terms):
    try:
        update_log(f"Starting rule-based NER for terms: {', '.join(selected_terms)}")
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
        entity_set = set()
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
            "YOUNGS_MODULUS": (0, 1000, "GPa"),
            "BATTERY": (0, 2000, "mAh/g")
        }
        similarity_threshold = 0.7
        ref_embeddings = {label: [get_scibert_embedding(term) for term in terms if get_scibert_embedding(term) is not None] for label, terms in reference_terms.items()}
        numerical_pattern = r"(\d+\.?\d*[eE]?-?\d*)\s*(mpa|gpa|kpa|pa|%|mAh/g)"
        term_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in selected_terms}
        batch_size = 2
        for batch_start in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_start:batch_start+batch_size]
            for _, row in batch_df.iterrows():
                try:
                    text = row["content"].lower()
                    if len(text) > nlp.max_length:
                        text = text[:nlp.max_length]
                        update_log(f"Truncated content for paper {row['id']}")
                    if not text.strip() or len(text) < 10:
                        update_log(f"Skipping paper {row['id']} due to empty/short content")
                        continue
                    doc = nlp(text)
                    spans = []
                    for sent in doc.sents:
                        for term in selected_terms:
                            if term_patterns[term].search(sent.text):
                                matches = re.finditer(numerical_pattern, sent.text, re.IGNORECASE)
                                for match in matches:
                                    start_char = sent.start_char + match.start()
                                    end_char = sent.start_char + match.end()
                                    span = doc.char_span(start_char, end_char, alignment_mode="expand")
                                    if span:
                                        spans.append(span)
                    if not spans:
                        update_log(f"No valid spans in paper {row['id']}")
                        continue
                    for span in spans:
                        span_text = span.text.lower().strip()
                        if not span_text:
                            update_log(f"Skipping empty span '{span_text}' in paper {row['id']}")
                            continue
                        term_matched = False
                        for term in selected_terms:
                            if term_patterns[term].search(span_text):
                                term_matched = True
                                break
                        if not term_matched:
                            span_embedding = get_scibert_embedding(span_text)
                            if span_embedding is None:
                                update_log(f"Skipping span '{span_text}' in paper {row['id']}: no embedding")
                                continue
                            term_embeddings = [get_scibert_embedding(term) for term in selected_terms if get_scibert_embedding(term) is not None]
                            similarities = [
                                np.dot(span_embedding, t_emb) / (np.linalg.norm(span_embedding) * np.linalg.norm(t_emb))
                                for t_emb in term_embeddings
                                if np.linalg.norm(span_embedding) != 0 and np.linalg.norm(t_emb) != 0
                            ]
                            if any(s > 0.6 for s in similarities):
                                term_matched = True
                        if not term_matched:
                            update_log(f"No term match for span '{span_text}' in paper {row['id']}")
                            continue
                        value_match = re.match(numerical_pattern, span_text, re.IGNORECASE)
                        if not value_match:
                            update_log(f"Skipping span '{span_text}' in paper {row['id']}: no numerical value")
                            continue
                        value = float(value_match.group(1))
                        unit = value_match.group(2).upper()
                        if unit == "GPA":
                            unit = "GPa"
                        elif unit == "KPA":
                            unit = "MPa"
                            value /= 1000
                        elif unit == "PA":
                            unit = "MPa"
                            value /= 1_000_000
                        elif unit == "MAH/G":
                            unit = "mAh/g"
                        span_embedding = get_scibert_embedding(span_text)
                        if span_embedding is None:
                            update_log(f"Skipping span '{span_text}' in paper {row['id']}: no embedding for label")
                            continue
                        best_label = None
                        best_similarity = 0
                        for label, ref_embeds in ref_embeddings.items():
                            for ref_embed in ref_embeds:
                                if np.linalg.norm(span_embedding) == 0 or np.linalg.norm(ref_embed) == 0:
                                    continue
                                similarity = np.dot(span_embedding, ref_embed) / (np.linalg.norm(span_embedding) * np.linalg.norm(ref_embed))
                                if similarity > similarity_threshold and similarity > best_similarity:
                                    best_label = label
                                    best_similarity = similarity
                        if not best_label:
                            update_log(f"No label match for span '{span_text}' in paper {row['id']}")
                            continue
                        if best_label in valid_ranges:
                            min_val, max_val, expected_unit = valid_ranges[best_label]
                            if unit == "GPA" and best_label == "YOUNGS_MODULUS":
                                unit = "GPa"
                            elif unit == "GPA":
                                unit = "MPa"
                                value *= 1000
                            if not (min_val <= value <= max_val and (unit == expected_unit or unit is None)):
                                update_log(f"Skipping span '{span_text}' in paper {row['id']}: invalid value/unit ({value} {unit})")
                                continue
                        entity_key = (row["id"], span_text, best_label, value, unit)
                        if entity_key in entity_set:
                            continue
                        entity_set.add(entity_key)
                        context_start = max(0, span.start_char - 100)
                        context_end = min(len(text), span.end_char + 100)
                        context_text = text[context_start:context_end].replace("\n", " ")
                        entities.append({
                            "paper_id": row["id"],
                            "title": row["title"],
                            "year": row["year"],
                            "entity_text": span.text,
                            "entity_label": best_label,
                            "value": value,
                            "unit": unit,
                            "context": context_text,
                            "score": best_similarity
                        })
                        update_log(f"Extracted entity: term='{span.text}', label={best_label}, value={value}, unit={unit}, paper_id={row['id']}")
                except MemoryError as e:
                    update_log(f"Memory error in paper {row['id']}: {str(e)}")
                    st.error("Memory exhausted. Try reducing text length or batch size.")
                    return pd.DataFrame()
                except Exception as e:
                    update_log(f"Error processing paper {row['id']}: {str(e)}")
                progress_bar.progress(min((batch_start + _ + 1) / len(df), 1.0))
        update_log(f"Completed NER analysis: extracted {len(entities)} entities")
        if not entities:
            update_log("No entities extracted. Possible issues: no numerical values, strict rules, or invalid content.")
        return pd.DataFrame(entities)
    except Exception as e:
        update_log(f"NER analysis failed: {str(e)}")
        st.error(f"NER analysis failed: {str(e)}")
        return pd.DataFrame()

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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud of Top {top_n} Terms")
    plt.tight_layout()
    return fig

@st.cache_data
def plot_term_histogram(terms, top_n):
    terms, counts = zip(*terms[:top_n])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(terms, counts, color="skyblue", edgecolor="black")
    ax.set_xlabel("Terms/Phrases")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top {top_n} Terms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    term_df = pd.DataFrame({"Term": terms, "Frequency": counts})
    csv_filename = f"term_histogram_{uuid.uuid4().hex}.csv"
    csv_path = os.path.join(DB_DIR, csv_filename)
    term_df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        csv_data = f.read()
    return fig, csv_data, csv_filename

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
                if term1 != term2:
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        if G.edges():
            fig, ax = plt.subplots(figsize=(8, 8))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_sizes = [500 + 3000 * (G.nodes[term]["freq"] / max(term_freqs.values())) for term in G.nodes]
            node_colors = [cm.get_cmap(colormap)(i / len(top_terms)) for i in range(len(top_terms))]
            edge_widths = [2 * G[u][v]["weight"] / max([d["weight"] for _, _, d in G.edges(data=True)]) for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"Term Co-occurrence Network (Top {top_n} Terms)")
            plt.tight_layout()
            nodes_df = pd.DataFrame([(n, d["type"]) for n, d in G.nodes(data=True)], columns=["node", "type"])
            edges_df = pd.DataFrame([(u, v, d["weight"]) for u, v, d in G.edges(data=True)], columns=["source", "target", "weight"])
            nodes_csv_filename = f"term_co_occurrence_nodes_{uuid.uuid4().hex}.csv"
            edges_csv_filename = f"term_co_occurrence_edges_{uuid.uuid4().hex}.csv"
            nodes_csv_path = os.path.join(DB_DIR, nodes_csv_filename)
            edges_csv_path = os.path.join(DB_DIR, edges_csv_filename)
            nodes_df.to_csv(nodes_csv_path, index=False)
            edges_df.to_csv(edges_csv_path, index=False)
            with open(nodes_csv_path, "rb") as f:
                nodes_csv_data = f.read()
            with open(edges_csv_path, "rb") as f:
                edges_csv_data = f.read()
            return fig, (nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename)
        update_log("No co-occurrences found for term network")
        return None, None
    except Exception as e:
        update_log(f"Error building term co-occurrence network: {str(e)}")
        return None, None

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
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        ax.bar(labels, counts, color=colors, edgecolor="black")
        ax.set_xlabel("Entity Labels")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Top {top_n} NER Entities")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER histogram: {str(e)}")
        return None

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
                if term1 != term2:
                    if G.has_edge(term1, term2):
                        G[term1][term2]["weight"] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        if G.edges():
            fig, ax = plt.subplots(figsize=(6, 6))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            node_colors = [cm.get_cmap(colormap)(i / len(entity_labels)) for i in range(len(entity_labels))]
            edge_widths = [2 * G[u][v]["weight"] for u, v in G.edges()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, width=edge_widths, font_size=font_size, font_weight="bold", ax=ax)
            ax.set_title(f"NER Co-occurrence Network (Top {top_n} Entities)")
            plt.tight_layout()
            return fig
        update_log("No co-occurrences found for NER network")
        return None
    except Exception as e:
        update_log(f"Error plotting NER co-occurrence network: {str(e)}")
        return None

@st.cache_data
def plot_ner_value_histogram(df, top_n, colormap):
    try:
        update_log(f"Building NER value histogram for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values in NER dataframe for value histogram")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities with numerical values and units for value histogram")
            return None
        label_counts = value_df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        for i, label in enumerate(labels):
            values = value_df[value_df["entity_label"] == label]["value"]
            unit = value_df[value_df["entity_label"] == label]["unit"].iloc[0] if not value_df[value_df["entity_label"] == label].empty else "Unknown"
            ax.hist(values, bins=10, alpha=0.5, label=f"{label} ({unit})", color=colors[i], edgecolor="black")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Numerical Values for Top {top_n} NER Entities")
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER value histogram: {str(e)}")
        return None

@st.cache_data
def plot_ner_value_radial(df, top_n, colormap):
    try:
        update_log(f"Building NER value radial chart for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values in NER dataframe for value radial chart")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities with numerical values and units for value radial chart")
            return None
        label_means = value_df.groupby("entity_label").agg({"value": "mean", "unit": "first"}).reset_index()
        label_means = label_means.sort_values("value", ascending=False).head(top_n)
        labels = label_means["entity_label"].tolist()
        values = label_means["value"].tolist()
        units = label_means["unit"].tolist()
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        widths = np.array([2 * np.pi / len(labels)] * len(labels))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        bars = ax.bar(theta, values, width=widths, color=colors, edgecolor="black")
        ax.set_xticks(theta)
        ax.set_xticklabels([f"{label} ({unit})" for label, unit in zip(labels, units)])
        ax.set_title(f"Radial Chart of Average Values for Top {top_n} NER Entities", pad=20)
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER value radial chart: {str(e)}")
        return None

@st.cache_data
def plot_ner_value_boxplot(df, top_n, colormap):
    try:
        update_log(f"Building NER value boxplot for top {top_n} entities")
        if df.empty or df["value"].isna().all():
            update_log("Empty or no numerical values in NER dataframe for value boxplot")
            return None
        value_df = df[df["value"].notna() & df["unit"].notna()]
        if value_df.empty:
            update_log("No entities with numerical values and units for value boxplot")
            return None
        label_counts = value_df["entity_label"].value_counts().head(top_n)
        labels = label_counts.index.tolist()
        data = [value_df[value_df["entity_label"] == label]["value"].values for label in labels]
        units = [value_df[value_df["entity_label"] == label]["unit"].iloc[0] if not value_df[value_df["entity_label"] == label].empty else "Unknown" for label in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [cm.get_cmap(colormap)(i / len(labels)) for i in range(len(labels))]
        box = ax.boxplot(data, patch_artist=True, labels=[f"{label} ({unit})" for label, unit in zip(labels, units)])
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlabel("Entity Labels")
        ax.set_ylabel("Value")
        ax.set_title(f"Box Plot of Numerical Values for Top {top_n} NER Entities")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Error plotting NER value boxplot: {str(e)}")
        return None

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
            font_type = st.selectbox("Font Type", ["None", "DejaVu Sans"], key="font_type")
            colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "hot", "cool", "rainbow"], key="colormap")
            network_font_size = st.slider("Network Font Size", min_value=6, max_value=12, value=8, key="network_font_size")
            analyze_terms_button = st.button("Extract Common Terms", key="analyze_terms")
        if analyze_terms_button:
            with st.spinner(f"Extracting terms from {os.path.basename(st.session_state.db_file)}..."):
                st.session_state.raw_common_terms = extract_common_terms(st.session_state.db_file, min_freq, phrase_weight, pmi_threshold)
        if st.session_state.raw_common_terms:
            st.session_state.common_terms = [(term, freq) for term, freq in st.session_state.raw_common_terms if not any(w in term.lower() for w in exclude_words)]
            if not st.session_state.common_terms:
                st.warning("No terms remain after applying exclude words.")
            else:
                st.success(f"Extracted **{len(st.session_state.common_terms)}** terms!")
                st.subheader("Visualizations")
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist, csv_data, csv_filename = plot_term_histogram(st.session_state.common_terms, top_n)
                    st.pyplot(fig_hist)
                    st.download_button(
                        label="Download Term Histogram Data",
                        data=csv_data,
                        file_name="term_histogram.csv",
                        mime="text/csv",
                        key="download_term_histogram"
                    )
                with col2:
                    fig_cloud = plot_word_cloud(st.session_state.common_terms, top_n, wordcloud_font_size, font_type, colormap)
                    st.pyplot(fig_cloud)
                fig_net, net_csv = plot_term_co_occurrence(st.session_state.common_terms, top_n, st.session_state.db_file, network_font_size, colormap)
                if fig_net:
                    st.pyplot(fig_net)
                    if net_csv:
                        nodes_csv_data, nodes_csv_filename, edges_csv_data, edges_csv_filename = net_csv
                        st.download_button(
                            label="Download Term Co-occurrence Nodes",
                            data=nodes_csv_data,
                            file_name="term_co_occurrence_nodes.csv",
                            mime="text/csv",
                            key="download_term_co_nodes"
                        )
                        st.download_button(
                            label="Download Term Co-occurrence Edges",
                            data=edges_csv_data,
                            file_name="term_co_occurrence_edges.csv",
                            mime="text/csv",
                            key="download_term_co_edges"
                        )
                term_df = pd.DataFrame(st.session_state.common_terms, columns=["Term/Phrase", "Frequency"])
                st.subheader("Common Terms")
                st.dataframe(term_df, use_container_width=True)
                term_csv = term_df.to_csv(index=False)
                st.download_button("Download Term CSV", term_csv, "terms.csv", "text/csv", key="download_terms")
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
            selected_terms = st.multiselect("Select Terms for NER", available_terms, default_terms, key="select_terms")
            if st.button("Run NER Analysis", key="analyze_ner"):
                if not selected_terms:
                    st.warning("Select at least one term for NER.")
                else:
                    with st.spinner(f"Processing NER for {len(selected_terms)} terms..."):
                        ner_df = perform_ner_on_terms(st.session_state.db_file, selected_terms)
                        st.session_state.ner_results = ner_df
                    if ner_df.empty:
                        st.warning("No entities found. Check logs.")
                        update_log("No entities extracted.")
                    else:
                        st.success(f"Extracted **{len(ner_df)}** entities!")
                        st.dataframe(
                            ner_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "context"]].head(100),
                            use_container_width=True
                        )
                        ner_csv = ner_df.to_csv(index=False)
                        st.download_button("Download NER Data CSV", ner_csv, "ner_data.csv", "text/csv", key="download_ner")
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
                            st.subheader("NER Frequency Histogram")
                            fig_hist = plot_ner_histogram(ner_df, top_n, colormap)
                            if fig_hist:
                                st.pyplot(fig_hist)
                            else:
                                st.warning("No entities for frequency histogram.")
                        st.subheader("NER Value Visualizations")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.subheader("Histogram of Numerical Values")
                            fig_value_hist = plot_ner_value_histogram(ner_df, top_n, colormap)
                            if fig_value_hist:
                                st.pyplot(fig_value_hist)
                            else:
                                st.warning("No numerical values for histogram.")
                        with col4:
                            st.subheader("Radial Chart of Average Values")
                            fig_value_radial = plot_ner_value_radial(ner_df, top_n, colormap)
                            if fig_value_radial:
                                st.pyplot(fig_value_radial)
                            else:
                                st.warning("No numerical values for radial chart.")
                        st.subheader("Box Plot of Numerical Values")
                        fig_value_box = plot_ner_value_boxplot(ner_df, top_n, colormap)
                        if fig_value_box:
                            st.pyplot(fig_value_box)
                        else:
                            st.warning("No numerical values for box plot.")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")
else:
    st.warning("Select or upload a database file.")