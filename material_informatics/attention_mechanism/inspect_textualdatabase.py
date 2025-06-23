import streamlit as st
import sqlite3
import pandas as pd
import os
import glob
import uuid
from pathlib import Path
import spacy
import re
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import logging

# Define database directory
DB_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'inspect_database.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Database Inspection and NER Tool", layout="wide")
st.title("SQLite Database Inspection and NER Analysis Tool")
st.markdown("""
This tool allows you to inspect SQLite database (.db) files and perform Named Entity Recognition (NER) analysis. Upload a database or select one from the local directory to:
- View all tables and the schema of the 'papers' table.
- See sample content and count valid papers.
- Search for specific terms in the content.
- Download a sample of the content as a CSV.
- Perform NER analysis on selected terms.
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

# Customize tokenizer for hyphenated phrases
from spacy.language import Language
@Language.component("custom_tokenizer")
def custom_tokenizer(doc):
    hyphenated_phrases = ["lithium-ion", "Li-ion"]
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
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "ner_results" not in st.session_state:
    st.session_state.ner_results = None

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

# Perform NER on selected terms
def perform_ner_on_terms(db_file, selected_terms):
    try:
        update_log(f"Starting NER analysis for terms: {', '.join(selected_terms)}")
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            update_log(f"No valid papers in {os.path.basename(db_file)}")
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
            "STRAIN": (0, 100, "%"), "STRESS": (0, 1000, "MPa"),
            "VOLUME_EXPANSION": (0, 500, "%"), "YOUNGS_MODULUS": (0, 1000, "GPa")
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
                    # Include tokens, noun chunks, and n-grams for broader coverage
                    spans = [span for span in doc.noun_chunks if len(span.text.split()) <= 5] + \
                            [token for token in doc if token.text.lower() in selected_terms] + \
                            [doc[start:end] for start, end in [(i, i+2) for i in range(len(doc)-1)] + [(i, i+3) for i in range(len(doc)-2)] if len(doc[start:end].text.split()) <= 5]
                    if not spans:
                        update_log(f"No valid spans in paper {row['id']}")
                        continue
                    for span in spans:
                        span_text = span.text.lower().strip()
                        if not span_text:
                            update_log(f"Skipping empty span in paper {row['id']}")
                            continue

                        # Check for exact or similar term match
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

                        # Match to reference labels
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
                                        unit_match = re.search(r"(?:mpa|gpa|%)", span_text, re.IGNORECASE)
                                        unit = unit_match.group(0).upper() if unit_match else None
                                        if unit == "GPA" and label == "YOUNGS_MODULUS":
                                            unit = "GPa"
                                        elif unit == "GPA":
                                            unit = "MPa"
                                            value *= 1000
                                    # Allow all entities, validate numerical ones
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
                                    update_log(f"Extracted entity '{span.text}' as {label} in paper {row['id']}")
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

# Inspect database
def inspect_database(db_path):
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        st.subheader("Tables in Database")
        if tables:
            st.write([table[0] for table in tables])
        else:
            st.warning("No tables found in the database.")
            conn.close()
            return

        # Check if 'papers' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
        if not cursor.fetchone():
            st.warning("No 'papers' table found in the database.")
            conn.close()
            return

        # Inspect schema of 'papers' table
        cursor.execute("PRAGMA table_info(papers);")
        schema = cursor.fetchall()
        st.subheader("Schema of 'papers' Table")
        schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)

        # Extract sample rows (first 5 papers)
        query = "SELECT id, title, year, substr(content, 1, 200) as sample_content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%' LIMIT 5"
        df = pd.read_sql_query(query, conn)
        st.subheader("Sample Rows from 'papers' Table (First 5 Papers)")
        if df.empty:
            st.warning("No valid papers found in the 'papers' table.")
        else:
            st.dataframe(df, use_container_width=True)

        # Count total valid papers
        cursor.execute("SELECT COUNT(*) as count FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'")
        total_papers = cursor.fetchone()[0]
        st.subheader("Total Valid Papers")
        st.write(f"{total_papers} papers")

        # Search for specific terms in content
        terms_to_search = ["lithium-ion battery", "stress", "strain", "young’s modulus", "volume expansion"]
        st.subheader("Term Frequency in 'content' Column")
        term_counts = {}
        for term in terms_to_search:
            cursor.execute(f"SELECT COUNT(*) FROM papers WHERE content LIKE '%{term}%' AND content IS NOT NULL AND content NOT LIKE 'Error%'")
            count = cursor.fetchone()[0]
            term_counts[term] = count
            st.write(f"'{term}': {count} papers")

        # Save full content sample to CSV (first 10 papers, limited content length)
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
        st.write(f"Uploaded database saved as {os.path.basename(temp_db_path)}")
else:
    if db_selection:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
        st.write(f"Selected database: {db_selection}")

# Tabs for inspection and NER
if st.session_state.db_file:
    tab1, tab2 = st.tabs(["Database Inspection", "NER Analysis"])

    with tab1:
        st.header("Database Inspection")
        if st.button("Inspect Database", key="inspect_button"):
            with st.spinner(f"Inspecting {os.path.basename(st.session_state.db_file)}..."):
                term_counts = inspect_database(st.session_state.db_file)
                if term_counts:
                    st.session_state.term_counts = term_counts
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="inspection_logs")

    with tab2:
        st.header("NER Analysis")
        if "term_counts" in st.session_state and st.session_state.term_counts:
            available_terms = [term for term, count in st.session_state.term_counts.items() if count > 0]
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
        else:
            st.warning("Run Database Inspection first to load available terms.")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="ner_logs")
else:
    st.warning("Select or upload a valid database file.")

