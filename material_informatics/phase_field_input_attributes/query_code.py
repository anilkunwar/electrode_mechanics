import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
import pickle
import torch
from datetime import datetime
import logging
import time
import h5py
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial.distance import cosine
from fuzzywuzzy import fuzz
import pyarrow.parquet as pq

# Initialize logging
logging.basicConfig(filename='phase_field_optimized_query_scibert.log', level=logging.DEBUG)

# Initialize Streamlit app
st.set_page_config(page_title="Optimized Phase Field Model Paper Query Tool with SciBERT", layout="wide")
st.title("Optimized arXiv Query for Phase Field Models in Li/Sn-based Battery Anodes")
st.markdown("""
This tool performs an optimized search on arXiv for papers on phase field models involving lithium (Li), tin (Sn), or Li-Sn systems for battery anodes, targeting 290 K to 340 K. It uses an expanded synonym dictionary (including terms like 'energies' and 'energy'), fuzzy matching, and a fallback query to maximize relevant papers, with SciBERT for ranking. PDFs are downloaded, and full text is saved to `phase_field_knowledgeuniverse.db` for NER analysis. Results are saved in CSV, SQLite, Parquet, JSON, .pkl, .h5, and .pt formats.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `arxiv`, `pymupdf`, `pandas`, `streamlit`, `fuzzywuzzy`, `python-Levenshtein`, `transformers`, `torch`, `scipy`, `pyarrow`, `h5py`, `pickle`
- Install with: `pip install arxiv pymupdf pandas streamlit fuzzywuzzy python-Levenshtein transformers torch scipy pyarrow h5py`
- SciBERT model is automatically downloaded via transformers.
""")

# Initialize SciBERT for ranking
try:
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
except Exception as e:
    st.error(f"Failed to load SciBERT: {str(e)}. Please ensure transformers and torch are installed.")
    logging.error(f"SciBERT loading failed: {str(e)}")
    st.stop()

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Function to compute simple SciBERT embeddings
def get_scibert_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token embedding
    except Exception as e:
        logging.error(f"SciBERT embedding failed for text: {text[:50]}...: {str(e)}")
        return None

# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    return 1 - cosine(embedding1, embedding2)

# Query arXiv function with optimized filtering
def query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases=[]):
    try:
        query_terms = query.strip().split()
        formatted_terms = []
        synonyms = {
            "phase field": ["phase-field", "phase field model", "phase field simulation", "diffuse interface", 
                            "phase-field simulation", "phase field method", "PFM", "phase-field approach", 
                            "phase field theory", "phase field modeling", "phase boundary dynamics", 
                            "phase-field method", "diffuse phase boundary"],
            "lithium": ["Li ", "Li-ion", "lithium-ion", "Li anode", "lithium battery", "Li-based", "Li-Sn", 
                        "lithium tin", "Li alloy", "lithium electrode", "Li metal", "lithium-based anode", 
                        "Li-ion battery"],
            "tin": ["Sn ", "tin anode", "Sn-based", "tin alloy", "Li-Sn", "lithium tin", "Sn electrode", 
                    "tin battery", "tin-based anode", "Sn alloy electrode"],
            "battery anode": ["anode material", "battery electrode", "electrode material", "anode", "Li-ion anode", 
                             "electrochemical anode", "battery material", "anode performance", "electrode performance", 
                             "negative electrode"],
            "temperature": ["Kelvin", "K ", "thermal", "temp", "temperature-dependent", "thermodynamic", 
                            "thermal properties", "temperature effects", "heat", "thermal behavior"],
            "interface energy": ["interfacial energy", "surface energy", "interface tension", "interfacial tension", 
                                 "boundary energy", "energies", "energy", "surface tension", "interface properties", 
                                 "interfacial energies", "boundary tension"],
            "diffuse interface width": ["interface thickness", "interfacial width", "diffuse interface", 
                                       "interface broadening", "gradient width", "interface layer", "diffuse boundary", 
                                       "interface width", "phase boundary width"],
            "model parameter": ["simulation parameter", "computational parameter", "model coefficient", 
                                "parameter estimation", "phase field parameter", "model calibration", 
                                "simulation coefficient", "computational model", "parameter optimization"]
        }
        api_terms = []
        for term in query_terms:
            if term.startswith('"') and term.endswith('"'):
                api_terms.append(term.strip('"').replace(" ", "+"))
            else:
                term = term.lower()
                api_terms.append(term)
                for key, syn_list in synonyms.items():
                    if term == key:
                        api_terms.extend(syn_list)
        api_query = " OR ".join(api_terms)
        for phrase in exact_phrases:
            api_query += f' AND "{phrase.replace(" ", "+")}"'
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=api_query,
            max_results=max_results * 2,  # Oversample
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        reference_terms = ["phase field", "temperature", "interface energy", "battery", "diffuse interface width", "model parameter"]
        reference_embeddings = {term: get_scibert_embedding(term) for term in reference_terms}
        
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract = result.summary.lower()
                title = result.title.lower()
                # Pre-filter for relevance (Li, Sn, or phase field)
                relevant_terms = ["phase field", "phase-field", "li ", "sn ", "li-sn", "battery", "anode", 
                                 "interface energy", "diffuse interface", "phase field modeling", "energies", 
                                 "energy", "surface tension", "phase boundary dynamics", "Li-ion battery", 
                                 "tin-based anode"]
                if not any(term in abstract or term in title for term in relevant_terms):
                    logging.debug(f"Skipping paper {result.entry_id}: no relevant terms in abstract or title")
                    continue
                
                query_words = set(word.lower() for word in re.split(r'\s+|\".*?\"', query) if word and not word.startswith('"'))
                for key, syn_list in synonyms.items():
                    if key in query_words:
                        query_words.update(syn_list)
                matched_terms = []
                for word in query_words:
                    if word in abstract or word in title:
                        matched_terms.append(word)
                    else:
                        for text in [abstract, title]:
                            words = text.split()
                            for w in words:
                                if fuzz.partial_ratio(word, w) > 75:
                                    matched_terms.append(word)
                                    break
                matched_terms = list(set(matched_terms))
                if len(matched_terms) >= 1:
                    match_score = len(matched_terms) / max(1, len(query_words))
                    # Compute SciBERT similarity for ranking
                    abstract_embedding = get_scibert_embedding(abstract)
                    similarity_scores = {
                        term: compute_similarity(abstract_embedding, reference_embeddings[term])
                        for term in reference_terms
                    }
                    # Temperature check (secondary filter)
                    temp_matches = re.findall(r'(\d+\.?\d*)\s*(K|kelvin)', abstract + " " + title)
                    temp_valid = any(290 <= float(temp) <= 340 for temp, unit in temp_matches if unit.lower() in ["k", "kelvin"])
                    temp_score = 1.0 if temp_valid else 0.8
                    
                    # Simple weighted score for ranking
                    weights = {
                        "phase_field": 0.3,
                        "temperature": 0.2,
                        "interface_energy": 0.15,
                        "battery": 0.15,
                        "diffuse_interface_width": 0.1,
                        "model_parameter": 0.1
                    }
                    comprehensive_score = sum(similarity_scores[term] * weights.get(term.replace(" ", "_"), 0.1) for term in reference_terms) * temp_score
                    
                    abstract_highlighted = abstract
                    for term in matched_terms:
                        abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                    
                    paper_data = {
                        "paper_id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "year": result.published.year,
                        "categories": ", ".join(result.categories),
                        "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                        "abstract_highlighted": abstract_highlighted[:200] + "..." if len(abstract_highlighted) > 200 else abstract_highlighted,
                        "pdf_url": result.pdf_url,
                        "download_status": "Not downloaded",
                        "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                        "match_score": round(match_score * 100),
                        "comprehensive_score": round(comprehensive_score, 3),
                        "pdf_path": None,
                        "full_text": None
                    }
                    for term in reference_terms:
                        paper_data[f"{term.replace(' ', '_')}_score"] = round(similarity_scores[term], 3)
                    papers.append(paper_data)
                    logging.debug(f"Paper {result.entry_id} included: score={comprehensive_score:.3f}, matched_terms={matched_terms}")
                else:
                    logging.debug(f"Paper {result.entry_id} excluded: insufficient matched terms ({matched_terms})")
            
            if len(papers) >= max_results:
                break
        
        # Fallback query if too few papers
        if len(papers) < 10:
            logging.info(f"Fallback query triggered: too few papers ({len(papers)})")
            fallback_query = "phase field OR lithium OR tin OR li-sn OR battery anode OR diffuse interface OR interface energy OR energies OR phase field modeling OR Li-ion battery"
            search = arxiv.Search(
                query=fallback_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            for result in client.results(search):
                if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                    abstract = result.summary.lower()
                    title = result.title.lower()
                    if not any(term in abstract or term in title for term in relevant_terms):
                        continue
                    matched_terms = []
                    for word in ["phase field", "li ", "sn ", "li-sn", "battery", "anode", "interface energy", "energies", "phase field modeling", "Li-ion battery"]:
                        if word in abstract or word in title:
                            matched_terms.append(word)
                        else:
                            for text in [abstract, title]:
                                words = text.split()
                                for w in words:
                                    if fuzz.partial_ratio(word, w) > 75:
                                        matched_terms.append(word)
                                        break
                    matched_terms = list(set(matched_terms))
                    if len(matched_terms) >= 1 and result.entry_id not in [p["paper_id"] for p in papers]:
                        match_score = len(matched_terms) / 10
                        abstract_embedding = get_scibert_embedding(abstract)
                        similarity_scores = {
                            term: compute_similarity(abstract_embedding, reference_embeddings[term])
                            for term in reference_terms
                        }
                        temp_matches = re.findall(r'(\d+\.?\d*)\s*(K|kelvin)', abstract + " " + title)
                        temp_valid = any(290 <= float(temp) <= 340 for temp, unit in temp_matches if unit.lower() in ["k", "kelvin"])
                        temp_score = 1.0 if temp_valid else 0.8
                        comprehensive_score = sum(similarity_scores[term] * weights.get(term.replace(" ", "_"), 0.1) for term in reference_terms) * temp_score
                        
                        abstract_highlighted = abstract
                        for term in matched_terms:
                            abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                        
                        paper_data = {
                            "paper_id": result.entry_id.split('/')[-1],
                            "title": result.title,
                            "year": result.published.year,
                            "categories": ", ".join(result.categories),
                            "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                            "abstract_highlighted": abstract_highlighted[:200] + "..." if len(abstract_highlighted) > 200 else abstract_highlighted,
                            "pdf_url": result.pdf_url,
                            "download_status": "Not downloaded",
                            "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                            "match_score": round(match_score * 100),
                            "comprehensive_score": round(comprehensive_score, 3),
                            "pdf_path": None,
                            "full_text": None
                        }
                        for term in reference_terms:
                            paper_data[f"{term.replace(' ', '_')}_score"] = round(similarity_scores[term], 3)
                        papers.append(paper_data)
                        logging.debug(f"Fallback: Paper {result.entry_id} included: score={comprehensive_score:.3f}, matched_terms={matched_terms}")
                
                if len(papers) >= max_results:
                    break
        
        # Sort by comprehensive score
        papers.sort(key=lambda x: x["comprehensive_score"], reverse=True)
        
        logging.info(f"Found {len(papers)} papers for query: {api_query}")
        st.write(f"Found {len(papers)} papers for query: {api_query}")
        return papers
    except Exception as e:
        logging.error(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}")
        return []

# Download PDF function
def download_pdf(pdf_url, paper_id):
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
        return f"Downloaded ({file_size:.2f} KB)", pdf_path
    except Exception as e:
        logging.error(f"PDF download failed for {paper_id}: {str(e)}")
        return f"Failed: {str(e)}", None

# Extract full text from PDF
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

# Save to SQLite database with full text
def save_to_sqlite(df, db_file="phase_field_knowledgeuniverse.db"):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS full_text (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                full_text TEXT
            )
        """)
        df = df.rename(columns={"id": "paper_id"})
        df[["paper_id", "title", "year", "categories", "abstract", "pdf_url", "full_text"]].to_sql(
            "full_text", conn, if_exists="replace", index=False
        )
        conn.commit()
        conn.close()
        return f"Saved metadata and full text to {db_file}"
    except Exception as e:
        logging.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Save to Parquet
def save_to_parquet(df, parquet_file="phase_field_papers_metadata.parquet"):
    try:
        df = df.rename(columns={"id": "paper_id"})
        df[["paper_id", "title", "year", "categories", "abstract", "pdf_url", "full_text"]].to_parquet(parquet_file, index=False)
        return f"Saved metadata to {parquet_file}"
    except Exception as e:
        logging.error(f"Parquet save failed: {str(e)}")
        return f"Failed to save to Parquet: {str(e)}"

# Save to additional formats (.pkl, .h5, .pt)
def save_additional_formats(df, base_name="phase_field_papers_metadata"):
    try:
        df = df.rename(columns={"id": "paper_id"})
        pkl_path = f"{base_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(df[["paper_id", "title", "year", "categories", "abstract", "pdf_url", "full_text"]], f)
        h5_path = f"{base_name}.h5"
        df[["paper_id", "title", "year", "categories", "abstract", "pdf_url", "full_text"]].to_hdf(h5_path, key="metadata", mode="w")
        pt_path = f"{base_name}.pt"
        torch.save(df[["paper_id", "title", "year", "categories", "pdf_url", "full_text"]].to_dict(orient="records"), pt_path)
        return h5_path, pkl_path, pt_path
    except Exception as e:
        logging.error(f"Failed to save additional formats: {str(e)}")
        return None, None, None

# Main Streamlit app
st.header("Optimized arXiv Query for Phase Field Model Papers")
st.markdown("Perform an optimized search for papers on phase field models involving Li, Sn, or Li-Sn systems for battery anodes (290 K to 340 K). Uses an expanded synonym dictionary (including 'energy' and 'energies'), fuzzy matching, and a fallback query to maximize relevant papers. Saves results in multiple formats for NER analysis.")

# Sidebar for search inputs
with st.sidebar:
    st.subheader("arXiv Search Parameters")
    st.markdown("Customize your search for phase field models in Li/Sn-based battery anodes.")
    
    query_option = st.radio(
        "Select Query Type",
        ["Default Query", "Custom Query", "Suggested Queries"],
        help="Choose how to specify the search query."
    )
    exact_phrases = []
    if query_option == "Default Query":
        query = "phase field lithium tin li-sn battery anode temperature interface energy diffuse interface width model parameter"
        st.write("Using default query: **" + query + "**")
    elif query_option == "Custom Query":
        query = st.text_input("Enter Custom Query", value="phase field li sn li-sn battery anode temperature")
        exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"phase field model\")", value="")
        exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
        st.write("Custom query: **" + query + "**")
        if exact_phrases:
            st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
    else:
        suggested_queries = [
            "phase field lithium battery anode temperature",
            "phase field tin li-sn anode interface energy",
            "phase field li-sn battery model parameter",
            "diffuse interface lithium anode temperature",
            "phase field model li sn 290 K 340 K"
        ]
        query = st.selectbox("Choose Suggested Query", suggested_queries)
        exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"phase field model\")", value="")
        exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
        st.write("Selected query: **" + query + "**")
        if exact_phrases:
            st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
    
    default_categories = ["cond-mat.mtrl-sci", "physics.chem-ph", "physics.app-ph"]
    extra_categories = ["physics.optics", "cs.CE", "eess.sy"]
    categories = st.multiselect(
        "Select arXiv Categories",
        default_categories + extra_categories,
        default=default_categories,
        help="Filter papers by categories."
    )
    
    max_results = st.slider(
        "Maximum Number of Papers",
        min_value=1,
        max_value=500,
        value=100,
        help="Set the maximum number of papers to retrieve."
    )
    
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=1900,
            max_value=current_year,
            value=2000,
            help="Earliest publication year."
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=start_year,
            max_value=current_year,
            value=current_year,
            help="Latest publication year."
        )
    
    output_formats = st.multiselect(
        "Select Output Formats",
        ["CSV", "SQLite (.db)", "Parquet (.parquet)", "JSON", "Pickle (.pkl)", "HDF5 (.h5)", "PyTorch (.pt)"],
        default=["CSV", "SQLite (.db)", "Pickle (.pkl)"],
        help="Choose formats for saving metadata and full text."
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
        with st.spinner("Querying arXiv and processing with optimized SciBERT ranking..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases)
        
        if not papers:
            st.warning("No papers found matching your criteria.")
            st.markdown("""
            **Suggestions to find more papers:**
            - Use broader terms (e.g., 'phase field battery' or 'energies').
            - Include more synonyms in exact phrases (e.g., "phase-field modeling", "interfacial energies").
            - Add more categories (e.g., 'cs.CE' for computational engineering).
            - Extend the year range (e.g., 1990–2025).
            - Increase the maximum number of papers (up to 500).
            - Check the log file (phase_field_optimized_query_scibert.log) for details.
            """)
        else:
            st.success(f"Found **{len(papers)}** papers matching your query!")
            exact_display = ', '.join(f'"{p}"' for p in exact_phrases) if exact_phrases else 'None'
            st.write(f"Query: **{query}** | Exact Phrases: **{exact_display}**")
            st.write(f"Categories: **{', '.join(categories)}** | Years: **{start_year}–{end_year}**")
            
            st.subheader("Downloading PDFs and Extracting Full Text")
            progress_bar = st.progress(0)
            for i, paper in enumerate(papers):
                if paper["pdf_url"]:
                    status, pdf_path = download_pdf(paper["pdf_url"], paper["paper_id"])
                    paper["download_status"] = status
                    paper["pdf_path"] = pdf_path
                    if pdf_path:
                        full_text = extract_text_from_pdf(pdf_path)
                        paper["full_text"] = full_text if not full_text.startswith("Error") else None
                    else:
                        paper["full_text"] = None
                else:
                    paper["download_status"] = "No PDF URL"
                    paper["pdf_path"] = None
                    paper["full_text"] = None
                progress_bar.progress((i + 1) / len(papers))
                time.sleep(0.1)
            
            df = pd.DataFrame(papers)
            st.subheader("Paper Details")
            st.dataframe(
                df[["paper_id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "match_score",
                    "comprehensive_score", "phase_field_score", "temperature_score", "interface_energy_score",
                    "battery_score", "diffuse_interface_width_score", "model_parameter_score", "download_status", "pdf_path"]],
                use_container_width=True,
                column_config={
                    "abstract_highlighted": st.column_config.TextColumn("Abstract (Highlighted)", help="Matched terms in bold orange."),
                    "comprehensive_score": st.column_config.NumberColumn("Comprehensive Score", help="Weighted SciBERT score for ranking."),
                    "phase_field_score": st.column_config.NumberColumn("Phase Field Score", help="SciBERT similarity score."),
                    "temperature_score": st.column_config.NumberColumn("Temperature Score", help="SciBERT similarity score."),
                    "interface_energy_score": st.column_config.NumberColumn("Interface Energy Score", help="SciBERT similarity score."),
                    "battery_score": st.column_config.NumberColumn("Battery Score", help="SciBERT similarity score."),
                    "diffuse_interface_width_score": st.column_config.NumberColumn("Diffuse Interface Width Score", help="SciBERT similarity score."),
                    "model_parameter_score": st.column_config.NumberColumn("Model Parameter Score", help="SciBERT similarity score.")
                }
            )
            
            # Save in selected formats
            if "CSV" in output_formats:
                csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                csv_path = "phase_field_papers_metadata.csv"
                with open(csv_path, "w") as f:
                    f.write(csv)
                st.info(f"Metadata CSV saved as {csv_path}. Automatic download starting...")
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="Download Paper Metadata CSV (Automatic)",
                        data=f,
                        file_name="phase_field_papers_metadata.csv",
                        mime="text/csv",
                        key=f"auto_download_{time.time()}"
                    )
                st.download_button(
                    label="Download Paper Metadata CSV (Manual)",
                    data=csv,
                    file_name="phase_field_papers_metadata.csv",
                    mime="text/csv",
                    key="manual_download"
                )
            
            if "SQLite (.db)" in output_formats:
                sqlite_status = save_to_sqlite(df)
                st.info(sqlite_status)
            
            if "Parquet (.parquet)" in output_formats:
                parquet_status = save_to_parquet(df)
                st.info(parquet_status)
            
            if "JSON" in output_formats:
                json_path = "phase_field_papers_metadata.json"
                df.rename(columns={"id": "paper_id"}).drop(columns=["abstract_highlighted"]).to_json(json_path, orient="records", lines=True)
                st.info(f"Saved metadata to {json_path}")
                with open(json_path, "rb") as f:
                    st.download_button(
                        label="Download Paper Metadata JSON",
                        data=f,
                        file_name="phase_field_papers_metadata.json",
                        mime="application/json",
                        key="json_download"
                    )
            
            if any(fmt in output_formats for fmt in ["Pickle (.pkl)", "HDF5 (.h5)", "PyTorch (.pt)"]):
                h5_path, pkl_path, pt_path = save_additional_formats(df)
                if h5_path:
                    st.info(f"Saved additional formats: {h5_path}, {pkl_path}, {pt_path}")
                    for path, mime in [
                        (h5_path, "application/x-hdf"),
                        (pkl_path, "application/octet-stream"),
                        (pt_path, "application/octet-stream")
                    ]:
                        if path and os.path.exists(path):
                            with open(path, "rb") as f:
                                st.download_button(
                                    label=f"Download {path}",
                                    data=f,
                                    file_name=path,
                                    mime=mime,
                                    key=f"download_{path}_{time.time()}"
                                )
            
            downloaded = sum(1 for p in papers if "Downloaded" in p["download_status"])
            extracted = sum(1 for p in papers if p["full_text"] and not p["full_text"].startswith("Error"))
            st.write(f"**Summary**: {len(papers)} papers found, {downloaded} PDFs downloaded successfully, {extracted} full texts extracted.")
            if downloaded < len(papers):
                st.warning("Some PDFs failed to download. Check 'download_status' for details.")
            if extracted < downloaded:
                st.warning("Some full texts failed to extract. Check log file for details.")
            common_terms = set()
            for terms in df["matched_terms"]:
                if terms and terms != "None":
                    common_terms.update(terms.split(", "))
            if common_terms:
                st.markdown(f"**Query Refinement Tip**: Common matched terms: {', '.join(common_terms)}. Try focusing on these (e.g., '{' '.join(list(common_terms)[:3])}').")

# Footer
st.markdown("---")
st.write("Developed for optimized querying of arXiv papers on phase field models for Li/Sn-based battery anodes using SciBERT ranking.")
st.markdown("**How to Run**:")
st.markdown("""
1. Install dependencies: `pip install arxiv pymupdf pandas streamlit fuzzywuzzy python-Levenshtein transformers torch scipy pyarrow h5py`
2. Save this code as `arxiv_phase_field_optimized_query_scibert.py`.
3. Run with: `streamlit run arxiv_phase_field_optimized_query_scibert.py --server.fileWatcherType none`
4. Use the generated `phase_field_knowledgeuniverse.db` or `.pkl` file for NER analysis with `phase_field_ner_analysis.py`.
""")