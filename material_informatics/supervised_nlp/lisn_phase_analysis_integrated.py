import streamlit as st
import PyPDF2
import sqlite3
import spacy
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
import os
import logging
from collections import Counter
import numpy as np
from datetime import datetime
import tempfile
from networkx.algorithms.community import greedy_modularity_communities

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state
if "log_buffer_db" not in st.session_state:
    st.session_state.log_buffer_db = []  # For Create DB tab
if "log_buffer_analysis" not in st.session_state:
    st.session_state.log_buffer_analysis = []  # For Phase Analysis tab
if "filtered_phases" not in st.session_state:
    st.session_state.filtered_phases = None
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []
if "db_created" not in st.session_state:
    st.session_state.db_created = False

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    st.error(f"Failed to load spaCy model: {e}. Please ensure 'en_core_web_sm' is installed.")
    st.stop()

# Define LiₓSnᵧ phase pattern
PHASE_PATTERN = r"Li\d*(?:\.\d+)?Sn\d*"
# Keywords from common_keywords.yaml
KEYWORDS = [
    "Sn anode", "Li-Sn phase transformation", "volume expansion",
    "microstructural evolution", "elastic-strain energy", "electrode potential",
    "Gibbs free energy", "cyclic property degradation"
]
# Additional relevant terms
ADDITIONAL_TERMS = ["SEI formation", "diffusion-induced stress", "reaction-controlled lithiation", "diffusion-controlled lithiation"]
# Mechanical terminology for highlighting
MECHANICAL_TERMS = ["volume expansion", "elastic-strain energy", "diffusion-induced stress", "microstructural evolution"]

def update_log(message, tab="db"):
    """Update the log buffer for the specified tab."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if tab == "db":
        st.session_state.log_buffer_db.append(f"[{timestamp}] {message}")
        if len(st.session_state.log_buffer_db) > 20:
            st.session_state.log_buffer_db.pop(0)
    else:
        st.session_state.log_buffer_analysis.append(f"[{timestamp}] {message}")
        if len(st.session_state.log_buffer_analysis) > 30:
            st.session_state.log_buffer_analysis.pop(0)

def extract_metadata_and_content(pdf_file):
    """Extract title, authors, year, and content from a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        os.unlink(tmp_file_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_file.name}")
            update_log(f"No text extracted from {pdf_file.name}", tab="db")
            return {
                "title": pdf_file.name,
                "authors": "Unknown",
                "year": str(datetime.now().year),
                "content": f"No text extracted from {pdf_file.name}."
            }
        
        # Extract metadata
        title = pdf_file.name
        authors = "Unknown"
        year = str(datetime.now().year)
        
        try:
            metadata = pdf_reader.metadata
            if metadata:
                if "/Title" in metadata and metadata["/Title"]:
                    title = metadata["/Title"]
                if "/Author" in metadata and metadata["/Author"]:
                    authors = metadata["/Author"]
        except Exception as e:
            logger.warning(f"Error accessing PDF metadata for {pdf_file.name}: {e}")
            update_log(f"Error accessing PDF metadata for {pdf_file.name}: {e}", tab="db")
        
        try:
            lines = text.splitlines()
            if lines:
                first_line = lines[0].strip()
                if len(first_line) > 10 and not first_line.lower().startswith(("abstract", "introduction")):
                    title = first_line[:100]
            title_match = re.search(r"(?:Title|TITLE)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()[:100]
            
            author_match = re.search(r"(?:Author|Authors|By)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if author_match:
                authors = author_match.group(1).strip()
            elif "abstract" in text.lower():
                pre_abstract = text[:text.lower().index("abstract")].strip()
                name_pattern = r"[A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)?(?:,\s[A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)?)*"
                author_match = re.search(name_pattern, pre_abstract)
                if author_match:
                    authors = author_match.group(0).strip()
            
            year_match = re.search(r"(?:Year|Published)\s*:\s*(\d{4})", text, re.IGNORECASE)
            if year_match:
                year = year_match.group(1)
            else:
                year_match = re.search(r"\b(20\d{2})\b", text)
                if year_match:
                    year = year_match.group(1)
        except Exception as e:
            logger.warning(f"Error extracting metadata from text for {pdf_file.name}: {e}")
            update_log(f"Error extracting metadata from text for {pdf_file.name}: {e}", tab="db")
        
        logger.info(f"Extracted metadata for {pdf_file.name}: Title={title}, Authors={authors}, Year={year}")
        update_log(f"Extracted metadata for {pdf_file.name}: Title={title}, Authors={authors}, Year={year}", tab="db")
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "content": text
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_file.name}: {e}")
        update_log(f"Error processing {pdf_file.name}: {e}", tab="db")
        return {
            "title": pdf_file.name,
            "authors": "Unknown",
            "year": str(datetime.now().year),
            "content": f"Error extracting text from {pdf_file.name}: {str(e)}"
        }

def create_database(papers, db_name="papers.db"):
    """Create a SQLite database and store the parsed papers."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                year TEXT,
                content TEXT
            )
        """)
        
        cursor.execute("DELETE FROM papers")
        
        for paper in papers:
            cursor.execute("""
                INSERT INTO papers (title, authors, year, content)
                VALUES (?, ?, ?, ?)
            """, (paper["title"], paper["authors"], paper["year"], paper["content"]))
        
        conn.commit()
        conn.close()
        logger.info(f"Database '{db_name}' created with {len(papers)} papers")
        update_log(f"Database '{db_name}' created with {len(papers)} papers", tab="db")
        return db_name
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        update_log(f"Error creating database: {e}", tab="db")
        raise

def extract_phases(text):
    """Extract and normalize LiₓSnᵧ phases using regex and spaCy NER."""
    if not isinstance(text, str):
        logger.warning("Non-string content detected, skipping phase extraction.")
        update_log("Non-string content detected, skipping phase extraction.", tab="analysis")
        return []
    
    phases = []
    try:
        matches = re.findall(PHASE_PATTERN, text, re.IGNORECASE)
        for match in matches:
            if match and "Li" in match and "Sn" in match:
                phases.append(match)
    except Exception as e:
        logger.error(f"Error in regex matching: {e}")
        update_log(f"Error in regex matching: {e}", tab="analysis")
    
    try:
        doc = nlp(text[:1000000])
        for ent in doc.ents:
            if re.match(PHASE_PATTERN, ent.text, re.IGNORECASE):
                phases.append(ent.text)
    except Exception as e:
        logger.error(f"Error in spaCy NER: {e}")
        update_log(f"Error in spaCy NER: {e}", tab="analysis")
    
    normalized_phases = []
    for phase in phases:
        try:
            phase_clean = phase.replace(" ", "").replace(".", "")
            if re.match(r"Li\d*Sn\d*", phase_clean, re.IGNORECASE):
                normalized_phases.append(phase_clean)
            elif phase_clean.startswith("Li") and "Sn" in phase_clean:
                match = re.match(r"Li(\d+)(?:Sn)(\d+)", phase_clean, re.IGNORECASE)
                if match:
                    x, y = match.groups()
                    normalized_phases.append(f"Li{x}Sn{y}")
        except Exception as e:
            logger.warning(f"Error normalizing phase {phase}: {e}")
            update_log(f"Error normalizing phase {phase}: {e}", tab="analysis")
    
    logger.debug(f"Extracted phases: {normalized_phases}")
    update_log(f"Extracted phases: {normalized_phases}", tab="analysis")
    return normalized_phases

def read_database(db_file):
    """Read papers from SQLite database."""
    try:
        conn = sqlite3.connect(db_file)
        query = "SELECT title, authors, year, content FROM papers"
        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Loaded {len(df)} papers from database.")
        update_log(f"Loaded {len(df)} papers from database.", tab="analysis")
        return df
    except Exception as e:
        logger.error(f"Error reading database: {e}")
        update_log(f"Error reading database: {e}", tab="analysis")
        raise

def count_phases(df):
    """Count occurrences of LiₓSnᵧ phases."""
    all_phases = []
    progress_bar = st.progress(0)
    for i, content in enumerate(df["content"]):
        try:
            phases = extract_phases(content)
            all_phases.extend(phases)
            update_log(f"Processed paper {i+1}/{len(df)} for phase counting.", tab="analysis")
            progress_bar.progress((i + 1) / len(df))
        except Exception as e:
            logger.error(f"Error counting phases in paper {i+1}: {e}")
            update_log(f"Error counting phases in paper {i+1}: {e}", tab="analysis")
    phase_counts = Counter(all_phases)
    logger.info(f"Phase counts: {dict(phase_counts)}")
    update_log(f"Phase counts: {dict(phase_counts)}", tab="analysis")
    return phase_counts

def filter_phases(phase_counts, exclude_phases):
    """Filter out unwanted phases."""
    filtered_counts = Counter({k: v for k, v in phase_counts.items() if k not in exclude_phases})
    logger.info(f"Filtered phase counts: {dict(filtered_counts)}")
    update_log(f"Filtered phase counts: {dict(filtered_counts)}", tab="analysis")
    return filtered_counts

def create_histogram_plotly(phase_counts, color, theme):
    """Create Plotly histogram."""
    df = pd.DataFrame(phase_counts.items(), columns=["Phase", "Count"])
    fig = px.bar(
        df, x="Phase", y="Count", title="Frequency of LiₓSnᵧ Phases (Plotly)",
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        xaxis_title="LiₓSnᵧ Phase",
        yaxis_title="Count",
        xaxis_tickangle=45,
        template=theme,
        font=dict(size=14),
        margin=dict(l=50, r=50, t=80, b=100)
    )
    return fig

def create_histogram_matplotlib(phase_counts, colormap, bar_width, title_font_size, label_font_size, tick_font_size, axes_thickness):
    """Create Matplotlib histogram with enhanced publication quality."""
    try:
        df = pd.DataFrame(phase_counts.items(), columns=["Phase", "Count"])
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
        
        if colormap in plt.colormaps():
            cmap = cm.get_cmap(colormap)
            colors = [cmap(i / len(df)) for i in range(len(df))]
        else:
            colors = colormap
        
        bars = ax.bar(df["Phase"], df["Count"], width=bar_width, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel("LiₓSnᵧ Phase", fontsize=label_font_size, weight='bold')
        ax.set_ylabel("Count", fontsize=label_font_size, weight='bold')
        ax.set_title("Frequency of LiₓSnᵧ Phases", fontsize=title_font_size, pad=30, weight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font_size, width=axes_thickness, length=8)
        ax.tick_params(axis='x', rotation=45)
        
        for spine in ax.spines.values():
            spine.set_linewidth(axes_thickness)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=600)
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        return buf_png, buf_svg
    except Exception as e:
        logger.error(f"Error creating Matplotlib histogram: {e}")
        update_log(f"Error creating Matplotlib histogram: {e}", tab="analysis")
        return BytesIO(), BytesIO()

def create_radar_chart(phase_counts, colormap, max_keywords, line_thickness, line_style):
    """Create radar chart of phase counts with customizable line thickness and style."""
    if len(phase_counts) < 3:
        logger.warning("Less than 3 phases found, skipping radar chart.")
        update_log("Less than 3 phases found, skipping radar chart.", tab="analysis")
        return None
    phases = list(phase_counts.keys())[:max_keywords]
    counts = list(phase_counts.values())[:max_keywords]
    phases += [phases[0]]
    counts += [counts[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=counts, theta=phases, fill="toself", name="Phase Counts",
        line=dict(color=colormap, width=line_thickness, dash=line_style)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(counts)])),
        showlegend=True,
        title="Radar Chart of LiₓSnᵧ Phase Frequencies",
        template="plotly_white",
        font=dict(size=14)
    )
    return fig

def create_network(df, phase_counts, layout_algorithm="spring", phase_shape="s", mech_term_shape="d", edge_style="solid", label_font_size=12, edge_thickness=1.0):
    """Create enhanced network graph for all filtered LiₓSnᵧ phases with mechanical terms."""
    G = nx.Graph()
    
    try:
        text = " ".join(df["content"].astype(str))
        all_phases = set(phase_counts.keys())
        all_terms = MECHANICAL_TERMS
        
        for phase in all_phases:
            G.add_node(phase, type="phase")
        for term in all_terms:
            G.add_node(term, type="mechanical")
        
        sentences = re.split(r"[.!?]", text)
        co_occurrences = Counter()
        progress_bar = st.progress(0)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_lower = sentence.lower()
            try:
                for phase in all_phases:
                    phase_lower = phase.lower()
                    for term in all_terms:
                        term_lower = term.lower()
                        if phase_lower in sentence_lower and term_lower in sentence_lower:
                            co_occurrences[(phase, term)] += 1
                update_log(f"Processed sentence {i+1}/{len(sentences)} for network analysis.", tab="analysis")
                progress_bar.progress((i + 1) / len(sentences))
            except Exception as e:
                logger.warning(f"Error processing sentence {i+1}: {e}")
                update_log(f"Error processing sentence {i+1}: {e}", tab="analysis")
                continue
        
        if not co_occurrences:
            logger.warning("No co-occurrences found between LiₓSnᵧ phases and mechanical terms. Generating graph with nodes only.")
            update_log("No co-occurrences found. Graph will show nodes without edges.", tab="analysis")
        else:
            for (node1, node2), weight in co_occurrences.items():
                G.add_edge(node1, node2, weight=weight)
        
        communities = greedy_modularity_communities(G)
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
        
        network_data = []
        for node in G.nodes:
            degree = G.degree[node]
            node_type = G.nodes[node]["type"]
            community = community_map.get(node, -1)
            network_data.append({"Node": node, "Type": node_type, "Degree": degree, "Community": community})
        for node1, node2, data in G.edges(data=True):
            network_data.append({"Node": f"{node1}-{node2}", "Type": "Edge", "Weight": data["weight"], "Community": -1})
        
        plt.figure(figsize=(16, 12), dpi=600)
        try:
            if layout_algorithm == "spring":
                pos = nx.spring_layout(G, k=1.2, seed=42)
            elif layout_algorithm == "circular":
                pos = nx.circular_layout(G)
            elif layout_algorithm == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_algorithm == "fruchterman_reingold":
                pos = nx.fruchterman_reingold_layout(G, k=1.2, seed=42)
            else:
                pos = nx.spring_layout(G, k=1.2, seed=42)
        except Exception as e:
            logger.warning(f"Error in layout {layout_algorithm}: {e}, falling back to spring")
            update_log(f"Error in layout {layout_algorithm}: {e}, falling back to spring", tab="analysis")
            pos = nx.spring_layout(G, k=1.2, seed=42)
        
        node_colors = []
        node_shapes = []
        node_sizes = []
        for node in G.nodes:
            if G.nodes[node]["type"] == "phase":
                node_colors.append("#FF9999")
                node_sizes.append(1500)
                node_shapes.append(phase_shape)
            elif G.nodes[node]["type"] == "mechanical":
                node_colors.append("#66CC66")
                node_sizes.append(1800)
                node_shapes.append(mech_term_shape)
        
        edge_widths = [edge_thickness + 3 * np.log1p(G.edges[e]["weight"]) if G.edges else edge_thickness for e in G.edges]
        edge_colors = ["#444444" for _ in G.edges]
        
        for shape in set(node_shapes):
            nodes = [n for n, s in zip(G.nodes, node_shapes) if s == shape]
            colors = [node_colors[i] for i, n in enumerate(G.nodes) if n in nodes]
            sizes = [node_sizes[i] for i, n in enumerate(G.nodes) if n in nodes]
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_shape=shape, node_color=colors,
                node_size=sizes, edgecolors="black", linewidths=1.5
            )
        
        if G.edges:
            nx.draw_networkx_edges(
                G, pos, edge_color=edge_colors, width=edge_widths, style=edge_style
            )
        
        nx.draw_networkx_labels(
            G, pos, font_size=label_font_size, font_weight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", linewidth=1.0)
        )
        
        plt.title("Network Analysis of LiₓSnᵧ Relations with Mechanical Terminology", fontsize=20, pad=40, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=600)
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        
        return buf_png, buf_svg, network_data
    
    except Exception as e:
        logger.error(f"Error creating network: {e}")
        update_log(f"Error creating network: {e}", tab="analysis")
        return BytesIO(), BytesIO(), []

def main():
    st.set_page_config(page_title="LiₓSnᵧ Phase Analysis and Database Creator", layout="wide")
    st.title("LiₓSnᵧ Phase Analysis and Database Creator")
    st.markdown("""
    This app provides two functionalities:
    - **PDF to Database**: Upload PDF files to extract text and metadata, creating a SQLite database.
    - **Phase Analysis**: Upload a SQLite database to analyze LiₓSnᵧ phases and their relations with mechanical terminology.
    """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["PDF to Database", "Phase Analysis"])
    
    # Tab 1: PDF to Database
    with tab1:
        st.header("PDF to SQLite Database Converter")
        st.markdown("""
        Upload one or more PDF files to extract text and create a SQLite database (.db) file compatible with LiₓSnᵧ phase NER analysis.
        The database will be updated whenever new PDFs are uploaded, overwriting the previous version.
        """)
        
        log_container_db = st.empty()
        def display_logs_db():
            log_container_db.text_area("Processing Logs (PDF to Database)", "\n".join(st.session_state.log_buffer_db), height=200)
        
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
        
        if uploaded_files:
            current_filenames = sorted([f.name for f in uploaded_files])
            last_filenames = sorted(st.session_state.last_uploaded_files)
            if current_filenames != last_filenames or not st.session_state.db_created:
                st.session_state.last_uploaded_files = current_filenames
                st.session_state.db_created = False
                
                with st.spinner("Processing PDFs and creating database..."):
                    papers = []
                    for uploaded_file in uploaded_files:
                        update_log(f"Processing {uploaded_file.name}...", tab="db")
                        paper_data = extract_metadata_and_content(uploaded_file)
                        papers.append(paper_data)
                        update_log(f"Extracted: Title={paper_data['title']}, Authors={paper_data['authors']}, Year={paper_data['year']}, Content length={len(paper_data['content'])}", tab="db")
                    
                    if not papers:
                        update_log("No valid papers extracted from PDFs.", tab="db")
                        st.error("No valid papers extracted from PDFs. Please check the files.")
                        display_logs_db()
                        return
                    
                    try:
                        db_name = "papers.db"
                        create_database(papers, db_name)
                        st.session_state.db_created = True
                        update_log(f"Database '{db_name}' created successfully with {len(papers)} papers.", tab="db")
                        st.success(f"Database '{db_name}' created with {len(papers)} papers!")
                        
                        st.subheader("Extracted Papers")
                        papers_df = pd.DataFrame(papers)
                        st.dataframe(papers_df[["title", "authors", "year"]])
                        csv_buffer = BytesIO()
                        papers_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        st.download_button(
                            label="Download Extracted Papers (CSV)",
                            data=csv_buffer,
                            file_name="extracted_papers.csv",
                            mime="text/csv"
                        )
                        
                        for i, paper in enumerate(papers, 1):
                            with st.expander(f"Paper {i}: {paper['title']}"):
                                st.write(f"**Title**: {paper['title']}")
                                st.write(f"**Authors**: {paper['authors']}")
                                st.write(f"**Year**: {paper['year']}")
                                st.write(f"**Content Preview**: {paper['content'][:200]}...")
                        
                        with open(db_name, "rb") as f:
                            db_bytes = f.read()
                        st.download_button(
                            label="Download Database File",
                            data=db_bytes,
                            file_name=db_name,
                            mime="application/x-sqlite3"
                        )
                    
                    except Exception as e:
                        update_log(f"Error creating database: {str(e)}", tab="db")
                        st.error(f"Error creating database: {str(e)}")
                    
                    display_logs_db()
                
            else:
                st.info("No new PDFs uploaded. Using existing database.")
                if os.path.exists("papers.db"):
                    with open("papers.db", "rb") as f:
                        db_bytes = f.read()
                    st.download_button(
                        label="Download Existing Database File",
                        data=db_bytes,
                        file_name="papers.db",
                        mime="application/x-sqlite3"
                    )
                display_logs_db()
        
        else:
            st.info("Please upload PDF files to create a database.")
            st.session_state.db_created = False
            st.session_state.last_uploaded_files = []
            display_logs_db()
    
    # Tab 2: Phase Analysis
    with tab2:
        st.header("LiₓSnᵧ Phase Analysis from SQLite Database")
        st.markdown("""
        Upload a SQLite database (.db) file to analyze LiₓSnᵧ phases and their relations with mechanical terminology.
        You can use the database created in the 'PDF to Database' tab or upload a different one.
        Customize visualizations for publication-quality output.
        """)
        
        log_container_analysis = st.empty()
        def display_logs_analysis():
            log_container_analysis.text_area("Processing Logs (Phase Analysis)", "\n".join(st.session_state.log_buffer_analysis), height=300)
        
        with st.sidebar:
            st.header("Visualization Settings")
            st.subheader("Histogram Settings")
            plotly_color = st.selectbox("Plotly histogram color", ["blue", "red", "green", "purple", "orange"], index=0, key="plotly_color")
            plotly_color_map = {"blue": "#1f77b4", "red": "#ff7f0e", "green": "#2ca02c", "purple": "#9467bd", "orange": "#ffbb78"}
            plotly_theme = st.selectbox("Plotly theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=1, key="plotly_theme")
            matplotlib_colormap = st.selectbox(
                "Matplotlib histogram colormap",
                ["viridis", "plasma", "inferno", "magma", "hot", "cool", "turbo", "jet", "rainbow",
                 "Blues", "Reds", "Greens", "Purples", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"],
                index=0, key="matplotlib_colormap"
            )
            bar_width = st.slider("Bar width", min_value=0.1, max_value=1.0, value=0.8, step=0.05, key="bar_width")
            title_font_size = st.slider("Title font size", min_value=12, max_value=30, value=20, step=1, key="title_font_size")
            label_font_size = st.slider("Axis label font size", min_value=10, max_value=24, value=16, step=1, key="label_font_size")
            tick_font_size = st.slider("Tick label font size", min_value=8, max_value=20, value=14, step=1, key="tick_font_size")
            axes_thickness = st.slider("Axes and tick thickness", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="axes_thickness")
            
            st.subheader("Radar Chart Settings")
            radar_colormap = st.selectbox(
                "Radar chart color",
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#ffbb78", "#d62728", "#8c564b", "#e377c2"],
                index=0, key="radar_colormap"
            )
            radar_line_thickness = st.slider("Radar line thickness", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="radar_line_thickness")
            radar_line_style = st.selectbox("Radar line style", ["solid", "dash", "dot"], index=0, key="radar_line_style")
            max_keywords = st.slider("Max phases in radar chart", min_value=3, max_value=12, value=6, step=1, key="max_keywords")
            
            st.subheader("Network Graph Settings")
            layout_algorithm = st.selectbox("Network layout", ["spring", "circular", "kamada_kawai", "fruchterman_reingold"], index=0, key="layout_algorithm")
            phase_shape = st.selectbox("Phase node shape", ["o", "s", "d", "^", "v", "p", "h"], index=1, key="phase_shape")
            mech_term_shape = st.selectbox("Mechanical term node shape", ["o", "s", "d", "^", "v", "p", "h"], index=2, key="mech_term_shape")
            edge_style = st.selectbox("Edge style", ["solid", "dashed", "dotted"], index=0, key="edge_style")
            network_label_font_size = st.slider("Network label font size", min_value=10, max_value=24, value=16, step=1, key="network_label_font_size")
            edge_thickness = st.slider("Edge thickness multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="edge_thickness")
        
        uploaded_file = st.file_uploader("Upload SQLite database (.db)", type=["db"], key="db_uploader")
        use_existing_db = st.checkbox("Use database created from PDF to Database tab", value=True)
        
        db_file = "papers.db" if use_existing_db and os.path.exists("papers.db") else None
        if uploaded_file is not None:
            with open("temp.db", "wb") as f:
                f.write(uploaded_file.read())
            db_file = "temp.db"
        
        if db_file:
            try:
                df = read_database(db_file)
                st.info(f"Loaded {len(df)} papers from the database.")
                display_logs_analysis()
                
                st.subheader("Initial Phase Counts")
                phase_counts = count_phases(df)
                if not phase_counts:
                    st.warning("No LiₓSnᵧ phases found in the database.")
                    display_logs_analysis()
                    return
                
                phase_df = pd.DataFrame(phase_counts.items(), columns=["Phase", "Count"])
                st.dataframe(phase_df)
                exclude_phases = st.multiselect(
                    "Select phases to exclude",
                    options=list(phase_counts.keys()),
                    default=[],
                    key="exclude_phases"
                )
                if st.button("Apply Phase Filter"):
                    st.session_state.filtered_phases = filter_phases(phase_counts, exclude_phases)
                    update_log(f"Applied filter, excluded phases: {exclude_phases}", tab="analysis")
                
                final_phase_counts = st.session_state.filtered_phases if st.session_state.filtered_phases else phase_counts
                
                st.subheader("Final Phase Counts (After Filtering)")
                final_phase_df = pd.DataFrame(final_phase_counts.items(), columns=["Phase", "Count"])
                st.dataframe(final_phase_df)
                csv_buffer = BytesIO()
                final_phase_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="Download Final Phase Counts (CSV)",
                    data=csv_buffer,
                    file_name="final_phase_counts.csv",
                    mime="text/csv"
                )
                
                st.subheader("Histogram of LiₓSnᵧ Phase Frequencies (Plotly)")
                fig_hist_plotly = create_histogram_plotly(final_phase_counts, plotly_color_map[plotly_color], plotly_theme)
                st.plotly_chart(fig_hist_plotly, use_container_width=True)
                
                st.subheader("Histogram of LiₓSnᵧ Phase Frequencies (Matplotlib)")
                hist_buf_png, hist_buf_svg = create_histogram_matplotlib(
                    final_phase_counts, matplotlib_colormap, bar_width, title_font_size,
                    label_font_size, tick_font_size, axes_thickness
                )
                if hist_buf_png.getvalue():
                    st.image(hist_buf_png, caption="Matplotlib Histogram", use_column_width=True)
                    st.download_button(
                        label="Download Matplotlib Histogram (PNG)",
                        data=hist_buf_png,
                        file_name="histogram_matplotlib.png",
                        mime="image/png"
                    )
                    st.download_button(
                        label="Download Matplotlib Histogram (SVG)",
                        data=hist_buf_svg,
                        file_name="histogram_matplotlib.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning("Failed to generate Matplotlib histogram.")
                
                st.subheader("Radar Chart of LiₓSnᵧ Phase Frequencies")
                fig_radar = create_radar_chart(
                    final_phase_counts, radar_colormap, max_keywords, radar_line_thickness, radar_line_style
                )
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Not enough phases for radar chart (minimum 3 required).")
                
                st.subheader("Network Analysis of LiₓSnᵧ Relations with Mechanical Terminology")
                network_buf_png, network_buf_svg, network_data = create_network(
                    df, final_phase_counts, layout_algorithm=layout_algorithm,
                    phase_shape=phase_shape, mech_term_shape=mech_term_shape,
                    edge_style=edge_style, label_font_size=network_label_font_size,
                    edge_thickness=edge_thickness
                )
                if network_buf_png.getvalue():
                    st.image(network_buf_png, caption="Network Graph of LiₓSnᵧ Relations with Mechanical Terminology", use_column_width=True)
                    st.download_button(
                        label="Download Network Graph (PNG)",
                        data=network_buf_png,
                        file_name="network_graph.png",
                        mime="image/png"
                    )
                    st.download_button(
                        label="Download Network Graph (SVG)",
                        data=network_buf_svg,
                        file_name="network_graph.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning("Failed to generate network graph. Check logs for details.")
                
                st.subheader("Network Quantification")
                if network_data:
                    network_df = pd.DataFrame(network_data)
                    st.dataframe(network_df)
                    network_csv_buffer = BytesIO()
                    network_df.to_csv(network_csv_buffer, index=False)
                    network_csv_buffer.seek(0)
                    st.download_button(
                        label="Download Network Quantification (CSV)",
                        data=network_csv_buffer,
                        file_name="network_quantification.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No network data available for quantification.")
                
                display_logs_analysis()
                
                if db_file == "temp.db" and os.path.exists("temp.db"):
                    os.remove("temp.db")
            
            except Exception as e:
                update_log(f"Error processing database: {str(e)}", tab="analysis")
                st.error(f"An error occurred: {str(e)}")
                if db_file == "temp.db" and os.path.exists("temp.db"):
                    os.remove("temp.db")
                display_logs_analysis()
        
        else:
            st.info("Please upload a .db file or create one in the 'PDF to Database' tab.")
            display_logs_analysis()

if __name__ == "__main__":
    main()
