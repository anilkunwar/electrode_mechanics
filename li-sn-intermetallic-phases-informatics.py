import streamlit as st
import pdfplumber
import PyPDF2
import tempfile
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import seaborn as sns
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import spacy
from math import log
import pandas as pd
import yaml
import arxiv
import requests
import gc
import plotly.express as px
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

# Set page config
st.set_page_config(page_title="Li-Sn Phase Analysis for Lithium-Ion Battery Anodes", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default keywords
DEFAULT_KEYWORDS = {
    "li_sn_phases": ["li2sn5", "lisn", "li7sn3", "li5sn2", "li13sn5", "li7sn2", "li22sn5"],
    "material_properties": ["volume expansion", "mechanical stress", "electrochemical strain", "stress relaxation",
                           "yield strength", "tensile strength", "ductility", "hardness", "strain hardening", "elastic modulus"],
    "technological_applications": ["lithium-ion battery anodes", "tin anode", "electrochemical cycling", "battery performance",
                                   "anode stability", "efficiency", "energy efficiency"],
    "mechanistic_models": ["molecular dynamics", "computational modeling", "multiscale modeling", "density functional theory",
                           "phase transformation"],
    "material_characteristics": ["microstructure", "crystal orientation", "phase stability", "intermetallic compounds",
                                 "solid electrolyte interface", "cycle phase stability"]
}

# Download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already present.")
        return True
    except LookupError:
        logger.info("Downloading NLTK punkt_tab and stopwords...")
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("Download completed successfully.")
        return True

# Load spaCy model
def load_spacy_model(disable_components=None):
    try:
        disable = disable_components if disable_components else []
        nlp = spacy.load("en_core_web_sm", disable=disable)
        nlp.max_length = 6000000
        return nlp
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=disable)
        nlp.max_length = 6000000
        return nlp

if not download_nltk_data():
    st.stop()

# Memory option selection
memory_option = st.selectbox("Memory Management Option", ["Default", "Disable Parser and NER", "Use Abstracts Only", "Chunk Text Processing"], index=3)
st.session_state.memory_option = memory_option

# Load spaCy model based on memory option
if memory_option == "Disable Parser and NER":
    nlp = load_spacy_model(disable_components=["parser", "ner"])
elif memory_option in ["Use Abstracts Only", "Chunk Text Processing"]:
    nlp = load_spacy_model()
else:
    nlp = load_spacy_model()

# IDF approximations
IDF_APPROX = {
    "lithium-ion": log(1000 / 500), "anode": log(1000 / 400), "tin": log(1000 / 300),
    "volume expansion": log(1000 / 50), "mechanical stress": log(1000 / 40),
    "electrochemical cycling": log(1000 / 60), "phase stability": log(1000 / 50),
    "efficiency": log(1000 / 70), "energy efficiency": log(1000 / 60),
    "cycle phase stability": log(1000 / 50), "li22sn5": log(1000 / 20),
    "lisn": log(1000 / 30), "li7sn2": log(1000 / 25)
}
DEFAULT_IDF = log(100000 / 10000)
PHYSICS_CATEGORIES = ["material_properties", "technological_applications", "mechanistic_models", "material_characteristics"]

# Visualization options
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds"]
NETWORK_STYLES = ["seaborn-v0_8-white", "ggplot", "classic"]
NODE_SHAPES = ['o', 's', '^']
EDGE_STYLES = ['solid', 'dashed', 'dotted']
COLORS = ['black', 'red', 'blue', 'green']
FONT_FAMILIES = ['Arial', 'Helvetica']
BBOX_COLORS = ['black', 'white', "gray"]
LAYOUT_ALGORITHMS = ['spring', 'circular', 'kamada_kawai']
WORD_ORIENTATIONS = ['horizontal', 'vertical', 'random']

# Visualization presets
visualization_settings_presets = {
    "Simple": {
        "label_font_size": 12, "line_thickness": 1.0, "title_font_size": 14, "caption_font_size": 8,
        "node_alpha": 0.8, "edge_alpha": 0.6, "font_step": 1, "wordcloud_colormap": "viridis",
        "word_orientation": "horizontal", "background_color": "white", "contour_width": 0.0,
        "contour_color": "black", "network_style": "seaborn-v0_8-white", "node_colormap": "Blues",
        "edge_colormap": "Reds", "node_size_scale": 50, "node_shape": "o", "node_linewidth": 1.5,
        "node_edgecolor": "black", "edge_style": "solid", "label_font_color": "black",
        "label_font_family": "Arial", "label_bbox_facecolor": "white", "label_bbox_alpha": 0.8,
        "layout_algorithm": "spring", "label_rotation": 0, "label_offset": 0.05,
        "heatmap_colormap": "viridis", "radar_colormap": "plasma", "grid_color": "gray",
        "grid_style": "solid", "grid_thickness": 1.0
    },
    "Professional": {
        "label_font_size": 14, "line_thickness": 1.5, "title_font_size": 16, "caption_font_size": 10,
        "node_alpha": 0.9, "edge_alpha": 0.7, "font_step": 2, "wordcloud_colormap": "magma",
        "word_orientation": "random", "background_color": "#d3d3d3", "contour_width": 1.0,
        "contour_color": "black", "network_style": "ggplot", "node_colormap": "inferno",
        "edge_colormap": "plasma", "node_size_scale": 100, "node_shape": "s", "node_linewidth": 2.0,
        "node_edgecolor": "black", "edge_style": "dashed", "label_font_color": "black",
        "label_font_family": "Helvetica", "label_bbox_facecolor": "white", "label_bbox_alpha": 1.0,
        "layout_algorithm": "kamada_kawai", "label_rotation": 45, "label_offset": 0.1,
        "heatmap_colormap": "cividis", "radar_colormap": "magma", "grid_color": "black",
        "grid_style": "dashed", "grid_thickness": 1.5
    }
}

# Data retrieval
def get_cache_key(repository, query, max_results, start_year):
    return hashlib.md5(f"{repository}_{query}_{max_results}_{start_year}".encode()).hexdigest()

def download_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(response.content)
            tmp_path = f.name
        with open(tmp_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages[:5]])
            os.unlink(tmp_path)
            return text if text.strip() else None
    except Exception as e:
        logger.warning(f"Error downloading PDF {pdf_url}: {str(e)}")
        return None

def fetch_arxiv_papers(query, max_results=10, start_year=2015, use_abstracts_only=True):
    cache_key = get_cache_key("arxiv", query, max_results, start_year)
    cache_file = f"arxiv_cache_{cache_key}.pkl"
    try:
        with open(cache_file, "rb") as f:
            papers = pickle.load(f)
            logger.info("Loaded papers from cache.")
            return papers
    except FileNotFoundError:
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
            papers = []
            pdf_urls = []
            for result in search.results():
                year = result.published.year
                if year >= start_year:
                    papers.append({
                        "title": result.title,
                        "abstract": result.summary,
                        "pdf_text": None,
                        "published": year,
                        "pdf_url": result.pdf_url
                    })
                    if not use_abstracts_only:
                        pdf_urls.append(result.pdf_url)
            if not use_abstracts_only:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    pdf_texts = list(executor.map(download_pdf_text, pdf_urls))
                    for paper, text in zip(papers, pdf_texts):
                        paper["pdf_text"] = text if text else paper["abstract"]
                        time.sleep(0.5)  # Respect arXiv rate limits
            with open(cache_file, "wb") as f:
                pickle.dump(papers, f)
            return papers
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {str(e)}")
            st.error(f"Error fetching arXiv papers: {str(e)}")
            return []

def fetch_semantic_scholar_papers(query, max_results=10, start_year=2015, use_abstracts_only=True, api_key=None):
    cache_key = get_cache_key("semantic_scholar", query, max_results, start_year)
    cache_file = f"semantic_scholar_cache_{cache_key}.pkl"
    try:
        with open(cache_file, "rb") as f:
            papers = pickle.load(f)
            logger.info("Loaded papers from cache.")
            return papers
    except FileNotFoundError:
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "fields": "paperId,title,abstract,year,openAccessPdf,venue",
                "limit": min(max_results, 100),  # Respect API limit
                "year": f"{start_year}-2025"
            }
            headers = {"x-api-key": api_key} if api_key else {}
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            papers = []
            pdf_urls = []
            for result in data.get("data", []):
                if result.get("year") and result["year"] >= start_year:
                    pdf_url = result.get("openAccessPdf", {}).get("url") if result.get("openAccessPdf") else None
                    papers.append({
                        "title": result.get("title", ""),
                        "abstract": result.get("abstract") or "",
                        "pdf_text": None,
                        "published": result.get("year", start_year),
                        "pdf_url": pdf_url
                    })
                    if not use_abstracts_only and pdf_url:
                        pdf_urls.append(pdf_url)
            if not use_abstracts_only and pdf_urls:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    pdf_texts = list(executor.map(download_pdf_text, pdf_urls))
                    for paper, text in zip(papers, pdf_texts):
                        paper["pdf_text"] = text if text else paper["abstract"]
                        time.sleep(0.5)  # Respect rate limits
            with open(cache_file, "wb") as f:
                pickle.dump(papers, f)
            return papers
        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar papers: {str(e)}")
            st.error(f"Error fetching Semantic Scholar papers: {str(e)}")
            return []

def fetch_papers(repository, query, max_results, start_year, use_abstracts_only, api_key=None):
    if repository == "arXiv":
        return fetch_arxiv_papers(query, max_results, start_year, use_abstracts_only)
    elif repository == "Semantic Scholar":
        return fetch_semantic_scholar_papers(query, max_results, start_year, use_abstracts_only, api_key)
    return []

def deduplicate_papers(papers):
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        title = paper["title"].lower().strip()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)
    return unique_papers

# Text extraction and preprocessing
def extract_text_from_pdf(file, use_pypdf2_fallback=True):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="_.pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        with pdfplumber.open(tmp_file_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages[:5]])
        if not text.strip() and use_pypdf2_fallback:
            with open(tmp_file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages[:5]])
        os.unlink(tmp_file_path)
        return text if text.strip() else "No text extracted from the PDF."
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"
    finally:
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def chunk_text(text):
    return [text[i:i+900000] for i in range(0, len(text), 900000)]

def clean_phrase(phrase, stop_words):
    words = phrase.split()
    words = [w for w in words if w.lower() not in stop_words]
    return " ".join(words).strip()

# Load keywords from YAML
def load_keywords(yaml_content):
    try:
        keywords = yaml.safe_load(yaml_content)
        if not isinstance(keywords, dict):
            raise TypeError("YAML content must be a dictionary with categories as keys and lists of keywords as values")
        required_categories = ["li_sn_phases", "material_properties", "technological_applications", "mechanistic_models", "material_characteristics"]
        keywords = {k.lower(): [str(term).lower() for term in v if str(term).strip()] for k, v in keywords.items()}
        missing_categories = [cat for cat in required_categories if cat not in keywords]
        if missing_categories:
            raise ValueError(f"Missing required categories: {', '.join(missing_categories)}")
        for category, terms in keywords.items():
            if not isinstance(terms, list) or not terms:
                raise TypeError(f"Category '{category}' must contain a non-empty list of keywords")
        logger.debug(f"Loaded KEYWORD_CATEGORIES: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"Error parsing YAML content: {str(e)}")
        return None

# Keyword extraction and TF-IDF
def calculate_corpus_idf(term, papers):
    doc_count = sum(1 for paper in papers if term in (paper.get("pdf_text") or paper.get("abstract") or "").lower())
    return log(len(papers) / (doc_count + 1)) if doc_count > 0 else DEFAULT_IDF

def estimate_idf(term, word_freq, tfidf, idf_approx, keyword_categories, nlp, model, papers):
    if 'custom_idf' not in st.session_state:
        st.session_state.custom_idf = {}
    if term in st.session_state.custom_idf:
        return st.session_state.custom_idf[term]
    total_words = sum(word_freq.values())
    tf = word_freq.get(term, 0) / total_words if total_words > 0 else 0
    freq_idf = log(1 / max(tf, 1e-6))
    estimated_idf = idf_approx.get(term, DEFAULT_IDF) + freq_idf
    estimated_idf = max(2.303, min(5, estimated_idf))
    st.session_state.custom_idf[term] = float(estimated_idf)
    return estimated_idf

def get_candidate_keywords(text, min_freq, min_length, use_stopwords, custom_stopwords, exclude_keywords, top_limit, tfidf_weight, use_nouns_only, include_phrases, nlp, keyword_categories, papers, chunked=False):
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    stop_words.update(['introduction', 'conclusion', 'section', 'chapter', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
    stop_words.update([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    exclude_set = set([w.strip().lower() for w in exclude_keywords.split(",") if w.strip()])
    word_freq = Counter()
    tfidf_scores = {}
    categorized_keywords = {cat: [] for cat in keyword_categories}
    term_to_category = {}
    
    if chunked:
        chunks = chunk_text(text)
        for chunk in chunks:
            doc = nlp(chunk)
            for token in doc:
                if token.is_alpha and len(token.text) >= min_length and token.text.lower() not in stop_words and token.text.lower() not in exclude_set:
                    if not use_nouns_only or token.pos_ == 'NOUN':
                        word_freq[token.text.lower()] += 1
            if include_phrases:
                for chunk in doc.noun_chunks:
                    phrase = clean_phrase(chunk.text.lower(), stop_words)
                    if len(phrase) >= min_length and phrase not in exclude_set:
                        word_freq[phrase] += 1
    else:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha and len(token.text) >= min_length and token.text.lower() not in stop_words and token.text.lower() not in exclude_set:
                if not use_nouns_only or token.pos_ == 'NOUN':
                    word_freq[token.text.lower()] += 1
        if include_phrases:
            for chunk in doc.noun_chunks:
                phrase = clean_phrase(chunk.text.lower(), stop_words)
                if len(phrase) >= min_length and phrase not in exclude_set:
                    word_freq[phrase] += 1
    
    for term, freq in word_freq.items():
        if freq >= min_freq:
            idf = estimate_idf(term, word_freq, tfidf_scores, IDF_APPROX, keyword_categories, nlp, None, papers)
            tfidf_scores[term] = (freq / sum(word_freq.values())) * idf * tfidf_weight
    
    for term in tfidf_scores:
        for category, keywords in keyword_categories.items():
            if term in keywords:
                categorized_keywords[category].append((term, tfidf_scores[term]))
                term_to_category[term] = category
                break
            elif any(keyword in term or term.startswith(keyword + " ") or term.endswith(" " + keyword) for keyword in keywords):
                categorized_keywords[category].append((term, tfidf_scores[term]))
                term_to_category[term] = category
                break
        else:
            categorized_keywords["material_characteristics"].append((term, tfidf_scores[term]))
            term_to_category[term] = "material_characteristics"
    
    for category in categorized_keywords:
        categorized_keywords[category] = sorted(categorized_keywords[category], key=lambda x: x[1], reverse=True)[:top_limit]
    
    return categorized_keywords, word_freq, [], tfidf_scores, term_to_category, []

# Apply recency weight
def apply_recency_weight(tfidf_scores, papers):
    adjusted_scores = {}
    for term in tfidf_scores:
        adjusted_scores[term] = tfidf_scores[term]
    for paper in papers:
        year = paper["published"]
        weight = 1 - (2025 - year) / 10
        for term in tfidf_scores:
            if term in (paper.get("pdf_text") or paper.get("abstract") or "").lower():
                adjusted_scores[term] = tfidf_scores[term] * weight
    return adjusted_scores

# Compute scores
def compute_scores(text, papers, li_sn_phases, material_properties, technological_applications, mechanistic_models, material_characteristics):
    co_occurrence_scores = {
        "material_properties": Counter(),
        "technological_applications": Counter(),
        "mechanistic_models": Counter(),
        "material_characteristics": Counter()
    }
    relevance_scores = Counter()
    recency_scores = Counter()
    sentiment_scores = Counter()
    
    for phase in li_sn_phases:
        co_occurrence_scores["material_properties"][phase] = 0
        co_occurrence_scores["technological_applications"][phase] = 0
        co_occurrence_scores["mechanistic_models"][phase] = 0
        co_occurrence_scores["material_characteristics"][phase] = 0
        relevance_scores[phase] = 0
        recency_scores[phase] = 0
        sentiment_scores[phase] = 0
    
    for paper in papers:
        year = paper["published"]
        recency_weight = 1 - (2025 - year) / 5
        text = (paper.get("pdf_text") or paper.get("abstract") or "").lower()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if any(phase in sentence.lower() for phase in li_sn_phases):
                for phase in li_sn_phases:
                    if phase in sentence.lower():
                        if any(term in sentence.lower() for term in material_properties):
                            co_occurrence_scores["material_properties"][phase] += 1
                        if any(term in sentence.lower() for term in technological_applications):
                            co_occurrence_scores["technological_applications"][phase] += 1
                        if any(term in sentence.lower() for term in mechanistic_models):
                            co_occurrence_scores["mechanistic_models"][phase] += 1
                        if any(term in sentence.lower() for term in material_characteristics):
                            co_occurrence_scores["material_characteristics"][phase] += 1
                        category_count = sum([
                            any(term in sentence.lower() for term in material_properties),
                            any(term in sentence.lower() for term in technological_applications),
                            any(term in sentence.lower() for term in mechanistic_models),
                            any(term in sentence.lower() for term in material_characteristics)
                        ])
                        relevance_score = 1.0 if category_count == 1 else 1.5 if category_count > 1 else 0
                        relevance_scores[phase] += relevance_score
                        sentiment = sid.polarity_scores(sentence)
                        sentiment_scores[phase] += sentiment["compound"]
                        recency_scores[phase] += recency_weight
    
    for category in co_occurrence_scores:
        max_score = max(co_occurrence_scores[category].values(), default=1)
        for phase in co_occurrence_scores[category]:
            co_occurrence_scores[category][phase] /= max_score if max_score > 0 else 1
    max_rel_score = max(relevance_scores.values(), default=1)
    for phase in relevance_scores:
        relevance_scores[phase] /= max_rel_score if max_rel_score > 0 else 1
    max_rec_score = max(recency_scores.values(), default=1)
    for phase in recency_scores:
        recency_scores[phase] /= max_rec_score if max_rec_score > 0 else 1
    max_sent_score = max(abs(v) for v in sentiment_scores.values()) if any(sentiment_scores.values()) else 1
    for phase in sentiment_scores:
        sentiment_scores[phase] /= max_sent_score if max_sent_score > 0 else 1
    
    return co_occurrence_scores, relevance_scores, recency_scores, sentiment_scores

def calculate_composite_score(word_freq, tfidf_scores, co_occurrence_scores, relevance_scores, recency_scores, sentiment_scores, max_freq, weights):
    composite_scores = {}
    w_material = weights.get("material_properties", 0.3)
    w_tech = weights.get("technological_applications", 0.4)
    w_models = weights.get("mechanistic_models", 0.2)
    w_char = weights.get("material_characteristics", 0.1)
    
    for phase in co_occurrence_scores["material_properties"]:
        freq_score = word_freq.get(phase, 0) / max_freq if max_freq > 0 else 0
        tfidf_score = tfidf_scores.get(phase, 0)
        co_score = (w_material * co_occurrence_scores["material_properties"].get(phase, 0) +
                    w_tech * co_occurrence_scores["technological_applications"].get(phase, 0) +
                    w_models * co_occurrence_scores["mechanistic_models"].get(phase, 0) +
                    w_char * co_occurrence_scores["material_characteristics"].get(phase, 0))
        rel_score = relevance_scores.get(phase, 0)
        rec_score = recency_scores.get(phase, 0)
        sent_score = sentiment_scores.get(phase, 0)
        composite_scores[phase] = (0.2 * freq_score + 0.15 * tfidf_score +
                                  0.3 * co_score + 0.15 * rel_score +
                                  0.1 * rec_score + 0.1 * sent_score)
    return composite_scores

# Visualization functions
def generate_word_cloud(text, selected_keywords, composite_scores, selection_criteria, colormap, title_font_size, caption_font_size, font_step, word_orientation, background_color, contour_width, contour_color):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['the', 'that', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        words = processed_text.split()
        filtered_words = [keyword_map.get(w, w) for w in words if keyword_map.get(w) in selected_keywords]
        if not filtered_words:
            logger.warning("No valid keywords found for word cloud.")
            return None, "No valid keywords found for word cloud."
        frequencies = {w: max(composite_scores.get(w, 0.01), 0.01) for w in filtered_words}
        max_freq_weight = max(frequencies.values(), default=1.0)
        if max_freq_weight <= 0:
            logger.warning("Invalid frequency weights for word cloud.")
            return None, "Invalid frequency weights for word cloud."
        for w in frequencies:
            frequencies[w] /= max_freq_weight
        wordcloud = WordCloud(
            width=1600, height=800, background_color=background_color, min_font_size=8,
            max_font_size=200, font_step=font_step, prefer_horizontal=1.0 if word_orientation == 'horizontal' else 0.5,
            colormap=colormap, contour_width=contour_width, contour_color=contour_color,
            margin=5, random_state=42
        ).generate_from_frequencies(frequencies)
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = ['Arial']
        fig, ax = plt.subplots(figsize=(12, 6), dpi=400)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Li-Sn Phase Word Cloud", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Word Cloud: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"
    finally:
        plt.close('all')
        gc.collect()
        
def generate_bibliometric_network(text, selected_keywords, composite_scores, label_font_size, selection_criteria, node_colormap, edge_colormap, network_style, line_thickness, node_alpha, edge_alpha, title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth, node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor, label_bbox_alpha, layout_algorithm, label_rotation, label_offset):
    try:
        G = nx.Graph()
        sentences = sent_tokenize(text.lower())
        co_occurrences = Counter()
        for sentence in sentences:
            present_keywords = [kw for kw in selected_keywords if kw in sentence]
            for kw1, kw2 in combinations(present_keywords, 2):
                co_occurrences[(kw1, kw2)] += 1
        for kw in selected_keywords:
            G.add_node(kw, weight=max(composite_scores.get(kw, 0.1), 0.1))
        for (kw1, kw2), count in co_occurrences.items():
            if count > 0:
                G.add_edge(kw1, kw2, weight=count)
        if G.number_of_nodes() == 0:
            logger.warning("No nodes in co-occurrence network.")
            return None, "No keywords found for network."
        plt.style.use(network_style)
        plt.rcParams['font.family'] = ['Arial']
        fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
        node_sizes = [G.nodes[kw]['weight'] * node_size_scale for kw in G.nodes]
        communities = list(greedy_modularity_communities(G))
        node_colors = {}
        cmap_nodes = plt.colormaps[node_colormap]
        for idx, community in enumerate(communities):
            color = cmap_nodes(idx / max(len(communities), 1))
            for node in community:
                node_colors[node] = color
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        max_weight = max(edge_weights, default=1.0)
        edge_widths = [w / max_weight * line_thickness for w in edge_weights]
        cmap_edges = plt.colormaps[edge_colormap]
        edge_colors = [cmap_edges(w / max_weight) for w in edge_weights]
        if layout_algorithm == 'spring':
            k = 1.0 / np.sqrt(G.number_of_nodes())
            pos = nx.spring_layout(G, k=k, iterations=50, seed=42)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=[node_colors[node] for node in G.nodes],
                               node_shape=node_shape, edgecolors=node_edgecolor, linewidths=node_linewidth, alpha=node_alpha, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, style=edge_style, alpha=edge_alpha, ax=ax)
        label_pos = {node: (pos[node][0], pos[node][1] + label_offset) for node in G.nodes}
        nx.draw_networkx_labels(G, pos=label_pos, font_size=label_font_size + 2, font_color=label_font_color,
                                font_family=label_font_family, font_weight='bold',
                                bbox=dict(facecolor=label_bbox_facecolor, alpha=label_bbox_alpha + 0.2, edgecolor='black', boxstyle='round,pad=0.3'), ax=ax)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges if G[u][v]['weight'] > max_weight * 0.5}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkgray')
        ax.set_title("Li-Sn Phase Co-occurrence Network", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Caption: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        ax.set_facecolor('#fafafa')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating network: {str(e)}")
        return None, f"Error generating network: {str(e)}"
    finally:
        plt.close('all')
        gc.collect()


def generate_stacked_bar_chart(co_occurrence_scores, selection_criteria, title_font_size, caption_font_size):
    try:
        li_sn_phases = sorted(co_occurrence_scores["material_properties"].keys())
        material_scores = [co_occurrence_scores["material_properties"].get(phase, 0) for phase in li_sn_phases]
        tech_scores = [co_occurrence_scores["technological_applications"].get(phase, 0) for phase in li_sn_phases]
        model_scores = [co_occurrence_scores["mechanistic_models"].get(phase, 0) for phase in li_sn_phases]
        char_scores = [co_occurrence_scores["material_characteristics"].get(p, 0) for p in li_sn_phases]

        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(12, 6), dpi=400)

        x = np.arange(len(li_sn_phases))
        ax.bar(x, material_scores, label='Material Properties', color='#1f77b4')
        ax.bar(x, tech_scores, bottom=material_scores, label='Technological Applications', color='#ff7f0e')
        ax.bar(x, model_scores, bottom=np.array(material_scores) + np.array(tech_scores), label='Mechanistic Models', color='#2ca02c')
        ax.bar(x, char_scores, bottom=np.array(material_scores) + np.array(tech_scores) + np.array(model_scores), label='Material Characteristics', color='#d62728')

        ax.set_xlabel('Li-Sn Phases', fontsize=12)
        ax.set_ylabel('Normalized Co-occurrence Score', fontsize=12)
        ax.set_title('Li-Sn Phase Co-occurrence by Category', fontsize=title_font_size, pad=20, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(li_sn_phases, rotation=45)
        ax.legend()

        caption = f"Stacked Bar Chart: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None

    except Exception as e:
        logger.error(f"Error generating stacked bar chart: {str(e)}")
        return None, f"Error generating stacked bar chart: {str(e)}"

    finally:
        plt.close('all')
        gc.collect()


def generate_heatmap_matrix(matrix, terms, li_sn_phases, selection_criteria, colormap, title_font_size, caption_font_size):
    try:
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = ['Arial']
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        sns.heatmap(matrix, xticklabels=terms, yticklabels=li_sn_phases, cmap=colormap, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=title_font_size, pad=20, fontweight='bold')
        plt.figtext(0.5, 0.02, f"Correlation Heatmap: {selection_criteria}", ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        return None, f"Error generating heatmap: {str(e)}"
    finally:
        plt.close('all')
        gc.collect()


def generate_sentiment_heatmap(scores, li_sn_phases, selection_criteria, colormap, title_font_size, caption_font_size):
    try:
        matrix = np.array([[scores.get(p, 0)] for p in li_sn_phases])
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = ['Arial']
        fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
        sns.heatmap(matrix, yticklabels=li_sn_phases, cmap=colormap, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Sentiment Heatmap", fontsize=title_font_size, pad=20, fontweight='bold')
        plt.figtext(0.5, 0.02, f"Sentiment Heatmap: {selection_criteria}", ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating sentiment heatmap: {str(e)}")
        return None, f"Error generating sentiment heatmap: {str(e)}"
    finally:
        plt.close('all')
        gc.collect()


def generate_parallel_coordinates_plot(data_df, selection_criteria, title_font_size, caption_font_size):
    try:
        fig = px.parallel_coordinates(
            data_df, dimensions=['Frequency', 'TF-IDF', 'Material', 'Tech', 'Models', 'Characteristics'],
            title="Parallel Coordinates Plot"
        )
        fig.update_layout(
            title_font_size=title_font_size,
            margin=dict(b=100)
        )
        fig.add_annotation(
            x=0.5, y=-0.1, xref="paper", yref="paper", showarrow=False,
            text=f"Parallel Coordinates: {selection_criteria}",
            font=dict(size=caption_font_size)
        )
        return fig, None
    except Exception as e:
        logger.error(f"Error generating parallel coordinates: {str(e)}")
        return None, f"Error generating parallel coordinates: {str(e)}"
    finally:
        gc.collect()

def generate_radar_chart(co_occurrence_scores, li_sn_phases, selection_criteria, colormap, title_font_size, caption_font_size, label_font_size, line_thickness, fill_alpha, label_rotation, label_offset, grid_color, grid_style, grid_thickness):
    try:
        categories = ['Material', 'Tech', 'Models', 'Characteristics']
        values = [[co_occurrence_scores[cat].get(p, 0) for p in li_sn_phases] for cat in ['material_properties', 'technological_applications', 'mechanistic_models', 'material_characteristics']]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = ['Arial']
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True), dpi=300)
        cmap = plt.colormaps[colormap]
        for i, phase_values in enumerate(zip(*values)):
            phase_values = list(phase_values) + [phase_values[0]]
            ax.plot(angles, phase_values, label=li_sn_phases[i], linewidth=line_thickness)
            ax.fill(angles, phase_values, alpha=fill_alpha, color=cmap(i / len(li_sn_phases)))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=label_font_size)
        ax.set_rlabel_position(0)
        ax.grid(color=grid_color, linestyle=grid_style, linewidth=grid_thickness)
        ax.set_title("Radar Chart of Li-Sn Phases", fontsize=title_font_size, pad=20, fontweight='bold')
        plt.figtext(0.5, 0.02, f"Radar Chart: {selection_criteria}", ha="center", fontsize=caption_font_size, wrap=True,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
        return None, f"Error generating radar chart: {str(e)}"
    finally:
        plt.close('all')
        gc.collect()

def save_figure(fig, filename):
    try:
        fig.savefig(f"{filename}.png", bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error saving figure {filename}: {str(e)}")
    finally:
        plt.close(fig)

# Streamlit app
st.title('Li-Sn Phase Analysis')
st.markdown('Analyze Li-Sn phases for lithium-ion battery anodes.')

# YAML uploader
st.subheader('Download YAML Template')
sample_yaml = yaml.dump(DEFAULT_KEYWORDS)
st.download_button(label="Download YAML Template", data=sample_yaml, file_name='keywords.yaml')

yaml_file = st.file_uploader('Upload YAML File', type=['yaml', 'yml'])
if yaml_file:
    KEYWORD_CATEGORIES = load_keywords(yaml_file.read())
else:
    KEYWORD_CATEGORIES = DEFAULT_KEYWORDS

# Weights
st.subheader('Scoring Weights')
col1, col2, col3, col4 = st.columns(4)
with col1:
    w_material = st.slider('Material Properties Weight', 0.0, 1.0, 0.3)
with col2:
    w_tech = st.slider('Technological Applications Weight', 0.0, 1.0, 0.4)
with col3:
    w_models = st.slider('Mechanistic Models Weight', 0.0, 1.0, 0.2)
with col4:
    w_char = st.slider('Material Characteristics Weight', 0.0, 1.0, 0.1)
weights = {
    'material_properties': w_material,
    'technological_applications': w_tech,
    'mechanistic_models': w_models,
    'material_characteristics': w_char
}

# Paper search
st.subheader('Paper Search')
repository = st.selectbox('Select Repository', ['arXiv', 'Semantic Scholar'], index=0)
st.session_state.repository = repository

if repository == 'Semantic Scholar':
    api_key = st.text_input('Semantic Scholar API Key (Optional)', type='password')
else:
    api_key = None

query = st.text_input('Search Query', 'tin anode AND lithium-ion battery')
max_results = st.slider('Maximum Results', 5, 50, 10 if repository == 'arXiv' else 5)
start_year = st.slider('Start Year', 2015, 2025, 2020)
custom_stopwords = st.text_input('Custom Stopwords (comma-separated)', 'et al,figure,equation')
exclude_keywords = st.text_input('Exclude Keywords (comma-separated)', 'preprint,submitted')

if st.button('Fetch Papers'):
    papers = fetch_papers(
        repository, query, max_results, start_year,
        memory_option == 'Use Abstracts Only',
        api_key=api_key
    )
    papers = deduplicate_papers(papers)
    st.session_state.papers = papers

    combined_text = '\n'.join([
        str(p.get('pdf_text', '') or p.get('abstract', '')) for p in papers
    ])
    st.session_state.combined_text = combined_text

    st.success(f'Fetched {len(papers)} unique papers from {repository}')

if 'papers' in st.session_state:
    papers = st.session_state.papers
    combined_text = st.session_state.combined_text
    st.subheader('Keyword Extraction Settings')
    min_freq = st.slider('Minimum Frequency', 1, 10, 5)
    min_length = st.slider('Minimum Keyword Length', 3, 30, 5)
    use_stopwords = st.checkbox('Use Stopwords', True)
    top_limit = st.slider('Top Keywords Limit', 10, 100, 50)
    tfidf_weight = float(st.slider('TF-IDF Weight', 0.0, 1.0, 1.0))
    use_nouns_only = st.checkbox('Extract Nouns Only', False)
    include_phrases = st.checkbox('Include Phrases', True)
    chunked = memory_option == 'Chunk Text Processing'
    criteria = [
        f'Min Freq: {min_freq}',
        f'Min Length: {min_length}',
        f'Stopwords: {"on" if use_stopwords else "off"}',
        f'Top Limit: {top_limit}',
        f'TF-IDF Weight: {tfidf_weight}',
        f'Nouns Only: {"on" if use_nouns_only else "off"}',
        f'Phrases: {"on" if include_phrases else "off"}'
    ]
    selection_criteria = ', '.join(criteria)
    
    st.subheader('Select Keywords for Analysis')
    categorized_keywords, word_freq, _, tfidf_scores, term_to_category, _ = get_candidate_keywords(
        combined_text, min_freq, min_length, use_stopwords, custom_stopwords, exclude_keywords,
        top_limit, tfidf_weight, use_nouns_only, include_phrases, nlp, KEYWORD_CATEGORIES, papers, chunked
    )
    li_sn_phases = KEYWORD_CATEGORIES['li_sn_phases']
    selected_keywords = set(li_sn_phases)
    for category in KEYWORD_CATEGORIES:
        if category != 'li_sn_phases':
            keywords = [t[0] for t in categorized_keywords[category]]
            with st.expander(f'{category.replace("_", " ").title()} ({len(keywords)} keywords)'):
                selected = st.multiselect(f'Select {category.replace("_", " ").title()} Keywords', keywords, default=keywords[:5])
                selected_keywords.update(selected)
    selected_keywords = list(selected_keywords)
    
    co_occurrence_scores, relevance_scores, recency_scores, sentiment_scores = compute_scores(
        combined_text, papers, li_sn_phases,
        KEYWORD_CATEGORIES['material_properties'],
        KEYWORD_CATEGORIES['technological_applications'],
        KEYWORD_CATEGORIES['mechanistic_models'],
        KEYWORD_CATEGORIES['material_characteristics']
    )
    
    adjusted_tfidf_scores = apply_recency_weight(tfidf_scores, papers)
    
    max_freq = max(word_freq.values(), default=1)
    composite_scores = calculate_composite_score(
        word_freq, adjusted_tfidf_scores, co_occurrence_scores, relevance_scores,
        recency_scores, sentiment_scores, max_freq, weights
    )
    
    # Numerical results
    st.subheader('Numerical Results')
    results_data = {
        'Phase': li_sn_phases,
        'Frequency': [word_freq.get(p, 0) for p in li_sn_phases],
        'TF-IDF': [adjusted_tfidf_scores.get(p, 0) for p in li_sn_phases],
        'Material': [co_occurrence_scores['material_properties'].get(p, 0) for p in li_sn_phases],
        'Tech': [co_occurrence_scores['technological_applications'].get(p, 0) for p in li_sn_phases],
        'Models': [co_occurrence_scores['mechanistic_models'].get(p, 0) for p in li_sn_phases],
        'Characteristics': [co_occurrence_scores['material_characteristics'].get(p, 0) for p in li_sn_phases],
        'Relevance': [relevance_scores.get(p, 0) for p in li_sn_phases],
        'Recency': [recency_scores.get(p, 0) for p in li_sn_phases],
        'Sentiment': [sentiment_scores.get(p, 0) for p in li_sn_phases],
        'Composite': [composite_scores.get(p, 0.0) for p in li_sn_phases]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df)
    st.download_button('Download Results as CSV', results_df.to_csv(index=False), 'results.csv')

    # Visualizations
    st.subheader('Visualizations')
    preset = st.selectbox('Visualization Preset', ['Simple', 'Professional'])
    vis_settings = visualization_settings_presets[preset]
    
    wordcloud_fig, wordcloud_error = generate_word_cloud(
        combined_text, selected_keywords, composite_scores, selection_criteria,
        vis_settings['wordcloud_colormap'], vis_settings['title_font_size'],
        vis_settings['caption_font_size'], vis_settings['font_step'],
        vis_settings['word_orientation'],
        vis_settings['background_color'],
        vis_settings['contour_width'], vis_settings['contour_color']
    )
    if wordcloud_fig:
        st.pyplot(wordcloud_fig)
        save_figure(wordcloud_fig, 'wordcloud')
        st.download_button('Download Word Cloud', data=open('wordcloud.png', 'rb').read(), file_name='wordcloud.png')
    elif wordcloud_error:
        st.error(wordcloud_error)
    
    network_fig, network_error = generate_bibliometric_network(
        combined_text, selected_keywords, composite_scores, vis_settings['label_font_size'],
        selection_criteria, vis_settings['node_colormap'], vis_settings['edge_colormap'],
        vis_settings['network_style'], vis_settings['line_thickness'], vis_settings['node_alpha'],
        vis_settings['edge_alpha'], vis_settings['title_font_size'], vis_settings['caption_font_size'],
        vis_settings['node_size_scale'], vis_settings['node_shape'], vis_settings['node_linewidth'],
        vis_settings['node_edgecolor'], vis_settings['edge_style'], vis_settings['label_font_color'],
        vis_settings['label_font_family'], vis_settings['label_bbox_facecolor'],
        vis_settings['label_bbox_alpha'], vis_settings['layout_algorithm'],
        vis_settings['label_rotation'], vis_settings['label_offset']
    )
    if network_fig:
        st.pyplot(network_fig)
        save_figure(network_fig, 'network')
        st.download_button('Download Network', data=open('network.png', 'rb').read(), file_name='network.png')
    elif network_error:
        st.error(network_error)
    
    stacked_bar_fig, stacked_bar_error = generate_stacked_bar_chart(
        co_occurrence_scores, selection_criteria, vis_settings['title_font_size'],
        vis_settings['caption_font_size']
    )
    if stacked_bar_fig:
        st.pyplot(stacked_bar_fig)
        save_figure(stacked_bar_fig, 'stacked_bar')
        st.download_button('Download Stacked Bar Chart', data=open('stacked_bar.png', 'rb').read(), file_name='stacked_bar.png')
    elif stacked_bar_error:
        st.error(stacked_bar_error)
    
    # Placeholder for co-occurrence matrix
    co_occurrence_matrix = np.random.rand(len(li_sn_phases), len(selected_keywords))
    terms = selected_keywords
    heatmap_fig, heatmap_error = generate_heatmap_matrix(
        co_occurrence_matrix, terms, li_sn_phases, selection_criteria,
        vis_settings['heatmap_colormap'], vis_settings['title_font_size'],
        vis_settings['caption_font_size']
    )
    if heatmap_fig:
        st.pyplot(heatmap_fig)
        save_figure(heatmap_fig, 'heatmap')
        st.download_button('Download Heatmap', data=open('heatmap.png', 'rb').read(), file_name='heatmap')
    elif heatmap_error:
        st.error(heatmap_error)
    
    sentiment_fig, sentiment_error = generate_sentiment_heatmap(
        sentiment_scores, li_sn_phases, selection_criteria, vis_settings['heatmap_colormap'],
        vis_settings['title_font_size'], vis_settings['caption_font_size']
    )
    if sentiment_fig:
        st.pyplot(sentiment_fig)
        save_figure(sentiment_fig, 'sentiment_heatmap')
        st.download_button('Download Sentiment Heatmap', data=open('sentiment_heatmap.png', 'rb').read(), file_name='sentiment_heatmap.png')
    elif sentiment_error:
        st.error(sentiment_error)
    
    parallel_df = results_df[['Frequency', 'TF-IDF', 'Material', 'Tech', 'Models', 'Characteristics']]
    parallel_fig, parallel_error = generate_parallel_coordinates_plot(
        parallel_df, selection_criteria, vis_settings['title_font_size'],
        vis_settings['caption_font_size']
    )
    if parallel_fig:
        st.plotly_chart(parallel_fig)
    elif parallel_error:
        st.error(parallel_error)
    
    radar_fig, radar_error = generate_radar_chart(
        co_occurrence_scores, li_sn_phases, selection_criteria, vis_settings['radar_colormap'],
        vis_settings['title_font_size'], vis_settings['caption_font_size'],
        vis_settings['label_font_size'], vis_settings['line_thickness'], 0.2,
        vis_settings['label_rotation'], vis_settings['label_offset'], vis_settings['grid_color'],
        vis_settings['grid_style'], vis_settings['grid_thickness']
    )
    if radar_fig:
        st.pyplot(radar_fig)
        save_figure(radar_fig, 'radar')
        st.download_button('Download Radar Chart', data=open('radar.png', 'rb').read(), file_name='radar.png')
    elif radar_error:
        st.error(radar_error)
    
    # Rankings
    st.subheader('Phase Rankings')
    sorted_phases = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (phase, score) in enumerate(sorted_phases, 1):
        st.markdown(f'{rank}. **{phase}**: Composite Score = {score:.4f}')
    
    st.markdown(f'**Powered by**: Streamlit, NLTK, spaCy, Matplotlib, NetworkX, Plotly, {repository} API')

