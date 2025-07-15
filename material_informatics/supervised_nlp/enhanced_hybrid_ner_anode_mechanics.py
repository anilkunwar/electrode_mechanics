import streamlit as st
import spacy
from spacy.pipeline import EntityRuler
import re
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import numpy as np
from matplotlib.patches import Patch

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "colormap" not in st.session_state:
    st.session_state.colormap = "viridis"
if "bar_width" not in st.session_state:
    st.session_state.bar_width = 0.9
if "title_font_size" not in st.session_state:
    st.session_state.title_font_size = 18
if "label_font_size" not in st.session_state:
    st.session_state.label_font_size = 14
if "tick_font_size" not in st.session_state:
    st.session_state.tick_font_size = 16
if "pie_explode" not in st.session_state:
    st.session_state.pie_explode = 0.1
if "pie_label_font_size" not in st.session_state:
    st.session_state.pie_label_font_size = 12
if "show_pie_percentages" not in st.session_state:
    st.session_state.show_pie_percentages = True
if "rule_entities" not in st.session_state:
    st.session_state.rule_entities = []
if "spacy_entities" not in st.session_state:
    st.session_state.spacy_entities = []
if "hybrid_entities" not in st.session_state:
    st.session_state.hybrid_entities = []
if "entity_probs" not in st.session_state:
    st.session_state.entity_probs = {}

# Load spaCy model and add EntityRuler for BCT Sn
try:
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": "PHASE", "pattern": [{"LOWER": "bct"}, {"LOWER": "sn"}]}])
    logger.info("spaCy model with EntityRuler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    st.error(f"Failed to load spaCy model: {e}. Please ensure 'en_core_web_sm' is installed.")
    st.stop()

# Define patterns and keywords for rule-based NER
PHASE_PATTERN = r"Li\d*(?:\.\d+)?Sn\d*"  # Matches LiₓSnᵧ phases (e.g., Li2Sn5, Li0.5Sn2)
ANODE_KEYWORDS = [
    "Sn anode", "Li-Sn phase transformation", "volume expansion",
    "microstructural evolution", "elastic-strain energy", "electrode potential",
    "Gibbs free energy", "cyclic property degradation", "SEI formation",
    "diffusion-induced stress", "reaction-controlled lithiation",
    "diffusion-controlled lithiation", "lithiation", "delithiation",
    "electrolyte interface", "mechanical stress", "crack formation",
    "capacity fade", "anode degradation", "phase boundary", "interphase layer"
]

# Comprehensive list of Matplotlib colormaps
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'rainbow', 'jet',
    'turbo', 'hsv', 'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys',
    'YlOrRd', 'YlOrBr', 'YlGnBu', 'BuPu', 'GnBu', 'PuBu', 'PuRd', 'RdPu',
    'OrRd', 'PuBuGn', 'BuGn', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm',
    'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv_r', 'flag', 'prism',
    'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2',
    'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow_r', 'nipy_spectral',
    'gist_ncar', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'afmhot',
    'autumn', 'spring', 'summer', 'winter'
]

def update_log(message):
    """Update the log buffer."""
    st.session_state.log_buffer.append(message)
    if len(st.session_state.log_buffer) > 20:
        st.session_state.log_buffer.pop(0)

def rule_based_ner(text):
    """Perform rule-based NER for LiₓSnᵧ phases and anode mechanics keywords."""
    entities = []
    try:
        phase_matches = re.findall(PHASE_PATTERN, text, re.IGNORECASE)
        for match in phase_matches:
            if match and "Li" in match and "Sn" in match:
                entities.append({"text": match, "type": "LiₓSnᵧ Phase"})
        logger.debug(f"Rule-based: Extracted phases: {phase_matches}")
        update_log(f"Rule-based: Extracted phases: {phase_matches}")
    except Exception as e:
        logger.error(f"Rule-based: Error in regex matching: {e}")
        update_log(f"Rule-based: Error in regex matching: {e}")
    
    text_lower = text.lower()
    for keyword in ANODE_KEYWORDS:
        if keyword.lower() in text_lower:
            entities.append({"text": keyword, "type": "Anode Keyword"})
            logger.debug(f"Rule-based: Found keyword: {keyword}")
            update_log(f"Rule-based: Found keyword: {keyword}")
    
    normalized_entities = []
    seen = set()
    for entity in entities:
        text = entity["text"].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            normalized_entities.append({"text": text, "type": entity["type"]})
    
    logger.info(f"Rule-based: Final entities: {normalized_entities}")
    update_log(f"Rule-based: Final entities: {normalized_entities}")
    return normalized_entities

def spacy_ner(text):
    """Perform spaCy NER with EntityRuler for general and domain-specific entities."""
    entities = []
    try:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({"text": ent.text, "type": ent.label_})
            logger.debug(f"spaCy: Detected entity '{ent.text}' as '{ent.label_}'")
        for token in doc:
            if re.match(PHASE_PATTERN, token.text, re.IGNORECASE):
                entities.append({"text": token.text, "type": "LiₓSnᵧ Phase"})
                logger.debug(f"spaCy: Detected regex entity '{token.text}' as 'LiₓSnᵧ Phase'")
        logger.debug(f"spaCy: Entities extracted: {[(e['text'], e['type']) for e in entities]}")
        update_log(f"spaCy: Entities extracted: {[(e['text'], e['type']) for e in entities]}")
    except Exception as e:
        logger.error(f"spaCy: Error in NER: {e}")
        update_log(f"spaCy: Error in NER: {e}")
    
    normalized_entities = []
    seen = set()
    for entity in entities:
        text = entity["text"].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            normalized_entities.append({"text": text, "type": entity["type"]})
    
    logger.info(f"spaCy: Final entities: {normalized_entities}")
    update_log(f"spaCy: Final entities: {normalized_entities}")
    return normalized_entities

def estimate_entity_probabilities(doc):
    """Estimate probabilities for all spaCy-detected entities using embeddings and context."""
    try:
        labels = ["ORG", "PERSON", "GPE", "PHASE", "LiₓSnᵧ Phase", "NONE"]
        entity_probs = {}
        
        for ent in doc.ents:
            span_text = ent.text
            span_tokens = [token for token in doc if token.idx >= ent.start_char and token.idx < ent.end_char]
            if not span_tokens:
                logger.warning(f"No tokens found for entity '{span_text}'")
                update_log(f"No tokens found for entity '{span_text}'")
                entity_probs[span_text] = {label: 1.0/len(labels) for label in labels}
                continue
            
            valid_vectors = [token.vector for token in span_tokens if token.vector.any()]
            if valid_vectors:
                feature_vector = np.mean(valid_vectors, axis=0)
            else:
                logger.warning(f"No valid vectors for entity '{span_text}'")
                update_log(f"No valid vectors for entity '{span_text}'")
                feature_vector = np.zeros(96)
            
            context_tokens = list(doc[max(0, ent.start - 2):ent.start]) + list(doc[ent.end:min(ent.end + 2, len(doc))])
            valid_context_vectors = [token.vector for token in context_tokens if token.vector.any()]
            context_vector = np.mean(valid_context_vectors, axis=0) if valid_context_vectors else np.zeros_like(feature_vector)
            feature_vector = (feature_vector + context_vector) / 2
            
            weights = {"PHASE": 1.0, "LiₓSnᵧ Phase": 0.9, "ORG": 0.6, "PERSON": 0.4, "GPE": 0.4, "NONE": 0.2}
            pos_scores = {"NOUN": 0.3, "PROPN": 0.5, "ADJ": 0.1}
            pos_boost = sum(pos_scores.get(token.pos_, 0) for token in span_tokens) / len(span_tokens) if span_tokens else 0
            
            scores = []
            for label in labels:
                score = weights[label] * (np.linalg.norm(feature_vector) + 1e-8) + pos_boost
                if label == ent.label_:
                    score += 1.0
                scores.append(score)
            
            exp_scores = np.exp(scores)
            probabilities = exp_scores / np.sum(exp_scores)
            entity_probs[span_text] = dict(zip(labels, probabilities))
            logger.debug(f"Probabilities for '{span_text}': {entity_probs[span_text]}")
        
        for token in doc:
            if re.match(PHASE_PATTERN, token.text, re.IGNORECASE) and token.text not in entity_probs:
                span_text = token.text
                feature_vector = token.vector if token.vector.any() else np.zeros(96)
                context_tokens = list(doc[max(0, token.i - 2):token.i]) + list(doc[token.i + 1:min(token.i + 3, len(doc))])
                valid_context_vectors = [t.vector for t in context_tokens if t.vector.any()]
                context_vector = np.mean(valid_context_vectors, axis=0) if valid_context_vectors else np.zeros_like(feature_vector)
                feature_vector = (feature_vector + context_vector) / 2
                pos_boost = pos_scores.get(token.pos_, 0)
                
                scores = []
                for label in labels:
                    score = weights[label] * (np.linalg.norm(feature_vector) + 1e-8) + pos_boost
                    if label == "LiₓSnᵧ Phase":
                        score += 1.0
                    scores.append(score)
                
                exp_scores = np.exp(scores)
                probabilities = exp_scores / np.sum(exp_scores)
                entity_probs[span_text] = dict(zip(labels, probabilities))
                logger.debug(f"Probabilities for regex entity '{span_text}': {entity_probs[span_text]}")
        
        logger.info(f"Estimated probabilities: {entity_probs}")
        update_log(f"Estimated probabilities: {entity_probs}")
        return entity_probs
    
    except Exception as e:
        logger.error(f"Error estimating probabilities: {e}")
        update_log(f"Error estimating probabilities: {e}")
        return {}

def create_entity_prob_plot(prob_dict, entity_text):
    """Create a bar plot for an entity's label probabilities."""
    try:
        if not prob_dict:
            logger.warning(f"No probabilities for {entity_text} bar plot.")
            update_log(f"No probabilities for {entity_text} bar plot.")
            return BytesIO(), BytesIO()
        
        labels = list(prob_dict.keys())
        probs = list(prob_dict.values())
        
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        ax.bar(labels, probs, color=matplotlib.colormaps.get_cmap(st.session_state.colormap)(0.5))
        
        ax.set_xlabel("Entity Labels", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_ylabel("Probability", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_title(f"Estimated Probabilities for '{entity_text}' (Bar)", fontsize=st.session_state.title_font_size, pad=15, weight='bold')
        
        ax.tick_params(axis='x', labelsize=st.session_state.tick_font_size, rotation=45)
        ax.tick_params(axis='y', labelsize=st.session_state.tick_font_size)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout(pad=2.0)
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=300, facecolor='white')
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        
        logger.info(f"Bar probability plot for {entity_text} generated successfully.")
        update_log(f"Bar probability plot for {entity_text} generated successfully.")
        return buf_png, buf_svg
    
    except Exception as e:
        logger.error(f"Error creating bar probability plot for {entity_text}: {e}")
        update_log(f"Error creating bar probability plot for {entity_text}: {e}")
        return BytesIO(), BytesIO()

def create_entity_prob_pie_plot(prob_dict, entity_text):
    """Create a pie chart for an entity's label probabilities with a colored legend."""
    try:
        if not prob_dict:
            logger.warning(f"No probabilities for {entity_text} pie plot.")
            update_log(f"No probabilities for {entity_text} pie plot.")
            return BytesIO(), BytesIO()
        
        labels = list(prob_dict.keys())
        probs = list(prob_dict.values())
        
        # Filter out zero-probability labels to avoid clutter
        filtered_labels = [label for label, prob in zip(labels, probs) if prob > 0.01]
        filtered_probs = [prob for prob in probs if prob > 0.01]
        
        if not filtered_probs:
            logger.warning(f"No non-zero probabilities for {entity_text} pie plot.")
            update_log(f"No non-zero probabilities for {entity_text} pie plot.")
            return BytesIO(), BytesIO()
        
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        # Create explode array: explode the highest probability slice
        explode = [st.session_state.pie_explode if prob == max(filtered_probs) else 0 for prob in filtered_probs]
        
        # Define colors from colormap
        cmap = matplotlib.colormaps.get_cmap(st.session_state.colormap)
        colors = [cmap(i / len(filtered_labels)) for i in range(len(filtered_labels))]
        
        # Plot pie chart without autopct
        wedges, texts = ax.pie(
            filtered_probs,
            labels=None,  # Remove labels from slices
            colors=colors,
            explode=explode,
            startangle=90
        )
        
        # Create legend with labels and percentages
        legend_labels = [
            f"{label}: {prob*100:.1f}%" if st.session_state.show_pie_percentages else label
            for label, prob in zip(filtered_labels, filtered_probs)
        ]
        legend_patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(filtered_labels))]
        ax.legend(
            handles=legend_patches,
            fontsize=st.session_state.pie_label_font_size,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            frameon=True,
            edgecolor='black'
        )
        
        ax.set_title(f"Estimated Probabilities for '{entity_text}' (Pie)", fontsize=st.session_state.title_font_size, pad=15, weight='bold')
        
        # Ensure circular shape
        ax.axis('equal')
        
        plt.tight_layout(pad=2.0, rect=[0, 0, 0.85, 1])  # Adjust for legend
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=300, facecolor='white')
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        
        logger.info(f"Pie probability plot for {entity_text} with colored legend generated successfully.")
        update_log(f"Pie probability plot for {entity_text} with colored legend generated successfully.")
        return buf_png, buf_svg
    
    except Exception as e:
        logger.error(f"Error creating pie probability plot for {entity_text}: {e}")
        update_log(f"Error creating pie probability plot for {entity_text}: {e}")
        return BytesIO(), BytesIO()

def generate_math_explanation(entity_probs):
    """Generate Markdown explanation of spaCy NER probability computation for all entities."""
    explanation = """
## Mathematical Formulation of spaCy NER Probability

spaCy's Named Entity Recognition (NER) model assigns probabilities to potential entity labels for each detected span using a statistical approach. The probability of a label \( l \) for a span \( s \) given its context \( C \) is computed as:

\[
P(l|s, C) = \frac{\exp(f(s, C, l))}{\sum_{l' \in L} \exp(f(s, C, l'))}
\]

### Components
- **\( l \)**: Entity label (e.g., ORG, PERSON, GPE, PHASE, LiₓSnᵧ Phase, NONE).
- **\( s \)**: Span, e.g., "BCT Sn" or "Li2Sn5".
- **\( C \)**: Context, including surrounding tokens, POS tags, and embeddings.
- **\( L \)**: Set of possible labels {ORG, PERSON, GPE, PHASE, LiₓSnᵧ Phase, NONE}.
- **\( f(s, C, l) \)**: Scoring function, approximated as:

\[
f(s, C, l) = W_l \cdot \phi(s, C) + b_l
\]

- **\( \phi(s, C) \)**: Feature vector combining token embeddings of the span and context tokens.
- **\( W_l, b_l \)**: Label-specific weights and biases, learned during training.

### Computation for All Entities
For each entity detected by spaCy (including those from EntityRuler and regex), the following steps are applied:

1. **Feature Extraction**:
   - Token embeddings for the span are averaged: \( \phi(s) = \text{mean}(\text{vector}(t_i)) \) for tokens \( t_i \) in the span.
   - Context embeddings from surrounding tokens (two tokens before and after) are averaged.
   - POS tags (e.g., PROPN, NOUN) add a heuristic boost.

2. **Scoring**:
   - For each label \( l \), compute \( f(s, C, l) \) using heuristic weights (since actual \( W_l \) is inaccessible):
     - PHASE: 1.0 (high due to EntityRuler for "BCT Sn").
     - LiₓSnᵧ Phase: 0.9 (high for regex-detected phases).
     - ORG, PERSON, GPE: 0.6, 0.4, 0.4.
     - NONE: 0.2.
   - POS boost: +0.5 for PROPN, +0.3 for NOUN, +0.1 for ADJ.
   - Predicted label boost: +1.0 if the label matches spaCy's prediction.

3. **Softmax Normalization**:
   - Compute \( \exp(f(s, C, l)) \) for each label.
   - Normalize: \( P(l|s, C) = \frac{\exp(f(s, C, l))}{\sum_{l' \in L} \exp(f(s, C, l'))} \).

### Results
Below are the estimated probabilities for each spaCy-detected entity in the input sentence:

"""
    if entity_probs:
        for entity_text, probs in entity_probs.items():
            explanation += f"#### {entity_text}\n"
            prob_list = [f"- **{label}**: {prob:.3f}" for label, prob in probs.items()]
            explanation += "\n".join(prob_list) + "\n\n"
    else:
        explanation += "- No entities detected or probabilities computed.\n\n"
    
    explanation += """
### Notes
- The EntityRuler ensures "BCT Sn" is labeled as PHASE, boosting its score.
- Regex ensures LiₓSnᵧ phases (e.g., Li2Sn5) are detected, with a high weight.
- Actual spaCy scores use neural network weights, approximated here with heuristics and embeddings.
- The bar plots and pie charts (with percentages in the legend) below visualize these probabilities for each entity.
"""
    return explanation

def hybrid_ner(text):
    """Combine rule-based and spaCy NER for hybrid approach."""
    rule_entities = rule_based_ner(text)
    spacy_entities = spacy_ner(text)
    
    hybrid_entities = rule_entities.copy()
    seen = {e["text"].lower() for e in hybrid_entities}
    for spacy_entity in spacy_entities:
        if spacy_entity["text"].lower() not in seen:
            hybrid_entities.append(spacy_entity)
            seen.add(spacy_entity["text"].lower())
    
    logger.info(f"Hybrid: Final entities: {hybrid_entities}")
    update_log(f"Hybrid: Final entities: {hybrid_entities}")
    
    rule_set = {(e["text"].lower(), e["type"]) for e in rule_entities}
    spacy_set = {(e["text"].lower(), e["type"]) for e in spacy_entities}
    rule_only = rule_set - spacy_set
    spacy_only = spacy_set - rule_set
    if rule_only:
        update_log(f"Entities unique to rule-based: {rule_only}")
    if spacy_only:
        update_log(f"Entities unique to spaCy: {spacy_only}")
    
    return hybrid_entities

def create_heatmap(rule_entities, spacy_entities, hybrid_entities):
    """Create a publication-quality heatmap comparing NER methods."""
    try:
        all_entities = set()
        for entities in [rule_entities, spacy_entities, hybrid_entities]:
            for entity in entities:
                all_entities.add((entity["text"], entity["type"]))
        
        if not all_entities:
            logger.warning("No entities found for heatmap.")
            update_log("No entities found for heatmap.")
            return BytesIO(), BytesIO()
        
        methods = ["Rule-Based", "spaCy", "Hybrid"]
        entity_labels = [f"{text} ({type_})" for text, type_ in sorted(all_entities)]
        data = np.zeros((len(methods), len(entity_labels)))
        
        for i, method_entities in enumerate([rule_entities, spacy_entities, hybrid_entities]):
            method_set = {(e["text"].lower(), e["type"]) for e in method_entities}
            for j, (text, type_) in enumerate(sorted(all_entities)):
                if (text.lower(), type_) in method_set:
                    data[i, j] = 1
        
        df = pd.DataFrame(data, index=methods, columns=entity_labels)
        
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        
        cmap = matplotlib.colormaps.get_cmap(st.session_state.colormap)
        cax = ax.imshow(df, cmap=cmap, interpolation='nearest')
        
        ax.set_xticks(np.arange(len(entity_labels)))
        ax.set_xticklabels(entity_labels, rotation=45, ha='right', fontsize=st.session_state.tick_font_size, weight='medium')
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods, fontsize=st.session_state.tick_font_size, weight='medium')
        
        ax.set_xlabel("Entities", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_ylabel("NER Methods", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_title("Entity Detection by NER Method", fontsize=st.session_state.title_font_size, pad=15, weight='bold')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        cbar = plt.colorbar(cax, label='Presence (1) / Absence (0)', pad=0.05)
        cbar.ax.tick_params(labelsize=st.session_state.tick_font_size, width=1.5)
        cbar.set_label('Presence', fontsize=st.session_state.label_font_size, weight='bold')
        cbar.outline.set_linewidth(1.5)
        
        ax.set_xticks(np.arange(len(entity_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(methods) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.tight_layout(pad=2.0)
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=300, facecolor='white')
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        
        logger.info("Heatmap generated successfully.")
        update_log("Heatmap generated successfully.")
        return buf_png, buf_svg
    
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        update_log(f"Error creating heatmap: {e}")
        return BytesIO(), BytesIO()

def create_bar_plot(rule_entities, spacy_entities, hybrid_entities):
    """Create a publication-quality bar plot comparing entity type counts."""
    try:
        rule_counts = Counter(e["type"] for e in rule_entities)
        spacy_counts = Counter(e["type"] for e in spacy_entities)
        hybrid_counts = Counter(e["type"] for e in hybrid_entities)
        
        all_types = sorted(set(rule_counts.keys()) | set(spacy_counts.keys()) | set(hybrid_counts.keys()))
        
        if not all_types:
            logger.warning("No entity types found for bar plot.")
            update_log("No entity types found for bar plot.")
            return BytesIO(), BytesIO()
        
        data = {
            "Rule-Based": [rule_counts.get(t, 0) for t in all_types],
            "spaCy": [spacy_counts.get(t, 0) for t in all_types],
            "Hybrid": [hybrid_counts.get(t, 0) for t in all_types]
        }
        df = pd.DataFrame(data, index=all_types)
        
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        x = np.arange(len(all_types))
        width = st.session_state.bar_width / 3
        
        ax.bar(x - width, df["Rule-Based"], width, label="Rule-Based", color=matplotlib.colormaps.get_cmap(st.session_state.colormap)(0.2))
        ax.bar(x, df["spaCy"], width, label="spaCy", color=matplotlib.colormaps.get_cmap(st.session_state.colormap)(0.5))
        ax.bar(x + width, df["Hybrid"], width, label="Hybrid", color=matplotlib.colormaps.get_cmap(st.session_state.colormap)(0.8))
        
        ax.set_xlabel("Entity Types", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_ylabel("Count", fontsize=st.session_state.label_font_size, weight='bold', labelpad=10)
        ax.set_title("Entity Type Counts by NER Method", fontsize=st.session_state.title_font_size, pad=15, weight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=45, ha='right', fontsize=st.session_state.tick_font_size, weight='medium')
        ax.tick_params(axis='y', labelsize=st.session_state.tick_font_size)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        ax.legend(fontsize=st.session_state.tick_font_size, frameon=True, edgecolor='black')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout(pad=2.0)
        
        buf_png = BytesIO()
        plt.savefig(buf_png, format="png", bbox_inches="tight", dpi=300, facecolor='white')
        buf_svg = BytesIO()
        plt.savefig(buf_svg, format="svg", bbox_inches="tight")
        plt.close()
        buf_png.seek(0)
        buf_svg.seek(0)
        
        logger.info("Bar plot generated successfully.")
        update_log("Bar plot generated successfully.")
        return buf_png, buf_svg
    
    except Exception as e:
        logger.error(f"Error creating bar plot: {e}")
        update_log(f"Error creating bar plot: {e}")
        return BytesIO(), BytesIO()

def display_entities(entities, title):
    """Display entities in Streamlit with summary and details."""
    if entities:
        entity_counts = Counter(e["type"] for e in entities)
        st.write(f"**{title} - Summary of Entity Types**")
        for entity_type, count in entity_counts.items():
            st.write(f"{entity_type}: {count}")
        st.write(f"**{title} - Detailed Entities**")
        entity_df = pd.DataFrame(entities)
        st.dataframe(entity_df[["text", "type"]])
    else:
        st.warning(f"No entities found for {title.lower()}.")

def main():
    st.set_page_config(page_title="NER Comparison for Sn Anode Mechanics", layout="wide")
    st.title("NER Comparison for Sn Anode Mechanics")
    st.markdown("""
    This app compares spaCy's statistical Named Entity Recognition (NER) with rule-based NER for a sentence related to anode mechanics of Sn electrodes in batteries. 
    It also demonstrates a hybrid approach combining both methods, prioritizing domain-specific rules for LiₓSnᵧ phases and anode keywords. Visualizations include a heatmap, entity type counts, and probability plots (bar and pie).
    A separate tab displays estimated probabilities for all spaCy-detected entities with a mathematical explanation.
    """)

    # Sidebar for visualization settings
    with st.sidebar:
        st.header("Visualization Settings")
        st.session_state.colormap = st.selectbox(
            "Colormap for Visualizations",
            COLORMAPS,
            index=COLORMAPS.index(st.session_state.colormap) if st.session_state.colormap in COLORMAPS else 0,
            key="colormap_select"
        )
        st.session_state.bar_width = st.slider(
            "Bar Width",
            min_value=0.5, max_value=1.5, value=st.session_state.bar_width, step=0.1,
            key="bar_width_slider"
        )
        st.session_state.title_font_size = st.slider(
            "Title Font Size",
            min_value=12, max_value=24, value=st.session_state.title_font_size, step=1,
            key="title_font_size_slider"
        )
        st.session_state.label_font_size = st.slider(
            "Label Font Size",
            min_value=10, max_value=20, value=st.session_state.label_font_size, step=1,
            key="label_font_size_slider"
        )
        st.session_state.tick_font_size = st.slider(
            "Tick Font Size",
            min_value=8, max_value=16, value=st.session_state.tick_font_size, step=1,
            key="tick_font_size_slider"
        )
        st.session_state.pie_explode = st.slider(
            "Pie Chart Explosion (Max Probability Slice)",
            min_value=0.0, max_value=0.5, value=st.session_state.pie_explode, step=0.05,
            key="pie_explode_slider"
        )
        st.session_state.pie_label_font_size = st.slider(
            "Pie Chart Legend Font Size",
            min_value=8, max_value=16, value=st.session_state.pie_label_font_size, step=1,
            key="pie_label_font_size_slider"
        )
        st.session_state.show_pie_percentages = st.checkbox(
            "Show Percentages in Pie Chart Legend",
            value=st.session_state.show_pie_percentages,
            key="show_pie_percentages_checkbox"
        )

    # Input sentence
    sentence = st.text_input(
        "Enter a sentence about Sn anode mechanics",
        value="During lithiation, the phase transformation from BCT Sn to Li2Sn5 causes significant volume expansion."
    )

    # Process button
    if st.button("Analyze Sentence"):
        if sentence.strip():
            with st.spinner("Analyzing sentence..."):
                st.session_state.rule_entities = rule_based_ner(sentence)
                st.session_state.spacy_entities = spacy_ner(sentence)
                st.session_state.hybrid_entities = hybrid_ner(sentence)
                doc = nlp(sentence)
                st.session_state.entity_probs = estimate_entity_probabilities(doc)
                logger.info("Analysis completed for sentence: " + sentence)
                update_log("Analysis completed for sentence: " + sentence)
        else:
            st.error("Please enter a non-empty sentence.")
            update_log("Error: Empty sentence provided.")

    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Rule-Based NER", "spaCy NER", "Hybrid NER", "Visualizations", "Entity Probabilities", "Logs"])

    with tab1:
        st.subheader("Rule-Based NER Results")
        st.markdown("Uses regex for LiₓSnᵧ phases and exact matching for anode mechanics keywords, including 'BCT Sn'.")
        display_entities(st.session_state.rule_entities, "Rule-Based NER")

    with tab2:
        st.subheader("spaCy NER Results")
        st.markdown("Uses spaCy's statistical model (`en_core_web_sm`) with EntityRuler for 'BCT Sn' and regex fallback for LiₓSnᵧ phases.")
        display_entities(st.session_state.spacy_entities, "spaCy NER")

    with tab3:
        st.subheader("Hybrid NER Results")
        st.markdown("Combines rule-based NER with spaCy NER, prioritizing domain-specific terms.")
        display_entities(st.session_state.hybrid_entities, "Hybrid NER")

    with tab4:
        st.subheader("Visualization of NER Comparison")
        st.markdown("Heatmap and bar plot comparing NER methods. Adjust settings in the sidebar to update visualizations.")

        if st.session_state.rule_entities or st.session_state.spacy_entities or st.session_state.hybrid_entities:
            # Heatmap
            heatmap_png, heatmap_svg = create_heatmap(
                st.session_state.rule_entities,
                st.session_state.spacy_entities,
                st.session_state.hybrid_entities
            )
            if heatmap_png.getvalue():
                st.image(heatmap_png, caption="Heatmap of Entity Detection", use_column_width=True)
                st.download_button(
                    label="Download Heatmap (PNG)",
                    data=heatmap_png,
                    file_name="ner_heatmap.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Heatmap (SVG)",
                    data=heatmap_svg,
                    file_name="ner_heatmap.svg",
                    mime="image/svg+xml"
                )
            else:
                st.warning("Failed to generate heatmap.")

            # Bar Plot
            bar_png, bar_svg = create_bar_plot(
                st.session_state.rule_entities,
                st.session_state.spacy_entities,
                st.session_state.hybrid_entities
            )
            if bar_png.getvalue():
                st.image(bar_png, caption="Bar Plot of Entity Type Counts", use_column_width=True)
                st.download_button(
                    label="Download Bar Plot (PNG)",
                    data=bar_png,
                    file_name="ner_bar_plot.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Bar Plot (SVG)",
                    data=bar_svg,
                    file_name="ner_bar_plot.svg",
                    mime="image/svg+xml"
                )
            else:
                st.warning("Failed to generate bar plot.")
        else:
            st.info("Run analysis to generate visualizations.")

    with tab5:
        st.subheader("Entity Probabilities")
        st.markdown("Estimated probabilities for each spaCy-detected entity, with bar and pie plots and mathematical explanation.")
        
        if "entity_probs" not in st.session_state or not st.session_state.entity_probs:
            if st.session_state.spacy_entities:
                st.warning("Entities detected but probability estimation failed. Check the 'Logs' tab for errors.")
            else:
                st.warning("No entities detected or analysis not run. Please click 'Analyze Sentence'.")
        else:
            prob_data = []
            for entity_text, probs in st.session_state.entity_probs.items():
                row = {"Entity": entity_text}
                row.update({label: f"{prob:.3f}" for label, prob in probs.items()})
                prob_data.append(row)
            prob_df = pd.DataFrame(prob_data)
            st.write("**Probability Table**")
            st.dataframe(prob_df)
            
            for entity_text, probs in st.session_state.entity_probs.items():
                st.write(f"**Probability Plots for '{entity_text}'**")
                
                # Bar Plot
                st.write("Bar Plot")
                prob_png, prob_svg = create_entity_prob_plot(probs, entity_text)
                if prob_png.getvalue():
                    st.image(prob_png, caption=f"Estimated Probabilities for '{entity_text}' (Bar)", use_column_width=True)
                    st.download_button(
                        label=f"Download Bar Plot for '{entity_text}' (PNG)",
                        data=prob_png,
                        file_name=f"prob_bar_plot_{entity_text.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
                    st.download_button(
                        label=f"Download Bar Plot for '{entity_text}' (SVG)",
                        data=prob_svg,
                        file_name=f"prob_bar_plot_{entity_text.lower().replace(' ', '_')}.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning(f"Failed to generate bar plot for '{entity_text}'. Check the 'Logs' tab.")
                
                # Pie Plot
                st.write("Pie Plot")
                pie_png, pie_svg = create_entity_prob_pie_plot(probs, entity_text)
                if pie_png.getvalue():
                    st.image(pie_png, caption=f"Estimated Probabilities for '{entity_text}' (Pie)", use_column_width=True)
                    st.download_button(
                        label=f"Download Pie Plot for '{entity_text}' (PNG)",
                        data=pie_png,
                        file_name=f"prob_pie_plot_{entity_text.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
                    st.download_button(
                        label=f"Download Pie Plot for '{entity_text}' (SVG)",
                        data=pie_svg,
                        file_name=f"prob_pie_plot_{entity_text.lower().replace(' ', '_')}.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.warning(f"Failed to generate pie plot for '{entity_text}'. Check the 'Logs' tab.")
            
            with st.expander("Mathematical Formulation of spaCy NER Probability"):
                st.markdown(generate_math_explanation(st.session_state.entity_probs))

    with tab6:
        st.subheader("Processing Logs")
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=300)

if __name__ == "__main__":
    main()
