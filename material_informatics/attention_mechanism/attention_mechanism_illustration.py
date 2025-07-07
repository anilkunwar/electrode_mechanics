# streamlit_scibert_score.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

st.set_page_config(page_title="SciBERT Relevance Scorer")

# Load SciBERT model
@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_scibert()
target_tokens = {"strain", "energy", "anode"}

def get_attention_scores(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions  # tuple of 12 layers

    tokens = inputs["input_ids"][0]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Use last layer, average over heads
    attn_last_layer = attentions[-1][0].mean(dim=0)  # shape: [seq_len, seq_len]
    
    token_scores = {}
    for i, tok in enumerate(token_strs):
        if tok in target_tokens:
            attn_score = attn_last_layer[:, i].sum().item()
            token_scores[tok] = attn_score

    return token_scores

def compute_relevance(token_scores):
    if not token_scores:
        return 0.0
    avg_score = np.mean(list(token_scores.values()))
    num_tokens = len(token_scores)
    relevance = min(0.5 + 0.4 * num_tokens * avg_score, 1.0)
    return relevance

# Streamlit UI
st.title("🔬 SciBERT Relevance Scorer")
text = st.text_area("Enter Abstract or Sentence or Phrase", 
                    "Elastic strain energy in lithium-ion battery anodes.")

if st.button("Compute Relevance Score"):
    try:
        token_scores = get_attention_scores(text)
        relevance = compute_relevance(token_scores)

        st.subheader("📌 Token Attention Scores")
        st.json(token_scores)

        st.subheader("📈 Relevance Score")
        st.metric("Relevance Probability", f"{relevance:.3f}")

    except Exception as e:
        st.error(f"Error during SciBERT inference: {e}")

