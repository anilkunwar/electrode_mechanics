# streamlit_scibert_score.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="SciBERT Relevance Explorer", layout="wide")

# Load model with cache
@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_scibert()
target_tokens = {"strain", "energy", "anode"}

# Available colormaps for matplotlib
available_cmaps = sorted([m for m in plt.colormaps() if not m.endswith("_r")])

# Compute attention matrix and token list
def get_attention_matrix(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions  # list of 12 layers

    tokens = inputs["input_ids"][0]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Last layer, averaged over all heads
    attn_last_layer = attentions[-1][0].mean(dim=0)  # [seq_len, seq_len]
    attn_matrix = attn_last_layer.cpu().numpy()
    return attn_matrix, token_strs

# Extract target token scores
def get_attention_scores(attn_matrix, token_strs):
    token_scores = {}
    for i, tok in enumerate(token_strs):
        if tok in target_tokens:
            attn_score = attn_matrix[:, i].sum()
            token_scores[tok] = attn_score
    return token_scores

# Relevance computation
def compute_relevance(token_scores):
    if not token_scores:
        return 0.0
    avg_score = np.mean(list(token_scores.values()))
    num_tokens = len(token_scores)
    relevance = min(0.5 + 0.4 * num_tokens * avg_score, 1.0)
    return relevance

# Matplotlib plot
def plot_matplotlib_heatmap(attn_matrix, token_strs, cmap, fontsize):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn_matrix[:len(token_strs), :len(token_strs)],
        xticklabels=token_strs,
        yticklabels=token_strs,
        cmap=cmap,
        cbar_kws={"label": "Attention Weight"},
        square=True,
        ax=ax
    )
    ax.set_title("Token-to-Token Attention (Matplotlib)", fontsize=fontsize + 4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("Key Tokens", fontsize=fontsize)
    ax.set_ylabel("Query Tokens", fontsize=fontsize)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Attention Weight", fontsize=fontsize)
    fig.tight_layout()
    return fig

# --- Streamlit UI ---
st.title("🔬 SciBERT Attention & Relevance Explorer")

text = st.text_area("Enter abstract, sentence, or phrase:",
                    "Elastic strain energy in lithium-ion battery anodes.")

if st.button("Compute Attention and Relevance"):
    try:
        attn_matrix, token_strs = get_attention_matrix(text)
        token_scores = get_attention_scores(attn_matrix, token_strs)
        relevance = compute_relevance(token_scores)

        # Cache result in session_state
        st.session_state["attn_matrix"] = attn_matrix
        st.session_state["token_strs"] = token_strs
        st.session_state["token_scores"] = token_scores
        st.session_state["relevance"] = relevance

    except Exception as e:
        st.error(f"Error during SciBERT inference: {e}")

# If available, render results
if "attn_matrix" in st.session_state:
    attn_matrix = st.session_state["attn_matrix"]
    token_strs = st.session_state["token_strs"]
    token_scores = st.session_state["token_scores"]
    relevance = st.session_state["relevance"]

    st.subheader("📌 Token Attention Scores")
    st.json(token_scores)

    st.subheader("📈 Relevance Score")
    st.metric("Relevance Probability", f"{relevance:.3f}")

    st.subheader("🌐 Attention Heatmap (Plotly)")
    fig_plotly = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=token_strs,
        y=token_strs,
        colorscale="Viridis",
        colorbar=dict(title="Attention Weight", tickfont=dict(size=14))
    ))
    fig_plotly.update_layout(
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        title="Token-to-Token Attention (Plotly)",
        font=dict(size=14)
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- Matplotlib UI options ---
    st.subheader("🎨 Customize Matplotlib Heatmap")
    fontsize = st.slider("Font Size for Labels", 10, 40, 20)
    cmap = st.selectbox("Matplotlib Colormap", available_cmaps)

    # Matplotlib plot
    st.subheader("🖼️ Attention Heatmap (Matplotlib)")
    fig_matplotlib = plot_matplotlib_heatmap(attn_matrix, token_strs, cmap, fontsize)
    st.pyplot(fig_matplotlib)

    # Download PNG
    buf = BytesIO()
    fig_matplotlib.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="📥 Download Matplotlib Heatmap (PNG)",
        data=buf.getvalue(),
        file_name="attention_heatmap.png",
        mime="image/png"
    )
