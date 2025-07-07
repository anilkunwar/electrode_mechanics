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
def get_attention_matrix(text, max_tokens=None):
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

    # Trim padding tokens
    pad_token_id = tokenizer.pad_token_id
    seq_len = (tokens != pad_token_id).sum().item()
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
    token_strs = token_strs[:seq_len]
    attn_matrix = attn_last_layer[:seq_len, :seq_len].cpu().numpy()

    return attn_matrix, token_strs, seq_len

# Extract target token scores
def get_attention_scores(attn_matrix, token_strs):
    token_scores = {}
    for i, tok in enumerate(token_strs):
        if tok in target_tokens:
            attn_score = attn_matrix[:, i].sum()
            token_scores[tok] = float(attn_score)
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
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        attn_matrix,
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

# Plotly plot with enhanced visualization
def plot_plotly_heatmap(attn_matrix, token_strs, colorscale, show_target_annotations):
    # Highlight target tokens in labels
    x_labels = [f"<b>{tok}</b>" if tok in target_tokens else tok for tok in token_strs]
    y_labels = [f"<b>{tok}</b>" if tok in target_tokens else tok for tok in token_strs]

    # Create hover text with attention weights
    hover_text = [[f"Query: {token_strs[i]}<br>Key: {token_strs[j]}<br>Attention: {attn_matrix[i, j]:.3f}"
                   for j in range(len(token_strs))] for i in range(len(token_strs))]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=x_labels,
        y=y_labels,
        text=hover_text,
        hoverinfo="text",
        colorscale=colorscale,
        colorbar=dict(
            title="Attention Weight",
            tickfont=dict(size=14),
            titleside="right"
        ),
        zmin=0,
        zmax=1
    ))

    # Add annotations for target tokens if enabled
    if show_target_annotations:
        annotations = []
        for i, tok in enumerate(token_strs):
            if tok in target_tokens:
                annotations.append(dict(
                    x=i,
                    y=i,
                    text=f"{tok}\n({attn_matrix[:, i].sum():.2f})",
                    showarrow=False,
                    font=dict(size=12, color="red"),
                    xanchor="center",
                    yanchor="middle"
                ))
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        title="Token-to-Token Attention (Plotly)",
        font=dict(size=14),
        xaxis=dict(tickangle=-45, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        height=600,
        width=600,  # Ensure square aspect ratio
        margin=dict(l=50, r=50, t=100, b=100),
        showlegend=False
    )
    return fig

# --- Streamlit UI ---
st.title("🔬 SciBERT Attention & Relevance Explorer")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter abstract, sentence, or phrase:",
                        "Elastic strain energy in lithium-ion battery anodes.")
with col2:
    max_tokens = st.slider("Max Tokens to Display", 5, 50, 20, help="Limit the number of tokens in the heatmap for clarity.")
    show_target_annotations = st.checkbox("Highlight Target Tokens", value=True)

if st.button("Compute Attention and Relevance"):
    try:
        attn_matrix, token_strs, seq_len = get_attention_matrix(text, max_tokens=max_tokens)
        token_scores = get_attention_scores(attn_matrix, token_strs)
        relevance = compute_relevance(token_scores)

        # Cache result in session_state
        st.session_state["attn_matrix"] = attn_matrix
        st.session_state["token_strs"] = token_strs
        st.session_state["token_scores"] = token_scores
        st.session_state["relevance"] = relevance
        st.session_state["seq_len"] = seq_len

    except Exception as e:
        st.error(f"Error during SciBERT inference: {e}")

# If available, render results
if "attn_matrix" in st.session_state:
    attn_matrix = st.session_state["attn_matrix"]
    token_strs = st.session_state["token_strs"]
    token_scores = st.session_state["token_scores"]
    relevance = st.session_state["relevance"]
    seq_len = st.session_state["seq_len"]

    st.subheader("📌 Token Attention Scores")
    st.json(token_scores)

    st.subheader("📈 Relevance Score")
    st.metric("Relevance Probability", f"{relevance:.3f}")

    # --- Plotly Heatmap ---
    st.subheader("🌐 Attention Heatmap (Plotly)")
    colorscale = st.selectbox("Plotly Colorscale", ["Viridis", "Plasma", "Inferno", "Magma", "Hot"], index=0)
    fig_plotly = plot_plotly_heatmap(attn_matrix, token_strs, colorscale, show_target_annotations)
    st.plotly_chart(fig_plotly, use_container_width=True)

    # Download Plotly heatmap as PNG
    buf_plotly = BytesIO()
    fig_plotly.write_image(buf_plotly, format="png", scale=2)
    st.download_button(
        label="📥 Download Plotly Heatmap (PNG)",
        data=buf_plotly.getvalue(),
        file_name="plotly_attention_heatmap.png",
        mime="image/png"
    )

    # --- Matplotlib UI options ---
    st.subheader("🎨 Customize Matplotlib Heatmap")
    fontsize = st.slider("Font Size for Labels", 10, 40, 20)
    cmap = st.selectbox("Matplotlib Colormap", available_cmaps)

    # Matplotlib plot
    st.subheader("🖼️ Attention Heatmap (Matplotlib)")
    fig_matplotlib = plot_matplotlib_heatmap(attn_matrix, token_strs, cmap, fontsize)
    st.pyplot(fig_matplotlib)

    # Download Matplotlib heatmap
    buf_matplotlib = BytesIO()
    fig_matplotlib.savefig(buf_matplotlib, format="png", dpi=300)
    st.download_button(
        label="📥 Download Matplotlib Heatmap (PNG)",
        data=buf_matplotlib.getvalue(),
        file_name="matplotlib_attention_heatmap.png",
        mime="image/png"
    )
