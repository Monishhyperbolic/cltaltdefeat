import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Article Recommendation System", layout="wide")
st.title("📰 Smart Article Recommender")
st.markdown("Enter text or tags, and get top recommended articles!")

# === CONFIG ===
# ✅ Your actual Google Drive file IDs:
ARTICLES_CSV_ID = '1raHQ1RYkCbhlzQSUuhuBq617DxDesA1m'        # data_cleaned3.csv
EMBEDDINGS_CSV_ID = '1sWffA9H9n1tFoZT3zh0TBTtPlqvGjYH4'      # pca_3d_embeddings.csv

# === HELPER FUNCTIONS ===

def load_csv_from_gdrive(file_id, output_filename):
    url = f'https://drive.google.com/uc?id={17we77-DBgOd4_MzsUV_pVd6SJ1CSjC1e}'
    gdown.download(url, output_filename, quiet=False)
    return pd.read_csv(output_filename)

@st.cache_data
def load_data():
    df = load_csv_from_gdrive(ARTICLES_CSV_ID, 'data_cleaned3.csv')
    embeddings_df = load_csv_from_gdrive(EMBEDDINGS_CSV_ID, 'pca_3d_embeddings.csv')

    # Validate embeddings
    embeddings = embeddings_df.values
    if not np.issubdtype(embeddings.dtype, np.number):
        st.error("🚨 Embeddings file is invalid. Please upload a valid numeric embeddings CSV.")
        st.stop()

    return df, embeddings

df, embeddings = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# === USER INPUT ===
st.subheader("Enter your search query")
user_input = st.text_input("Enter keywords, tags, or text:", placeholder="e.g., AI climate research")

if user_input:
    # Embed user input
    user_embedding = model.encode([user_input])

    # Reduce user embedding to match embeddings shape
    user_embedding_reduced = user_embedding[:, :embeddings.shape[1]]

    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding_reduced, embeddings).flatten()

    # Get top N recommendations
    top_n = 5
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Display recommendations
    st.subheader(f"🔍 Top {top_n} Recommendations")
    for idx in top_indices:
        article = df.iloc[idx]
        st.markdown(f"### {article['title']}")
        st.markdown(f"**Authors:** {article['authors']}")
        st.markdown(f"**Date:** {article['timestamp']}")
        st.markdown(f"**Tags:** {article['tags']}")
        st.markdown(f"{article['text'][:300]}...")  # show snippet
        st.markdown("---")

    # Optional: Similarity scores bar chart
    fig, ax = plt.subplots()
    ax.bar(range(top_n), similarities[top_indices], tick_label=[f"Doc {i+1}" for i in range(top_n)])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Top Similarity Scores")
    st.pyplot(fig)

else:
    st.info("Please enter a query to get recommendations.")

# === FOOTER ===
st.markdown("---")
st.markdown("Built with ❤️ by Team clt+alt+defeat")
