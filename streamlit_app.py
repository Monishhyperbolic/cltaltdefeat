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

# Helper function to load CSV from Google Drive
def load_csv_from_gdrive(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Load data
@st.cache_data
def load_data():
    # ✅ Replace with your actual file IDs
    csv_file_id = '1raHQ1RYkCbhlzQSUuhuBq617DxDesA1m'  # Articles CSV
    embeddings_file_id = '1raHQ1RYkCbhlzQSUuhuBq617DxDesA1m'  # <- Add your ID!

    df = load_csv_from_gdrive(csv_file_id, 'articles.csv')
    embeddings = load_csv_from_gdrive(embeddings_file_id, 'embeddings.csv').values
    return df, embeddings

df, embeddings = load_data()

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# User input
st.subheader("Enter your search query")
user_input = st.text_input("Enter keywords, tags, or text:", placeholder="e.g., AI climate research")

if user_input:
    # Embed user input
    user_embedding = model.encode([user_input])

    # Reduce user embedding to match PCA shape
    user_embedding_reduced = user_embedding[:, :embeddings.shape[1]]

    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding_reduced, embeddings).flatten()

    # Get top 5 indices
    top_n = 5
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Show results
    st.subheader(f"🔍 Top {top_n} Recommendations")
    for idx in top_indices:
        article = df.iloc[idx]
        st.markdown(f"### {article['title']}")
        st.markdown(f"**Authors:** {article['authors']}")
        st.markdown(f"**Date:** {article['timestamp']}")
        st.markdown(f"**Tags:** {article['tags']}")
        st.markdown(f"{article['text'][:300]}...")  # show snippet
        st.markdown("---")

    # Optional: Visualize similarity scores
    fig, ax = plt.subplots()
    ax.bar(range(top_n), similarities[top_indices], tick_label=[f"Doc {i+1}" for i in range(top_n)])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Top Similarity Scores")
    st.pyplot(fig)

else:
    st.info("Please enter a query to get recommendations.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Team clt+alt+defeat")
