import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="AI-Powered Content Recommendation with SHAP", layout="wide")
st.title("🤖 AI-Powered Content Analysis & Recommendation System with Explainability")
st.markdown("Upload your embeddings CSV, apply PCA, get recommendations, and explain them with SHAP!")

    # Upload embeddings
    # Load embeddings directly from the bundled CSV
    embeddings = pd.read_csv("embeddings.csv", header=None)
    st.write("### Raw Embeddings Data", embeddings)

    st.write("### Raw Embeddings Data", embeddings)

    # PCA Components selector
    n_components = st.slider("Select number of PCA components", min_value=2, max_value=min(embeddings.shape[1], 10), value=3)

    # Scale + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

    st.write("### PCA Reduced Data", pca_df)

    # User input
    st.subheader("🔍 Find Similar Content")
    user_input = st.text_area("Enter your input vector (comma-separated numbers):", "0.1, 0.2, -0.3")

    try:
        user_vector = np.array([float(i) for i in user_input.split(",")]).reshape(1, -1)
        user_vector_scaled = scaler.transform(user_vector)
        user_vector_pca = pca.transform(user_vector_scaled)

        # Cosine similarity
        similarities = cosine_similarity(user_vector_pca, X_pca).flatten()
        top_n = st.slider("Select number of recommendations", 1, 20, 5)
        top_indices = similarities.argsort()[-top_n:][::-1]

        st.write(f"### 🏆 Top {top_n} Recommendations")
        recs = embeddings.iloc[top_indices]
        st.write(recs)

        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.3, label='Content')
        ax.scatter(pca_df.iloc[top_indices]['PC1'], pca_df.iloc[top_indices]['PC2'], color='red', label='Recommended', s=100)
        ax.scatter(user_vector_pca[:, 0], user_vector_pca[:, 1], color='green', label='Your Input', s=100, marker='X')
        ax.legend()
        ax.set_title('PCA Visualization with Recommendations')
        st.pyplot(fig)

        # SHAP Explainability
        st.subheader("🔍 SHAP Explainability of Recommendations")

        # Define a simple model: cosine similarity function
        def similarity_model(X):
            return cosine_similarity(X, user_vector_pca)

        # Initialize SHAP explainer
        explainer = shap.Explainer(similarity_model, X_pca)
        shap_values = explainer(X_pca)

        # Visualize SHAP for top recommendation
        st.write("### 🔬 SHAP Explanation for Top Recommendation")
        shap_matplotlib = shap.plots.bar(shap_values[top_indices[0]], show=False)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

        # Optional: Waterfall plot for deeper explanation
        st.write("### 💧 Detailed Waterfall Plot")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[top_indices[0]], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing input: {e}")

    # Download PCA CSV
    st.subheader("⬇️ Download PCA Data")
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(pca_df)
    st.download_button(
        label="Download PCA CSV",
        data=csv,
        file_name='embeddings_pca.csv',
        mime='text/csv',
    )

else:
    st.warning("👆 Please upload your embeddings CSV to proceed.")

# Footer
st.markdown("---")
st.markdown("Built for **Codrelate 2025** 🚀 | By Team Monish")
