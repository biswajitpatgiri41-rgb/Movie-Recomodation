import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
import altair as alt

# -------------------------------
# Load CSV
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "tmdb_movies1.csv")
df = pd.read_csv(csv_path)

# -------------------------------
# Detect title and text columns
# -------------------------------
title_candidates = ["title", "movie_title", "name"]
text_candidates = ["overview", "description", "plot", "summary"]

title_col = next((col for col in title_candidates if col in df.columns), df.columns[0])
text_col = next((col for col in text_candidates if col in df.columns), df.select_dtypes(include="object").columns[1])

# -------------------------------
# Prepare dataframe
# -------------------------------
df = df[[title_col, text_col]].dropna()
df[text_col] = df[text_col].astype(str).str.lower()

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df[text_col])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in df[title_col].values:
        return []
    idx = df[df[title_col] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return df[title_col].iloc[top_indices].tolist()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="🎬 Advanced Movie Recommender", layout="wide")
st.title("🎥 Futuristic Movie Recommendation Dashboard")

# Movie Selection
movie_list = df[title_col].sort_values().tolist()
selected_movie = st.selectbox("Select a movie to get recommendations", movie_list)

num_recs = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie, num_recs)
    if not recommendations:
        st.warning("Movie not found!")
    else:
        st.success(f"Top {num_recs} movies similar to '{selected_movie}':")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")

# -------------------------------
# Dashboard Graphs
# -------------------------------
st.markdown("---")
st.header("📊 Movies Overview")

# Movie description length
df['desc_length'] = df[text_col].apply(len)
fig1 = px.histogram(df, x='desc_length', nbins=50, title="Distribution of Movie Description Length")
st.plotly_chart(fig1, use_container_width=True)

# Top 20 movies with longest descriptions
top_desc = df.nlargest(20, 'desc_length')
fig2 = alt.Chart(top_desc).mark_bar().encode(
    x=alt.X('desc_length', title='Description Length'),
    y=alt.Y(f'{title_col}:N', sort='-x', title='Movie Title'),
    tooltip=[title_col, 'desc_length']
).properties(title="Top 20 Movies with Longest Descriptions", height=500)
st.altair_chart(fig2, use_container_width=True)

# Word cloud for movie descriptions (optional, more advanced)
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    text = " ".join(df[text_col])
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2').generate(text)
    st.subheader("🌐 Word Cloud of Movie Descriptions")
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
except:
    st.info("Install 'wordcloud' and 'matplotlib' for Word Cloud visualization.")