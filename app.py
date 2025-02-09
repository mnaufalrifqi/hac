import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Title of the app
st.title("Agglomerative Clustering Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=';')
    
    st.write("Dataset Information:")
    st.write(data.info())
    
    data_features = data[['Price', 'Number Sold', 'Total Review']].fillna(data[['Price', 'Number Sold', 'Total Review']].median())
    
    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_features)
    
    # Applying Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=4, linkage='single')
    clusters = clustering.fit_predict(data_features)
    data['Cluster'] = clusters
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data_features.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Elbow Method for HAC
    st.subheader("Elbow Method for HAC")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        hac = AgglomerativeClustering(n_clusters=k, linkage='single')
        hac.fit(data_features)
        linkage_matrix_k = linkage(data_features, method='single')
        inertia.append(sum(linkage_matrix_k[:, 2][-k:]))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertia, marker='o')
    ax.set_title("Elbow Method for HAC")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Pseudo-Inertia")
    st.pyplot(fig)
    
    # Silhouette Score
    silhouette_avg = silhouette_score(data_features, clusters)
    st.write(f"Silhouette Score for 4 Clusters: {silhouette_avg}")
    
    # PCA Visualization
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data_features)
    pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = clusters + 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set2', s=100, edgecolor='black', ax=ax)
    ax.set_title("PCA Visualization of Clusters")
    st.pyplot(fig)
    
    # Scatter Plot Number Sold vs Price
    st.subheader("Number Sold vs Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Price', y='Number Sold', hue='Category', data=data, palette='Set2', s=100, edgecolor='black', ax=ax)
    ax.set_title("Number Sold vs Price (2D Visualization)")
    st.pyplot(fig)
    
    # Scatter Plot Number Sold vs Total Review
    st.subheader("Number Sold vs Total Review")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Total Review', y='Number Sold', hue='Category', data=data, palette='Set2', s=100, edgecolor='black', ax=ax)
    ax.set_title("Number Sold vs Total Review (2D Visualization)")
    st.pyplot(fig)
    
    # Bar Chart for Category Statistics
    st.subheader("Average Values per Category")
    avg_data = data.groupby('Category')[['Number Sold', 'Price', 'Total Review']].mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    avg_data['Number Sold'].plot(kind='bar', ax=axes[0], color='lightblue', edgecolor='black')
    axes[0].set_title('Single Number Sold by Category')
    avg_data['Price'].plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_title('Single Price by Category')
    avg_data['Total Review'].plot(kind='bar', ax=axes[2], color='salmon', edgecolor='black')
    axes[2].set_title('Single Total Review by Category')
    st.pyplot(fig)
    
    # Dendrogram
    st.subheader("Dendrogram")
    linkage_matrix = linkage(scaled_data, method='single')
    fig, ax = plt.subplots(figsize=(10, 9))
    dendrogram(linkage_matrix, color_threshold=3, ax=ax)
    ax.axhline(y=3, color='r', linestyle='--', label='Threshold = 3')
    ax.set_title("Dendrogram for Agglomerative Clustering")
    ax.set_xlabel("Index Data")
    ax.set_ylabel("Distance")
    ax.legend()
    st.pyplot(fig)
