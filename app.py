import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# Streamlit UI for model selection
st.title("Clustering Visualization : ARIMA vs LSTM")

model_choice = st.selectbox('Pilih Model Prediksi:', ['KMEANS', 'HAC'])

if model_choice == 'HAC':
    st.subheader("HAC Model")
    
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

elif model_choice == 'KMEANS':
    st.subheader("KMEANS Model")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=';')
    
        st.write("Dataset Information:")
        st.write(data.info())
    
    # Selecting relevant features
    features = data[['Price', 'Number Sold', 'Total Review']]
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(features.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8}, ax=ax)
    st.pyplot(fig)
    
    # Handling missing values
    features = features.dropna()
    
    # Standardizing the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    
    # Elbow Method
    st.subheader("Elbow Method")
    wcss = []
    range_n_clusters = range(1, 11)
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range_n_clusters, wcss, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
    st.pyplot(fig)
    
    # Applying KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Silhouette Score
    silhouette_avg = silhouette_score(data_scaled, clusters)
    st.write(f"Silhouette Score for 4 Clusters: {silhouette_avg}")
    
    # Assign clusters to data
    data['Cluster'] = kmeans.labels_
    
    # PCA Visualization
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data_scaled)
    pca_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = data['Cluster']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='coolwarm', s=100, edgecolor='k', ax=ax)
    ax.set_title("K-means Clustering with PCA Reduction")
    st.pyplot(fig)
    
    # Scatter plots
    st.subheader("Number Sold vs Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Price', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
    ax.set_title("K-means Clustering: Number Sold vs Price")
    st.pyplot(fig)
    
    st.subheader("Number Sold vs Total Review")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total Review', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
    ax.set_title("K-means Clustering: Number Sold vs Total Review")
    st.pyplot(fig)
    
    st.subheader("Price vs Total Review")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total Review', y='Price', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
    ax.set_title("K-means Clustering: Price vs Total Review")
    st.pyplot(fig)

