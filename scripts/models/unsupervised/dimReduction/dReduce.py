import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pacmap



def graphPCA(X,n,Y):

    pca = PCA(n_components=n)  # Reduce to 2D
    X_pca = pca.fit_transform(X)


    # Plot
    plt.figure(figsize=(8,6))
    ax=sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y.ravel(), palette="tab10")
    sns.move_legend(ax, "upper right", fontsize=4, markerscale=0.35)

    plt.title("PCA Projection")
    plt.tight_layout()
    plt.show()


def graphtSNE(X,z):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    ax=sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=z.ravel(), palette="tab10")
    sns.move_legend(ax, "upper right", fontsize=4, markerscale=0.35)
    
    
    plt.title("t-SNE Projection")
    plt.tight_layout()
    plt.show()

def graphUMAP(X,z):
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.05, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    ax=sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=z.ravel(), palette="tab10")
    sns.move_legend(ax, "upper right", fontsize=4, markerscale=0.35)
    plt.title("UMAP Projection")
    plt.show()


def graphPaCMAP(X,z):
    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=15, MN_ratio=0.6, FP_ratio=1.5, random_state=42)
    X_pacmap = pacmap_reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    ax=sns.scatterplot(x=X_pacmap[:, 0], y=X_pacmap[:, 1], hue=z.ravel(), palette="tab10")
    sns.move_legend(ax, "upper right", fontsize=4, markerscale=0.35)
    plt.title("PaCMAP Projection")
    plt.show()
