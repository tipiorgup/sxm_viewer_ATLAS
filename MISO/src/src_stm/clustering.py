import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

def cluster_with_gmm(df, exclude_features=None, variance_threshold=0.01, 
                     max_clusters=5, use_pca=True, n_pca_components=None):
    """
    Perform Gaussian Mixture Model clustering on DataFrame features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature DataFrame with samples as rows
    exclude_features : list
        List of feature names to exclude from clustering
    variance_threshold : float
        Minimum variance threshold for feature selection
    max_clusters : int
        Maximum number of clusters to test (default: min(10, n_samples//2))
    use_pca : bool
        Whether to apply PCA before clustering
    n_pca_components : int
        Number of PCA components (default: min(n_samples-1, n_features))
    
    Returns:
    --------
    dict containing clustering results and analysis
    """
    
    # 1. Feature Selection and Preprocessing
    # Remove specified features
    if exclude_features is None:
        exclude_features = []
    
    # Automatically exclude constant or near-constant features
    constant_features = []
    for col in df.columns:
        if col not in exclude_features:
            if df[col].nunique() <= 1 or df[col].std() < 1e-6:
                constant_features.append(col)
    
    if constant_features:
        print(f"Excluding constant features: {constant_features}")
        exclude_features.extend(constant_features)
    
    # Select features for clustering
    feature_cols = [col for col in df.columns if col not in exclude_features]
    X = df[feature_cols].copy()
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Using features: {feature_cols}")
    
    # Handle missing values
    if X.isnull().any().any():
        print("Warning: Found missing values, filling with median")
        X = X.fillna(X.median())
    
    # 2. Variance-based feature selection
    if variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        X_selected = selector.fit_transform(X)
        selected_features = np.array(feature_cols)[selector.get_support()]
        print(f"After variance filtering ({variance_threshold}): {X_selected.shape[1]} features")
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    # 3. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 4. PCA (optional)
    if use_pca and X_scaled.shape[1] > 3:
        if n_pca_components is None:
            n_pca_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1], 15)
        
        pca = PCA(n_components=n_pca_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components")
        print(f"Explained variance: {pca.explained_variance_ratio_[:5]}")
        print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)[:5]}")
        
        # Use PCA features for clustering
        clustering_features = X_pca
        feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    else:
        pca = None
        clustering_features = X_scaled
        feature_names = X.columns.tolist()
    
    # 5. Determine cluster range
    n_samples = clustering_features.shape[0]
    if max_clusters is None:
        max_clusters = min(8, n_samples - 1)  # Conservative upper limit
    
    cluster_range = range(2, max_clusters + 1)
    print(f"Testing {min(cluster_range)} to {max(cluster_range)} clusters")
    
    # 6. Model Selection with AIC/BIC
    models = []
    aics = []
    bics = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, 
                                covariance_type='full', max_iter=100)
            gmm.fit(clustering_features)
            
            models.append(gmm)
            aics.append(gmm.aic(clustering_features))
            bics.append(gmm.bic(clustering_features))
            
            # Calculate silhouette score
            labels = gmm.predict(clustering_features)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(clustering_features, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)
                
        except Exception as e:
            print(f"Failed to fit {n_clusters} clusters: {e}")
            models.append(None)
            aics.append(np.inf)
            bics.append(np.inf)
            silhouette_scores.append(-1)
    

    optimal_n_sil = cluster_range[np.argmax(silhouette_scores)]

    optimal_n = optimal_n_sil
    final_gmm = GaussianMixture(n_components=optimal_n, random_state=42, covariance_type='full')
    cluster_labels = final_gmm.fit_predict(clustering_features)
    cluster_probs = final_gmm.predict_proba(clustering_features)
    
    if len(np.unique(cluster_labels)) > 1:
        final_silhouette = silhouette_score(clustering_features, cluster_labels)
        print(f"Final silhouette score: {final_silhouette:.3f}")
    
    results_df = df.copy()
    results_df['cluster'] = cluster_labels
    results_df['cluster_probability'] = np.max(cluster_probs, axis=1)

    return {
        'results_df': results_df,
        'cluster_labels': cluster_labels,
        'cluster_probabilities': cluster_probs,
        'gmm_model': final_gmm,
        'pca_model': pca,
        'scaler': scaler,
        'selected_features': X.columns.tolist(),
        'clustering_features': clustering_features,
        'optimal_clusters': optimal_n,
        'model_selection': {
            'cluster_range': list(cluster_range),
            'aics': aics,
            'bics': bics,
            'silhouette_scores': silhouette_scores
        }
    }

import matplotlib.pyplot as plt
import numpy as np

def visualize_clusters(features, labels, feature_names=None, sample_names=None):
    """
    Clean visualization of clustering results
    
    Parameters:
    -----------
    features : array-like
        Feature matrix (n_samples, n_features)
    labels : array-like
        Cluster labels for each sample
    feature_names : list, optional
        Names of features for axis labels
    sample_names : list, optional
        Names of samples for point labels
    """
    
    n_clusters = len(np.unique(labels))
    
    # Use default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
    
    # Create subplots based on feature dimensions
    if features.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 8))
        
        # 2D plot
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(features[:, 0], features[:, 1], c=labels, 
                             cmap='tab10', s=100, alpha=0.7)
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_title('2D Cluster View')
        
        # Add sample labels if provided
        if sample_names is not None:
            for i, name in enumerate(sample_names):
                ax1.annotate(name, (features[i, 0], features[i, 1]), 
                           xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # 3D plot
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(features[:, 0], features[:, 1], features[:, 2], 
                   c=labels, cmap='tab10', s=60)
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_zlabel(feature_names[2])
        ax2.set_title('3D Cluster View')
        
        # Cluster sizes
        ax3 = fig.add_subplot(2, 2, 3)
        
    else:
        # Only 2D available
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D plot
        scatter = axes[0].scatter(features[:, 0], features[:, 1], c=labels, 
                                 cmap='tab10', s=100, alpha=0.7)
        axes[0].set_xlabel(feature_names[0])
        axes[0].set_ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2')
        axes[0].set_title('Cluster Visualization')
        
        # Add sample labels if provided
        if sample_names is not None:
            for i, name in enumerate(sample_names):
                axes[0].annotate(name, (features[i, 0], features[i, 1]), 
                               xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # Cluster sizes
        ax3 = axes[1]
    
    # Cluster size bar chart
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    ax3.bar(unique_labels, counts, color=[colors[i] for i in unique_labels])
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Cluster Sizes')
    ax3.set_xticks(unique_labels)
    
    plt.tight_layout()
    plt.show()
    

def analyze_clusters_df(results_df, feature_columns):
    """Analyze cluster characteristics"""
    
    print(f"\n=== Cluster Analysis ===")
    
    # Group by cluster and calculate statistics
    cluster_stats = results_df.groupby('cluster')[feature_columns].agg(['mean', 'std'])
    
    print("Cluster means:")
    print(cluster_stats.xs('mean', axis=1, level=1).round(3))
    
    # Show which samples are in each cluster
    print(f"\nSample assignments:")
    for cluster_id in sorted(results_df['cluster'].unique()):
        samples = results_df[results_df['cluster'] == cluster_id].index.tolist()
        probs = results_df[results_df['cluster'] == cluster_id]['cluster_probability'].values
        print(f"Cluster {cluster_id}: {samples}")
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs]}")