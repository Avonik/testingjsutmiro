import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="Spotify Clustering Dashboard")



@app.cell(hide_code=True)
async def _():
    import sys
    import marimo as mo

    # Installation for WASM/GitHub Pages
    if "pyodide" in sys.modules:
        import micropip
        await micropip.install(["polars", "pyarrow", "pandas", "altair"])

    # Core imports
    import polars as pl
    import altair as alt
    from sklearn.cluster import KMeans
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.cm as cm
    from sklearn.ensemble import RandomForestClassifier
    from math import pi

    alt.data_transformers.disable_max_rows()
    return (
        KMeans,
        PCA,
        RandomForestClassifier,
        StandardScaler,
        alt,
        cm,
        cs,
        dendrogram,
        linkage,
        mo,
        np,
        pd,
        pi,
        pl,
        plt,
        silhouette_samples,
        silhouette_score,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    """Dashboard Header"""
    mo.callout(mo.md("""
    # 🎵 Spotify Song Clustering Dashboard

    **Explore 5,000 songs through the lens of machine learning**

    This interactive dashboard uses K-Means clustering to group Spotify songs based on their audio features 
    (energy, danceability, tempo, etc.). Adjust the number of clusters using the sidebar and explore how 
    different groupings reveal patterns in music.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    """Sidebar Controls"""
    cluster_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label="Number of Clusters (k)"
    )

    mo.sidebar([
        mo.md("# ⚙️ Settings"),
        mo.md("**Clustering Parameters**"),
        cluster_slider,
        mo.md("---"),
        mo.callout(mo.md("""
        **Tips:**
        - Start with k=3 for clear patterns
        - Higher k = more granular groups
        - Watch the quality metrics below
        """), kind="info")
    ])
    return (cluster_slider,)


@app.cell(hide_code=True)
def _(pl):
    """Data Loading"""
    csv_url = "https://raw.githubusercontent.com/datagus/ASDA2025/refs/heads/main/datasets/homework_week11/6.3.3_spotify_5000_songs.csv"

    data = pl.read_csv(csv_url, null_values=["NA"])
    data = data.rename({col: col.strip() for col in data.columns})
    return (data,)


@app.cell(hide_code=True)
def _(StandardScaler, cs, data, pl):
    """Feature Scaling"""
    scaler = StandardScaler()
    df_features = data.select(cs.numeric())
    scaled_features = scaler.fit_transform(df_features)
    df_scaled = pl.DataFrame(scaled_features, schema=df_features.columns)
    return df_features, df_scaled


@app.cell(hide_code=True)
def _(mo):
    """Section 1: Finding Optimal Clusters"""
    mo.callout(mo.md("""
    ## Question 1: How Many Clusters Should We Use?

    Before clustering, we need to determine the optimal number of groups. Two methods help us decide:

    - **Elbow Method**: Look for the "elbow" where adding more clusters shows diminishing returns
    - **Silhouette Score**: Higher scores (closer to 1) indicate better-defined clusters
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(KMeans, df_scaled, mo, plt):
    """Elbow Method Visualization"""
    inertia_values = []
    K_elbow = range(2, 11)

    for k_elbow in K_elbow:
        kmeans_elbow = KMeans(n_clusters=k_elbow, random_state=42, n_init=10)
        kmeans_elbow.fit(df_scaled)
        inertia_values.append(kmeans_elbow.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4.5))
    ax_elbow.plot(K_elbow, inertia_values, 'o-', color='#1DB954', linewidth=2, markersize=8)
    ax_elbow.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax_elbow.set_ylabel('Inertia', fontsize=11)
    ax_elbow.set_title('Elbow Method: Finding the Optimal k', fontsize=13, fontweight='bold')
    ax_elbow.grid(alpha=0.3)
    ax_elbow.set_xticks(K_elbow)
    plt.tight_layout()

    mo.vstack([
        mo.callout(
            mo.md(
                "**📉 Elbow Method**: Look for the point where the curve starts to flatten (the 'elbow'). For this dataset, k=3 or k=4 often shows a clear elbow."),
            kind="info"
        ),
        mo.as_html(fig_elbow)
    ])
    return


@app.cell(hide_code=True)
def _(KMeans, df_scaled, mo, plt, silhouette_score):
    """Silhouette Analysis"""
    sil_scores_list = []
    K_sil_range = range(2, 11)

    for k_sil in K_sil_range:
        kmeans_sil = KMeans(n_clusters=k_sil, random_state=42, n_init=10)
        labels_sil = kmeans_sil.fit_predict(df_scaled)
        sil_scores_list.append(silhouette_score(df_scaled, labels_sil))

    fig_sil_scores, ax_sil_scores = plt.subplots(figsize=(10, 4.5))
    ax_sil_scores.plot(K_sil_range, sil_scores_list, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
    ax_sil_scores.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax_sil_scores.set_ylabel('Silhouette Score', fontsize=11)
    ax_sil_scores.set_title('Silhouette Score: Cluster Quality Assessment', fontsize=13, fontweight='bold')
    ax_sil_scores.grid(alpha=0.3)
    ax_sil_scores.set_xticks(K_sil_range)
    ax_sil_scores.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Good threshold (0.2)')
    ax_sil_scores.legend()
    plt.tight_layout()

    mo.vstack([
        mo.callout(
            mo.md(
                "**📊 Silhouette Score Analysis**: Higher scores indicate better-separated clusters. Scores above 0.2 are generally considered good for complex, real-world data."),
            kind="info"
        ),
        mo.as_html(fig_sil_scores)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 2 Header"""
    mo.callout(mo.md("""
    ## Question 2: What Do The Clusters Look Like?

    Now let's visualize the clusters in 2D space using Principal Component Analysis (PCA). 
    The arrows show which audio features influence each dimension.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(KMeans, cluster_slider, data, df_scaled, pl):
    """Clustering with Current Slider Value"""
    kmeans_model = KMeans(n_clusters=cluster_slider.value, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(df_scaled)

    data_with_clusters = data.with_columns(pl.Series("cluster", clusters))
    return clusters, data_with_clusters


@app.cell(hide_code=True)
def _(
    PCA,
    cluster_slider,
    data_with_clusters,
    df_features,
    df_scaled,
    mo,
    np,
    plt,
    sns,
):
    """PCA Biplot Visualization"""
    pca2 = PCA(n_components=2)
    pca_data = pca2.fit_transform(df_scaled)

    fig_pca, ax_pca = plt.subplots(figsize=(11, 6))

    sns.scatterplot(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        hue=data_with_clusters.get_column('cluster').to_numpy(),
        palette='Set2',
        alpha=0.6,
        s=30,
        ax=ax_pca
    )

    # Feature vectors (reduced for clarity)
    loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
    important_features = ['energy', 'danceability', 'acousticness', 'valence', 'loudness']

    for feat_idx, feature in enumerate(df_features.columns):
        if feature in important_features:
            arrow_scale = 4
            ax_pca.arrow(0, 0, loadings[feat_idx, 0] * arrow_scale, loadings[feat_idx, 1] * arrow_scale,
                         color='darkred', alpha=0.7, width=0.05, head_width=0.2)
            ax_pca.text(loadings[feat_idx, 0] * arrow_scale * 1.15, loadings[feat_idx, 1] * arrow_scale * 1.15,
                        feature, color='black', fontsize=10, fontweight='bold')

    ax_pca.set_title(f'Cluster Visualization (k={cluster_slider.value})', fontsize=14, fontweight='bold')
    ax_pca.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}% variance)', fontsize=11)
    ax_pca.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}% variance)', fontsize=11)
    ax_pca.grid(alpha=0.3)
    ax_pca.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    total_var = pca2.explained_variance_ratio_.sum() * 100

    mo.vstack([
        mo.callout(
            mo.md(
                f"**🗺️ PCA Cluster Map**: PCA reduces {len(df_features.columns)} dimensions to 2, capturing {total_var:.1f}% of variance. Red arrows show feature influence: songs toward 'energy' are more energetic, toward 'acousticness' are more acoustic."),
            kind="success"
        ),
        mo.as_html(fig_pca)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 3 Header"""
    mo.callout(mo.md("""
    ## Question 3: What Makes Clusters Different?

    Which audio features best separate the clusters? This analysis shows the importance of each feature.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(RandomForestClassifier, alt, cluster_slider, data_with_clusters, mo, pd):
    """Feature Importance Analysis"""
    feature_cols = ['danceability', 'energy', 'valence', 'acousticness',
                    'speechiness', 'instrumentalness', 'liveness', 'tempo']

    X_train = data_with_clusters.select(feature_cols).to_pandas()
    y_train = data_with_clusters['cluster'].to_pandas()

    rf_model = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    top_feature = importance_df.iloc[0]['Feature']
    top_score = importance_df.iloc[0]['Importance']

    importance_chart = (
        alt.Chart(importance_df)
        .mark_bar()
        .encode(
            x=alt.X('Importance:Q', title='Importance Score'),
            y=alt.Y('Feature:N', sort='-x', title=None),
            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Feature', alt.Tooltip('Importance:Q', format='.2%')]
        )
        .properties(width=600, height=280)
    )

    mo.vstack([
        mo.callout(
            mo.md(
                f"**🔑 Feature Importance Ranking**: '{top_feature}' is the most important feature ({top_score:.1%}) for distinguishing between your {cluster_slider.value} clusters."),
            kind="success"
        ),
        importance_chart
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 4 Header"""
    mo.callout(mo.md("""
    ## Question 4: How Are Features Distributed?

    Explore how individual features vary across clusters. Select a feature to compare.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    """Feature Selection Dropdown"""
    feature_dropdown = mo.ui.dropdown(
        options={
            'Energy': 'Energy',
            'Danceability': 'danceability',
            'Valence (Positivity)': 'valence',
            'Acousticness': 'acousticness',
            'Tempo': 'tempo',
            'Loudness': 'loudness'
        },
        value='Energy',
        label="Select Audio Feature:"
    )

    mo.hstack([feature_dropdown], justify="start")
    return (feature_dropdown,)


@app.cell(hide_code=True)
def _(alt, data_with_clusters, feature_dropdown, mo):
    """Box Plot Visualization"""
    selected_feature = feature_dropdown.value

    chart_box = (
        alt.Chart(data_with_clusters.to_pandas())
        .mark_boxplot(size=40, color='#1DB954')
        .encode(
            x=alt.X('cluster:O', title='Cluster'),
            y=alt.Y(f'{selected_feature}:Q', title=selected_feature.capitalize()),
            color=alt.Color('cluster:O', legend=None, scale=alt.Scale(scheme='category10'))
        )
        .properties(width=600, height=320, title=f'Distribution of {selected_feature.capitalize()} Across Clusters')
    )

    mo.vstack([
        mo.callout(
            mo.md(
                "**📦 Feature Distribution**: Each box shows the median (center line) and spread (box height) for that cluster. Compare boxes to see which clusters have distinct feature values."),
            kind="info"
        ),
        chart_box
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 5 Header"""
    mo.callout(mo.md("""
    ## Question 5: What Is Each Cluster's "Fingerprint"?

    Radar charts show the unique audio profile of each cluster across multiple features.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(data_with_clusters, mo, pi, plt):
    """Radar Chart (Cluster Fingerprints)"""
    radar_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'liveness']

    df_radar = data_with_clusters.select(radar_features + ['cluster']).to_pandas()
    cluster_means = df_radar.groupby('cluster').mean().reset_index()

    def make_spider(row_idx, title, color, ax):
        categories = list(df_radar.columns[1:])
        N_cats = len(categories)
        angles = [n / float(N_cats) * 2 * pi for n in range(N_cats)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.sca(ax)
        plt.xticks(angles[:-1], categories, size=9)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)

        values = df_radar.loc[row_idx].drop('cluster').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.5)
        ax.fill(angles, values, color=color, alpha=0.25)
        plt.title(title, size=12, color=color, weight='bold', y=1.08)

    n_clusters_radar = len(cluster_means)
    n_cols = min(n_clusters_radar, 4)
    n_rows = (n_clusters_radar + n_cols - 1) // n_cols

    fig_radar = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    colors = plt.cm.Set2(range(n_clusters_radar))

    for radar_idx in range(n_clusters_radar):
        ax_radar = plt.subplot(n_rows, n_cols, radar_idx + 1, polar=True)
        make_spider(row_idx=radar_idx, title=f"Cluster {radar_idx}", color=colors[radar_idx], ax=ax_radar)

    plt.tight_layout()

    mo.vstack([
        mo.callout(
            mo.md(
                "**🕸️ Cluster Radar Charts**: Each shape represents a cluster's average profile. Large areas in a direction indicate high values for that feature (e.g., a cluster extending toward 'energy' contains energetic songs)."),
            kind="info"
        ),
        mo.as_html(fig_radar)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 6 Header"""
    mo.callout(mo.md("""
    ## Question 6: How Good Is Our Clustering?

    The silhouette plot shows how well each song fits into its assigned cluster.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(
    cluster_slider,
    clusters,
    cm,
    df_scaled,
    mo,
    np,
    plt,
    silhouette_samples,
    silhouette_score,
):
    """Silhouette Analysis Per Sample"""
    n_clusters_val = cluster_slider.value
    silhouette_avg = silhouette_score(df_scaled, clusters)
    sample_silhouette_values = silhouette_samples(df_scaled, clusters)

    # Quality Assessment
    if silhouette_avg > 0.20:
        quality_status = "✅ Strong Separation"
        desc_text = "Clusters are well-defined with clear boundaries."
        alert_color = "success"
    elif silhouette_avg > 0.13:
        if n_clusters_val <= 5:
            quality_status = "✓ Good Balance"
            desc_text = f"For {n_clusters_val} clusters, this is solid separation without over-segmentation."
            alert_color = "info"
        else:
            quality_status = "⚠️ Over-Segmented"
            desc_text = f"With {n_clusters_val} clusters, you may have too many groups without clear gains."
            alert_color = "warn"
    else:
        quality_status = "❌ Weak Structure"
        desc_text = "Clusters overlap heavily. Consider fewer clusters."
        alert_color = "danger"

    # Plotting
    fig_sil_detail, ax_sil_detail = plt.subplots(figsize=(10, 5))
    y_low = 10

    for cluster_idx in range(n_clusters_val):
        ith_cluster_sil_values = sample_silhouette_values[clusters == cluster_idx]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_up = y_low + size_cluster_i

        color = cm.nipy_spectral(float(cluster_idx) / n_clusters_val)
        ax_sil_detail.fill_betweenx(
            np.arange(y_low, y_up), 0, ith_cluster_sil_values,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        ax_sil_detail.text(-0.05, y_low + 0.5 * size_cluster_i, str(cluster_idx), fontsize=10)
        y_low = y_up + 10

    ax_sil_detail.set_title(f'Silhouette Plot (k={n_clusters_val})', fontsize=13, fontweight='bold')
    ax_sil_detail.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax_sil_detail.set_ylabel('Cluster', fontsize=11)
    ax_sil_detail.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
                          label=f'Average: {silhouette_avg:.2f}')
    ax_sil_detail.set_yticks([])
    ax_sil_detail.legend(fontsize=10)
    plt.tight_layout()

    mo.vstack([
        mo.callout(
            mo.md(f"""
            **🎯 Silhouette Quality Check**

            Average Silhouette Score: **{silhouette_avg:.3f}**  
            Status: **{quality_status}**

            {desc_text}
            """),
            kind=alert_color
        ),
        mo.as_html(fig_sil_detail),
        mo.callout(
            mo.md(
                "**Reading the plot**: Each colored section represents a cluster. Width shows cluster size. Values near 1 indicate well-separated songs; near 0 means overlap with other clusters."),
            kind="info"
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 7 Header"""
    mo.callout(mo.md("""
    ## Question 7: How Do Clusters Relate Hierarchically?

    This dendrogram shows how clusters merge together as we reduce the number of groups.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(cluster_slider, data_with_clusters, dendrogram, linkage, mo, plt):
    """Hierarchical Clustering Dendrogram"""
    # Use sample for performance
    df_dend_sample = data_with_clusters.sample(n=1500, seed=42)

    features_dend = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
    X_dend = df_dend_sample.select(features_dend).to_numpy()

    Z_dend = linkage(X_dend, method='ward')

    # Calculate threshold
    n_k_dend = cluster_slider.value
    if n_k_dend > 1:
        dist_now = Z_dend[-(n_k_dend), 2]
        dist_prev = Z_dend[-(n_k_dend - 1), 2]
        threshold_dend = (dist_now + dist_prev) / 2
    else:
        threshold_dend = 0

    fig_dend, ax_dend = plt.subplots(figsize=(12, 5))

    dendrogram(
        Z_dend,
        truncate_mode='lastp',
        p=25,
        leaf_rotation=0,
        leaf_font_size=10,
        show_contracted=True,
        color_threshold=threshold_dend,
        ax=ax_dend
    )

    ax_dend.axhline(y=threshold_dend, c='red', ls='--', lw=2, label=f'Cut for k={n_k_dend}')
    ax_dend.set_title('Hierarchical Clustering Dendrogram', fontsize=13, fontweight='bold')
    ax_dend.set_xlabel('Cluster Index (number of songs in parentheses)', fontsize=11)
    ax_dend.set_ylabel('Distance (Ward Linkage)', fontsize=11)
    ax_dend.legend()
    plt.tight_layout()

    mo.vstack([
        mo.callout(
            mo.md(
                f"**🌳 Hierarchical Clustering View**: The red line shows where to 'cut' the tree to get {n_k_dend} clusters. Branches below the line are separate clusters; those above would merge if we reduced k."),
            kind="info"
        ),
        mo.as_html(fig_dend)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Section 8: Data Explorer"""
    mo.callout(mo.md("""
    ## Interactive Data Explorer

    Filter clusters to explore individual songs and their features.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(data_with_clusters, mo):
    """Cluster Filter UI"""
    available_clusters = sorted(data_with_clusters['cluster'].unique().to_list())
    cluster_options = [str(c) for c in available_clusters]

    cluster_filter = mo.ui.multiselect(
        options=cluster_options,
        value=cluster_options,
        label="Select Clusters to Display:"
    )

    mo.vstack([
        mo.callout(
            mo.md("**🔍 Cluster Filter**: Select one or more clusters to view their songs in detail below."),
            kind="info"
        ),
        cluster_filter
    ])
    return (cluster_filter,)


@app.cell(hide_code=True)
def _(cluster_filter, data_with_clusters, mo, pl):
    """Filtered Data Table"""
    selected_indices = [int(c) for c in cluster_filter.value]

    filtered_df = data_with_clusters.filter(
        pl.col('cluster').is_in(selected_indices)
    )

    display_cols = ['name', 'artist', 'cluster', 'danceability', 'energy',
                    'valence', 'acousticness', 'tempo', 'loudness']

    mo.vstack([
        mo.callout(
            mo.md(f"**📋 Song Details**: Showing {len(filtered_df):,} songs from selected clusters"),
            kind="success"
        ),
        mo.ui.table(
            filtered_df.select(display_cols).to_pandas(),
            pagination=True,
            page_size=15
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """Footer"""
    mo.callout(mo.md("""
    ### 📚 About This Dashboard

    This dashboard uses **K-Means clustering** to group 5,000 Spotify songs based on audio features. 
    Key techniques include:

    - **Standardization**: Features scaled to mean=0, std=1
    - **PCA**: Dimensionality reduction for visualization
    - **Silhouette Analysis**: Cluster quality measurement
    - **Random Forest**: Feature importance ranking

    **Data Source**: [Spotify 5000 Songs Dataset](https://github.com/datagus/ASDA2025)

    *Built with marimo | Deployed on GitHub Pages*
    """), kind="neutral")
    return


if __name__ == "__main__":
    app.run()

