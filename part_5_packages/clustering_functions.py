"""Functions for HDBSCAN clustering and visualization"""

# pyspark
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import split, explode
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
import pyspark.ml.feature as ftr
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline

# nlp
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk.stem
from nltk.corpus import stopwords

# for visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import colorsys
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

# python standard libraries
from statistics import mean, variance
from math import log10, sqrt
from operator import add
from functools import reduce
import builtins
import pickle
import time
from collections import defaultdict, Counter
import os
import string
import warnings
import random
from pathlib import Path
import joblib
import ast
import re

# for clustering
from scipy.spatial import distance
import sklearn.cluster
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
from sklearn import metrics
import hdbscan

# our functions
from part_5_packages.udf_functions import *


# returns pca components vector, given a column of vectors
def get_PCA_components(embeddings_df, inputCol="embedding", outputCol="embedding_pca", k=50):
    pca = PCA(k=k, inputCol=inputCol).setOutputCol(outputCol)
    df_pca = pca.fit(embeddings_df).transform(embeddings_df)
    return df_pca


# This is a wrapper for hdbscan function (from the hdbscan package), works bad with the cosine similarity (at least in our case), so don't use the "precomputed" metric. Optionally, reduces the embeddings dimension using pca (we didn't use it in the final version)
def hdbscan_clustering(embeddings_df, spark, metric='precomputed', min_samples=2, min_cluster_size=5, cluster_selection_method='leaf',
                           cluster_selection_epsilon=0.75, prediction_data=True, pca=False, pca_k=50):
    if pca:
        embedding_col = "embedding_pca"
        pca_distinct = get_PCA_components(df.select("embedding").distinct(), k=pca_k)
        df = df.join(pca_distinct, "embedding", "inner")
    else:
        df, embedding_col = embeddings_df, "embedding"

    # indexed_embeddings = list(zip(*[(row[embedding_col], row["index"]) for row in 
    #                                     df.select("index", embedding_col).collect()]))

    embeddings_list = [row.embedding for row in df.select("embedding").distinct().collect()]

    X = np.array(embeddings_list)

    if metric == "precomputed":
        X = metrics.pairwise.cosine_similarity(X, X)
        np.fill_diagonal(X, 0)
    
    # clusterer is our clustering model that we will use for cluster prediction of new points (aka empbedded sentences)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, 
                                cluster_selection_method=cluster_selection_method, metric=metric, cluster_selection_epsilon=cluster_selection_epsilon, prediction_data=prediction_data,
                                gen_min_span_tree=True).fit(X)
    
    clusters = clusterer.labels_
    probs = clusterer.probabilities_
    # the soft clusters are clusters which contain the original clustered points, and also points that are marked previously as noise. The noise points are appended to the clusters to which they are most probably belong (considering the existing clusters).
    all_points_membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
    soft_clusters = [np.argmax(x) for x in all_points_membership_vectors]
    soft_probs = [np.max(x) for x in all_points_membership_vectors]
    
    # These are different evaluation scores. The most relevant for hdbscan is DBCV, although it only helps to choose the right tunning, as it is not an absolute score and can't be used for comparing different clustering models on different kinds of data.
    eval_score = metrics.silhouette_score(X, clusters, metric=metric)
    print(f"Silhouette score: {eval_score}")
    eval_score = metrics.calinski_harabasz_score(X, clusters)
    print(f"Calinski-Harabasz score: {eval_score}")
    eval_score = metrics.davies_bouldin_score(X, clusters)
    print(f"Davies-Bouldin score: {eval_score}")
    try:
        eval_score = clusterer.relative_validity_
        print(f"DBCV relative score: {eval_score}")
    except Exception as e:
        print(f"DBCV:\n{e}")
    print(f"Number of clusters: {len(set(clusters)) - 1}")
    print(f"Number of unique sentences clustered: {len([x for x in clusters if x != -1])}")


    # mapping clusters to sentences
    clusters_map = [(embedding, int(cluster), int(soft_cluster), 
                     float(probability), float(soft_probability)) 
                    for i, (embedding, cluster, soft_cluster, probability, soft_probability) 
                    in enumerate(zip(embeddings_list, clusters, soft_clusters, probs, soft_probs))]
    clusters_map_df = spark.createDataFrame(clusters_map, ["embedding", "cluster", "soft_cluster", "probability", "soft_probability"])

    df_clustered = df.join(clusters_map_df, "embedding", "inner")

    return df_clustered, clusterer


# tsne embeddings are 2D vectors, obtained from embeddings of high dimension, used for visualisation of high dimensional vectors/points
def get_tsne_embeddings(df_clustered, spark, inputCol="embedding", metric='cosine', 
                            random_state=None, early_exaggeration=12, perplexity=20):
    df = df_clustered.select(inputCol).distinct()
    df_tsne = df_clustered

    # it's recommended to reduce the dimensionality to 50 with pca before applying tsne, to achieve better results
    if len(df_clustered.first().asDict()[inputCol]) > 50:
        df = get_PCA_components(df, inputCol=inputCol, k=50)
        df_tsne = df_tsne.join(df, inputCol, "inner")
        inputCol = "embedding_pca"
        
    vectors = df.select(inputCol).rdd.map(lambda x: tuple(x[inputCol].toArray())).collect()
    vectors_tsne = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,
                   metric=metric, n_jobs=-1, random_state=random_state).fit_transform(vectors)
    
    # schema = StructType([StructField(inputCol, ArrayType(DoubleType(), True), True),
    #                     StructField("vector_TSNE", ArrayType(DoubleType(), True), True)])
    vectors_df = spark.createDataFrame([Row(**{inputCol: Vectors.dense(v1), "vector_TSNE": Vectors.dense(v2.tolist())}) 
                                for v1, v2 in zip(vectors, vectors_tsne)], [inputCol, "vector_TSNE"])
    df_tsne = df_tsne.join(vectors_df, inputCol, "inner").drop("embedding_pca")

    return df_tsne


def generate_n_unique_colors(indexes, saturation=1, lightness="random"):
        saturation = random.uniform(0.9, 1) if saturation == "random" else saturation
        lightness = random.uniform(0.2, 0.3) if lightness == "random" else lightness

        colors = {}
        for i in indexes:
            r, g, b = colorsys.hls_to_rgb(random.uniform(0, 1), lightness, saturation)
            r_hex = int(r * 255)
            g_hex = int(g * 255)
            b_hex = int(b * 255)
            color_code = "#{:02x}{:02x}{:02x}".format(r_hex, g_hex, b_hex)
            colors[i] = color_code
        return colors


# this function visualizes the clusters in 2D, but not as pretty as the datamapplot library we used for final visualization.
# yet, we use it to analyze the clusters
def visualize_tsne_clusters(vectors, clusters, probs, subtitle, soft_clusters=None, soft_probs=None, show_soft_clusters=True,
                            show_noise=False, annotate=True, show_simplices=True, clusters_to_show=[], fig_size=(10, 10), s=200, fontsize=10, satur_accent=1, cent_prob=0.99):
    x_coords, y_coords = zip(*vectors)
    x_coords, y_coords = np.array(x_coords), np.array(y_coords)
    centroids = {cluster: (np.mean(np.array([x for i, x in enumerate(x_coords) if clusters[i] == cluster and probs[i] > cent_prob])),
                           np.mean(np.array([y for i, y in enumerate(y_coords) if clusters[i] == cluster and probs[i] > cent_prob])))
                 for cluster in set(clusters)}
    
    if len(clusters_to_show) == 0:
        clusters_to_show =  clusters
    unique_clusters = list(set([x for x in clusters if x != -1]))
    
    # Calculate convex hulls for each cluster
    hulls = {}
    for i, cluster in enumerate(unique_clusters):
        cluster_points = np.array([vector for i, vector in enumerate(vectors)  if clusters[i] == cluster])
        try:
            hull = ConvexHull(cluster_points)
            hull_points = np.vstack([cluster_points[hull.vertices], cluster_points[hull.vertices[0]]])
            hulls[i] = hull_points
        except:
             hulls[i] = None

    # Interpolate hull points to create smooth area
    smooth_hulls = {}
    for i in range(len(unique_clusters)):
        hull_points = hulls[i]
        if hull_points is not None:
            x, y = hull_points[:, 0], hull_points[:, 1]
            t = np.linspace(0, 1, len(x))
            interp = interp1d(t, np.vstack([x, y]), kind='cubic', axis=1)
            t_smooth = np.linspace(0, 1, 100)
            smooth_points = interp(t_smooth).T
            smooth_hulls[i] = smooth_points
        else:
            smooth_hulls[i] = None

    color_palette = generate_n_unique_colors(unique_clusters)
    color_palette = {key: (value if key in clusters_to_show else 'white') for key, value in color_palette.items()}
    color_palette[-1] = (0.5, 0.5, 0.5)
    zorders = {key: (2 if key in clusters_to_show else 0) for key in clusters}
    alphas = {"cluster": 0.1, "soft_cluster": 0.1, "noise": 0.03}

    cluster_colors = [color_palette[x] for x in clusters]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, probs)]
    
    plt.figure(figsize=fig_size)
    if show_noise:
        for i, (x, y, cluster) in enumerate(zip(x_coords, y_coords, clusters)):
            if cluster == -1:
                plt.scatter(x, y, color=cluster_member_colors[i], alpha=alphas["noise"], s=s/2, zorder=1)

    if show_soft_clusters and soft_clusters is not None and soft_probs is not None:
        alphas["cluster"] = alphas["soft_cluster"] * 2
        soft_cluster_colors = [sns.desaturate(color_palette[x], min(1, p * satur_accent)) for x, p in zip(soft_clusters, soft_probs)]
        for i, (x, y, soft_cluster) in enumerate(zip(x_coords, y_coords, soft_clusters)):
            if clusters[i] == -1 and soft_cluster in clusters_to_show:
                plt.scatter(x, y, color=soft_cluster_colors[i], alpha=alphas["soft_cluster"], s=s, zorder=3)
                
    for i, (x, y, cluster) in enumerate(zip(x_coords, y_coords, clusters)):
        if cluster != -1:
            plt.scatter(x, y, color=cluster_member_colors[i], alpha=alphas["cluster"], s=s, zorder=zorders[cluster])
    
    for i, cluster in enumerate(unique_clusters):
        x, y = centroids[cluster]
        smooth_points = smooth_hulls[i]
        if show_simplices and smooth_points is not None:
            plt.fill(smooth_points[:, 0], smooth_points[:, 1], color=color_palette[cluster], alpha=0.05, zorder=zorders[cluster])
        if annotate:
            plt.text(x, y, str(cluster), color=color_palette[cluster], ha='center', va='center', 
                     fontsize=fontsize, zorder=zorders[cluster])

    plt.title(f"2D Approximate Relative Visual Representation of Clusters ([0-{max(clusters)}])\n({subtitle})", 
              fontsize=fig_size[0] + 3)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
# This creates names "layers" for clusters. The method is simple: the most inner layer is the sentence itself.
# The second layer is the max_ most frequent words in the sentences in each clusters. The third is the max_-1 most frequent, etc.
# We filtered and preprocessed some of the words before layering, and also it is possible to start from n-th most frequent word, using start_from, 
# to obtain more meaningful names (sometimes it helps).
def create_layers(clusters_df, spark, min_=2, max_=5, min_word_len=2, start_from=0, lemmatize=True, soft=True):
    df = clusters_df.withColumn("sentence_cleaned", filter_nouns(F.col("sentence_cleaned")))
    cluster_col = "soft_cluster" if soft else "cluster"
    sent_col = "sentence_cleaned"

    words = df.withColumn("word", F.explode(F.split(F.col(sent_col), r"\s+|,\s*")))
    words = words.withColumn("word", F.regexp_replace("word", "[^a-zA-Z+#/]", "")) \
                    .filter(F.length(F.col("word")) >= min_word_len)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    @udf (StringType())
    def lemmatize_udf(word):
        return lemmatizer.lemmatize(word)

    if lemmatize:
        words = words.withColumn("word", lemmatize_udf("word"))
                    
    word_counts = words.groupBy("word", cluster_col).count()
    most_frequent_words = word_counts.orderBy(cluster_col, F.desc("count"))

    top_words_per_cluster = most_frequent_words.groupBy(cluster_col) \
        .agg(F.collect_list("word").alias("words")) \
        .select(cluster_col, get_first_n(F.col("words"), F.lit(max_), F.lit(start_from)).alias(f"top_{max_}_words"))

    clusters_range = list(range(clusters_df.agg(F.min(cluster_col).alias("max_cluster")).collect()[0]["max_cluster"],
                    clusters_df.agg(F.max(cluster_col).alias("max_cluster")).collect()[0]["max_cluster"] + 1))
    remaining_clusters = [row[cluster_col] for row in top_words_per_cluster.select(cluster_col).distinct().collect()]
    clusters_to_supplement = [c for c in clusters_range if c not in remaining_clusters]
    if clusters_to_supplement != []:
        top_words_per_cluster = top_words_per_cluster.union(
            spark.createDataFrame([(c, v) for c, v in zip(clusters_to_supplement, ["Unlabelled"] * len(clusters_to_supplement))], [cluster_col, f"top_{max_}_words"]))
        
    df = top_words_per_cluster
    df.persist()

    for i in range(max_ - min_ + 1):
        df = df.withColumn(f"layer_{i}", F.concat_ws(", ", get_first_n(F.col(f"top_{max_}_words"), F.lit(max_ - i), F.lit(0))))
    
    clusters_df = clusters_df.join(df.drop(f"top_{max_}_words"), [cluster_col], "left_outer")
    clusters_df.persist()

    return clusters_df
