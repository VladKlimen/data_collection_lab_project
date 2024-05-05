"""Functions for sentence embeddings, including some preprocessing functions"""
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

# Preprocessing functions
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# our functions
from part_5_packages.udf_functions import *


def remove_non_english_chars(text):
        return re.sub(r'[^\x00-\x7F]+', '', text).strip()
    

# This removes the (...) parts from the given sentence (also [...] and {...}). For example, the "(also [...] and {...})"" from the previous sentence will be removed.
def remove_between_brackets(text):
    pattern = r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}"
    result = re.sub(pattern, "", text)
    return result


# removes punctuation (replaces with spaces), removes repeating spaces
def clean_text(text):
    punctuation = '!"$%&\'()*-/:;<=>?[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    text = text.translate(translator)
    text = ' '.join(text.split())
    return text


def filter_nouns_adjectives(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    filtered_words = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('J')]
    filtered_sentence = ' '.join(filtered_words)
    if filtered_sentence == "":
        return sentence
    return filtered_sentence


def filter_nouns_adjectives_verbs(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    filtered_words = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]
    filtered_sentence = ' '.join(filtered_words)
    if filtered_sentence == "":
        return sentence
    return filtered_sentence
    

# The prep_pipeline is a list of functions from the above (just put the function in the list, as object)
# The functions are applied on the sentence, by the order in the list (the order matters sometimes)
def optional_preprocess(sentence, prep_pipeline, lower=True):
    # example of prep_pipeline: [remove_non_english_chars, remove_between_brackets]
    if lower:
        prep_pipeline = [lambda s: s.lower()] + prep_pipeline
    for func in prep_pipeline:
        sentence = func(sentence)

    return sentence

    
# Sentence embeddings
def embed_sentences(data, pretrained_embeddings, prep_pipeline=[], 
                    remove_stopwords=True, clean=True, lower=True, normalize=False):
    # Optional preprocessing / cleaning
    optional_preprocess_udf = udf(lambda sentence: optional_preprocess(sentence, prep_pipeline, lower=lower), StringType())
    data = data.withColumn("sentence_cleaned", optional_preprocess_udf(F.col("sentence")))
    if clean:
        data = data.withColumn("sentence_cleaned", F.regexp_replace(F.col("sentence_cleaned"), r"[^a-zA-Z+#/-_ ]", ""))

    # Optional stop words removing. Some words are worth keeping in some cases. E.g "it" can stand for "IT" (Information Technologies), wchich is important in our case
    keep_stopwords = ["it"]
    if remove_stopwords:
        english_stopwords = [word for word in stopwords.words('english') if word not in keep_stopwords]

        tokenizer = ftr.Tokenizer(inputCol="sentence_cleaned", outputCol="tokenized")
        data = tokenizer.transform(data)

        remover = ftr.StopWordsRemover(inputCol="tokenized", outputCol="filtered", stopWords=english_stopwords)
        data = remover.transform(data).withColumn("sentence_cleaned", F.concat_ws(" ", F.col("filtered")))
    
    # The embedding part
    documentAssembler = DocumentAssembler().setInputCol("sentence_cleaned").setOutputCol("document")
    embeddings = pretrained_embeddings.setInputCols(["document"]).setOutputCol("sentence_embeddings")
    pipeline = Pipeline().setStages([
        documentAssembler,
        embeddings
    ])

    distinct_data = data.select("sentence_cleaned").distinct()

    result = pipeline.fit(distinct_data).transform(distinct_data)
    result = result.select("sentence_cleaned", "sentence_embeddings") \
            .withColumn("embedding", F.expr("transform(sentence_embeddings, x -> x.embeddings)")).drop("sentence_embeddings")

    # Optional normalization
    if normalize:
        normalizer = ftr.Normalizer(p=2.0).setInputCol("embedding_raw").setOutputCol("embedding")
        result = normalizer.transform(result).select("sentence_cleaned", "embedding")

    # Convert the embedding to spark vector
    array_to_vector_udf = udf(lambda x: Vectors.dense(x[0]), VectorUDT())
    result = result.select("sentence_cleaned", "embedding").withColumn("embedding", array_to_vector_udf(F.col("embedding")))

    embeddings_df = data.select("index", "sentence_cleaned").join(result, "sentence_cleaned", "inner")

    return embeddings_df


# This function is relevant if you want to create a dict of embeddings by some category (in our case, industry was a category, but in the end we choose to not divide the data into categories. Yet, we used this function, with category "all", to not rewrite the code)
def get_sentence_embeddings_dict(sent_dict, pretrained_embeddings, prep_pipeline=[], cache=True, n_partitions=100,
                                 remove_stopwords=True, clean=True, lower=True, normalize=False):
    sentence_embeddings_dict = {}

    for category in sent_dict.keys():
        data = sent_dict[category].select("index", "sentence")
        embeddings_df = embed_sentences(data, pretrained_embeddings, prep_pipeline=prep_pipeline,
                                remove_stopwords=remove_stopwords, clean=clean, lower=lower, normalize=normalize)

        sentence_embeddings_dict[category] = sent_dict[category] \
                .join(embeddings_df, "index", "inner") \
                .repartition(n_partitions)
        if cache:
            sentence_embeddings_dict[category].persist()

    return sentence_embeddings_dict
