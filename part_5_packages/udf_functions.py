"""UDF functions for pyspark"""
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

@udf (VectorUDT())
def elementwise_avg(vectors):
    num_vectors = len(vectors)
    vector_size = len(vectors[0])
    avg_values = [0.0] * vector_size

    for vector in vectors:
        for i in range(vector_size):
            avg_values[i] += vector[i] / num_vectors

    return Vectors.dense(avg_values)


@udf (DoubleType())
def cosine_similarity(v1, v2):
    return float(v1.dot(v2) / ((v1.norm(2) * v2.norm(2))))


@udf (DoubleType())
def euqlidean_dist(v1, v2):
    return float(v1.squared_distance(v2) ** 0.5)


@udf (ArrayType(StringType()))
def get_first_n(arr, n, start_from):
    return list(arr)[start_from : n + start_from]


# This used for correcting the list wrapped in string, we got from Gemini as the answer to the promt
@udf (ArrayType(StringType()))
def string_to_list(list_string):
    return [s.strip() for s in list_string[1:-1].split(r', ')]


@udf (StringType())
def filter_nouns(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    filtered_words = [word for word, pos in tagged_words if pos.startswith('N')]
    filtered_sentence = ' '.join(filtered_words)
    if filtered_sentence == "":
        return sentence
    return filtered_sentence


@udf (StringType())
def extract_job_function(s):
    result = re.findall(r"Job function=(.*?), Seniority level", s)
    if len(result) > 0:
        return result[0]
    return ""


@udf (ArrayType(StringType()))
def split_job_functions(s):
    result = re.split(r', and | and |, ', s)
    if result == ['']:
        return ["Unlabelled"]
    return result