import json
import numpy as np


def get_dataset(path):
    dataset = {}
    with open(path, 'r') as f:
        for line in f:
            dataset.update(json.loads(line))
    return dataset


def calculate_mrr(similarity_matrix, relevant_indices):
    num_queries = similarity_matrix.shape[0]
    reciprocal_ranks = []

    for i in range(num_queries):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]

        rank = np.where(sorted_indices == relevant_indices[i])[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)

    return reciprocal_ranks


def calculate_hit_rate_at_k(similarity_matrix, relevant_indices, k):
    num_queries = similarity_matrix.shape[0]
    hits = []

    for i in range(num_queries):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        hits.append(1 if relevant_indices[i] in sorted_indices[:k] else 0)

    return hits
