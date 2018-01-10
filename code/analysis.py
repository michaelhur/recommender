from math import sqrt
import pandas as pd
import json
import numpy as np
from preprocessing import *
import os.path


def load_data():
    movies_df = pd.read_csv('../data/movies.csv', encoding='utf-8')
    ratings_df = pd.read_csv('../data/ratings.csv', encoding='utf-8')

    movies_df = clean_movies(movies_df, ratings_df)
    movies_dict, movies_id_to_title, user_to_movie_ratings, movie_to_user_ratings = generate_data(movies_df, ratings_df)

    return movies_dict, movies_id_to_title, user_to_movie_ratings, movie_to_user_ratings

print "Loading data..."
movies_dict, movies_id_to_title, user_to_movie_ratings, movie_to_user_ratings = load_data()

## Move this to preprocessing.py
def transformdictof_X(dictof_X):
    result = {}
    for entry in dictof_X:
        for item in dictof_X[entry]:
            result.setdefault(item, {})

            ## Flip item and entry
            result[item][entry] = dictof_X[entry][item]
    return result


def sim_distance(dictof_X, entry1, entry2):
    """
    sim_distance() function returns the similarity, defined by the Euclidean distance, between two entries.
    """
    
    ## The class definition of dictionary in Python 2 make changes to the original dictionary even when we create the copy of it.
    ## So we create a new dictionary
    d = {}
    
    ## Get the list of all items
    entry1_items = set(dictof_X[entry1].keys())
    entry2_items = set(dictof_X[entry2].keys())
    
    all_items = entry1_items.union(entry2_items)
    common_items = entry1_items.intersection(entry2_items)
    
    ## If the item is rated by only one user:
    for item in all_items.difference(entry1_items):
        d[item] = dictof_X[entry2][item]
        
    for item in all_items.difference(entry2_items):
        d[item] = dictof_X[entry1][item]
    
    ## If the item is rated by both users
    for item in common_items:
        d[item] = dictof_X[entry1][item] - dictof_X[entry2][item]
        
    ## Add up the squares of all the differences
    sum_of_squares = sum([pow(d[item], 2) for item in d.keys()])

    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(dictof_X, entry1, entry2): 
    """
    sim_pearson() returns the Pearson correlation coefficient for entry1 and entry2
    """
    
    entry1_items = set(dictof_X[entry1].keys())
    entry2_items = set(dictof_X[entry2].keys())
        
    common_items = entry1_items.intersection(entry2_items)
    
    if len(common_items) == 0:
        return 0
    
    entry1_avg = np.mean([rating for item, rating in dictof_X[entry1].items() if item in common_items])
    entry2_avg = np.mean([rating for item, rating in dictof_X[entry2].items() if item in common_items])
    
    num = sum([(dictof_X[entry1][item] - entry1_avg) * (dictof_X[entry2][item] - entry2_avg) for item in common_items])
    
    sum1sq = sqrt(sum([pow(dictof_X[entry1][item] - entry1_avg, 2) for item in common_items]))
    sum2sq = sqrt(sum([pow(dictof_X[entry2][item] - entry2_avg, 2) for item in common_items]))
    
    den = sum1sq * sum2sq
    
    if den == 0:
        return 0
    
    r = num / den
    
    return r


def sim_cosine(dictof_X, entry1, entry2):
    
    ## Get the list of mutually rated items
    entry1_items = set(dictof_X[entry1].keys())
    entry2_items = set(dictof_X[entry2].keys())
        
    common_items = entry1_items.intersection(entry2_items)
    
    num = sum([dictof_X[entry1][item] * dictof_X[entry2][item] for item in common_items])

#     sum1sq = sqrt(float(sum([pow(dictof_X[entry1][item], 2) for item in common_items])))
#     sum2sq = sqrt(float(sum([pow(dictof_X[entry2][item], 2) for item in common_items])))
    
    sum1sq = sqrt(float(sum([pow(dictof_X[entry1][item], 2) for item in entry1_items])))
    sum2sq = sqrt(float(sum([pow(dictof_X[entry2][item], 2) for item in entry2_items])))
    
    den = sum1sq * sum2sq
    
    if den == 0:
        return 0
    
    cosine_similarity = num / den
    
    return cosine_similarity


def sim_jaccard(dictof_X, entry1, entry2):
 
    ##Get the list of all items and common items
    entry1_items = set(dictof_X[entry1].keys())
    entry2_items = set(dictof_X[entry2].keys())
    
    all_items = entry1_items.union(entry2_items)
    common_items = entry1_items.intersection(entry2_items)

    ##Get the total number of items for intersection and union
    num, den = len(common_items), len(all_items)

    ## return jaccard distance
    return float(num) / float(den) 
                                                                                             

def topMatches(dictof_X, entry, k = None, similarity = sim_pearson):
    """
    topMatches() returns the k most similar entries for the input entry from the dictof_X dictionary.
    The number of results and similarity function are optional parameters.
    """
    
    ## By default, the function returns all items, sorted from the most similar to the lease similar items
    if k == None:
        k = len(dictof_X)

    scores = [(similarity(dictof_X, entry, other), other) for other in dictof_X if other != entry]
    scores.sort()
    scores.reverse()
    
    return scores[:k]


def calculateSimilarity(dictof_X, k = 10, similarity = sim_pearson):
    """
    calculateSimilarItems() creates a dictionary of items showing which other items they are most similar to.
    This function will be used to pre-compute pairwise similarity score such that 
    we do not have to compute the similarity score every time we restart the app.
    """

    result = {}
    
    c = 0
    
    for item in dictof_X:
        ## Status updates for large datasets
        c += 1
        if c % 100 == 0: print "%d / %d" % (c, len(dictof_X))
        ## Find the most similar items to this one
        scores = topMatches(dictof_X, item, k = k, similarity = similarity)
        result[item] = scores

    return result


def getRecommendations(dictof_X, entry, n = 10, k = 10, similarity = sim_pearson):
    """
    getRecommendations() returns n recommendations for an entry by using a weighted average every other user's rankings
    """

    ## This returns k most similar users
    similarity_scores = topMatches(dictof_X, entry, k, similarity)
    
    totals = {}
    similarity_sum = {}
    
    for sim_score, other_entry in similarity_scores:
        
        if sim_score <= 0:
            continue
        
        ## Only recommend the movies that the user haven't yet seen
        for rated_items in dictof_X[other_entry]:
            
            if rated_items not in dictof_X[entry] or dictof_X[entry][rated_items] == 0:
                ## Similarity * Score
                totals.setdefault(rated_items, 0)
                totals[rated_items] += dictof_X[other_entry][rated_items] * sim_score
                ## Sum of similarities
                similarity_sum.setdefault(rated_items, 0)
                similarity_sum[rated_items] += sim_score
    
    ## Create the normalized list
    rankings = [(round(total / similarity_sum[item],2), movies_id_to_title[item]) for item, total in totals.items()]
    
    ## Return the sorted list
    rankings.sort()
    rankings.reverse()
#     rankings = [(tup[0], tup[2]) for tup in rankings]

    ## This returns n most similar movies based on 10 most similar user preferences.
    return rankings[:n]


def featurewise_getRecommendations(dictof_X, entry, k = 10, similarity = sim_pearson):
    """
    featurewise_getRecommendations()
    1. Computes the pairwise similarity between input movie and other movies in the dictionary
    2. The similarity is computed based on the features of the movie
    3. Returns average rating of the n most similar movies
    """
        
    similarity_scores = topMatches(dictof_X, entry, k, similarity)
    
    rankings = []

    for sim_score, similar_movie in similarity_scores:
        rating = dictof_X[similar_movie]['ratings']
        rankings.append((movies_id_to_title[similar_movie], round(rating,2)))

    return rankings