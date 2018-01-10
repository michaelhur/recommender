import pandas as pd
import re
import numpy as np
import json
import os
import os.path


def extract_year(title):
    year = re.search(r'\(\d{4}\)', title)
    if year:
        year = year.group()
        return int(year[1:5])
    else:
        return 0


def extract_genre(x):
    if x == '(no genres listed)':
        keys = []
    else:
        #keys = re.sub('[|]', ' ', x)
        keys = str(x).replace('|', ' ')
        keys = keys.split()
    return keys

# def create_genre_dict(listof_g):
#     """
#     create_genre_dict function takes list of genre as an input and,
#     returns dictionary of genres where (genre, [0,1]) as a key-value pair.
#     """
    
#     genre_dict = {}

#     for genre in all_genres:
#     	if genre in listof_g:
#     		genre_dict[genre] = 1
#     	else:
# 	        genre_dict[genre] = 0
    
#     return genre_dict

#     ## This will return weird Piccard Similarity Score, for it computes common non-genre as interception

def create_genre_dict(listof_g):
    """
    create_genre_dict function takes list of genre as an input and,
    returns dictionary of genres where (genre, [0,1]) as a key-value pair.
    """
    
    genre_dict = {}

    for genre in listof_g:
    	genre_dict[genre] = 1
    
    return genre_dict    


def clean_movies(movies_df, ratings_df):
	"""
	Clean movies_df dataframe such that it could be used for analysis.
	1. Extract year from the title
	2. Remove year from the title
	3. Extract genre
	4. Add average user rating
	"""

	## Title take ths form 
	movies_df['year'] = movies_df['title'].apply(extract_year)

	movies_df.loc[movies_df.year != 0, 'title'] = movies_df.title.apply(lambda x: x[:-7])

	# ## We remove all the cases where the year the movie was released is unknown. 
	# movies_df = movies_df[movies_df.year != 0]

	movies_df.loc[22368, 'title'] = "Diplomatic Immunity"
	movies_df.loc[22669, 'title'] = "Bing Bang Theory"
	movies_df.loc[22679, 'title'] = "Fawlty Towers"

	movies_df['genre_list'] = movies_df.genres.apply(extract_genre)

	global all_genres
	all_genres = list(set([genre for genre_list in movies_df.genre_list.tolist() for genre in genre_list]))
	
	summary = ratings_df.groupby(['movieId'])
	summary_df = pd.DataFrame()
	summary_df['average_rating'] = summary['rating'].mean()
	# summary_df['total_rating'] = summary['rating'].sum()
	# summary_df['number_of_ratings'] = summary['rating'].count()

	movies_df = pd.merge(movies_df, summary_df, left_on= 'movieId', right_index= True, how = 'outer')
	movies_df = movies_df.fillna(0)

	## We remove all the cases where the year the movie was released is unknown. 
	movies_df = movies_df[movies_df.year != 0]
	movies_df.reset_index(drop = True, inplace = True)

	return movies_df


def generate_data(movies_df, ratings_df):
	"""
	Consumes movies_df and ratings_df, then generates the data needed for analyses
	"""

	# movies_df = clean_movies(movies_df)

	movies_dict = {}
	movies_id_to_title = {}
	# movie_ratings = {}

	for index, row in movies_df.iterrows():
	    #movieId, title, year, genre_list, avg_rating, total_rating, number_of_ratings = int(row[0]), row[1], int(row[3]), row[4], row[5], row[6], int(row[7])
	    movieId, title, year, genre_list, avg_rating = int(row['movieId']), row['title'], int(row['year']), row['genre_list'], row['average_rating']

	    movies_dict.setdefault(movieId, {})
	    movies_id_to_title.setdefault(movieId, '')
	    # movie_ratings.setdefault(movieId, {})

	    movies_id_to_title[movieId] = title

	    movies_dict[movieId]['year'] = year
	    
	    movies_dict[movieId]['ratings'] = avg_rating
	    # movies_dict[movieId]['popularity'] =  number_of_ratings

	    # movie_ratings[movieId]['ratings'] = avg_rating
	    # movie_ratings[movieId]['total_ratings'] = total_rating
	    # movie_ratings[movieId]['number_of_ratings'] = number_of_ratings
	    
	    movies_dict[movieId].update(create_genre_dict(genre_list))

	"""Move this ratings_df to clean_movies, and change function name to clean_df"""
	eligible_movieId = movies_df.movieId.tolist()

	filtered_ratings_df = ratings_df[ratings_df.movieId.isin(eligible_movieId)]
	filtered_ratings_df.reset_index(drop = True, inplace = True)

	user_to_movie_ratings = {}
	movie_to_user_ratings = {}

	for index, row in filtered_ratings_df.iterrows():
	    userId, movieId, rating = int(row[0]), int(row[1]), int(row[2])
	    user_to_movie_ratings.setdefault(userId, {})
	    user_to_movie_ratings[userId][movieId] = rating

	    movie_to_user_ratings.setdefault(movieId, {})
	    movie_to_user_ratings[movieId][userId] = rating

#	return user_to_movie_ratings
	return movies_dict, movies_id_to_title, user_to_movie_ratings, movie_to_user_ratings