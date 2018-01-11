# Recommendation System
A movie recommendation system, in which it recommends the movies that the user have not yet seen or those that are most similar to the interested one.

It incorporates the 'user-based' collaborative filtering to find other users with similar preferences to a specific user, then recommends the movies based on their preferences.

For the "similar movies" recommendation, it computes the pairwise similarity among all movies then recommends ones that are most similar.

### Original Data

The Movielens (small) data was used in this project, and the dataset as well as the details can be found under the "data" folder. It was originally retreived from the following url:

```
https://grouplens.org/datasets/movielens/
```

### How to run

1. Clone repository

```
$ git clone https://github.com/michaelhur/recommender.git
```

2. Install the required packages using pip:

```
$ pip install -r requirements.txt
```

3. Run following line of code

```
$ python recommend.py
```

