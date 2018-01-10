from analysis import *
from preprocessing import *
import pandas as pd

while True:
    Quest = raw_input("\nSelect one or type 'done' to quit:\n 1- User recommendation \n 2- Item recommendation\n")
    if Quest == '1':
        ID= raw_input("Enter the user ID\n")
        try:
            recommended = getRecommendations(user_to_movie_ratings, int(ID))
            print "Recommended movies with the ratings are:\n", recommended
        except:
            print "Please check the ID"

    elif Quest == '2':
        Item = raw_input("Enter the movie ID':\n")

        try:
            recommended = featurewise_getRecommendations(movies_dict, int(Item))
            print 'Recommended movies with the ratings are:\n',recommended

        except:
            print "lease check the ID"

    elif Quest == 'done':
        break

    else:
        print "please try again"