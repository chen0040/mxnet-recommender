from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import numpy as np


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    data_dir_path = patch_path('data/ml-latest-small')
    trained_model_dir_path = patch_path('models')

    all_ratings = pd.read_csv(data_dir_path + '/ratings.csv')
    print(all_ratings.describe())

    user_id_test = all_ratings.as_matrix(columns=['userId'])
    item_id_test = all_ratings.as_matrix(columns=['movieId'])
    rating_test = all_ratings.as_matrix(columns=['rating'])

    from mxnet_recommender.library.cf import CollaborativeFilteringV2
    cf = CollaborativeFilteringV2()
    cf.load_model(trained_model_dir_path)

    predicted_ratings = cf.predict(user_id_test, item_id_test)
    print(predicted_ratings)

    for i in range(20):
        user_id = user_id_test[i]
        item_id = item_id_test[i]
        rating = rating_test[i]
        predicted_rating = cf.predict_single(user_id, item_id)
        print('predicted: ', predicted_rating, ' actual: ', rating)


if __name__ == '__main__':
    main()
