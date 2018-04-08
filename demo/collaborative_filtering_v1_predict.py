from sklearn.model_selection import train_test_split
import pandas as pd
from keras_recommender.library.cf import CollaborativeFilteringV1
import numpy as np


def main():
    data_dir_path = './data/ml-latest-small'
    trained_model_dir_path = './models'

    all_ratings = pd.read_csv(data_dir_path + '/ratings.csv')
    print(all_ratings.describe())

    user_id_test = all_ratings['userId']
    item_id_test = all_ratings['movieId']
    rating_test = all_ratings['rating']

    cf = CollaborativeFilteringV1()
    cf.load_model(CollaborativeFilteringV1.get_config_file_path(trained_model_dir_path),
                  CollaborativeFilteringV1.get_weight_file_path(trained_model_dir_path))

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
