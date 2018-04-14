from sklearn.model_selection import train_test_split
import pandas as pd
import mxnet
import os
import sys


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    data_dir_path = patch_path('data/ml-latest-small')
    output_dir_path = patch_path('models')

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    ratings_train, ratings_test = train_test_split(records, test_size=0.2, random_state=0)

    timestamp_train = ratings_train.as_matrix(columns=['timestamp'])
    user_id_train = ratings_train.as_matrix(columns=['userId'])
    item_id_train = ratings_train.as_matrix(columns=['movieId'])
    rating_train = ratings_train.as_matrix(columns=['rating'])

    timestamp_test = ratings_test.as_matrix(columns=['timestamp'])
    user_id_test = ratings_test.as_matrix(columns=['userId'])
    item_id_test = ratings_test.as_matrix(columns=['movieId'])
    rating_test = ratings_test.as_matrix(columns=['rating'])

    max_user_id = records['userId'].max()
    max_item_id = records['movieId'].max()

    from mxnet_recommender.library.cf import CollaborativeFilteringWithTemporalInformation
    cf = CollaborativeFilteringWithTemporalInformation(model_ctx=mxnet.gpu(0))
    cf.max_user_id = max_user_id
    cf.max_item_id = max_item_id
    history = cf.fit(user_id_train=user_id_train,
                     item_id_train=item_id_train,
                     timestamp_train=timestamp_train,
                     rating_train=rating_train,
                     model_dir_path=output_dir_path,
                     epochs=20,
                     checkpoint_interval=2,
                     batch_size=256)

    metrics = cf.evaluate_mae(user_id_test=user_id_test,
                              item_id_test=item_id_test,
                              timestamp_test=timestamp_test,
                              rating_test=rating_test)


if __name__ == '__main__':
    main()
