# mxnet-recommender

Collaborative Filtering NN and CNN based recommender implemented with MXNet

The dataset is taken from  [ml-latest-small (MovieLens)](https://grouplens.org/datasets/movielens/)

The trained models an be found in demo/models

# Deep Learning Models

### Collaborative Filtering Models

* Collaborative Filtering V1: hidden factor analysis implementation of CF
    * training: [demo/collaborative_filtering_v1.py](demo/collaborative_filtering_v1.py)
    * predicting: [demo/collaborative_filtering_v1_predict.py](demo/collaborative_filtering_v1_predict.py)
* Collaborative Filtering V2: CF with feedforward dense layer
    * training: [demo/collaborative_filtering_v2.py](demo/collaborative_filtering_v2.py)
    * predicting: [demo/collaborative_filtering_v2_predict.py](demo/collaborative_filtering_v2_predict.py)
* Collaborative Filtering with Temporal Information: CF with feedforward dense layer and incorporate timestamp as input
    * training: [demo/collaborative_filtering_temporal.py](demo/collaborative_filtering_temporal.py)
    * predicting: [demo/collaborative_filtering_temporal_predict.py](demo/collaborative_filtering_temporal_predict.py)
    
### Content-based Filtering Models
    
* Item-based Content-Based Filtering: Use timestamp information and item on content-based filtering
    * trainng: [demo/temporal_content_based_filtering.py](demo/temporal_content_based_filtering.py)
    * predicting: [demo/temporal_content_based_filtering_predict.py](demo/temporal_content_based_filtering_predict.py)
    
# Usage

The following code samples provide an illustration on both training and prediction using a deep 
learning model in the mxnet_recommender/library. Other deep learning models follow the similar
training and prediction patterns.

### Train CF model

To train a CF model, say [CollaborativeFilteringV1](mxnet_recommender/library/cf.py), run the following commands:

```bash
pip install requirements.txt

cd demo
python collaborative_filtering_v1.py 
```

The training code in [collaborative_filtering_v1.py](demo/collaborative_filtering_v1.py) is quite straightforward and 
illustrated below:

```python
from sklearn.model_selection import train_test_split
import pandas as pd
from mxnet_recommender.library.cf import CollaborativeFilteringV1

data_dir_path = './data/ml-latest-small' # refers to demo/data/ml-latest-small folder
trained_model_dir_path = './models' # refers to demo/models folder

records = pd.read_csv(data_dir_path + '/ratings.csv')
print(records.describe())

ratings_train, ratings_test = train_test_split(records, test_size=0.2, random_state=0)

user_id_train = ratings_train.as_matrix(columns=['userId'])
item_id_train = ratings_train.as_matrix(columns=['movieId'])
rating_train = ratings_train.as_matrix(columns=['rating'])

user_id_test = ratings_test.as_matrix(['userId'])
item_id_test = ratings_test.as_matrix(['movieId'])
rating_test = ratings_test.as_matrix(['rating'])

max_user_id = records['userId'].max()
max_item_id = records['movieId'].max()

# default context for the recommender is mxnet.cpu() which uses CPU for the model context and data context
# change this line to cf = CollaborativeFilteringV1(model_ctx=mxnet.gpu(0)) if you want to use GPU instead
cf = CollaborativeFilteringV1() 
cf.max_user_id = max_user_id
cf.max_item_id = max_item_id
history = cf.fit(user_id_train=user_id_train,
                 item_id_train=item_id_train,
                 rating_train=rating_train,
                 model_dir_path=trained_model_dir_path)

metrics = cf.evaluate_mae(user_id_test=user_id_test,
                      item_id_test=item_id_test,
                      rating_test=rating_test)

```

After the training is completed, the trained models will be saved as cf-v1-*.* in the demo/models.

### Predict Rating using CF trained model

To use the trained CF model to predict the rating of an item by a user, you can use the following 
[code](demo/collaborative_filtering_v1_predict.py):

```python

from mxnet_recommender.library.cf import CollaborativeFilteringV1
import pandas as pd

data_dir_path = './data/ml-latest-small' # refers to demo/data/ml-latest-small folder
trained_model_dir_path = './models' # refers to demo/models folder

records = pd.read_csv(data_dir_path + '/ratings.csv')
print(records.describe())

user_id_test = records['userId']
item_id_test = records['movieId']

cf = CollaborativeFilteringV1()
cf.load_model(trained_model_dir_path)

# batch prediction
predicted_ratings = cf.predict(user_id_test, item_id_test)
print(predicted_ratings)

# individual (user_id, item_id) prediction
for i in range(20):
    user_id = user_id_test[i]
    item_id = item_id_test[i]
    predicted_rating = cf.predict_single(user_id, item_id)
    print('predicted rating: ', predicted_rating)
```

### Train CF model with Temporal Information

To train a CF model to also take timestamp into consideration, say [CollaborativeFilteringWithTemporalInformation](mxnet_recommender/library/cf.py), run the following commands:

```bash
pip install requirements.txt

cd demo
python collaborative_filtering_temporal.py 
```

The training code in [collaborative_filtering_temporal.py](demo/collaborative_filtering_temporal.py) is 
illustrated below:

```python
from sklearn.model_selection import train_test_split
import pandas as pd
from mxnet_recommender.library.cf import CollaborativeFilteringWithTemporalInformation


def main():
    data_dir_path = './data/ml-latest-small'
    output_dir_path = './models'

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

    cf = CollaborativeFilteringWithTemporalInformation()
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


```

After the training is completed, the trained models will be saved as temporal-cf-*.* in the demo/models.

### Predict Rating with Temporal Information

To use the trained CF model to predict the rating of an item by a user at a particular time, you can use the following 
[code](demo/collaborative_filtering_temporal_predict.py):

```python
import pandas as pd
from mxnet_recommender.library.cf import CollaborativeFilteringWithTemporalInformation


def main():
    data_dir_path = './data/ml-latest-small'
    trained_model_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    timestamp_test = records.as_matrix(columns=['timestamp'])
    user_id_test = records.as_matrix(columns=['userId'])
    item_id_test = records.as_matrix(columns=['movieId'])
    rating_test = records.as_matrix(columns=['rating'])

    cf = CollaborativeFilteringWithTemporalInformation()
    cf.load_model(model_dir_path=trained_model_dir_path)

    predicted_ratings = cf.predict(user_id_test, item_id_test, timestamp_test)
    print(predicted_ratings)
    
    for i in range(20):
        user_id = user_id_test[i]
        item_id = item_id_test[i]
        timestamp = timestamp_test[i]
        rating = rating_test[i]
        predicted_rating = cf.predict_single(user_id, item_id, timestamp)
        print('predicted: ', predicted_rating, ' actual: ', rating)


if __name__ == '__main__':
    main()
```

### Train Content-based Filtering model with Temporal Information

To train a content-based filtering model, say [TemporalContentBasedFiltering](mxnet_recommender/library/cf.py), run the following commands:

```bash
pip install requirements.txt

cd demo
python temporal_content_based_filtering.py 
```

The training code in [temporal_content_based_filtering.py](demo/temporal_content_based_filtering.py) is 
illustrated below:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mxnet_recommender.library.content_based_filtering import TemporalContentBasedFiltering


def main():
    data_dir_path = './data/ml-latest-small'
    output_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    ratings_train, ratings_test = train_test_split(records, test_size=0.2, random_state=0)

    timestamp_train = ratings_train.as_matrix(columns=['timestamp'])
    item_id_train = ratings_train.as_matrix(columns=['movieId'])
    rating_train = ratings_train.as_matrix(columns=['rating'])

    timestamp_test = ratings_test.as_matrix(columns=['timestamp'])
    item_id_test = ratings_test.as_matrix(columns=['movieId'])
    rating_test = ratings_test.as_matrix(columns=['rating'])

    max_item_id = records['movieId'].max()

    cf = TemporalContentBasedFiltering()
    cf.max_item_id = max_item_id
    history = cf.fit(timestamp_train=timestamp_train,
                     item_id_train=item_id_train,
                     rating_train=rating_train,
                     model_dir_path=output_dir_path,
                     epochs=20,
                     checkpoint_interval=2,
                     batch_size=256)

    metrics = cf.evaluate_mae(timestamp_test=timestamp_test,
                              item_id_test=item_id_test,
                              rating_test=rating_test)


if __name__ == '__main__':
    main()

```

After the training is completed, the trained models will be saved as temporal-cbf-*.* in the demo/models.

### Predict Item Rating with Temporal Information

To use the trained CF model to predict the rating of an item at a particular time, you can use the following 
[code](demo/temporal_content_based_filtering_predict.py):

```python
import pandas as pd
from mxnet_recommender.library.content_based_filtering import TemporalContentBasedFiltering


def main():
    data_dir_path = './data/ml-latest-small'
    trained_model_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    timestamp_test = records.as_matrix(columns=['timestamp'])
    item_id_test = records.as_matrix(columns=['movieId'])
    rating_test = records.as_matrix(columns=['rating'])

    max_item_id = records['movieId'].max()

    config = dict()
    config['max_item_id'] = max_item_id

    cf = TemporalContentBasedFiltering()
    cf.load_model(model_dir_path=trained_model_dir_path)

    predicted_ratings = cf.predict(item_id_test, timestamp_test)
    print(predicted_ratings)

    for i in range(20):
        date = timestamp_test[i]
        item_id = item_id_test[i]
        rating = rating_test[i]
        predicted_rating = cf.predict_single(item_id, date)
        print('predicted: ', predicted_rating, ' actual: ', rating)


if __name__ == '__main__':
    main()

```

# Note

Note that the default training scripts in the [demo](demo) folder use GPU for training, therefore, you must configure your
graphic card for this (or remove the "model_ctx=mxnet.gpu(0)" in the training scripts). 


* Step 1: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (you should download CUDA® Toolkit 9.0)
* Step 2: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
