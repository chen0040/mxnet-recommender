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

### Predict Rating

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
