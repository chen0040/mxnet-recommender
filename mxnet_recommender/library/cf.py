from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import math


class CFV1(gluon.nn.Block):

    def __init__(self, embedding_size, user_input_dim, item_input_dim, **kwargs):
        super(CFV1, self).__init__(**kwargs)
        with self.name_scope():
            # initializer: random with minval = -0.05, maxval = 0.05
            self.user_embedding = gluon.nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.item_embedding = gluon.nn.Embedding(input_dim=item_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))

    def forward(self, x):
        user_vecs = nd.flatten(self.user_embedding(x[:, 0]))
        item_vecs = nd.flatten(self.item_embedding(x[:, 1]))
        return nd.sum(nd.multiply(user_vecs, item_vecs), axis=1)


class CFV2(gluon.nn.Block):

    def __init__(self, embedding_size, user_input_dim, item_input_dim, **kwargs):
        super(CFV2, self).__init__(**kwargs)
        with self.name_scope():
            # initializer: random with minval = -0.05, maxval = 0.05
            self.user_embedding = gluon.nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.item_embedding = gluon.nn.Embedding(input_dim=item_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.input_dropout = gluon.nn.Dropout(rate=0.5)
            self.dense0 = gluon.nn.Dense(units=64, activation='relu')
            self.dense1 = gluon.nn.Dense(units=1)

    def forward(self, x):
        user_vecs = nd.flatten(self.user_embedding(x[:, 0]))
        item_vecs = nd.flatten(self.item_embedding(x[:, 1]))
        input_vecs = nd.concat(user_vecs, item_vecs)
        input_vecs_dropout = self.input_dropout(input_vecs)
        dense0_output = self.dense0(input_vecs_dropout)
        return self.dense1(dense0_output)


class CFWithTemporalInformation(gluon.nn.Block):

    def __init__(self, embedding_size, user_input_dim, item_input_dim, **kwargs):
        super(CFWithTemporalInformation, self).__init__(**kwargs)
        with self.name_scope():
            # initializer: random with minval = -0.05, maxval = 0.05
            self.user_embedding = gluon.nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.item_embedding = gluon.nn.Embedding(input_dim=item_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.input_dropout = gluon.nn.Dropout(rate=0.5)
            self.dense0 = gluon.nn.Dense(units=64, activation='relu')
            self.dense1 = gluon.nn.Dense(units=1)

    def forward(self, x):
        user_vecs = nd.flatten(self.user_embedding(x[:, 0]))
        item_vecs = nd.flatten(self.item_embedding(x[:, 1]))
        input_vecs = nd.concat(user_vecs, item_vecs, x[:, 2])
        input_vecs_dropout = self.input_dropout(input_vecs)
        dense0_output = self.dense0(input_vecs_dropout)
        return self.dense1(dense0_output)


class CollaborativeFilteringV1(object):
    model_name = 'cf-v1'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None
        self.embedding_size = 100

    @staticmethod
    def create_model(embedding_size, user_input_dim, item_input_dim):
        return CFV1(embedding_size, user_input_dim, item_input_dim)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-config.npy'

    @staticmethod
    def get_params_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-net.params'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.embedding_size = self.config['embedding_size']
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']

        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        params_file_path = self.get_params_file_path(model_dir_path)
        self.model.load_params(params_file_path, ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def predict_single(self, user_id, item_id):
        data = nd.array([[user_id, item_id]])
        return self.model(data).asscalar()

    def predict(self, user_id_test, item_id_test):
        test_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_test, item_id_test)],
                          ctx=self.data_ctx)
        prediction = self.model(test_x)
        return prediction.asnumpy()

    def evaluate_mae(self, user_id_test, item_id_test, rating_test, batch_size=64):
        test_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_test, item_id_test)],
                          ctx=self.data_ctx)
        test_y = nd.array(rating_test, ctx=self.data_ctx)

        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_x, test_y), batch_size=batch_size,
                                          shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        mae = mx.metric.MAE()
        loss_avg = 0.
        for i, (data, rating) in enumerate(test_data):
            data = data.as_in_context(self.model_ctx)
            rating = rating.as_in_context(self.model_ctx)
            predictions = self.model(data)
            loss = mae_loss(predictions, rating)
            mae.update(preds=predictions, labels=rating)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
        return mae.get()[1], loss_avg

    def fit(self, user_id_train, item_id_train, rating_train, model_dir_path, learning_rate=0.001, batch_size=64,
            epochs=20, checkpoint_interval=5, test_data=None):
        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        self.config = dict()
        self.config['embedding_size'] = self.embedding_size
        self.config['max_user_id'] = self.max_user_id
        self.config['max_item_id'] = self.max_item_id
        np.save(self.get_config_file_path(model_dir_path), self.config)

        train_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_train, item_id_train)],
                           ctx=self.data_ctx)
        train_y = nd.array(rating_train, ctx=self.data_ctx)

        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_x, train_y), batch_size=batch_size,
                                           shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.model_ctx)

        trainer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_params={
            'learning_rate': learning_rate
        })

        num_samples = len(user_id_train)
        batch_count = int(math.ceil(num_samples / batch_size))

        history = dict()
        loss_train_seq = []
        mae_train_seq = []
        loss_test_seq = []
        mae_test_seq = []

        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, rating) in enumerate(train_data):
                data = data.as_in_context(self.model_ctx)
                rating = rating.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    loss = mae_loss(output, rating)
                loss.backward()
                trainer.step(batch_size)
                batch_loss = nd.sum(loss).asscalar()
                batch_avg_loss = batch_loss / data.shape[0]
                cumulative_loss += batch_loss
                print("Epoch %s / %s, Batch %s / %s. Loss: %s" %
                      (e + 1, epochs, i + 1, batch_count, batch_avg_loss))
            train_mae, train_avg_loss = self.evaluate_mae(user_id_train, item_id_train, rating_train,
                                                          batch_size=batch_size)
            mae_train_seq.append(train_mae)
            if test_data is None:
                print("Epoch %s / %s. Loss: %s. MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae))
            else:
                user_id_test, item_id_test, rating_test = test_data

                test_mae, test_avg_loss = self.evaluate_mae(user_id_test, item_id_test, rating_test,
                                                            batch_size=batch_size)
                mae_test_seq.append(test_mae)
                loss_test_seq.append(test_avg_loss)

                print("Epoch %s / %s. Loss: %s. MAE: %s. Test MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae, test_mae))

            if e % checkpoint_interval == 0:
                self.checkpoint(model_dir_path)
            loss_train_seq.append(cumulative_loss)

        self.checkpoint(model_dir_path)

        history['loss_train'] = loss_train_seq
        history['mae_train'] = mae_train_seq
        history['loss_test'] = loss_test_seq
        history['mae_test'] = mae_test_seq
        np.save(model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-train-history.npy', history)

        return history


class CollaborativeFilteringV2(object):
    model_name = 'cf-v2'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None
        self.embedding_size = 100

    @staticmethod
    def create_model(embedding_size, user_input_dim, item_input_dim):
        return CFV2(embedding_size, user_input_dim, item_input_dim)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-config.npy'

    @staticmethod
    def get_params_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-net.params'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.embedding_size = self.config['embedding_size']
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']

        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        params_file_path = self.get_params_file_path(model_dir_path)
        self.model.load_params(params_file_path, ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def predict_single(self, user_id, item_id):
        data = nd.array([[user_id, item_id]])
        return self.model(data)[0].asscalar()

    def predict(self, user_id_test, item_id_test):
        test_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_test, item_id_test)],
                          ctx=self.data_ctx)
        prediction = self.model(test_x)
        return prediction.asnumpy()[0]

    def evaluate_mae(self, user_id_test, item_id_test, rating_test, batch_size=64):
        test_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_test, item_id_test)],
                          ctx=self.data_ctx)
        test_y = nd.array(rating_test, ctx=self.data_ctx)

        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_x, test_y), batch_size=batch_size,
                                          shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        mae = mx.metric.MAE()
        loss_avg = 0.
        for i, (data, rating) in enumerate(test_data):
            data = data.as_in_context(self.model_ctx)
            rating = rating.as_in_context(self.model_ctx)
            predictions = self.model(data)
            loss = mae_loss(predictions, rating)
            mae.update(preds=predictions, labels=rating)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
        return mae.get()[1], loss_avg

    def fit(self, user_id_train, item_id_train, rating_train, model_dir_path, learning_rate=0.001, batch_size=64,
            epochs=20, checkpoint_interval=5, test_data=None):
        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        self.config = dict()
        self.config['embedding_size'] = self.embedding_size
        self.config['max_user_id'] = self.max_user_id
        self.config['max_item_id'] = self.max_item_id
        np.save(self.get_config_file_path(model_dir_path), self.config)

        train_x = nd.array([[user_id, item_id] for user_id, item_id in zip(user_id_train, item_id_train)],
                           ctx=self.data_ctx)
        train_y = nd.array(rating_train, ctx=self.data_ctx)

        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_x, train_y), batch_size=batch_size,
                                           shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.model_ctx)

        trainer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_params={
            'learning_rate': learning_rate
        })

        num_samples = len(user_id_train)
        batch_count = int(math.ceil(num_samples / batch_size))

        history = dict()
        loss_train_seq = []
        mae_train_seq = []
        loss_test_seq = []
        mae_test_seq = []

        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, rating) in enumerate(train_data):
                data = data.as_in_context(self.model_ctx)
                rating = rating.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    loss = mae_loss(output, rating)
                loss.backward()
                trainer.step(batch_size)
                batch_loss = nd.sum(loss).asscalar()
                batch_avg_loss = batch_loss / data.shape[0]
                cumulative_loss += batch_loss
                print("Epoch %s / %s, Batch %s / %s. Loss: %s" %
                      (e + 1, epochs, i + 1, batch_count, batch_avg_loss))
            train_mae, train_avg_loss = self.evaluate_mae(user_id_train, item_id_train, rating_train,
                                                          batch_size=batch_size)
            mae_train_seq.append(train_mae)
            if test_data is None:
                print("Epoch %s / %s. Loss: %s. MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae))
            else:
                user_id_test, item_id_test, rating_test = test_data

                test_mae, test_avg_loss = self.evaluate_mae(user_id_test, item_id_test, rating_test,
                                                            batch_size=batch_size)
                mae_test_seq.append(test_mae)
                loss_test_seq.append(test_avg_loss)

                print("Epoch %s / %s. Loss: %s. MAE: %s. Test MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae, test_mae))

            if e % checkpoint_interval == 0:
                self.checkpoint(model_dir_path)
            loss_train_seq.append(cumulative_loss)

        self.checkpoint(model_dir_path)

        history['loss_train'] = loss_train_seq
        history['mae_train'] = mae_train_seq
        history['loss_test'] = loss_test_seq
        history['mae_test'] = mae_test_seq
        np.save(model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-train-history.npy', history)

        return history


class CollaborativeFilteringWithTemporalInformation(object):
    model_name = 'temporal-cf'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.max_user_id = 0
        self.max_item_id = 0
        self.max_date = 0
        self.min_date = 0
        self.config = None
        self.embedding_size = 100

    @staticmethod
    def create_model(embedding_size, user_input_dim, item_input_dim):
        return CFWithTemporalInformation(embedding_size, user_input_dim, item_input_dim)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-config.npy'

    @staticmethod
    def get_params_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-net.params'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.embedding_size = self.config['embedding_size']
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']
        self.max_date = self.config['max_date']
        self.min_date = self.config['min_date']

        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        params_file_path = self.get_params_file_path(model_dir_path)
        self.model.load_params(params_file_path, ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def predict_single(self, user_id, item_id, timestamp):
        data = nd.array([[user_id, item_id, (timestamp-self.min_date) / (self.max_date - self.min_date)]])
        return self.model(data)[0].asscalar()

    def predict(self, user_id_test, item_id_test, timestamp_test):
        date_range = self.max_date - self.min_date
        test_x = nd.array([[user_id, item_id, (timestamp - self.min_date) / date_range]
                           for user_id, item_id, timestamp in zip(user_id_test, item_id_test, timestamp_test)],
                          ctx=self.data_ctx)
        prediction = self.model(test_x)
        return prediction.asnumpy()[0]

    def evaluate_mae(self, user_id_test, item_id_test, timestamp_test, rating_test, batch_size=64):

        date_range = self.max_date - self.min_date

        test_x = nd.array([[user_id, item_id, (timestamp - self.min_date) / date_range]
                           for user_id, item_id, timestamp in zip(user_id_test, item_id_test, timestamp_test)],
                          ctx=self.data_ctx)
        test_y = nd.array(rating_test, ctx=self.data_ctx)

        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_x, test_y), batch_size=batch_size,
                                          shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        mae = mx.metric.MAE()
        loss_avg = 0.
        for i, (data, rating) in enumerate(test_data):
            data = data.as_in_context(self.model_ctx)
            rating = rating.as_in_context(self.model_ctx)
            predictions = self.model(data)
            loss = mae_loss(predictions, rating)
            mae.update(preds=predictions, labels=rating)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
        return mae.get()[1], loss_avg

    def fit(self, user_id_train, item_id_train, timestamp_train, rating_train, model_dir_path, learning_rate=0.001,
            batch_size=64,
            epochs=20, checkpoint_interval=5, test_data=None):

        self.max_date = max(timestamp_train)
        self.min_date = min(timestamp_train)

        self.model = self.create_model(self.embedding_size, self.max_user_id + 1, self.max_item_id + 1)

        self.config = dict()
        self.config['embedding_size'] = self.embedding_size
        self.config['max_user_id'] = self.max_user_id
        self.config['max_item_id'] = self.max_item_id
        self.config['max_date'] = self.max_date
        self.config['min_date'] = self.min_date
        np.save(self.get_config_file_path(model_dir_path), self.config)

        date_range = self.max_date - self.min_date
        train_x = nd.array([[user_id, item_id, (timestamp - self.min_date) / date_range]
                            for user_id, item_id, timestamp in zip(user_id_train, item_id_train, timestamp_train)],
                           ctx=self.data_ctx)
        train_y = nd.array(rating_train, ctx=self.data_ctx)

        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_x, train_y), batch_size=batch_size,
                                           shuffle=True)

        mae_loss = gluon.loss.L1Loss()

        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.model_ctx)

        trainer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_params={
            'learning_rate': learning_rate
        })

        num_samples = len(user_id_train)
        batch_count = int(math.ceil(num_samples / batch_size))

        history = dict()
        loss_train_seq = []
        mae_train_seq = []
        loss_test_seq = []
        mae_test_seq = []

        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, rating) in enumerate(train_data):
                data = data.as_in_context(self.model_ctx)
                rating = rating.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    loss = mae_loss(output, rating)
                loss.backward()
                trainer.step(batch_size)
                batch_loss = nd.sum(loss).asscalar()
                batch_avg_loss = batch_loss / data.shape[0]
                cumulative_loss += batch_loss
                print("Epoch %s / %s, Batch %s / %s. Loss: %s" %
                      (e + 1, epochs, i + 1, batch_count, batch_avg_loss))
            train_mae, train_avg_loss = self.evaluate_mae(user_id_train, item_id_train, timestamp_train, rating_train,
                                                          batch_size=batch_size)
            mae_train_seq.append(train_mae)
            if test_data is None:
                print("Epoch %s / %s. Loss: %s. MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae))
            else:
                user_id_test, item_id_test, timestamp_test, rating_test = test_data

                test_mae, test_avg_loss = self.evaluate_mae(user_id_test, item_id_test, timestamp_test, rating_test,
                                                            batch_size=batch_size)
                mae_test_seq.append(test_mae)
                loss_test_seq.append(test_avg_loss)

                print("Epoch %s / %s. Loss: %s. MAE: %s. Test MAE: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_mae, test_mae))

            if e % checkpoint_interval == 0:
                self.checkpoint(model_dir_path)
            loss_train_seq.append(cumulative_loss)

        self.checkpoint(model_dir_path)

        history['loss_train'] = loss_train_seq
        history['mae_train'] = mae_train_seq
        history['loss_test'] = loss_test_seq
        history['mae_test'] = mae_test_seq
        np.save(model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-train-history.npy',
                history)

        return history
