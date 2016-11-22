import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


def read_train(filename='data/train.csv'):
    return pd.read_csv(filename, encoding='utf-8', dialect='excel', lineterminator='\n')


def read_test(filename='data/test.csv'):
    return pd.read_csv(filename, encoding='utf-8', dialect='excel', lineterminator='\n')


class Features:
    @staticmethod
    def get_user_lang_feature(data, train_data):
        table = train_data[['user.lang', 'retweet_count']].groupby(by='user.lang').mean()
        table = table.to_dict()['retweet_count']
        return pd.Series.from_array([table.get(x, np.mean(list(table.values()))) for x in data['user.lang']])

    @staticmethod
    def get_text_feature(data, i):
        if i == 0:
            return pd.Series.from_array([x.count('t.co') for x in data['text']])
        elif i == 1:
            return pd.Series.from_array([x.count('@') for x in data['text']])

    @staticmethod
    def get_in_reply_to_user_id_feature(data):
        return pd.Series.from_array([np.bool(x) for x in data['in_reply_to_user_id']])

    @staticmethod
    def get_user_time_zone_feature(data, train_data):
        table = train_data[['user.time_zone', 'retweet_count']].groupby(by='user.time_zone').mean()
        table = table.to_dict()['retweet_count']
        return pd.Series.from_array([table.get(x, np.mean(list(table.values()))) for x in data['user.time_zone']])


def df2features(data, train_data):
    return np.array([
        Features.get_text_feature(data, 0),
        Features.get_text_feature(data, 1),
        Features.get_in_reply_to_user_id_feature(data),
        Features.get_user_lang_feature(data, train_data),
        Features.get_user_time_zone_feature(data, train_data),
        data['user.utc_offset'],
        data['user.statuses_count'],
        data['user.followers_count'],
        data['user.friends_count'],
        data['user.favourites_count'],
        data['user.is_translation_enabled'] * data['user.geo_enabled'],
        data['user.listed_count']
    ]).transpose()


class Models:
    @staticmethod
    def model_1(train_X, test_X, train_y, test_y, real_test_X):
        param_grid = {'min_samples_leaf': [i for i in range(10, 101, 10)]}
        est = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=4)
        est.fit(train_X, train_y)

        proba = est.predict_proba(train_X)
        print(roc_auc_score(train_y, proba[:, 1]))

        proba = est.predict_proba(test_X)
        print(roc_auc_score(test_y, proba[:, 1]))

        proba = est.predict_proba(real_test_X)
        return proba[:, 1]

    @staticmethod
    def model_2(train_X, test_X, train_y, test_y, real_test_X):
        est = SVC(probability=True, verbose=True)
        est.fit(train_X, train_y)

        proba = est.predict_proba(train_X)
        print(roc_auc_score(train_y, proba[:, 1]))

        proba = est.predict_proba(test_X)
        print(roc_auc_score(test_y, proba[:, 1]))

        proba = est.predict_proba(real_test_X)
        return proba[:, 1]


if __name__ == '__main__':
    data, test_data = read_train(), read_test()
    data_y = data['retweet_count'] > 20
    train_X, test_X, train_y, test_y = train_test_split(data, data_y, test_size=0.33)
    train_X, test_X, real_test_X = df2features(train_X, train_X), df2features(test_X, train_X), df2features(test_data, train_X)

    proba = Models.model_1(train_X, test_X, train_y, test_y, real_test_X)
    proba = Models.model_2(train_X, test_X, train_y, test_y, real_test_X)

    prediction = pd.DataFrame(data={'id': test_data['id'], 'probability': proba})
    prediction.to_csv('data/prediction.csv', index=False)
    print(prediction.head())
