import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def read_train(filename='data/train.csv'):
    return pd.read_csv(filename, encoding='utf-8', dialect='excel', lineterminator='\n')


def read_test(filename='data/test.csv'):
    return pd.read_csv(filename, encoding='utf-8', dialect='excel', lineterminator='\n')


class Features:
    @staticmethod
    def get_user_lang_feature(data):
        table = data[['user.lang', 'retweet_count']].groupby(by='user.lang').mean()
        table = table.to_dict()['retweet_count']
        return pd.Series.from_array([table[x] for x in data['user.lang']])

    @staticmethod
    def get_text_feature(data, i):
        if i == 0:
            return pd.Series.from_array([x.count('t.co') for x in data['text']])
        elif i == 1:
            return pd.Series.from_array([x.count('@') for x in data['text']])


def df2features(data):
    return np.array([
        Features.get_text_feature(data, 0),
        data['in_reply_to_user_id'],
        Features.get_text_feature(data, 1),
        Features.get_user_lang_feature(data),
        data['user.utc_offset'],
        data['user.statuses_count'],
        data['user.followers_count'],
        data['user.friends_count'],
        data['user.favourites_count'],
        data['user.listed_count']
    ]).transpose()


if __name__ == '__main__':
    data = read_train()
    xs, ys = df2features(data), data['retweet_count'] > 20
    train_X, test_X, train_y, test_y = train_test_split(xs, ys, test_size=0.33)

    param_grid = {'min_samples_leaf': [i for i in range(10, 101, 10)]}
    est = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=4)
    est.fit(train_X, train_y)

    proba = est.predict_proba(train_X)
    print(roc_auc_score(train_y, proba[:,1]))

    proba = est.predict_proba(test_X)
    print(roc_auc_score(test_y, proba[:, 1]))


    # prediction = pd.read_csv('sample_prediction.csv')
    # prediction.head()
    #
    # proba = est.predict_proba(test_X)
    # prediction['probability'] = proba[:,1]
    # prediction.to_csv('prediction.csv', index=False)
    # prediction.head()

