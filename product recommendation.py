import pandas as pd
import numpy as np
import time
import graphlab as gl
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
# import scripts.data_layer as data_layer

customer = pd.read_csv('custid.csv')
transaction = pd.read_csv('data.csv')

print(customer.head())
print(transaction.head())

print(customer.describe())
print(transaction.describe())

transaction['products'] = transaction['products'].apply(lambda x: [int(i) for i in x.split('|')])
transaction.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index()

# pd.melt(transaction.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index(),
#              id_vars=['customerId'],
#              value_name='products') \
#     .dropna().drop(['variable'], axis=1) \
#     .groupby(['customerId', 'products']) \
#     .agg({'products': 'count'}) \
#     .rename(columns={'products': 'purchase_count'}) \
#     .reset_index() \
#     .rename(columns={'products': 'productId'})

s = time.time()

data = pd.melt(transaction.set_index('customerId')['products'].apply(pd.Series).reset_index(),
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})
data['productId'] = data['productId'].astype(np.int64)

print("Execution time:", round((time.time()-s)/60,2), "minutes")

print(data.shape)
print(data.head())


def create_data_dummy(data):

    data_dummy = data.copy()
    data_dummy['purchase_count'] = 1

    return data_dummy


data_dummy = create_data_dummy(data)


def normalize_data(data):

    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())

    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']

    return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()


data_norm = normalize_data(data)

print(data_norm.head())

# gl.get_dependencies()


def split_data(data):

    train, test = train_test_split(data, test_size=.2)
    train_data = gl.SFrame(train)
    test_data = gl.SFrame(test)
    return train_data, test_data


train_data, test_data = split_data(data)
train_data_norm, test_data_norm = split_data(data_norm)


def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'Popularity':
        model = gl.popularity_recommender.create(train_data, user_id=user_id, item_id=item_id, target=target)

    elif name == 'Similarity':
        model = gl.item_similarity_recommender.create(train_data, user_id=user_id, item_id=item_id, target=target)

    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model


user_id = 'customerId'
item_id = 'productId'
users_to_recommend = list(customer[user_id])
n_rec = 10
n_display = 30

# These items are products with the highest number of sells across customers
print('Product Recommender Using Popularity Model')
name = 'Popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)
print(pop_norm)

print('Product Recommender Using Collaborative Model')
name = 'Similarity'
target = 'purchase_count'
sim_norm = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)
print(sim_norm)
