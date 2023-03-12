# import os
import json
import numpy as np

num_class = 4

with open('../data/femnist-data-clustered-alt/test/data.json', 'r') as json_testfile:
    dataset_test = json.load(json_testfile)
    users = dataset_test['users']

for i, j in enumerate(users):
    X = np.array(dataset_test['user_data'][j]['x'])
    Y = np.array(dataset_test['user_data'][j]['y'])
    if i <= 0:
        temp = X
        temp_ = Y
    else:
        X_test = np.concatenate((temp, X), axis=0)
        y_test = np.concatenate((temp_, Y), axis=0)
        temp = X_test
        temp_ = y_test
    print("i:", i)

np.save('fmnist_test.npy', {'X': X_test, 'y': y_test}, allow_pickle=True)

with open('../data/femnist-data-clustered-alt/train/data.json', 'r') as json_trainfile:
    dataset_train = json.load(json_trainfile)
    users = dataset_train['users']

for i, j in enumerate(users):
    X = np.array(dataset_train['user_data'][j]['x'])
    Y = np.array(dataset_train['user_data'][j]['y'])
    if i <= 0:
        temp = X
        temp_ = Y
    else:
        X_train = np.concatenate((temp, X), axis=0)
        y_train = np.concatenate((temp_, Y), axis=0)
        temp = X_train
        temp_ = y_train
    print("i:", i)

np.save('fmnist_train.npy', {'X': X_train, 'y': y_train}, allow_pickle=True)

# if not os.path.exists(f'{num_class}_kmeans.npy'):
#     label_test = np.array(dataset_test["cluster_ids"])
#     label_train = np.array(dataset_train["cluster_ids"])
#     label = np.concatenate((label_test, label_train), axis=0)
#     np.save(f'{num_class}_kmeans.npy', label)
# else:
#     label = np.load(f'{num_class}_kmeans.npy')
#
# if not os.path.exists(f'{num_class}_users.npy'):
#     users = np.array(dataset['users'])
#     np.save(f'{num_class}_users.npy', users)
# else:
#     users = np.load(f'{num_class}_users.npy')
