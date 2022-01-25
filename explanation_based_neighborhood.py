import pandas as pd
import numpy as np
from PyALE import ale
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from frequency_based_random_sampling import FrequencyBasedRandomSampling
from encoding_utils import *
from sklearn.metrics import pairwise_distances

class ExplanationBasedNeighborhood():
    def __init__(self,
                X,
                y,
                model,
                dataset):

        XX = FrequencyBasedRandomSampling(X, 10000)
        yy = model.predict(XX)

        class_set = np.unique(yy)
        self.class_set = class_set

        self.X = XX
        self.y = yy
        self.model = model
        self.dataset = dataset
        self.discrete_indices = dataset['discrete_indices']
        self.continuous_indices = dataset['continuous_indices']
        self.numerical_width = dataset['feature_width'][dataset['continuous_indices']]

    def categoricalSimilarity(self):

        categorical_similarity = {}
        categorical_width = {}
        categorical_importance = {}
        class_data = {}
        for c in self.class_set:
            categorical_similarity.update({c: {}})
            categorical_width.update({c: {}})
            categorical_importance.update({c: {}})
            class_data.update({c: {}})

        for c in self.class_set:
            ind_c = np.where(self.y==c)[0]
            X_c = self.X[ind_c, :]
            class_data[c] = X_c
            X_df = pd.DataFrame(data= np.r_[self.X, X_c], columns=range(0, self.X.shape[1]))
            for f in self.discrete_indices:

                exp = ale(X_df, self.model, [f], feature_type="discrete", include_CI=True, C=0.95)
                categorical_similarity[c][f] = exp['eff']

                categorical_width[c][f] = max(exp['eff']) - min(exp['eff'])

                categorical_importance[c][f] = max(exp['eff'])

        self.categorical_similarity = categorical_similarity
        self.categorical_width = categorical_width
        self.categorical_importance = categorical_importance
        self.class_data = class_data

    def neighborhoodModel(self):
        models = {}
        for c, X_c in self.class_data.items():
            X_c_ohe = ord2ohe(X_c, self.dataset)
            model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
            model.fit(X_c_ohe)
            models[c] = model
        self.neighborhood_models = models

    def fit(self):
        self.categoricalSimilarity()
        self.neighborhoodModel()


    def cat2numConverter(self,
                          x,
                          feature_list=None):

        if feature_list == None:
            feature_list = self.discrete_indices

        x_exp = x.copy()

        if x_exp.shape.__len__() == 1:
            x_c = self.model.predict(x.reshape(1,-1))[0]
            for f in feature_list:
                x_exp[f] = self.categorical_similarity[x_c][f][x[f]]
        else:
            x_c = self.model.predict(x)
            for f in feature_list:
                vec = x[:,f]
                vec_converted = np.asarray(list(map(lambda c,v: self.categorical_similarity[c][f][v], x_c, vec)))
                x_exp[:,f] = vec_converted
        return x_exp

    def neighborhoodSampling(self, x, N_samples):

        # finding the label of x
        x_c = self.model.predict(x.reshape(1,-1))[0]
        x_ohe = ord2ohe(x, self.dataset)

        # finding the closest neighbors in the other classes
        x_hat = {}
        for c in self.class_set:
            if c == x_c:
                x_hat[c] = x
            else:
                distances, indices = self.neighborhood_models[c].kneighbors(x_ohe.reshape(1, -1))
                x_hat[c] = self.class_data[c][indices[0][0]].copy()

        # converting input samples from categorical to numerical representation
        x_hat_exp = {}
        for c, instance in x_hat.items():
            x_hat_exp[c] = self.cat2numConverter(instance)

        # generating random samples from the distribution of training data
        X_sampled = FrequencyBasedRandomSampling(self.X, N_samples * 5)
        X_sampled_c = self.model.predict(X_sampled)

        # converting random samples from categorical to numerical representation
        X_sampled_exp = self.cat2numConverter(X_sampled)

        # calculating the distance between inputs and the random samples
        distance = np.zeros(X_sampled.shape[0])
        for i, c in enumerate(X_sampled_c):
            feature_width = np.r_[self.numerical_width, np.asarray(list(self.categorical_width[c].values()))]
            dist = (1 / feature_width) * abs(x_hat_exp[c] - X_sampled_exp[i,:])
            distance[i] = np.mean(dist)

        # selecting N_samples based on the calculated distance
        sorted_indices = np.argsort(distance)
        selected_indices = sorted_indices[:N_samples]
        sampled_data = X_sampled[selected_indices, :]

        # merging neighborhood data with x
        neighborhood_data = np.r_[x.reshape(1, -1), sampled_data]

        # predicting the label and probability of the neighborhood data
        neighborhood_labels = self.model.predict(neighborhood_data)
        neighborhood_proba = self.model.predict_proba(neighborhood_data)
        neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]

        return neighborhood_data, neighborhood_labels, neighborhood_proba