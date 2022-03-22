import pandas as pd
from sklearn.neighbors import NearestNeighbors
from frequency_based_random_sampling import FrequencyBasedRandomSampling
from alibi.explainers import ALE
from encoding_utils import *

class ExplanationBasedNeighborhood():
    def __init__(self,
                X,
                y,
                model,
                dataset,
                N_samples):

        self.X = X # FrequencyBasedRandomSampling(X, N_samples * 20)
        self.y = y # model.predict(self.X)
        self.model = model
        self.dataset = dataset
        self.discrete_indices = dataset['discrete_indices']
        self.continuous_indices = dataset['continuous_indices']
        self.numerical_width = dataset['feature_width'][dataset['continuous_indices']]
        self.class_set = np.unique(y)

    def categoricalSimilarity(self):

        # initializing the variables
        categorical_similarity = {}
        categorical_width = {}
        categorical_importance = {}
        class_data = {}
        for c in self.class_set:
            categorical_similarity.update({c: {}})
            categorical_width.update({c: {}})
            categorical_importance.update({c: {}})
            class_data.update({c: {}})

        # creating ALE explainer
        ale_explainer = ALE(self.model.predict_proba,
                            feature_names=self.discrete_indices,
                            target_names=self.class_set,
                            low_resolution_threshold=100)
        ale_exp = ale_explainer.explain(self.X)
        # plot_ale(ale_exp)

        # extracting global effect values
        for c in self.class_set:
            ind_c = np.where(self.y==c)[0]
            X_c = self.X[ind_c, :]
            class_data[c] = X_c
            for f in self.discrete_indices:
                categorical_similarity[c][f] = pd.Series(ale_exp.ale_values[f][:,c])
                categorical_width[c][f] = max(ale_exp.ale_values[f][:,c]) - min(ale_exp.ale_values[f][:,c])
                categorical_importance[c][f] = max(ale_exp.ale_values[f][:,c])

        # returning the results
        self.categorical_similarity = categorical_similarity
        self.categorical_width = categorical_width
        self.categorical_importance = categorical_importance
        self.class_data = class_data

    def neighborhoodModel(self):
        models = {}
        for c, X_c in self.class_data.items():
            X_c_ohe = ord2ohe(X_c, self.dataset)
            model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='matching')
            model.fit(X_c_ohe)
            models[c] = model
        self.neighborhood_models = models

    def fit(self):
        self.categoricalSimilarity()
        self.neighborhoodModel()

    def cat2numConverter(self,
                          x,
                          feature_list=None):

        # converting features in categorical representation to explanation representation
        if feature_list == None:
            feature_list = self.discrete_indices

        x_exp = x.copy()
        if x_exp.shape.__len__() == 1:
            # the input is a single instance
            x_c = self.model.predict(x.reshape(1,-1))[0]
            for f in feature_list:
                x_exp[f] = self.categorical_similarity[x_c][f][x[f]]
        else:
            # the input is a matrix of instances
            x_c = self.model.predict(x)
            for f in feature_list:
                vec = x[:,f]
                vec_converted = np.asarray(list(map(lambda c,v: self.categorical_similarity[c][f][v], x_c, vec)))
                x_exp[:,f] = vec_converted
        return x_exp

    def neighborhoodSampling(self, x, N_samples):

        # finding the label of x
        x_c = self.model.predict(x.reshape(1,-1))[0]

        # finding the closest neighbors in the other classes
        x_hat = {}
        x_ohe = ord2ohe(x, self.dataset)
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
        X_sampled = FrequencyBasedRandomSampling(self.X, N_samples * 10)
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
        neighborhood_data = np.r_[x.reshape(1, -1), sampled_data]

        # predicting the label and probability of the neighborhood data
        neighborhood_labels = self.model.predict(neighborhood_data)
        neighborhood_proba = self.model.predict_proba(neighborhood_data)
        neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]

        return neighborhood_data, neighborhood_labels, neighborhood_proba