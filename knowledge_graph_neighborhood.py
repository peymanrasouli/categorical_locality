import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from frequency_based_random_sampling import FrequencyBasedRandomSampling
from alibi.explainers import ALE
from encoding_utils import *

class KnowledgeGraphNeighborhood():
    def __init__(self,
                 X,
                 y,
                 model,
                 dataset):

        # splitting the data into train and test set with the same random state used for training the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # check whether the training data contains all possible values for the features; add extra samples in case
        for f in range(X_train.shape[1]):
            for fv in dataset['feature_values'][f]:
                if fv in np.unique(X_train[:,f]):
                    pass
                else:
                    idx = np.where(X_test[:, f] == fv)[0][0]
                    X_train = np.r_[X_train, X_test[idx, :].reshape(1,-1)]
                    y_train = np.r_[y_train, y_test[idx]]

        self.X_train = X_train
        self.y_train = model.predict(X_train)
        self.model = model
        self.dataset = dataset
        self.discrete_indices = dataset['discrete_indices']
        self.class_set = np.unique(y_train)

    def categoricalSimilarity(self):

        # initializing the variables
        categorical_similarity = {}
        categorical_width = {}
        categorical_importance = {}
        for c in self.class_set:
            categorical_similarity.update({c: {}})
            categorical_width.update({c: {}})
            categorical_importance.update({c: {}})

        # creating ALE explainer
        ale_explainer = ALE(self.model.predict_proba,
                            feature_names=self.discrete_indices,
                            target_names=self.class_set,
                            low_resolution_threshold=100)
        ale_exp = ale_explainer.explain(self.X_train)

        # extracting global effect values
        for c in self.class_set:
            for f in self.discrete_indices:
                categorical_similarity[c][f] = pd.Series(ale_exp.ale_values[f][:,c])
                categorical_width[c][f] = max(ale_exp.ale_values[f][:,c]) - min(ale_exp.ale_values[f][:,c])
                categorical_importance[c][f] = max(ale_exp.ale_values[f][:,c])

        # returning the results
        self.categorical_similarity = categorical_similarity
        self.categorical_width = categorical_width
        self.categorical_importance = categorical_importance

    def neighborhoodModel(self):

        # creating neighborhood models based on class-wise ground-truth data
        class_data = {}
        for c in self.class_set:
            class_data.update({c: {}})

        class_data = {}
        models = {}
        for c in self.class_set:
            ind_c = np.where(self.y_train == c)[0]
            X_c = self.X_train[ind_c, :]
            class_data[c] = X_c
            X_c_ohe = ord2ohe(X_c, self.dataset)
            model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='matching')
            model.fit(X_c_ohe)
            models[c] = model
        self.class_data = class_data
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
        distance_hat = {}
        x_ohe = ord2ohe(x, self.dataset)
        for c in self.class_set:
            if c == x_c:
                x_hat[c] = x
                distance_hat[c] = (x_hat[c] !=  x_hat[c]).astype(int)
            else:
                distances, indices = self.neighborhood_models[c].kneighbors(x_ohe.reshape(1, -1))
                x_hat[c] = self.class_data[c][indices[0][0]].copy()
                distance_hat[c] =  (x_hat[c] !=  x).astype(int)

        # converting input samples from categorical to numerical representation
        x_hat_exp = {}
        for c, instance in x_hat.items():
            x_hat_exp[c] = self.cat2numConverter(instance)

        # generating random samples from the distribution of training data
        X_sampled = FrequencyBasedRandomSampling(self.X_train, N_samples * 5)
        X_sampled_c = self.model.predict(X_sampled)

        # converting random samples from categorical to numerical representation
        X_sampled_exp = self.cat2numConverter(X_sampled)

        # calculating the distance between inputs and the random samples
        distance = np.zeros(X_sampled.shape[0])
        for i, c in enumerate(X_sampled_c):
            feature_width = np.asarray(list(self.categorical_width[c].values()))
            dist = (abs(x_hat_exp[c] - X_sampled_exp[i,:]))# + (x_hat[c] != X_sampled[i,:]).astype(int)

            # dist = (x_hat[c] != X_sampled[i,:]).astype(int)

            distance[i] = np.mean(dist) + np.mean(distance_hat[c])

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


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors
# from frequency_based_random_sampling import FrequencyBasedRandomSampling
# from domain_knowledge.csv2graph import CSV2GRAPH
# from domain_knowledge.graph2embedding import GRAPH2EMBEDDING
# from encoding_utils import *
#
# class KnowledgeGraphNeighborhood():
#     def __init__(self,
#                  X,
#                  y,
#                  model,
#                  dataset):
#
#         # splitting the data into train and test set with the same random state used for training the model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         self.X_train = X_train
#         self.y_train = y_train
#         self.model = model
#         self.dataset = dataset
#         self.feature_names = dataset['feature_names']
#         self.class_set = np.unique(y_train)
#
#     def neighborhoodModel(self):
#
#         # creating neighborhood models based on class-wise ground-truth data
#         class_data = {}
#         for c in self.class_set:
#             class_data.update({c: {}})
#
#         class_data = {}
#         models = {}
#         for c in self.class_set:
#             ind_c = np.where(self.y_train == c)[0]
#             X_c = self.X_train[ind_c, :]
#             class_data[c] = X_c
#             X_c_ohe = ord2ohe(X_c, self.dataset)
#             model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='matching')
#             model.fit(X_c_ohe)
#             models[c] = model
#         self.class_data = class_data
#         self.neighborhood_models = models
#
#     def fit(self):
#         self.neighborhoodModel()
#
#     def transformCSV2GRAPH(self):
#
#         # path of the rdf graph
#         rdf_file = "domain_knowledge/ontologies/adult_ontology.owl"
#         csv_file = "domain_knowledge/csv_data/adult_categorical.csv"
#
#         # instantiating the CSV2GRAPH class
#         csv2graph = CSV2GRAPH(rdf_file)
#
#         # Create RDF triples
#         csv2graph.Convert(csv_file)
#
#         # Graph with only data
#         csv2graph.saveGraph("domain_knowledge/ontologies/adult_ontology_instantiated.owl")
#
#     def transformGRAPH2EMBEDDING(self):
#         # path of the knowledge graph file
#         owl_file = "domain_knowledge/ontologies/adult_ontology_instantiated.owl"
#
#         # instantiating the GRAPH2EMBEDDING class
#         self.embedding_model = GRAPH2EMBEDDING(owl_file)
#
#     def neighborhoodSampling(self, x, N_samples):
#
#         # finding the label of x
#         x_c = self.model.predict(x.reshape(1,-1))[0]
#
#         # finding the closest neighbors in the other classes
#         x_hat = {}
#         x_ohe = ord2ohe(x, self.dataset)
#         for c in self.class_set:
#             if c == x_c:
#                 x_hat[c] = x
#             else:
#                 distances, indices = self.neighborhood_models[c].kneighbors(x_ohe.reshape(1, -1))
#                 x_hat[c] = self.class_data[c][indices[0][0]].copy()
#
#         # generating random samples from the distribution of training data
#         X_sampled = FrequencyBasedRandomSampling(self.X_train, N_samples * 10)
#         X_sampled_c = self.model.predict(X_sampled)
#
#         # creating a csv file from original and random data
#         index = []
#         data = []
#         for idx, inst in x_hat.items():
#             index.append('original' + str(idx))
#             data.append(ord2org(inst, self.dataset))
#         for idx, inst in enumerate(X_sampled):
#             index.append('random' + str(idx))
#             data.append(ord2org(inst, self.dataset))
#
#         df = pd.DataFrame(data=data,columns=self.feature_names,index=index)
#         df.to_csv("domain_knowledge/csv_data/adult_categorical.csv")
#
#         # converting input and random samples to knowledge graph
#         self.transformCSV2GRAPH()
#
#         # converting knowledge graph to embeddings
#         self.transformGRAPH2EMBEDDING()
#
#         # finding similarity between original input and random data
#         similarity_classwise = {}
#         for c in self.class_set:
#             sim_vec = np.zeros(N_samples * 10)
#             instances = self.embedding_model.FindSimilarInstances(instance='original'+str(c), N=N_samples * 20)
#             for instance in instances:
#                 if instance[0].startswith('random'):
#                     idx = int(instance[0].split('random',1)[1])
#                     sim = instance[1]
#                     sim_vec[idx] = sim
#             similarity_classwise[c] = sim_vec
#
#         # obtaining similarity between inputs and the random samples
#         similarity = np.zeros(X_sampled.shape[0])
#         for i, c in enumerate(X_sampled_c):
#             similarity[i] = similarity_classwise[c][i]
#
#         # selecting N_samples based on the calculated similarity
#         sorted_indices = np.argsort(-similarity)
#         selected_indices = sorted_indices[:N_samples]
#         sampled_data = X_sampled[selected_indices, :]
#         neighborhood_data = np.r_[x.reshape(1, -1), sampled_data]
#
#         # predicting the label and probability of the neighborhood data
#         neighborhood_labels = self.model.predict(neighborhood_data)
#         neighborhood_proba = self.model.predict_proba(neighborhood_data)
#         neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]
#
#         return neighborhood_data, neighborhood_labels, neighborhood_proba