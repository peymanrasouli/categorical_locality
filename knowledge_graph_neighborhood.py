import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from frequency_based_random_sampling import FrequencyBasedRandomSampling
from domain_knowledge.csv2graph import CSV2GRAPH
from domain_knowledge.graph2embedding import GRAPH2EMBEDDING
from encoding_utils import *

class KnowledgeGraphNeighborhood():
    def __init__(self,
                 X,
                 y,
                 model,
                 dataset):

        # splitting the data into train and test set with the same random state used for training the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.dataset = dataset
        self.feature_names = dataset['feature_names']
        self.class_set = np.unique(y_train)

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
        self.neighborhoodModel()

    def transformCSV2GRAPH(self):

        # path of the rdf graph
        rdf_file = "domain_knowledge/ontologies/adult_ontology.owl"
        csv_file = "domain_knowledge/csv_data/adult_categorical.csv"

        # instantiating the CSV2GRAPH class
        csv2graph = CSV2GRAPH(rdf_file)

        # Create RDF triples
        csv2graph.Convert(csv_file)

        # Graph with only data
        csv2graph.saveGraph("domain_knowledge/ontologies/adult_ontology_instantiated.owl")

    def transformGRAPH2EMBEDDING(self):
        # path of the knowledge graph file
        owl_file = "domain_knowledge/ontologies/adult_ontology_instantiated.owl"

        # instantiating the GRAPH2EMBEDDING class
        self.embedding_model = GRAPH2EMBEDDING(owl_file)

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

        # generating random samples from the distribution of training data
        X_sampled = FrequencyBasedRandomSampling(self.X_train, N_samples * 10)
        X_sampled_c = self.model.predict(X_sampled)

        # creating a csv file from original and random data
        index = []
        data = []
        for idx, inst in x_hat.items():
            index.append('original' + str(idx))
            data.append(ord2org(inst, self.dataset))
        for idx, inst in enumerate(X_sampled):
            index.append('random' + str(idx))
            data.append(ord2org(inst, self.dataset))

        df = pd.DataFrame(data=data,columns=self.feature_names,index=index)
        df.to_csv("domain_knowledge/csv_data/adult_categorical.csv")

        # converting input and random samples to knowledge graph
        self.transformCSV2GRAPH()

        # converting knowledge graph to embeddings
        self.transformGRAPH2EMBEDDING()

        # finding similarity between original input and random data
        similarity_classwise = {}
        for c in self.class_set:
            sim_vec = np.zeros(N_samples * 10)
            instances = self.embedding_model.FindSimilarInstances(instance='original'+str(c), N=N_samples * 20)
            for instance in instances:
                if instance[0].startswith('random'):
                    idx = int(instance[0].split('random',1)[1])
                    sim = instance[1]
                    sim_vec[idx] = sim
            similarity_classwise[c] = sim_vec

        print()

        # calculating the distance between inputs and the random samples


        # distance = np.zeros(X_sampled.shape[0])
        # for i, c in enumerate(X_sampled_c):
        #     feature_width = np.r_[self.numerical_width, np.asarray(list(self.categorical_width[c].values()))]
        #     dist = (1 / feature_width) * abs(x_hat_exp[c] - X_sampled_exp[i,:])
        #     distance[i] = np.mean(dist)
        #
        # # selecting N_samples based on the calculated distance
        # sorted_indices = np.argsort(distance)
        # selected_indices = sorted_indices[:N_samples]
        # sampled_data = X_sampled[selected_indices, :]
        # neighborhood_data = np.r_[x.reshape(1, -1), sampled_data]
        #
        # # predicting the label and probability of the neighborhood data
        # neighborhood_labels = self.model.predict(neighborhood_data)
        # neighborhood_proba = self.model.predict_proba(neighborhood_data)
        # neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]
        #
        # return neighborhood_data, neighborhood_labels, neighborhood_proba
        return 0