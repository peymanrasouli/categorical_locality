import sys
sys.path.append('LORE')
import numpy as np
from LORE.neighbor_generator import *

class RandomSamplingNeighborhood():
    def __init__(self,
                 X,
                 y,
                 model,
                 dataset):
        self.X = X
        self.y = y
        self.model = model
        self.dataset = dataset

    def fit(self):
        dfZ, _ = dataframe2explain(self.X, self.dataset, 0, self.model)
        self.dfZ = dfZ

    def neighborhoodSampling(self, x, N_samples):

        # generating random neighborhood data
        Z_df, Z = random_neighborhood(self.dfZ, x, self.model, self.dataset)
        sampled_data = Z_df[self.dataset['feature_names']].values
        neighborhood_data = np.r_[x.reshape(1, -1), sampled_data]

        # predicting the label and probability of the neighborhood data
        neighborhood_labels = self.model.predict(neighborhood_data)
        neighborhood_proba = self.model.predict_proba(neighborhood_data)
        neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]

        return neighborhood_data, neighborhood_labels, neighborhood_proba
