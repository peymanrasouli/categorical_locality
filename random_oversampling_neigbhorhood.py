import sys
sys.path.append('LORE')
from LORE.neighbor_generator import *

class RandomOversamplingNeighborhood():
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

        # generating random oversampling neighborhood data
        Z_df, Z = random_oversampling(self.dfZ, x, self.model, self.dataset)
        neighborhood_data = Z_df[self.dataset['feature_names']].values

        # predicting the label and probability of the neighborhood data
        neighborhood_labels = self.model.predict(neighborhood_data)
        neighborhood_proba = self.model.predict_proba(neighborhood_data)
        neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]

        return neighborhood_data, neighborhood_labels, neighborhood_proba
