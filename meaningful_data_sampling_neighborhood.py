import sys
sys.path.append('MDS')
from MDS.meaningful_sampling import MeaningfulSampling

class MeaningfulDataSamplingNeighborhood():
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
        pass

    def neighborhoodSampling(self, x, N_samples):

        # generating random neighborhood data
        _ , neighborhood_data = MeaningfulSampling(x, self.model, self.X, N_samples)

        # predicting the label and probability of the neighborhood data
        neighborhood_labels = self.model.predict(neighborhood_data)
        neighborhood_proba = self.model.predict_proba(neighborhood_data)
        neighborhood_proba = neighborhood_proba[:, neighborhood_labels[0]]

        return neighborhood_data, neighborhood_labels, neighborhood_proba
