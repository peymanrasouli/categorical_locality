import collections
import numpy as np

def RandomSampling(training_data, num_samples):


    # Collecting values and frequency of values for every feature
    feature_values = {}
    feature_frequencies = {}
    feature_list = list(range(training_data.shape[1]))
    for feature in feature_list:
        column = training_data[:, feature]
        feature_count = collections.Counter(column)
        values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
        feature_values[feature] = values
        feature_frequencies[feature] = (np.array(frequencies) / float(sum(frequencies)))

    # Generating random data for every feature
    np.random.seed(0)
    random_samples = np.zeros([num_samples, training_data.shape[1]])
    for column in feature_list:
        values = feature_values[column]
        frequencies = feature_frequencies[column]
        inverse_column = np.random.choice(values, size=num_samples, replace=True, p=frequencies)
        random_samples[:, column] = inverse_column

    return random_samples