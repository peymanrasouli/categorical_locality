from prepare_datasets import *
from create_model import CreateModel
from result_format import resultFormat
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from explanation_based_neighborhood import ExplanationBasedNeighborhood
from random_sampling_neighborhood import RandomSamplingNeighborhood
from random_oversampling_neigbhorhood import RandomOversamplingNeighborhood
from random_instance_selection_neighborhood import RandomInstanceSelectionNeighborhood
from genetic_neighborhood import GeneticNeighborhood
from meaningful_data_sampling_neighborhood import MeaningfulDataSamplingNeighborhood
from sklearn.preprocessing import OneHotEncoder
from console_progressbar.progressbar import ProgressBar
import warnings
warnings.filterwarnings('ignore')

def forward_selection(data, labels, N_features, ohe_encoder=None):
    clf = Ridge()
    used_features = []
    for _ in range(min(N_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue

            data_ohe = []
            for f in  used_features + [feature]:
                if ohe_encoder[f] is None:
                    data_ohe.append(data[:, f].reshape(-1, 1))
                else:
                    data_ohe.append(ohe_encoder[f].transform(data[:,f].reshape(-1, 1)))
            data_ohe = np.hstack(data_ohe)

            clf.fit(data_ohe,
                    labels)
            score = clf.score(data_ohe,
                              labels)
            if score > max_:
                best = feature
                max_ = score
        used_features.append(best)
    return np.array(used_features)

def interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba, N_features=5, ohe_encoder=None):

    used_features = forward_selection(neighborhood_data, neighborhood_proba, N_features, ohe_encoder)
    data_ohe = []
    for f in used_features:
        if ohe_encoder[f] is None:
            data_ohe.append(neighborhood_data[:, f].reshape(-1, 1))
        else:
            data_ohe.append(ohe_encoder[f].transform(neighborhood_data[:, f].reshape(-1, 1)))
    data_ohe = np.hstack(data_ohe)
    lr = Ridge(random_state=42)
    lr.fit(data_ohe, neighborhood_proba)
    lr_preds = lr.predict(data_ohe)
    local_model_pred = float(lr.predict(data_ohe[0, :].reshape(1, -1)))
    local_model_score = r2_score(neighborhood_proba, lr_preds)
    return local_model_pred, local_model_score

def data_sampling(sampling_method, instance2explain, N_samples=1000):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = sampling_method(instance2explain, N_samples)

    return neighborhood_data, neighborhood_labels, neighborhood_proba


def explain_instance(instance2explain, N_samples=1000, N_features=5, ohe_encoder = None, sampling_method=None):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = data_sampling(sampling_method, instance2explain, N_samples)

    local_model_pred, \
    local_model_score = interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba,
                                            N_features=N_features, ohe_encoder=ohe_encoder)

    return local_model_pred, local_model_score

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult),
        'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPAS),
        'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefault),
        'german-credit': ('german-credit.csv', PrepareGermanCredit),
        'breast-cancer': ('breast-cancer.data', PrepareBreastCancer),
        'heart-disease': ('heart-disease.csv', PrepareHeartDisease),
        'nursery': ('nursery.data', PrepareNursery),
        'car': ('car.data', PrepareCar),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        'gb': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        'svm': SVC
    }

    # defining the number of neighborhood samples
    N_samples = {
        'adult': 500,
        'compas-scores-two-years': 500,
        'credit-card-default': 500,
        'german-credit': 500,
        'breast-cancer': 500,
        'heart-disease': 500,
        'nursery': 500,
        'car':500,
    }

    # defining the number of selected features for explanation
    N_features = {
        'adult': 5,
        'compas-scores-two-years': 5,
        'credit-card-default': 5,
        'german-credit': 5,
        'breast-cancer': 5,
        'heart-disease': 5,
        'nursery': 5,
        'car':5,
    }

    # creating a comprehensive dictionary for storing the results
    results = resultFormat(type_format='lr')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)

        # reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # creating one-hot encoder for discrete features
        ohe_encoder = {}
        for f in range(X.shape[1]):
            if f in dataset['discrete_indices']:
                enc = OneHotEncoder(sparse=False)
                enc.fit(X[:,f].reshape(-1, 1))
                ohe_encoder[f] = enc
            else:
                ohe_encoder[f] = None

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)

            # creating sampling methods
            sampling_methods = {}

            # explanation based neighborhood
            exp = ExplanationBasedNeighborhood(X, y, blackbox, dataset)
            exp.fit()
            sampling_methods['exp'] = exp.neighborhoodSampling

            # random neighborhood
            rnd = RandomSamplingNeighborhood(X, y, blackbox, dataset)
            rnd.fit()
            sampling_methods['rnd'] = rnd.neighborhoodSampling

            # random oversampling neighborhood
            ros = RandomOversamplingNeighborhood(X, y, blackbox, dataset)
            ros.fit()
            sampling_methods['ros'] = ros.neighborhoodSampling

            # random instance selection neighborhood
            ris = RandomInstanceSelectionNeighborhood(X, y, blackbox, dataset)
            ris.fit()
            sampling_methods['ris'] = ris.neighborhoodSampling

            # random genetic neighborhood
            gp = GeneticNeighborhood(X, y, blackbox, dataset)
            gp.fit()
            sampling_methods['gp'] = gp.neighborhoodSampling

            # meaningful data sampling neighborhood
            mds = MeaningfulDataSamplingNeighborhood(X, y, blackbox, dataset)
            mds.fit()
            sampling_methods['mds'] = mds.neighborhoodSampling

            # Generating explanations for the samples in the explain set
            methods_output = {'exp': {'local_model_pred':[], 'local_model_score':[]},
                              'rnd': {'local_model_pred': [], 'local_model_score': []},
                              'ros': {'local_model_pred':[], 'local_model_score':[]},
                              'ris': {'local_model_pred':[], 'local_model_score':[]},
                              'gp': {'local_model_pred': [], 'local_model_score': []},
                              'mds': {'local_model_pred': [], 'local_model_score': []}
                              }

            # setting the number of explained instances
            N_explain = min(X_test.shape[0], 300)

            # explaining instances
            pb = ProgressBar(total=N_explain, prefix='Progress:', suffix='Complete', decimals=1, length=50,
                             fill='█', zfill='-')
            X_explain = []
            i = 0
            while i < N_explain:
                try:
                    for method, output in methods_output.items():
                        local_model_pred, \
                        local_model_score = explain_instance(X_test[i, :],
                                                             N_samples=N_samples[dataset_kw],
                                                             N_features=N_features[dataset_kw],
                                                             ohe_encoder=ohe_encoder,
                                                             sampling_method=sampling_methods[method])
                        methods_output[method]['local_model_pred'].append(local_model_pred)
                        methods_output[method]['local_model_score'].append(local_model_score)
                    X_explain.append(X_test[i, :])
                    i += 1
                    pb.print_progress_bar(i)
                except Exception:
                    pass
                if i == X_test.shape[0]:
                    break

            # calculating the performance of different sampling strategy
            X_explain = np.vstack(X_explain)
            prediction = blackbox.predict_proba(X_explain)
            bb_pred = np.max(prediction, axis=1)
            for method, output in methods_output.items():
                # R2 score
                local_r2 = r2_score(bb_pred, np.asarray(output['local_model_pred']))
                results[dataset_kw][blackbox_name][method]['r2_score'] = local_r2

                # MAE
                mae = mean_absolute_error(bb_pred, np.asarray(output['local_model_pred']))
                results[dataset_kw][blackbox_name][method]['mae'] = mae

                # MSE
                mse = mean_squared_error(bb_pred, np.asarray(output['local_model_pred']))
                results[dataset_kw][blackbox_name][method]['mse'] = mse

                # Average score of local model
                avg_local_model_score = np.mean(np.asarray(output['local_model_score']))
                results[dataset_kw][blackbox_name][method]['model_score'] = avg_local_model_score

            for key, val in results[dataset_kw][blackbox_name].items():
                print(key, ':', val)

if __name__ == '__main__':
    main()