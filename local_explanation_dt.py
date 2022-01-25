from prepare_datasets import *
from create_model import CreateModel
from result_format import resultFormat
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from explanation_based_neighborhood import ExplanationBasedNeighborhood
from random_sampling_neighborhood import RandomSamplingNeighborhood
from random_oversampling_neigbhorhood import RandomOversamplingNeighborhood
from random_instance_selection_neighborhood import RandomInstanceSelectionNeighborhood
from genetic_neighborhood import GeneticNeighborhood
from meaningful_data_sampling_neighborhood import MeaningfulDataSamplingNeighborhood
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import _tree
import warnings
warnings.filterwarnings('ignore')


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules

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
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    data_ohe = []
    for f in used_features:
        if ohe_encoder[f] is None:
            data_ohe.append(neighborhood_data[:, f].reshape(-1, 1))
        else:
            data_ohe.append(ohe_encoder[f].transform(neighborhood_data[:, f].reshape(-1, 1)))
    data_ohe = np.hstack(data_ohe)
    dt.fit(data_ohe, neighborhood_labels)
    dt_labels = dt.predict(data_ohe)
    local_model_pred = int(dt.predict(data_ohe[0,:].reshape(1, -1)))
    local_model_score = f1_score(neighborhood_labels, dt_labels, average='weighted')
    # rules = get_rules(dt, range(data_ohe.shape[1]), 'class')
    # for r in rules:
    #     print(r)
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
        # 'breast-cancer': ('breast-cancer.data', PrepareBreastCancer),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease),
        # 'nursery': ('nursery.data', PrepareNursery),
        # 'car': ('car.data', PrepareCar),
        # 'wine': ('wine.data', PrepareWine),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        # 'gb': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        # 'svm': SVC
    }

    N_samples = {
        'adult': 1000,
        'compas-scores-two-years': 1000,
        'credit-card-default': 1000,
        'german-credit': 1000,
        'heart-disease': 1000,
        'breast-cancer': 1000,
        'nursery': 1000,
        'car':1000,
        'wine':1000
    }

    N_features = {
        'adult': 5,
        'compas-scores-two-years': 5,
        'credit-card-default': 5,
        'german-credit': 5,
        'heart-disease': 5,
        'breast-cancer': 5,
        'nursery': 5,
        'car':5,
        'wine':5
    }


    # creating a comprehensive dictionary for storing the results
    results = resultFormat(type_format='dt')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)

        # reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_explain = X_test[:min(X_test.shape[0], 100),:]

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

            # # random neighborhood
            # rnd = RandomSamplingNeighborhood(X, y, blackbox, dataset)
            # rnd.fit()
            # sampling_methods['rnd'] = rnd.neighborhoodSampling
            #
            # # random oversampling neighborhood
            # ros = RandomOversamplingNeighborhood(X, y, blackbox, dataset)
            # ros.fit()
            # sampling_methods['ros'] = ros.neighborhoodSampling
            #
            # # random instance selection neighborhood
            # ris = RandomInstanceSelectionNeighborhood(X, y, blackbox, dataset)
            # ris.fit()
            # sampling_methods['ris'] = ris.neighborhoodSampling

            # # random genetic neighborhood
            # gp = GeneticNeighborhood(X, y, blackbox, dataset)
            # gp.fit()
            # sampling_methods['gp'] = gp.neighborhoodSampling

            # # meaningful data sampling neighborhood
            # mds = MeaningfulDataSamplingNeighborhood(X, y, blackbox, dataset)
            # mds.fit()
            # sampling_methods['mds'] = mds.neighborhoodSampling

            # Generating explanations for the samples in the explain set
            methods_output = {'exp': {'local_model_pred':[], 'local_model_score':[]},
                              # 'rnd': {'local_model_pred': [], 'local_model_score': []},
                              # 'ros': {'local_model_pred':[], 'local_model_score':[]},
                              # 'ris': {'local_model_pred':[], 'local_model_score':[]},
                              # 'gp': {'local_model_pred': [], 'local_model_score': []},
                              # 'mds': {'local_model_pred': [], 'local_model_score': []}
                              }

            for x in X_explain:
                for method, output in methods_output.items():
                    local_model_pred, \
                    local_model_score = explain_instance(x,
                                                         N_samples=N_samples[dataset_kw],
                                                         N_features=N_features[dataset_kw],
                                                         ohe_encoder=ohe_encoder,
                                                         sampling_method=sampling_methods[method])
                    methods_output[method]['local_model_pred'].append(local_model_pred)
                    methods_output[method]['local_model_score'].append(local_model_score)

            # calculating the performance of different sampling strategy
            bb_pred = blackbox.predict(X_explain)
            for method, output in methods_output.items():
                # F1 score
                local_f1 = f1_score(bb_pred, np.asarray(output['local_model_pred']), average='weighted')
                results[dataset_kw][blackbox_name][method]['f1_score'] = local_f1

                # Precision score
                local_precision = precision_score(bb_pred, np.asarray(output['local_model_pred']), average='weighted')
                results[dataset_kw][blackbox_name][method]['precision'] = local_precision

                # Accuracy score
                local_accuracy = accuracy_score(bb_pred, np.asarray(output['local_model_pred']))
                results[dataset_kw][blackbox_name][method]['accuracy'] = local_accuracy

                # Average score of local model
                avg_local_model_score = np.mean(np.asarray(output['local_model_score']))
                results[dataset_kw][blackbox_name][method]['model_score'] = avg_local_model_score

            for key, val in results[dataset_kw][blackbox_name].items():
                print(key, ':', val)

if __name__ == '__main__':
    main()
