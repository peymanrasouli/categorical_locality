from prepare_datasets import *
from create_model import CreateModel
from result_format import resultFormat
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from random_sampling import RandomSampling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from overlap_distance import OverlapDistance
from of_distance import OFDistance
from iof_distance import IOFDistance
from eskin_distance import EskinDistance
from goodall1_distance import Goodall1Distance
from global_explanation_sampling import GlobalExplanationSampling
from sklearn import tree
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

def forward_selection(data, labels, num_features, ohe_encoder=None):
    clf = Ridge()
    used_features = []
    for _ in range(min(num_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue

            data_ohe = []
            for f in  used_features + [feature]:
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

def interpretable_model(sampled_data, sampled_labels, sampled_proba, num_locality_samples=1000,
                        num_features=5, distance_metric=None, ohe_encoder=None):

    distances = distance_metric.pairwiseDistance(
        sampled_data[0,:],
        sampled_data,
    )

    selected_samples = np.argsort(distances)
    selected_samples = selected_samples[:num_locality_samples]
    sampled_data = sampled_data[selected_samples,:]
    sampled_labels = sampled_labels[selected_samples]
    sampled_proba = sampled_proba[selected_samples]
    used_features = forward_selection(sampled_data, sampled_proba, num_features, ohe_encoder)
    lr = Ridge(random_state=42)
    data_ohe = []
    for f in used_features:
        data_ohe.append(ohe_encoder[f].transform(sampled_data[:, f].reshape(-1, 1)))
    data_ohe = np.hstack(data_ohe)
    lr.fit(data_ohe, sampled_proba)
    lr_preds = lr.predict(data_ohe)
    local_model_pred = float(lr.predict(data_ohe[0, :].reshape(1, -1)))
    local_model_score = r2_score(sampled_proba, lr_preds)
    return local_model_pred, local_model_score


def data_sampling(instance2explain, blackbox, training_data, num_random_samples=2000):
    sampled_data = RandomSampling(training_data, num_random_samples)
    sampled_data = np.r_[instance2explain.reshape(1, -1), sampled_data]
    sampled_labels = blackbox.predict(sampled_data)
    sampled_proba = blackbox.predict_proba(sampled_data)
    sampled_proba = sampled_proba[:,sampled_labels[0]]

    return sampled_data, sampled_labels, sampled_proba


def explain_instance(blackbox, instance2explain, training_data, num_random_samples=2000,
                     num_locality_samples=500, num_features=5, ohe_encoder = None,
                     distance_metric=None):
    sampled_data, \
    sampled_labels, \
    sampled_proba = data_sampling(instance2explain, blackbox, training_data, num_random_samples)

    local_model_pred, \
    local_model_score = interpretable_model(sampled_data, sampled_labels, sampled_proba,
                                            num_locality_samples=num_locality_samples,
                                            num_features=num_features,
                                            distance_metric=distance_metric,
                                            ohe_encoder=ohe_encoder)

    return local_model_pred, local_model_score

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'

    # defining the list of data sets
    datsets_list = {
        # 'adult': ('adult.csv', PrepareAdult),
        # 'adult': ('adult.csv', PrepareAdultCat),
        # 'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPAS),
        # 'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPASCat),
        # 'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefault),
        # 'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefaultCat),
        # 'german-credit': ('german-credit.csv', PrepareGermanCredit),
        'german-credit': ('german-credit.csv', PrepareGermanCreditCat),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDisease),
        # 'heart-disease': ('heart-disease.csv', PrepareHeartDiseaseCat),
        'breast-cancer': ('breast-cancer.data', PrepareBreastCancer),
        # 'nursery': ('nursery.data', PrepareNursery),
        'car': ('car.data', PrepareCar),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        # 'gb': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        # 'svm': SVC
    }

    num_locality_samples = {
        'adult': 1000,
        'compas-scores-two-years': 1000,
        'credit-card-default': 1000,
        'german-credit': 500,
        'heart-disease': 1000,
        'breast-cancer': 500,
        'nursery': 1000,
        'car':500
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
        X_explain = X_test[:min(X_test.shape[0], 100),:]

        # creating one-hot encoder for discrete features
        ohe_encoder = {}
        for f in dataset['discrete_indices']:
            enc = OneHotEncoder(sparse=False)
            enc.fit(X[:,f].reshape(-1, 1))
            ohe_encoder[f] = enc

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)

            # creating distance metrics
            distance_metrics = {}
            semantic = GlobalExplanationSampling(X, y, blackbox, dataset,
                                                 dataset['discrete_indices'], dataset['continuous_indices'])
            semantic.fit()
            distance_metrics['semantic'] = semantic

            overlap = OverlapDistance()
            distance_metrics['overlap'] = overlap

            of = OFDistance(X)
            distance_metrics['of'] = of

            iof = IOFDistance(X)
            distance_metrics['iof'] = iof

            eskin = EskinDistance(X)
            distance_metrics['eskin'] = eskin

            goodall1 = Goodall1Distance(X)
            distance_metrics['goodall1'] = goodall1

            # Generating explanations for the samples in the explain set
            methods_output = {'semantic': {'local_model_pred':[], 'local_model_score':[]},
                              # 'overlap': {'local_model_pred': [], 'local_model_score': []},
                              'of': {'local_model_pred':[], 'local_model_score':[]},
                              # 'iof': {'local_model_pred':[], 'local_model_score':[]},
                              # 'eskin': {'local_model_pred': [], 'local_model_score': []},
                              # 'goodall1': {'local_model_pred': [], 'local_model_score': []}
                              }
            for x in X_explain:
                for method, output in methods_output.items():
                    local_model_pred, \
                    local_model_score = explain_instance(blackbox, x, X_train, num_random_samples=2000,
                                                         num_locality_samples=num_locality_samples[dataset_kw],
                                                         num_features=5, distance_metric=distance_metrics[method],
                                                         ohe_encoder=ohe_encoder)
                    methods_output[method]['local_model_pred'].append(local_model_pred)
                    methods_output[method]['local_model_score'].append(local_model_score)

            # calculating the performance of different sampling strategy
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