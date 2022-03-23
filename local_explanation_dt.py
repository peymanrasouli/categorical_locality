from prepare_datasets import *
from create_model import CreateModel
from result_format import resultFormat
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from explanation_based_neighborhood import ExplanationBasedNeighborhood
from random_sampling_neighborhood import RandomSamplingNeighborhood
from random_oversampling_neigbhorhood import RandomOversamplingNeighborhood
from random_instance_selection_neighborhood import RandomInstanceSelectionNeighborhood
from genetic_neighborhood import GeneticNeighborhood
from meaningful_data_sampling_neighborhood import MeaningfulDataSamplingNeighborhood
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import _tree
from console_progressbar.progressbar import ProgressBar
from datetime import datetime
from encoding_utils import *
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
        rule = "IF "

        for p in path[:-1]:
            if rule != "IF ":
                rule += " AND "
            p = str(p)
            if '<= 0.5' in p:
                p = p.replace('<= 0.5', '')
                last_char_index = p.rfind("_")
                p = p[:last_char_index] +  ' != ' + p[last_char_index + 1:]
            elif '> 0.5' in p:
                p = p.replace('> 0.5', '')
                last_char_index = p.rfind("_")
                p = p[:last_char_index] + ' == ' + p[last_char_index + 1:]
            rule += p
        rule += " THEN "
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

def interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba, N_features=5,
                        dataset=None, ohe_encoder=None, print_rules=False):

    neighborhood_data_org = ord2org(neighborhood_data, dataset)
    used_features = forward_selection(neighborhood_data_org, neighborhood_proba, N_features, ohe_encoder)

    data_ohe = []
    data_features = []
    for f in used_features:
        data_ohe.append(ohe_encoder[f].transform(neighborhood_data_org[:, f].reshape(-1, 1)))
        data_features.append(ohe_encoder[f].get_feature_names(input_features=[dataset['discrete_features'][f]]))
    data_ohe = np.hstack(data_ohe)
    data_features = np.hstack(data_features)

    # param_grid = {
    #     "max_depth": [1, 3, 5, 7, 10],
    #     "min_samples_split": [3, 5, 7, 10],
    #     "min_samples_leaf": [1, 2, 5]
    # }
    # clf = DecisionTreeClassifier(random_state=42)
    # grid_cv = GridSearchCV(clf, param_grid, scoring="f1_macro", n_jobs=-1, cv=3).fit(data_ohe, neighborhood_labels)
    # # print("Param for GS", grid_cv.best_params_)
    # # print("CV score for GS", grid_cv.best_score_)
    # # print("AUC ROC Score for GS: ", roc_auc_score(neighborhood_labels, grid_cv.predict(data_ohe)))
    # dt = grid_cv.best_estimator_

    dt = DecisionTreeClassifier(random_state=42, max_depth=5)

    dt.fit(data_ohe, neighborhood_labels)
    dt_labels = dt.predict(data_ohe)
    local_model_pred = int(dt.predict(data_ohe[0,:].reshape(1, -1)))
    local_model_score = f1_score(neighborhood_labels, dt_labels, average='macro')

    if print_rules:
        rules = get_rules(dt, data_features, list(dataset['labels'].values()))
        for r in rules:
            print(r)
        print('\n')

    return local_model_pred, local_model_score

def data_sampling(sampling_method, instance2explain, N_samples=1000):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = sampling_method(instance2explain, N_samples)

    return neighborhood_data, neighborhood_labels, neighborhood_proba


def explain_instance(instance2explain, N_samples=1000, N_features=5, dataset=None, ohe_encoder=None,
                     sampling_method=None, print_rules=False):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = data_sampling(sampling_method, instance2explain, N_samples)

    if print_rules:
        print(sampling_method, ':')

    local_model_pred, \
    local_model_score = interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba,
                                            N_features=N_features,  dataset=dataset, ohe_encoder=ohe_encoder,
                                            print_rules=print_rules)

    return local_model_pred, local_model_score

def main():
    # defining path of data sets and experiment results
    path = './'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult),
        'compas-scores-two-years': ('compas-scores-two-years.csv', PrepareCOMPAS),
        'credit-card-default': ('credit-card-default.csv', PrepareCreditCardDefault),
        'german-credit': ('german-credit.csv', PrepareGermanCredit),
        'breast-cancer': ('breast-cancer.data', PrepareBreastCancer),
        'heart-disease': ('heart-disease.csv', PrepareHeartDisease),
        'car': ('car.data', PrepareCar),
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn': MLPClassifier,
        'gb': GradientBoostingClassifier
    }

    # defining the number of neighborhood samples
    N_samples = {
        'adult': 1000,
        'compas-scores-two-years': 1000,
        'credit-card-default': 1000,
        'german-credit': 1000,
        'breast-cancer': 1000,
        'heart-disease': 1000,
        'car': 1000,
    }

    # defining the number of selected features for explanation
    N_features = {
        'adult': 5,
        'compas-scores-two-years': 5,
        'credit-card-default': 5,
        'german-credit': 5,
        'breast-cancer': 5,
        'heart-disease': 5,
        'car':5,
    }

    # creating a comprehensive dictionary for storing the results
    results = resultFormat(type_format='dt')

    # creating a csv file to store the results
    results_csv = open(experiment_path + 'local_explanation_dt_%s.csv' % (datetime.now()), 'a')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        results_csv.write('%s\n' % ('dataset= ' + dataset_kw))
        results_csv.write('%s, %s\n' % ('N_samples='+ str(N_samples[dataset_kw]),
                                        'N_features='+ str(N_features[dataset_kw])))
        results_csv.flush()

        # reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # creating one-hot encoder for discrete features
        ohe_encoder = {}
        X_org = ord2org(X, dataset)
        for f_id, f_name in enumerate(dataset['discrete_features']):
            enc = OneHotEncoder(sparse=False, categories='auto')
            enc.fit(X_org[:,f_id].reshape(-1, 1))
            ohe_encoder[f_id] = enc

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)
            results_csv.write('%s\n' % ('blackbox= ' + blackbox_name))
            results_csv.flush()

            # creating black-box model
            blackbox = CreateModel(X_train, X_test, Y_train, Y_test, blackbox_name, blackbox_constructor)

            # creating sampling methods
            sampling_methods = {}

            # explanation based neighborhood
            exp = ExplanationBasedNeighborhood(X, y, X_train, Y_train, blackbox, dataset)
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
            gen = GeneticNeighborhood(X, y, blackbox, dataset)
            gen.fit()
            sampling_methods['gen'] = gen.neighborhoodSampling

            # meaningful data sampling neighborhood
            mds = MeaningfulDataSamplingNeighborhood(X, y, blackbox, dataset)
            mds.fit()
            sampling_methods['mds'] = mds.neighborhoodSampling

            # Generating explanations for the samples in the explain set
            methods_output = {'exp': {'local_model_pred':[], 'local_model_score':[]},
                              'rnd': {'local_model_pred': [], 'local_model_score': []},
                              'ros': {'local_model_pred':[], 'local_model_score':[]},
                              'ris': {'local_model_pred':[], 'local_model_score':[]},
                              'gen': {'local_model_pred': [], 'local_model_score': []},
                              'mds': {'local_model_pred': [], 'local_model_score': []}
                              }

            # setting the number of explained instances
            N_explain = min(X_test.shape[0], 300)

            # explaining instances
            pb = ProgressBar(total=N_explain, prefix='Progress:', suffix='Complete', decimals=1, length=50,
                             fill='█', zfill='-')

            X_explain = []
            tried = 0
            explained = 0
            while explained < N_explain:
                try:
                    local_model_pred = {}
                    local_model_score = {}
                    for method, output in methods_output.items():
                        local_model_pred[method], \
                        local_model_score[method] = explain_instance(X_test[tried, :],
                                                                     N_samples=N_samples[dataset_kw],
                                                                     N_features=N_features[dataset_kw],
                                                                     dataset=dataset,
                                                                     ohe_encoder=ohe_encoder,
                                                                     sampling_method=sampling_methods[method],
                                                                     print_rules=False)
                    for method, pred in local_model_pred.items():
                        methods_output[method]['local_model_pred'].append(pred)
                    for method, score in local_model_score.items():
                        methods_output[method]['local_model_score'].append(score)
                    X_explain.append(X_test[tried, :])
                    tried += 1
                    explained += 1
                    pb.print_progress_bar(explained)

                except Exception:
                    tried += 1
                    pass

                if tried == X_test.shape[0]:
                    break

            # calculating the performance of different sampling strategy
            X_explain = np.vstack(X_explain)
            bb_pred = blackbox.predict(X_explain)
            for method, output in methods_output.items():
                # F1 score
                local_f1 = f1_score(bb_pred, np.asarray(output['local_model_pred']), average='macro')
                results[dataset_kw][blackbox_name][method]['f1_score'] = local_f1

                # Precision score
                local_precision = precision_score(bb_pred, np.asarray(output['local_model_pred']), average='macro')
                results[dataset_kw][blackbox_name][method]['precision'] = local_precision

                # Accuracy score
                local_accuracy = accuracy_score(bb_pred, np.asarray(output['local_model_pred']))
                results[dataset_kw][blackbox_name][method]['accuracy'] = local_accuracy

                # Average score of local model
                avg_local_model_score = np.mean(np.asarray(output['local_model_score']))
                results[dataset_kw][blackbox_name][method]['model_score'] = avg_local_model_score

            for key, val in results[dataset_kw][blackbox_name].items():
                print(key, ':', val)
                results_csv.write('%s\n' % (str(key) + ' : ' + str(val)))
                results_csv.flush()

        results_csv.write('\n')
        results_csv.flush()

    results_csv.close()

if __name__ == '__main__':
    main()
