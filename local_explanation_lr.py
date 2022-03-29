from prepare_datasets import *
from create_model import CreateModel
from result_format import resultFormat
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from explanation_based_neighborhood import ExplanationBasedNeighborhood
from random_sampling_neighborhood import RandomSamplingNeighborhood
from random_oversampling_neigbhorhood import RandomOversamplingNeighborhood
from random_instance_selection_neighborhood import RandomInstanceSelectionNeighborhood
from genetic_neighborhood import GeneticNeighborhood
from meaningful_data_sampling_neighborhood import MeaningfulDataSamplingNeighborhood
from sklearn.preprocessing import OneHotEncoder
from console_progressbar.progressbar import ProgressBar
from datetime import datetime
from encoding_utils import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(model, feature_names, class_name, input, label, experiment_path,
                           dataset_name, blackbox_name, sampling_name, index_instance):

    coef = model.coef_
    ind_category = np.where(input==1.0)[0]

    feature_names_ = feature_names[ind_category]
    coef_ = coef[ind_category]

    sorted_coef_ind = np.argsort(-abs(coef_))
    feature_names_ = feature_names_[sorted_coef_ind]
    coef_ = coef_[sorted_coef_ind]

    for i in range(len(feature_names_)):
        last_char_index = feature_names_[i].rfind("_")
        feature_names_[i] = feature_names_[i][:last_char_index] + " = " + feature_names_[i][last_char_index+1:]

    feature_importance = {}
    for f,c in zip(feature_names_, coef_):
        feature_importance[f] = c
    plt.close('all')
    vals = list(feature_importance.values())
    names = list(feature_importance.keys())
    vals.reverse()
    names.reverse()
    colors = ['#FF0051' if x > 0 else '#008BFB' for x in vals]
    pos = np.arange(len(vals)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names,fontsize=13)
    plt.title('Local explanation for class %s' % class_name[label], fontsize=15)
    plt.xlabel('Regression\'s coefficients (impact on model output)')
    plt.savefig(experiment_path + str(index_instance) + '_' + sampling_name + '_' + dataset_name + '_' + blackbox_name
                +'.pdf', bbox_inches = 'tight')

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

def interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba, N_features=5, dataset=None,
                        ohe_encoder=None, experiment_path=None, dataset_name=None, blackbox_name=None,
                        sampling_name=None, index_instance=None, plot_explanations=False):

    neighborhood_data_org = ord2org(neighborhood_data, dataset)
    used_features = forward_selection(neighborhood_data_org, neighborhood_proba, N_features, ohe_encoder)

    data_ohe = []
    data_features = []
    for f in used_features:
        data_ohe.append(ohe_encoder[f].transform(neighborhood_data_org[:, f].reshape(-1, 1)))
        data_features.append(ohe_encoder[f].get_feature_names(input_features=[dataset['discrete_features'][f]]))
    data_ohe = np.hstack(data_ohe)
    data_features = np.hstack(data_features)

    lr = Ridge(random_state=42)
    lr.fit(data_ohe, neighborhood_proba)
    lr_preds = lr.predict(data_ohe)
    local_model_pred = float(lr.predict(data_ohe[0, :].reshape(1, -1)))
    local_model_score = r2_score(neighborhood_proba, lr_preds)

    if plot_explanations:
        plot_feature_importance(lr, data_features, list(dataset['labels'].values()), data_ohe[0,:],
                                neighborhood_labels[0], experiment_path, dataset_name, blackbox_name,
                                sampling_name, index_instance)
    return local_model_pred, local_model_score

def data_sampling(sampling_method, instance2explain, N_samples=1000):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = sampling_method(instance2explain, N_samples)

    return neighborhood_data, neighborhood_labels, neighborhood_proba


def explain_instance(instance2explain, N_samples=1000, N_features=5, dataset=None, ohe_encoder=None,
                     sampling_method=None, experiment_path=None, dataset_name=None, blackbox_name=None,
                     sampling_name=None, index_instance=None, plot_explanations=False):

    neighborhood_data, \
    neighborhood_labels, \
    neighborhood_proba = data_sampling(sampling_method, instance2explain, N_samples)

    local_model_pred, \
    local_model_score = interpretable_model(neighborhood_data, neighborhood_labels, neighborhood_proba,
                                            N_features=N_features, dataset=dataset, ohe_encoder=ohe_encoder,
                                            experiment_path=experiment_path, dataset_name=dataset_name,
                                            blackbox_name=blackbox_name, sampling_name=sampling_name,
                                            index_instance=index_instance, plot_explanations=plot_explanations)
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
    results = resultFormat(type_format='lr')

    # creating a csv file to store the results
    results_csv = open(experiment_path + 'local_explanation_lr_%s.csv' % (datetime.now()), 'a')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        results_csv.write('%s\n' % ('dataset= ' + dataset_kw))
        results_csv.write('%s, %s\n' % ('N_samples=' + str(N_samples[dataset_kw]),
                                        'N_features=' + str(N_features[dataset_kw])))
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
            N_explain = min(X_test.shape[0], 500)

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
                                                                     experiment_path=experiment_path,
                                                                     dataset_name=dataset_kw,
                                                                     blackbox_name=blackbox_name,
                                                                     sampling_name=method,
                                                                     index_instance=tried,
                                                                     plot_explanations=False
                                                                     )
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
                results_csv.write('%s\n' % (str(key) + ' : ' + str(val)))
                results_csv.flush()

        results_csv.write('\n')
        results_csv.flush()

    results_csv.close()

if __name__ == '__main__':
    main()