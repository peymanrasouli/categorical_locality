import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.datasets import load_diabetes, load_iris, make_moons, load_wine, fetch_california_housing
import sys
sys.path.append('LORE')
from util import *
from LORE.gpdatagenerator import *

# ## Preparing Adult dataset
# def PrepareAdult(dataset_path, dataset_name):
#
#     ## Reading data from a csv file
#     df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values=' ?')
#
#     ## Handling missing values
#     df = df.dropna().reset_index(drop=True)
#
#     ## Recognizing inputs
#     class_name = 'class'
#     df_X_org = df.loc[:, df.columns!=class_name]
#     df_y = df.loc[:, class_name]
#
#     continuous_features = ['age','fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
#     discrete_features = ['work-class', 'education', 'marital-status', 'occupation', 'relationship',
#                          'race', 'sex', 'native-country']
#
#     continuous_availability = True
#     discrete_availability = True
#
#     df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)
#
#     continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
#     discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]
#
#     feature_values = []
#     for c in continuous_features:
#         feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
#     for d in discrete_features:
#         feature_values.append({d: set(df_X_org[d].unique())})
#
#     ## Extracting the precision of continuous features
#     types = df_X_org[continuous_features].dtypes
#     continuous_precision = []
#     for c in continuous_features:
#         if types[c] == float:
#             len_dec = []
#             for val in df_X_org[c]:
#                 len_dec.append(len(str(val).split('.')[1]))
#             len_dec = max(set(len_dec), key=len_dec.count)
#             continuous_precision.append(len_dec)
#         else:
#             continuous_precision.append(0)
#
#     precision = pd.Series(continuous_precision, index=continuous_features)
#     df_X_org = df_X_org.round(precision)
#
#     ## Scaling continuous features
#     num_feature_scaler =StandardScaler()
#     scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
#     scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)
#
#     ## Encoding discrete features
#     # Ordinal feature transformation
#     ord_feature_encoder = OrdinalEncoder()
#     ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
#     ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)
#
#     # One-hot feature transformation
#     ohe_feature_encoder = OneHotEncoder(sparse=False)
#     ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
#     ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)
#
#     # Creating ordinal and one-hot data frames
#     df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
#     df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)
#
#     ## Encoding labels
#     df_y_le = df_y.copy(deep=True)
#     label_encoder = {}
#     le = LabelEncoder()
#     df_y_le = le.fit_transform(df_y_le)
#     label_encoder[class_name] = le
#
#     ## Extracting raw data and labels
#     X_org = df_X_org.values
#     X_ord = df_X_ord.values
#     X_ohe = df_X_ohe.values
#     y = df_y_le
#
#     ## Indexing labels
#     labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}
#
#     ## Indexing features
#     feature_names = list(df_X_org.columns)
#     feature_indices = {i: feature for i, feature in enumerate(feature_names)}
#     feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
#     feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)
#
#     n_cat_discrete = ord_encoded_data.nunique().to_list()
#
#     len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
#     len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]
#
#     len_continuous_ord = [0, scaled_data.shape[1]]
#     len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]
#
#     len_continuous_ohe = [0, scaled_data.shape[1]]
#     len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]
#
#     # LORE
#     df = pd.concat([pd.Series(df_y_le,name=class_name), df_X_ord], axis=1)
#     df_le, label_encoder = label_encode(df, discrete_features+[class_name])
#     type_features, features_type = recognize_features_type(df, class_name)
#     idx_features = feature_indices
#     possible_outcomes = list(df[class_name].unique())
#
#     feature_values = calculate_feature_values(X_ord, [class_name]+feature_names, class_name,
#                                               discrete_features + [class_name], continuous_features,
#                                               discrete_use_probabilities=False,
#                                               continuous_function_estimation=False)
#
#     ## Returning dataset information
#     dataset = {
#         'name': dataset_name.replace('.csv', ''),
#         'df': df,
#         'df_y': df_y,
#         'df_X_org': df_X_org,
#         'df_X_ord': df_X_ord,
#         'df_X_ohe': df_X_ohe,
#         'df_y_le': df_y_le,
#         'class_name': class_name,
#         # 'label_encoder': label_encoder,
#         'labels': labels,
#         'ord_feature_encoder': ord_feature_encoder,
#         'ohe_feature_encoder': ohe_feature_encoder,
#         'num_feature_scaler': num_feature_scaler,
#         'feature_names': feature_names,
#         # 'feature_values': feature_values,
#         'feature_indices': feature_indices,
#         'feature_ranges': feature_ranges,
#         'feature_width': feature_width,
#         'continuous_availability': continuous_availability,
#         'discrete_availability': discrete_availability,
#         'discrete_features': discrete_features,
#         'discrete_indices': discrete_indices,
#         'continuous_features': continuous_features,
#         'continuous_indices': continuous_indices,
#         'continuous_precision': continuous_precision,
#         'n_cat_discrete': n_cat_discrete,
#         'len_discrete_ord': len_discrete_ord,
#         'len_continuous_ord': len_continuous_ord,
#         'len_discrete_ohe': len_discrete_ohe,
#         'len_continuous_ohe': len_continuous_ohe,
#         'len_discrete_org': len_discrete_org,
#         'len_continuous_org': len_continuous_org,
#         'X_org': X_org,
#         'X_ord': X_ord,
#         'X_ohe': X_ohe,
#         'y': y,
#
#         # LORE
#         'columns': [class_name]+feature_names,
#         'possible_outcomes': possible_outcomes,
#         'type_features': type_features,
#         'features_type': features_type,
#         'discrete': discrete_features+[class_name],
#         'continuous': continuous_features,
#         'idx_features': idx_features,
#         'label_encoder': label_encoder,
#         'feature_values': feature_values
#     }
#
#     return dataset

## Preparing Adult Categorical dataset
def PrepareAdult(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values=' ?')

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = []
    discrete_features = ['age', 'hours-per-week',
                         'work-class', 'education', 'marital-status', 'occupation', 'relationship',
                         'race', 'sex', 'native-country']

    continuous_availability = False
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    continuous_precision = []

    ## Scaling continuous features
    num_feature_scaler = None

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = ord_encoded_data
    df_X_ohe = ohe_encoded_data

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = []
    len_discrete_org = [0,  X_org.shape[1]]

    len_continuous_ord = []
    len_discrete_ord = [0,  X_ord.shape[1]]

    len_continuous_ohe = []
    len_discrete_ohe = [0,  X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset


## Preparing COMPAS dataset
def PrepareCOMPAS(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', skipinitialspace=True)

    ## Data cleaning
    # handling missing values and converting feature types
    columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest',
               'c_jail_in', 'c_jail_out', 'c_charge_degree', 'is_recid', 'is_violent_recid',
               'two_year_recid', 'decile_score', 'score_text']
    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
    cat_to_num = {
                  "sex": {"Male": 0,
                          "Female": 1}
                  }

    df = df.replace(cat_to_num)

    # classifying instances with respect to recidivism score
    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'

    df['class'] = df['decile_score'].apply(get_class)

    # removing useless columns
    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age', 'priors_count', 'days_b_screening_arrest', 'length_of_stay']
    discrete_features = [ 'is_recid', 'age_cat', 'c_charge_degree', 'is_violent_recid', 'two_year_recid', 'sex', 'race']

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c: [min(df_X_org[c]), max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset

## Preparing Default of Credit Card Clients dataset
def PrepareCreditCardDefault(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=',')

    ## Recognizing inputs
    class_name = 'default payment next month'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['LIMIT_BAL', 'AGE',  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                           'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    discrete_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset

## Preparing German Credit dataset
def PrepareGermanCredit(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=',')

    ## Recognizing inputs
    class_name = 'default'
    df_X_org = df.loc[:, df.columns != class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['duration_in_month', 'credit_amount', 'installment_as_income_perc', 'present_res_since',
                           'age', 'credits_this_bank', 'people_under_maintenance']
    discrete_features = ['account_check_status', 'credit_history', 'purpose', 'savings', 'present_emp_since',
                         'personal_status_sex', 'other_debtors', 'property', 'other_installment_plans', 'housing',
                         'job', 'telephone', 'foreign_worker']

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler = StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset

## Preparing Breast Cancer dataset
def PrepareBreastCancer(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values='?', header=None)
    df.columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
                  'breast', 'breast-quad', 'irradiat']

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = []
    discrete_features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
                        'breast', 'breast-quad', 'irradiat']

    continuous_availability = False
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    continuous_precision = []

    ## Scaling continuous features
    num_feature_scaler = None

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = ord_encoded_data
    df_X_ohe = ohe_encoded_data

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = []
    len_discrete_org = [0,  X_org.shape[1]]

    len_continuous_ord = []
    len_discrete_ord = [0,  X_ord.shape[1]]

    len_continuous_ohe = []
    len_discrete_ohe = [0,  X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset


## Preparing Nursery dataset
def PrepareNursery(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values='?', header=None)
    df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df.loc[df['class'] == 'recommend', 'class'] = 'not_recom'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = []
    discrete_features = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']

    continuous_availability = False
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    continuous_precision = []

    ## Scaling continuous features
    num_feature_scaler = None

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = ord_encoded_data
    df_X_ohe = ohe_encoded_data

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = []
    len_discrete_org = [0,  X_org.shape[1]]

    len_continuous_ord = []
    len_discrete_ord = [0,  X_ord.shape[1]]

    len_continuous_ohe = []
    len_discrete_ohe = [0,  X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset


## Preparing Car dataset
def PrepareCar(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values='?', header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = []
    discrete_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    continuous_availability = False
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    continuous_precision = []

    ## Scaling continuous features
    num_feature_scaler = None

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = ord_encoded_data
    df_X_ohe = ohe_encoded_data

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = []
    len_discrete_org = [0,  X_org.shape[1]]

    len_continuous_ord = []
    len_discrete_ord = [0,  X_ord.shape[1]]

    len_continuous_ohe = []
    len_discrete_ohe = [0,  X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset


## Preparing Heart Disease dataset
def PrepareHeartDisease(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path + dataset_name, delimiter=',', na_values='?')

    ## Handling missing values
    df = df.dropna().reset_index(drop=True)

    ## Recognizing inputs
    class_name = 'num'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age','trestbps','chol','thalach','oldpeak']
    discrete_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset

## Preparing Iris dataset
def PrepareWine(dataset_path, dataset_name):

    ## Importing data from sklearn library
    data = load_wine()
    df = pd.DataFrame(data=np.c_[data.data,data.target], columns=data.feature_names+['class'])

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                           'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                           'od280/od315_of_diluted_wines', 'proline']
    discrete_features = []

    continuous_availability = True
    discrete_availability = False

    df_X_org = df_X_org[continuous_features]

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = []

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = None

    # One-hot feature transformation
    ohe_feature_encoder = None

    # Creating ordinal and one-hot data frames
    df_X_ord = scaled_data.copy(deep=True)
    df_X_ohe = scaled_data.copy(deep=True)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = []

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = []

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = []

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = []

    # LORE
    df = pd.concat([pd.Series(df_y_le, name=class_name), df_X_ord], axis=1)
    df_le, label_encoder = label_encode(df, discrete_features + [class_name])
    type_features, features_type = recognize_features_type(df, class_name)
    idx_features = feature_indices
    possible_outcomes = list(df[class_name].unique())

    feature_values = calculate_feature_values(X_ord, [class_name] + feature_names, class_name,
                                              discrete_features + [class_name], continuous_features,
                                              discrete_use_probabilities=False,
                                              continuous_function_estimation=False)

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        # 'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        # 'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y,

        # LORE
        'columns': [class_name] + feature_names,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete_features + [class_name],
        'continuous': continuous_features,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'feature_values': feature_values
    }

    return dataset
