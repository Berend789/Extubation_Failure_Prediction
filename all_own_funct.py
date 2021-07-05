
from math import sqrt
from sklearn.metrics import roc_auc_score
import pylab as plt
import pandas as pd
import numpy as np
import locale
import scipy as sp
import typing
import datetime as dt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, f1_score, roc_curve, roc_auc_score, plot_confusion_matrix


import os


def split_stratified_into_train_val_test(df_input, stratify_colname=['y'],
                                         frac_train=0.7, frac_val=0.1, frac_test=0.2,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    X = df_input  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(
                                                              1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def cnfl(x):
    import numpy as np
    try:
        if ',' in x:
            x = float(locale.atof(x))
        elif x == 'NULL':
            x = np.nan
        else:
            try:
                x = int(x)
            except:

                x = float(locale.atof(x))
    except:
        x = float(x)
    return x


def memory_downscale(df):
    float_columns = list(df.select_dtypes(
        include=['float64', 'float32']).columns)
    for column in float_columns:
        df[column] = df[column].astype('float16')
    int_columns = list(df.select_dtypes(
        include=['int64', 'int32']).columns)
    for column in int_columns:
        df[column] = df[column].astype('int64')
    #df['OK_datum'] = df['OK_datum'].astype('datetime64[ns]')
    return df


def memory_upscale(df):
    float_columns = list(df.select_dtypes(
        include=['float64', 'float32', 'float16']).columns)
    for column in float_columns:
        df[column] = df[column].astype('float64')
    int_columns = list(df.select_dtypes(
        include=['int64', 'int32', 'int16']).columns)
    for column in int_columns:
        df[column] = df[column].astype('int64')
    return df


def x_modelling(df):
    temp = df[['Age', 'mis', 'Adnum', 'idx_1hr']]
    df = df.drop(['Age', 'pat_datetime', 'Reintub'], axis=1)
    df = df.groupby(['Adnum', 'idx_1hr'], sort=False).agg(['mean', 'std'])
    df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
    df = df.stack().unstack([2, 1])
    df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
    df = df.reset_index().drop_duplicates().set_index('Adnum')
    #df = df.fillna(method='ffill').fillna(method='bfill')

    temp = temp.groupby('Adnum', sort=False).agg(['mean'])
    temp.columns = ["_".join(a) for a in temp.columns.to_flat_index()]
    temp = temp.reset_index().drop_duplicates().set_index('Adnum')
    temp = temp[~temp.index.duplicated(keep='last')]
    df = df.merge(temp, right_index=True, left_index=True, how='left')
    return df


def y_modelling(df):
    y = df[['Reintub', 'Adnum', 'idx_1hr']]
    y = y.groupby(['Adnum', 'idx_1hr'], sort=False).agg(['max'])
    y.columns = ["_".join(a) for a in y.columns.to_flat_index()]
    y.reset_index(drop=True, inplace=True, level='idx_1hr')
    y = y.reset_index().drop_duplicates().set_index(
        ['Adnum'])
    y = y[~y.index.duplicated(keep='last')]
    y = y.fillna(method='ffill').fillna(method='bfill')
    return y


def y_modelling_12(df):
    y = df[['Reintub', 'Adnum']]
    y = y.groupby('Adnum', sort=False).agg(['max'])
    y.columns = ["_".join(a) for a in y.columns.to_flat_index()]
    y = y.reset_index().drop_duplicates().set_index('Adnum')
    y = y[~y.index.duplicated(keep='last')]
    y = y.fillna(method='ffill').fillna(method='bfill')
    return y


def x_modelling_12(df):
    temp = df[['Age', 'mis']]
    df = df.drop(['Age', 'pat_datetime', 'Reintub'], axis=1)
    df = df.groupby(level=['pat_hosp_id', 'AdmissionDate',
                           'DischargeDate'], sort=False).agg(['mean', 'std'])
    df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
    df = df.reset_index().drop_duplicates().set_index(
        level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'])
    df = df.fillna(method='ffill').fillna(method='bfill')

    temp = temp.groupby(
        level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False).agg(['mean'])
    temp.columns = ["_".join(a) for a in temp.columns.to_flat_index()]
    temp = temp.reset_index().drop_duplicates().set_index(
        level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'])
    temp = temp[~temp.index.duplicated(keep='last')]
    return df


def extub_group(group):
    temp = pd.DataFrame()
    group_2 = pd.DataFrame()
    group.sort_values('pat_datetime', inplace=True)
    if np.isnat(group['Extubation_date'].unique()):
        group = group.tail(720)
        group['Reintub'] = group['Reintub'].fillna(value=0)
        group['Extubation_date'] = pd.to_datetime(dt.datetime(
            2050, 1, 1, 0, 0, 0), format=('%d%m%Y %H:%M:%S'), dayfirst=True)
    else:
        group = group[group['pat_datetime'] <=
                      group['Extubation_date'].max()]
        group = group.tail(2160)
        group['Reintub'] = group['Reintub'].fillna(value=1)
        group = group.sort_values('pat_datetime')

        temp['gap'] = (group['pat_datetime'].diff() /
                       pd.Timedelta(minutes=15)).sub(1).fillna(0)
        if temp['gap'].max() > 0:
            temp = (group[temp['gap'] > 0][['pat_datetime']].head(1))

            group_1 = group[group['pat_datetime'] < temp['pat_datetime'].max()]
            group_1 = group_1.tail(720)

            if len(group_1) < 60:
                group.sort_values('pat_datetime', inplace=True)
                group = group.tail(720)
            else:
                group = group_1
        else:
            group.sort_values('pat_datetime', inplace=True)
            group = group.tail(720)
        group.sort_values('pat_datetime', inplace=True)
    return group


def OK_group(group):
    group.sort_values('pat_datetime', inplace=True)
    group = group[group['pat_datetime'] > group['OK_datum'].max()]
    group.sort_values('pat_datetime', inplace=True)
    return group


def cnfl2(x):
    import numpy as np
    if ',' in x:
        x = np.float16(locale.atof(x))
    elif x == None:
        x = np.nan
    elif x == 'NULL':
        x = np.nan
    else:
        x = np.float16()
    return x


def value_filtering(df):
    # df.dropna(axis=1,how='all',inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df.replace(to_replace='NULL', value=np.nan, inplace=True)
    df.replace(to_replace='nan', value=np.nan, inplace=True)
    df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
    return df

# Calculate streaks, filter out streaks that are too short, apply global mask


def nan_mask(df, column, nanlength=15):
    nan_spots = np.where(np.isnan(df[column]))
    diff = np.diff(nan_spots)[0]
    streaks = np.split(nan_spots[0], np.where(diff != 1)[0]+1)
    long_streaks = set(
        np.hstack([streak for streak in streaks if len(streak) >= nanlength]))
    mask = [item not in long_streaks for item in range(len(df[column]))]
    print("Filtered (without long streaks): ", len(df[column][mask]))
    return mask


class Wraptastic:
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(data=transformed, columns=df.columns, index=df.index)


# print(df.groupby('Admissionnumber')['idx'].nunique())


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) +
                   (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)


def age_calc_bron(group):
    if group['pat_weight_act'].isnull().all():
        if group['Age'].max() < 1.0:
            group['pat_weight_act'] = group['Age'].apply(lambda x: (x*12+9)/2)
        elif [(group['Age'].max() >= 1.0) & (group['Age'].max() < 5.0)]:
            group['pat_weight_act'] = group['Age'].apply(lambda x: (x+5)*2)
        elif [(group['Age'].max() >= 5.0) & (group['Age'].max() < 14.0)]:
            group['pat_weight_act'] = group['Age'].apply(lambda x: (x*4))
        else:
            group['pat_weight_act'] = group['Age'].apply(lambda x: x*3)
    else:
        group['pat_weight_act'] = group['pat_weight_act'].fillna(
            method='ffill').fillna(method='bfill')
    return group


def age_calc_where(group):
    if group['pat_weight_act'].isnull().all():
        group['pat_weight'] = group['Age'].where(
            (group['Age'] < 1.0), ((group['Age'] * 12+9)/2))
        group['pat_weight'] = group['Age'].where(
            [(group['Age'].max() >= 1.0) & (group['Age'].max() < 5.0)], ((group['Age']+5)*2))
        group['pat_weight'] = group['Age'].where(
            [(group['Age'].max() >= 5.0) & (group['Age'].max() < 14.0)], ((group['Age'] * 4)))
        group['pat_weight'] = group['Age'].where(
            (group['Age'] >= 14.0), ((group['Age'] * 3)))
    else:
        group['pat_weight'] = group['pat_weight_act'].fillna(
            method='ffill').fillna(method='bfill')
    return group


def evaluate(model, X_TEST, Y_TEST, name, output_folder):
    clf = model
    y_pred_clas = clf.predict(X_TEST)
    # Predict the probabilities, function depends on used classifier

    try:
        y_pred_prob = clf.predict_proba(X_TEST)
        y_pred_prob = y_pred_prob[:, 1]
    except:
        try:
            y_pred_prob = clf.decision_function(X_TEST)
        except:
            y_pred_prob = y_pred_clas

    average_precision = average_precision_score(Y_TEST, y_pred_prob)
    f1_s = f1_score(Y_TEST, y_pred_clas)
    AUC = roc_auc_score(Y_TEST, y_pred_prob)
    print('Model Performance')
    print(f'average_precision = {average_precision}')
    print(f'F1 score: {f1_s} ')
    print(f'AUC = {AUC}')
    with open(os.path.join(output_folder, f"Result_scores_all.txt"), 'a') as file:
        file.write(f"Results for {name}\n\n")
        file.write(f"AUC = {AUC} \n")
        file.write(f"F1 score: {f1_s} \n")
        file.write(f"Average precision score {average_precision} \n")
    return AUC
