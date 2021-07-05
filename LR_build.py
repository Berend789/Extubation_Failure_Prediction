from sklearn.model_selection import train_test_split
from all_stand_var import conv_dict, vent_cols3
from all_own_funct import extub_group, age_calc_bron
import os
import pandas as pd
import numpy as np
import locale
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import pickle


locale.setlocale(locale.LC_ALL, 'fr_FR')
output_folder = os.path.join(os.getcwd(), 'Results_bron_EDA')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class data_preprocessing():

    def __init__():
        """self, df, scaler=StandardScaler(), nan_cols=['vent_m_ppeak', 'vent_m_tv_exp', 'vent_m_fio2'], quantiles=(0.005, 0.995), nan_len=15, timestep=60):
        self.df = df
        self.nan_cols = nan_cols
        self.nan_len = nan_len
        self.timestep = timestep
        self.quantiles = quantiles
        self.scaler = scaler
        self.timestep = timestep
        """
        pass

    @staticmethod
    def value_filtering(df):
        # df.dropna(axis=1,how='all',inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df.replace(to_replace='NULL', value=np.nan, inplace=True)
        df.replace(to_replace='nan', value=np.nan, inplace=True)
        df.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
        df.sort_values(['pat_hosp_id', 'pat_datetime'], inplace=True)
        print(len(df['AdmissionDate'].unique()))
        df['Adnum'] = df.groupby(
            ['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False, as_index=False).ngroup()
        df = df.dropna(how='all', subset=['vent_m_rr', 'vent_m_tv_exp'])
        #df['Admissionnumber'] = df['Admissionnumber'].add(1)

        return df

    @staticmethod
    def Age_calc_cat(df):
        df['pat_datetime_temp'] = pd.to_datetime(df['pat_datetime']).dt.date
        df['pat_bd'] = pd.to_datetime(df['pat_bd']).dt.date
        df['Age'] = (df['pat_datetime_temp'] -
                     df['pat_bd']).dt.days.divide(365)
        df = df.drop('pat_datetime_temp', axis=1)
        df = df.drop('pat_bd', axis=1)
        df['Age'] = df['Age'].astype('float64')
        df['Age'] = np.where((df['Age'] > 25), 1, df['Age'])
        print(df['Age'].describe())
        # df['pat_weight_act'] = df.groupby(['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False, as_index=False)[
        #    'pat_weight_act'].fillna(method='ffill', inplace=True).fillna(method='bfill', inplace=True)
        df = df.groupby(['pat_hosp_id', 'AdmissionDate', 'DischargeDate'],
                        sort=False, as_index=False).apply(age_calc_bron)
        df['pat_weight_act'] = df['pat_weight_act'].astype('float64')
        return df

    @staticmethod
    def scale_dataframe(df, scaler, quantiles):
        float_columns = list(df.select_dtypes(
            include=['float64', 'float32']).columns)
        to_remove = ['pat_hosp_id', 'Reintub', 'Detub_fail']
        float_columns = list(
            (Counter(float_columns)-Counter(to_remove)).elements())
        for column in float_columns:
            df.loc[df[column] < 0, column] = 0
            df[column] = df[column].astype('float32')
        df['mon_hr'] = pd.to_numeric(df['mon_hr'], errors='coerce')
        df['mon_rr'].mask(df['mon_rr'] > 100, 100, inplace=True)
        df['vent_m_tv_exp'].mask(df['vent_m_tv_exp'] >
                                 1000, 1000, inplace=True)

        """
        down_quantiles = df[float_columns].quantile(quantiles[0])
        up_quantiles = df[float_columns].quantile(quantiles[1])
        print('up_quantile')
        print(100*'-')
        print('down_quantile')
        df[float_columns] = df[float_columns].mask(
            (df[float_columns] < down_quantiles), down_quantiles, axis=1)
        df[float_columns] = df[float_columns].mask(
            (df[float_columns] > up_quantiles), up_quantiles, axis=1)
        """

        #ss = func.Wraptastic(scaler)
        df_temp = df[float_columns]
        scaled_features = scaler.fit_transform(df_temp.values)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=df_temp.index, columns=df_temp.columns)
        df[float_columns] = scaled_features_df

        return df

    @staticmethod
    def nan_mask(df, nan_cols, nan_len):
        mask_list = []
        for column in nan_cols:
            nan_spots = np.where(np.isnan(df[column]))
            diff = np.diff(nan_spots)[0]
            streaks = np.split(nan_spots[0], np.where(diff != 1)[0]+1)
            long_streaks = set(
                np.hstack([streak for streak in streaks if len(streak) >= nan_len]))
            mask = np.array([item not in long_streaks for item in range(
                len(df[column]))])
            #print("Filtered (without long streaks): ", len(df[column][mask]))
            mask_list.append(mask)
        combmask = np.logical_or.reduce(mask_list)
        df = df[combmask]
        return df

    @staticmethod
    def time_step_df(df, timestep):
        grouped = pd.Grouper(key='pat_datetime', freq=timestep, origin='start')
        df['idx'] = df.groupby(level=['pat_hosp_id', 'AdmissionDate',
                                      'DischargeDate'], by=grouped, as_index=False, sort=False).ngroup()
        return df

    @staticmethod
    def data_merge(df, hours):
        add = dt.datetime.strptime("23:59:00", '%H:%M:%S')
        df2 = pd.read_csv(r'Results\admissiondate_v2_ext.csv', delimiter=';', usecols=['Reintub', 'AdmissionDate', 'DischargeDate', 'Extubation_date', 'pat_hosp_id'], parse_dates=[
            'AdmissionDate', 'DischargeDate', 'Extubation_date'], dayfirst=True)
        print(len(df2['Extubation_date'].unique()))
        df2['Extubation_date'] = df2['Extubation_date'].apply(
            lambda x: x+dt.timedelta(hours=add.hour, minutes=add.minute))
        df2.set_index(['pat_hosp_id', 'AdmissionDate',
                       'DischargeDate'], inplace=True)
        df.set_index(['pat_hosp_id', 'AdmissionDate',
                      'DischargeDate'], inplace=True)
        print(df.groupby(
            ['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False).ngroup().max())
        df_merged = df.merge(
            df2, how='left', left_index=True, right_index=True, validate="many_to_one")
        print(100*'merg')
        print(len(df_merged['Extubation_date'].unique()))
        if hours == True:
            print(df_merged.info())
            df_trim = df_merged.groupby(
                level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False, as_index=False).apply(extub_group)
        #df_merged.set_index(['Admissionnumber', 'idx'], inplace=True)
        print(df_trim.groupby(level=[
              'pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False).ngroup().max())
        print('Trim')
        print(len(df_trim['Extubation_date'].unique()))
        return df_trim


def data_pp_function(df, path='False', hours=True, scaler=StandardScaler(), nan_cols=['vent_m_ppeak', 'vent_m_tv_exp', 'vent_m_peep', 'vent_m_fio2'], quantiles=(0.005, 0.995), nan_len=15, timestep='12H'):
    df = data_preprocessing.value_filtering(df)
    df = data_preprocessing.Age_calc_cat(df)
    df = data_preprocessing.scale_dataframe(
        df, scaler=scaler, quantiles=quantiles)
    #df = data_preprocessing.nan_mask(df=df, nan_cols=nan_cols, nan_len=nan_len)
    df = data_preprocessing.data_merge(df=df, hours=hours)
    # df = df.groupby(level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False, as_index=False).fillna(
    #    method='ffill').fillna(method='bfill')
    df.sort_values('pat_datetime', inplace=True)
    #df = data_preprocessing.time_step_df(df=df, timestep=timestep)

    print(df.info())

    if path is not 'False':
        f = open(os.path.join(path, 'processed_df.txt'), 'wb')
        pickle.dump(df, f)
        f.close()

    return df


if __name__ == "__main__":
    dtype_dict = {'vent_cat': 'category',
                  'vent_machine': 'category', 'vent_mode': 'category'}
    df = pd.read_csv(r'data\sorted_bron_date.csv', delimiter=';', converters=conv_dict, usecols=vent_cols3, dtype=dtype_dict, parse_dates=[
        'pat_bd', 'pat_datetime', 'AdmissionDate', 'DischargeDate'], na_values=['NULL', 'null', 'Null', 'nUll', 'nuLl', 'nulL'], nrows=500000)
    df = data_pp_function(df, path=os.path.join(os.getcwd(), 'Results_RNN'))


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


class PadSequences(object):

    def __init__(self):
        self.name = 'padder'

    def pad(self, df, lb, time_steps, pad_value=-100):
        ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard 
            ub is an upper bound to truncate on. All entries are padded to their ubber bound '''

        # df = pd.read_csv(path):
        self.uniques = pd.unique(df['Adnum'])
        df = df.groupby(['Adnum']).filter(
            lambda group: len(group) > lb).reset_index(drop=True)
        df = df.groupby(['Adnum']).apply(
            lambda group: group[0:time_steps]).reset_index(drop=True)
        df = df.groupby(['Adnum']).apply(lambda group: pd.concat([group, pd.DataFrame(
            pad_value*np.ones((time_steps-len(group), len(df.columns))), columns=df.columns)], axis=0)).reset_index(drop=True)

        return df
