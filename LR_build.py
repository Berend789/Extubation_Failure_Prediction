from sklearn.model_selection import train_test_split
from all_stand_var import conv_dict, vent_cols3
from all_own_funct import extub_group, age_calc_bron, split_stratified_into_train_val_test
import os
import pandas as pd
import numpy as np
import locale
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import pickle

"""
File which transforms the raw input from sql server to useable dataframe
Data path= folder in which the raw data in the csv file is stored
label_data = file in which the label data is stored
output_folder = Folder in which the transformed dataframe should be stored
"""
Data_path = 'data\sorted_bron_date.csv'
label_data = 'Results\admissiondate_v2_ext.csv'
locale.setlocale(locale.LC_ALL, 'fr_FR')
output_folder = os.path.join(os.getcwd(), 'Results_bron_EDA')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class data_preprocessing():

    def __init__():
        pass

    @staticmethod
    def value_filtering(df):
        """
        Function filters out empty rows, replaces null with NaN and replaces inf with NaN
        Adds admission number per admission
        Drops all rows in which the respiratory rate and expiratory tidal volume is NaN
        """
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
        """
        Calculate the age and weight for every admission
        """
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
    def scale_dataframe(df, scaler):
        """
        Function which removes physiological impossible values and scales all numerical features with the use of z-score normalisation
        """
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
        df_temp = df[float_columns]
        scaled_features = scaler.fit_transform(df_temp.values)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=df_temp.index, columns=df_temp.columns)
        df[float_columns] = scaled_features_df
        return df

    @staticmethod
    def data_merge(df):
        """
        Function to merge the label dataset with the parameter dataset
        Returns only the last 12 hours on the ventilator data
        """
        add = dt.datetime.strptime("23:59:00", '%H:%M:%S')
        df2 = pd.read_csv(label_data, delimiter=';', usecols=['Reintub', 'AdmissionDate', 'DischargeDate', 'Extubation_date', 'pat_hosp_id'], parse_dates=[
            'AdmissionDate', 'DischargeDate', 'Extubation_date'], dayfirst=True)
        df2['Extubation_date'] = df2['Extubation_date'].apply(
            lambda x: x+dt.timedelta(hours=add.hour, minutes=add.minute))
        df2.set_index(['pat_hosp_id', 'AdmissionDate',
                       'DischargeDate'], inplace=True)
        df.set_index(['pat_hosp_id', 'AdmissionDate',
                      'DischargeDate'], inplace=True)

        df_merged = df.merge(
            df2, how='left', left_index=True, right_index=True, validate="many_to_one")
        print(100*'merg')
        print(len(df_merged['Extubation_date'].unique()))

        df_trim = df_merged.groupby(
            level=['pat_hosp_id', 'AdmissionDate', 'DischargeDate'], sort=False, as_index=False).apply(extub_group)
        return df_trim


def data_pp_function(df, path='False', hours=True, scaler=StandardScaler()):
    df = data_preprocessing.value_filtering(df)
    df = data_preprocessing.Age_calc_cat(df)
    df = data_preprocessing.scale_dataframe(
        df, scaler=scaler)
    df = data_preprocessing.data_merge(df=df, hours=hours)
    df.sort_values('pat_datetime', inplace=True)

    if path is not 'False':
        f = open(os.path.join(path, 'processed_df.txt'), 'wb')
        pickle.dump(df, f)
        f.close()

    return df


if __name__ == "__main__":
    dtype_dict = {'vent_cat': 'category',
                  'vent_machine': 'category', 'vent_mode': 'category'}
    df = pd.read_csv(Data_path, delimiter=';', converters=conv_dict, usecols=vent_cols3, dtype=dtype_dict, parse_dates=[
        'pat_bd', 'pat_datetime', 'AdmissionDate', 'DischargeDate'], na_values=['NULL', 'null', 'Null', 'nUll', 'nuLl', 'nulL'])
    df = data_pp_function(df, path=output_folder)


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
