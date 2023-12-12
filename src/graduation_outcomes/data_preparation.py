import requests
import pandas as pd
from requests.exceptions import HTTPError
from scipy.stats.mstats import winsorize

class APIClient:
    def __init__(self, api_url, limit=1000):
        self.api_url = api_url
        self.limit = limit
        self.df = None

    def fetch_data(self, offset=0):
        params = {'$limit': self.limit, '$offset': offset}
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            print(f"An error occurred: {err}")
            return None

    def fetch_all_data(self):
        data_frames = [] 
        offset = 0
        while True:
            df = self.fetch_data(offset)
            if df is not None and not df.empty:
                data_frames.append(df)
                offset += self.limit
            else:
                break
        self.df = pd.concat(data_frames, ignore_index=True)  

    def filter_data(self, desired_years):
        if self.df is None:
            raise ValueError("Data not loaded. Please call fetch_all_data first.")
        df_filtered = self.df[self.df['schoolyear'].isin(desired_years)].copy()
        columns_to_drop = [f'grade{i}' for i in range(1, 13)] + [
            'prek', 'k', 'frl_percent','ell_num', 'sped_num',
            'ctt_num', 'selfcontained_num', 'asian_num', 'black_num', 
            'hispanic_num', 'white_num', 'male_num', 'female_num'
        ]
        df_filtered.drop(columns_to_drop, axis=1, inplace=True)
        return df_filtered

    def convert_columns_to_numeric(self, df, columns_to_convert):
        """
        Converts specified columns to numeric data type.

        Parameters:
        columns_to_convert (list): A list of column names to be converted.
        """
        for column in columns_to_convert:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df


class DataProcess:
    def __init__(self, csv_url):
        self.df = self.read_csv_from_github(csv_url)

    def read_csv_from_github(self, csv_url):
        return pd.read_csv(csv_url)

    def filter_data(self, desired_year, demographic):
        df_filtered_year = self.df[self.df['Cohort'].isin(desired_year)].copy()
        df_filtered = df_filtered_year[df_filtered_year['Demographic'] == demographic].copy()
        columns_to_drop = [
            'Total Cohort', 'Total Grads - n', 'Total Regents - n',
            'Total Regents - % of grads', 'Advanced Regents - n',
            'Advanced Regents - % of grads', 'Regents w/o Advanced - n',
            'Regents w/o Advanced - % of grads', 'Local - n', 'Local - % of grads',
            'Still Enrolled - n', 'Dropped Out - n'
        ]
        df_filtered.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        return df_filtered

    def process_and_merge_data(self, df1, df2):
        # Map Cohort to School Year
        cohort_to_schoolyear = {
            '2004': '20082009',
            '2005': '20092010'
        }
        df2['mapped_schoolyear'] = df2['Cohort'].map(cohort_to_schoolyear)

        # Columns to drop
        columns_to_drop = ['Demographic', 'DBN', 'School Name', 'Cohort', 'mapped_schoolyear']

        # Merge and process the data
        merged_df = pd.merge(df1, df2, left_on=['dbn', 'schoolyear'], right_on=['DBN', 'mapped_schoolyear'], how='left')
        merged_df.drop(columns=columns_to_drop, axis=1, inplace=True)

        merged_df.dropna(subset=['Total Grads - % of cohort'], inplace=True)

        return merged_df

    def replace_ell_percent_nulls_with_median(self, df):
        median_ell_percent = df['ell_percent'].median()

        df['ell_percent'].fillna(median_ell_percent, inplace=True)

        return df

    def apply_winsorization(self, df):
        for column in df:
            df[column] = winsorize(df[column], limits=[0.05, 0.05])
        return df

    

    



