import requests
import pandas as pd
from requests.exceptions import HTTPError
from scipy.stats.mstats import winsorize

class APIClient:
    """
    A class for fetching and processing data from a specified API.

    Attributes:
        api_url (str): URL of the API to fetch data from.
        limit (int): The maximum number of records to fetch per request.
        df (pandas.DataFrame): DataFrame holding the fetched data.
    """

    def __init__(self, api_url, limit=1000):
        """
        Initializes the APIClient with a specified API URL and limit.

        Parameters:
            api_url (str): URL of the API to fetch data from.
            limit (int): The maximum number of records to fetch per request.
        """
        self.api_url = api_url
        self.limit = limit
        self.df = None

    def fetch_data(self, offset=0):
        """
        Fetches data from the API with a specified offset.

        Parameters:
            offset (int): The offset to start fetching data from.

        Returns:
            pandas.DataFrame: DataFrame containing fetched data.
        """
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
        """
        Fetches all available data from the API.

        This method will continuously fetch data until no more data is available.
        The fetched data is concatenated into a single DataFrame.
        """
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
        """
        Filters the fetched data for specified years.

        Parameters:
            desired_years (list): A list of years to filter the data by.

        Returns:
            pandas.DataFrame: A DataFrame filtered by the specified years.
        """
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
            df (pandas.DataFrame): The DataFrame in which to convert columns.
            columns_to_convert (list): A list of column names to be converted.

        Returns:
            pandas.DataFrame: DataFrame with specified columns converted to numeric.
        """
        for column in columns_to_convert:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df


class DataProcess:
    """
    A class for processing and analyzing data from a CSV file.

    Attributes:
        df (pandas.DataFrame): The DataFrame loaded from the given CSV URL.
    """
    def __init__(self, csv_url):
        """
        Initializes the DataProcess with a URL to a CSV file.

        Parameters:
            csv_url (str): URL of the CSV file to be read.
        """
        self.df = self.read_csv_from_github(csv_url)

    def read_csv_from_github(self, csv_url):
        """
        Reads a CSV file from a given URL and returns a pandas DataFrame.

        Parameters:
            csv_url (str): URL of the CSV file to be read.

        Returns:
            pandas.DataFrame: DataFrame containing the data from the CSV file.
        """
        return pd.read_csv(csv_url)

    def filter_data(self, desired_year, demographic):
        """
        Filters data for a specified year and demographic.

        Parameters:
            desired_year (list): A list of years to filter the data by.
            demographic (str): The demographic category to filter by.

        Returns:
            pandas.DataFrame: A DataFrame filtered by the specified year and demographic.
        """
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
        """
        Processes and merges two dataframes based on specific criteria.

        Parameters:
            df1 (pandas.DataFrame): The first DataFrame to merge.
            df2 (pandas.DataFrame): The second DataFrame to merge.

        Returns:
            pandas.DataFrame: The merged DataFrame after processing.
        """
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
        """
        Replaces null values in the 'ell_percent' column with the column's median.

        Parameters:
            df (pandas.DataFrame): The DataFrame in which the replacements are to be made.

        Returns:
            pandas.DataFrame: The DataFrame with null values replaced in 'ell_percent' column.
        """
        median_ell_percent = df['ell_percent'].median()

        df['ell_percent'].fillna(median_ell_percent, inplace=True)

        return df

    def apply_winsorization(self, df):
        """
        Applies winsorization to all columns of a DataFrame to limit extreme values.

        Parameters:
            df (pandas.DataFrame): The DataFrame to apply winsorization to.

        Returns:
            pandas.DataFrame: The DataFrame after applying winsorization.
        """
        for column in df:
            df[column] = winsorize(df[column], limits=[0.05, 0.05])
        return df

    

    



