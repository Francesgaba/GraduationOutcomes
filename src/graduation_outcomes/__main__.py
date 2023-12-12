"""
Main module for running all components in the graduation_outcomes project.

This module integrates various classes from the project to fetch, prepare,
and analyze graduation outcome data. It includes steps for data retrieval,
exploratory data analysis, data preparation, and investigative analysis.

"""
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from graduation_outcomes.data_preparation import APIClient
from graduation_outcomes.data_preparation import DataProcess
from graduation_outcomes.graduation_eda import EDAPerformer
from graduation_outcomes.graduation_investigative import InvestigativeAnalysis

def main():
    print("Data Source")
    api_url = "https://data.cityofnewyork.us/resource/ihfw-zy9j.json"
    api = APIClient(api_url)
    try:
        df1 = api.fetch_all_data()
    except HTTPError as e:
        print("Error fetching data from API:", e)
        return
    df1_filtered = api.filter_data(['20082009'])
    print(df1_filtered.head())

    csv_url = "https://raw.githubusercontent.com/qdou14/GraduationOutcomes/main/dataset/2005-2010_Graduation_Outcomes_-_School_Level_20231209.csv"
    processor = DataProcess(csv_url)
    filtered_df2 = processor.filter_data(['2004'], 'Total Cohort')
    print(filtered_df2.head())

    print("-------------------------------------")
    print("Exploratory Data Analysis")
    converted_df = api.convert_columns_to_numeric(df1_filtered, [
        'fl_percent', 'total_enrollment', 'ell_percent',
        'sped_percent', 'asian_per', 'black_per',
        'hispanic_per', 'white_per', 'male_per', 'female_per'
    ])
    print(converted_df.head())
    eda1 = EDAPerformer(converted_df)
    eda1.describe_stats('ell_percent')
    eda1.plot_histogram('ell_percent')
    eda1.describe_stats(['total_enrollment', ])
    eda1.plot_histogram('total_enrollment')
    eda2 = EDAPerformer(filtered_df2)
    eda2.describe_stats('Total Grads - % of cohort')
    eda2.plot_boxplot('Total Grads - % of cohort')

    print("-------------------------------------")
    print("Data Preparation")
    merged_df = processor.process_and_merge_data(converted_df, filtered_df2)
    merged_df.to_csv('merged_data.csv', index=False)
    print(merged_df.head())
    df = processor.replace_ell_percent_nulls_with_median(merged_df)
    df = processor.apply_winsorization(df)

    print("-------------------------------------")
    print("Prepped Data Review")
    eda = EDAPerformer(df)
    eda.plot_dropout_rates()
    eda.plot_pairplot(['male_per', 'female_per', 'asian_per', 'black_per', 'hispanic_per', 'ell_percent', 'total_enrollment', 'Total Grads - % of cohort'])
    eda.plot_heatmap()

    print("-------------------------------------")
    print("Investigative Analysis & Results")
    analysis = InvestigativeAnalysis(df)
    analysis.plot_graduation_rates()
    analysis.plot_exam_pass_rates()
    analysis.analyze_and_plot_data()
    analysis.plot_graduation_vs_ell_sped()
    
    print("Destination after graduation")
    selected_school_names = [
        'BUSHWICK COMMUNITY HIGH SCHOOL',
        'LEGACY SCHOOL FOR INTEGRATED STUDIES',  
        'BRONX HIGH SCHOOL FOR LAW AND COMMUNITY SERVICE' 
    ]
    analysis.plot_school_outcomes(selected_school_names)

    print("Analysis of School Data Using Linear Regression")
    features = ['asian_per', 'white_per', 'Total Regents - % of cohort']
    target = 'Total Grads - % of cohort'
    model, mse, r2 = analysis.train_and_evaluate(features, target)

main()