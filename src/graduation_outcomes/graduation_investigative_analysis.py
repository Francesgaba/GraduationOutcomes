import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class InvestigativeAnalysis:
    """
    A class for performing investigative analysis on school data.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing school data.
    """
    def __init__(self, df):
        """
        Initializes the InvestigativeAnalysis with school data.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing school data.
        """
        self.df = df

    def plot_graduation_rates(self):
        """
        Plots the relationship between family economic status and graduation rates.

        This method creates a scatter plot showing the correlation between the percentage 
        of students with free or reduced-price lunch (fl_percent) and graduation rates.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='fl_percent', y='Total Grads - % of cohort', data=self.df)
        plt.title('Relationship Between Family Economic Status and Graduation Rates')
        plt.xlabel('Free or Reduced-Price Lunch Percentage')
        plt.ylabel('Total Graduates - % of Cohort')
        plt.show()

    def plot_exam_pass_rates(self):
        """
        Plots the relationship between family economic status and exam pass rates.

        This method creates a regression plot showing the correlation between the percentage 
        of students with free or reduced-price lunch (fl_percent) and exam pass rates.
        """
        plt.figure(figsize=(10, 6))
        sns.regplot(x='fl_percent', y='Total Regents - % of cohort', data=self.df)
        plt.title('Relationship Between Family Economic Status and Exam Pass Rates')
        plt.xlabel('Free or Reduced-Price Lunch Percentage')
        plt.ylabel('Total Regents - % of Cohort')
        plt.show()

    def plot_school_outcomes(self, selected_school_names):
        """
        Plots the outcomes for selected schools using a polar chart.

        Parameters:
            selected_school_names (list): A list of school names to plot outcomes for.

        This method creates polar charts showing various school outcomes like enrollment rate,
        local graduation rate, and dropout rate for the selected schools.
        """
        selected_schools = self.df[self.df['name'].isin(selected_school_names)]

        categories = ['Still Enrolled - % of cohort', 'Local - % of cohort', 'Dropped Out - % of cohort']
        N = len(categories)

        fig = plt.figure(figsize=(12, 8))

        rows = int(np.ceil(len(selected_schools) / 3))
        cols = 3

        subplot_num = 1

        for _, row in selected_schools.iterrows():
            values = row[categories].tolist()
            values += values[:1]
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            ax = plt.subplot(rows, cols, subplot_num, polar=True)
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            ax.plot(angles, values, label=row['name'])
            ax.fill(angles, values, alpha=0.1)
            plt.title(row['name'], size=11, color='blue', y=1.1)

            subplot_num += 1

        plt.tight_layout()
        plt.show()

    def plot_graduation_vs_ell_sped(self):
        """
        Plots the relationship between ELL/SpEd percentages and graduation rates.

        This method creates two scatter plots in a single figure, one showing the correlation 
        between English Language Learner (ELL) percentage and graduation rates, and the other 
        showing the correlation between Special Education (SpEd) percentage and graduation rates.
        """
        plt.figure(figsize=(10,4))

        plt.subplot(1, 2, 1)
        plt.scatter(self.df['ell_percent'], self.df['Total Grads - % of cohort'], color='blue', label='ELL vs. Graduation Rate')
        m, b = np.polyfit(self.df['ell_percent'], self.df['Total Grads - % of cohort'], 1)
        plt.plot(self.df['ell_percent'], m*self.df['ell_percent'] + b, color='blue', linestyle='-', linewidth=1)
        plt.title('ELL Percentage vs. Graduation Rate')
        plt.xlabel('ELL Percentage')
        plt.ylabel('Graduation Rate (%)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(self.df['sped_percent'], self.df['Total Grads - % of cohort'], color='green', label='SpEd vs. Graduation Rate')
        m, b = np.polyfit(self.df['sped_percent'], self.df['Total Grads - % of cohort'], 1)
        plt.plot(self.df['sped_percent'], m*self.df['sped_percent'] + b, color='green', linestyle='-', linewidth=1)
        plt.title('Special Education Percentage vs. Graduation Rate')
        plt.xlabel('Special Education Percentage')
        plt.ylabel('Graduation Rate (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def analyze_and_plot_data(self):
        """
        Performs data analysis and plots the relationship between racial diversity and school outcomes.

        This method first cleans the data by converting specific columns to numeric and dropping NaN values.
        It then calculates the racial diversity index and plots its relationship with graduation rates 
        and exam pass rates.
        """
        self.df['Total Grads - % of cohort'] = pd.to_numeric(self.df['Total Grads - % of cohort'], errors='coerce')
        self.df['Total Regents - % of cohort'] = pd.to_numeric(self.df['Total Regents - % of cohort'], errors='coerce')
        self.df.dropna(subset=['Total Grads - % of cohort', 'Total Regents - % of cohort'], inplace=True)

        diversity_index = 1 - ((self.df['asian_per'] / 100) ** 2 + (self.df['black_per'] / 100) ** 2 +
                            (self.df['hispanic_per'] / 100) ** 2 + (self.df['white_per'] / 100) ** 2)

        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=diversity_index, y=self.df['Total Grads - % of cohort'])
        plt.title('Graduation Rates vs. Racial Diversity Index')
        plt.xlabel('Racial Diversity Index')
        plt.ylabel('Total Graduates - % of Cohort')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.regplot(x=diversity_index, y=self.df['Total Regents - % of cohort'], scatter_kws={'alpha':0.5})
        plt.title('Exam Pass Rates vs. Racial Diversity Index')
        plt.xlabel('Racial Diversity Index')
        plt.ylabel('Total Regents - % of Cohort')
        plt.show()

    def prepare_data(self, features, target, test_size=0.2, random_state=42):
        """
        Prepares data for machine learning by splitting it into training and testing sets.

        Parameters:
            features (list): A list of column names to be used as features.
            target (str): The name of the target column.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            tuple: A tuple containing training and testing splits of features and target.
        """
        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, features, target):
        """
        Trains a linear regression model on the given features and target, and evaluates its performance.

        Parameters:
            features (list): A list of column names to be used as features.
            target (str): The name of the target column.

        Returns:
            tuple: A tuple containing the trained model, mean squared error, and R^2 score.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(features, target)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        return model, mse, r2


