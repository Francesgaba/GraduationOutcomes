import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class InvestigativeAnalysis:
    def __init__(self, df):
        self.df = df

    def plot_graduation_rates(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='fl_percent', y='Total Grads - % of cohort', data=self.df)
        plt.title('Relationship Between Family Economic Status and Graduation Rates')
        plt.xlabel('Free or Reduced-Price Lunch Percentage')
        plt.ylabel('Total Graduates - % of Cohort')
        plt.show()

    def plot_exam_pass_rates(self):
        plt.figure(figsize=(10, 6))
        sns.regplot(x='fl_percent', y='Total Regents - % of cohort', data=self.df)
        plt.title('Relationship Between Family Economic Status and Exam Pass Rates')
        plt.xlabel('Free or Reduced-Price Lunch Percentage')
        plt.ylabel('Total Regents - % of Cohort')
        plt.show()

    def plot_school_outcomes(self, selected_school_names):
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

    def analyze_and_plot_data(self):
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

    def plot_school_outcomes(self, selected_school_names):
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

    def prepare_data(self, features, target, test_size=0.2, random_state=42):
        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, features, target):
        X_train, X_test, y_train, y_test = self.prepare_data(features, target)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        return model, mse, r2


