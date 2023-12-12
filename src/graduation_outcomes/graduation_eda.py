import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EDAPerformer:
    """
    A class for performing various exploratory data analysis (EDA) tasks on a given pandas DataFrame.

    Attributes:
        df (pandas.DataFrame): The DataFrame on which EDA tasks will be performed.
    """

    def __init__(self, df):
        """
        Initialize the EDAPerformer with a pandas DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame to analyze.
        """
        self.df = df

    def describe_stats(self, column_name):
        """
        Provides descriptive statistics for a specified column in the DataFrame.

        Parameters:
            column_name (str): The name of the column for which to provide descriptive statistics.

        Returns:
            pandas.Series: Descriptive statistics of the specified column.
        """
        return self.df[column_name].describe()

    def plot_histogram(self, column, bins=30, kde=True):
        """
        Plots a histogram for a specified column in the DataFrame.

        Parameters:
            column (str): The name of the column to plot.
            bins (int, optional): The number of bins in the histogram. Default is 30.
            kde (bool, optional): Whether to display a Kernel Density Estimate (KDE). Default is True.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, bins=bins, kde=kde)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_all_columns(self):
        """
        Plots histograms and boxplots for all numeric columns in the DataFrame.

        This method identifies all numeric columns in the DataFrame and
        calls the plot_histogram_and_boxplot method for each column to
        display both histogram and boxplot side by side.
        """
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            self.plot_histogram_and_boxplot(column)

    def plot_boxplot(self, column):
        """
        Plots a boxplot for a specified column in the DataFrame.

        Parameters:
            column (str): The name of the column to plot a boxplot for.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x=column)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

    def plot_scatter(self, column_x, column_y):
        """
        Plots a scatter plot for two specified columns in the DataFrame.

        Parameters:
            column_x (str): The name of the column to use for the x-axis.
            column_y (str): The name of the column to use for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=column_x, y=column_y, data=self.df)
        plt.title(f'Scatter Plot of {column_x} vs {column_y}')
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.show()

    def plot_pairplot(self, columns):
        """
        Plots pairwise relationships in a dataset.

        This method creates a grid of Axes such that each variable in `columns` 
        will be shared across the y-axes across a single row and the x-axes 
        across a single column.

        Parameters:
            columns (list): List of column names to be included in the pairplot.
        """
        sns.pairplot(data=self.df, vars=columns)
        plt.suptitle('Pairwise Relationships Among Selected Variables', y=1.02)
        plt.show()

    def plot_heatmap(self):
        """
        Plots a heatmap to visualize the correlation matrix of the DataFrame.

        This method calculates the correlation matrix of the DataFrame and
        uses seaborn's heatmap function to visualize it. Each cell in the heatmap
        shows the correlation coefficient between two variables.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Heatmap of Correlation')
        plt.show()

    def plot_histogram_and_boxplot(self, column):
        """
        Plots both a histogram and a boxplot for a specified column in the DataFrame.

        This method is useful for a comprehensive overview of a single variable's
        distribution and outliers. It displays a histogram with a Kernel Density Estimate (KDE)
        and a boxplot side by side for the same variable.

        Parameters:
            column (str): The name of the column to plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(data=self.df, x=column, bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')

        sns.boxplot(data=self.df, x=column, ax=axes[1])
        axes[1].set_title(f'Boxplot of {column}')
        axes[1].set_xlabel(column)

        plt.tight_layout()
        plt.show()

    def plot_dropout_rates(self):
        """
        Plots the average dropout rates for the top 20 schools in the DataFrame.

        The method groups the data by school name, calculates the mean dropout rate for each school, 
        and then plots the top 20 schools based on their average dropout rate.
        """
        school_dropout_rates = self.df.groupby('name')['Dropped Out - % of cohort'].mean().sort_values(ascending=False)
        top_schools = school_dropout_rates.head(20) 

        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_schools.values, y=top_schools.index)

        plt.title('Average Dropout Rates by School')
        plt.xlabel('Average Dropout Rate (%)')
        plt.ylabel('name')

        plt.show()