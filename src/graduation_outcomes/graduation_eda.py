import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EDAPerformer:
    def __init__(self,df):
        self.df=df

    def describe_stats(self, column_name):
        return self.df[column_name].describe()

    def plot_histogram(self, column, bins=30, kde=True):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, bins=bins, kde=kde)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_bar(self, column):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, data=self.df)
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    def plot_boxplot(self, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x=column)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

    def plot_scatter(self, column_x, column_y):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=column_x, y=column_y, data=self.df)
        plt.title(f'Scatter Plot of {column_x} vs {column_y}')
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.show()

    def plot_regression(self, x_column, y_column):
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.df, x=x_column, y=y_column, line_kws={"color": "red"})
        plt.title(f'{x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_violin(self, x_column, y_column):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.df, x=x_column, y=y_column, scale='width', inner='quartile')
        plt.title(f'{x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_pairplot(self, columns):
        sns.pairplot(data=self.df, vars=columns)
        plt.suptitle('Pairwise Relationships Among Selected Variables', y=1.02)
        plt.show()

    def plot_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Heatmap of Correlation')
        plt.show()

    def plot_histogram_and_boxplot(self, column):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制直方图
        sns.histplot(data=self.df, x=column, bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')

        # 绘制箱线图
        sns.boxplot(data=self.df, x=column, ax=axes[1])
        axes[1].set_title(f'Boxplot of {column}')
        axes[1].set_xlabel(column)

        plt.tight_layout()
        plt.show()

    def plot_all_columns(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            self.plot_histogram_and_boxplot(column)