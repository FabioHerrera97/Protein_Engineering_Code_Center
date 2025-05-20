import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MeanAndStd:
    '''
    A class to compute row-wise mean and standard deviation for selected replicate columns in a DataFrame.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing replicate data.

    Methods:
        get_mean_and_std(columns_replicates):
            Computes the mean and standard deviation across specified replicate columns for each row.
    '''

    def __init__(self, data):
        self.data = data

    def get_mean_and_std(self, columns_replicates):
        '''
        Calculates the row-wise mean and standard deviation for the specified replicate columns.

        Args:
            columns_replicates (list of str): List of column names representing replicate measurements.

        Returns:
            tuple of pd.Series: Two Series containing the mean and standard deviation for each row.
        '''

        replicates = self.data[columns_replicates]
        mean = replicates.mean(axis=1)
        std = replicates.std(axis=1)

        return mean, std

class DataOutliers:
    '''
    A class for outlier detection.

    Parameters:
    data (pandas.DataFrame): The input data.

    Methods:
    create_outlier_report(data, threshold, output_file): Creates an outlier report based on the data.
    '''

    def __init__(self, data):
        self.data = data
    
    def detect_outliers_and_report(self, column, id_column, std_dev=None):

        '''
            Detects outliers in a specified numerical column using Z-score and IQR methods,
            generates a textual report, and creates an interactive Plotly visualization.

            The function performs the following:
            - Identifies outliers using Z-score and IQR.
            - Saves a detailed `outlier_report.txt` file with a brief explanation of each method and outlier listings.
            - Displays an interactive Plotly subplot showing outliers for each method, with labels based on an identifier column.

            Args:
                df (pd.DataFrame): The input DataFrame containing the data.
                column (str): Name of the numerical column to analyze for outliers.
                id_column (str, optional): Name of the identifier column used for labeling data points.
                std_dev (list): standard deviation for each data point
            Raises:
                ValueError: If either the `column` or `id_column` is not found in the DataFrame.

            Outputs:
                - A text file named `outlier_report.txt` containing explanations and outlier listings.
                - An interactive Plotly subplot visualization highlighting outliers for both methods.
        '''
            
        report_lines = []

        if column not in self.data.columns or id_column not in self.data.columns:
            raise ValueError('Ensure both the column and id_column exist in the DataFrame.')

        z_scores = zscore(self.data[column].dropna())
        z_threshold = 3
        z_outliers = self.data.loc[self.data[column].dropna().index[np.abs(z_scores) > z_threshold]]

        report_lines.append('=== Z-Score Method ===')
        report_lines.append(f"Outliers are defined as values with a Z-score > {z_threshold} or < -{z_threshold}.")
        report_lines.append(f"Total Outliers: {len(z_outliers)}")
        report_lines.extend([f"{row[id_column]}: {row[column]}" for _, row in z_outliers.iterrows()])
        report_lines.append("")

        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]

        report_lines.append("=== IQR Method ===")
        report_lines.append("Outliers are defined as values outside 1.5 * IQR from Q1 and Q3.")
        report_lines.append(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
        report_lines.append(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
        report_lines.append(f"Total Outliers: {len(iqr_outliers)}")
        report_lines.extend([f"{row[id_column]}: {row[column]}" for _, row in iqr_outliers.iterrows()])
        report_lines.append("")

        with open("../Reports/outlier_report.txt", "w") as f:
            f.write("\n".join(report_lines))

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Z-Score Method", "IQR Method"])

        if std_dev is not None:

            fig.add_trace(go.Scatter(
                x = self.data[id_column],
                y = self.data[column],
                mode = 'markers+text',
                text = std_dev.round(2).astype(str),
                textposition = 'top center',
                marker = dict(color=np.where(self.data[id_column].isin(z_outliers[id_column]), 'red', 'blue')),
                name = 'Z-Score'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = self.data[id_column],
                y = self.data[column],
                mode = 'markers+text',
                text = std_dev.round(2).astype(str),
                textposition = 'top center',
                marker = dict(color=np.where(self.data[id_column].isin(iqr_outliers[id_column]), 'red', 'green')),
                name = 'IQR'
            ), row=1, col=2)

        else:
            fig.add_trace(go.Scatter(
                x = self.data[id_column],
                y = self.data[column],
                mode = 'markers+text',
                text = self.data[id_column].where(self.data[id_column].isin(z_outliers[id_column])),
                textposition = 'top center',
                name = 'Z-Score'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = self.data[id_column],
                y = self.data[column],
                mode = 'markers+text',
                text = self.data[id_column].where(self.data[id_column].isin(iqr_outliers[id_column])),
                textposition = 'top center',
                name = 'IQR'
            ), row=1, col=2)

        fig.update_layout(showlegend=False)
        fig.show()


class LogarithmicTransform:

    def __init__(self, data):
        self.data = data

    def transform_to_logarithmic(self, column, wild_type):
        '''
        Transforms the labels to a logarithmic score.

        Parameters:
        data (pandas.DataFrame): The input data.
        wild_type (float): property in the wildtype

        Returns:
        pandas.DataFrame: The transformed data.
        '''
        transformed_data = np.log(self.data[column] / wild_type)

        return transformed_data
    
