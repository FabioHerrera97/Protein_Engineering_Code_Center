import numpy as np

class DataProcessing:
    """
    A class for data processing operations.

    Parameters:
    data (pandas.DataFrame): The input data.

    Methods:
    drop_columns(columns): Drops specified columns from the data.
    create_outlier_report(data, threshold, output_file): Creates an outlier report based on the data.
    transform_to_logarithmic(data): Transforms the data to logarithmic scale.
    """

    def __init__(self, data):
        self.data = data
    
    def drop_columns(self, columns):
        """
        Drops specified columns from the data.

        Parameters:
        columns (list): The list of column names to be dropped.

        Returns:
        pandas.DataFrame: The updated data after dropping the columns.
        """
        self.data = self.data.drop(columns, axis=1)
        return self.data
    
    def create_outlier_report(data, threshold, output_file):
        """
        Creates an outlier report based on the data.

        Parameters:
        data (pandas.DataFrame): The input data.
        threshold (float): The threshold value for identifying outliers.
        output_file (str): The path to the output file where the report will be saved.
        """
        outliers = data[(data < data.quantile(threshold)) | (data > data.quantile(1 - threshold))]
        outlier_count = len(outliers)

        with open(output_file, 'w') as file:
            file.write(f'Outlier Report\n\n')
            file.write(f'Total Outliers: {outlier_count}\n\n')
            file.write(f'Outlier Details:\n')
            file.write(f'{outliers}\n')

        print(f'Outlier report created successfully at {output_file}')

    
    def transform_to_logarithmic(data):
        """
        Transforms the data to logarithmic scale.

        Parameters:
        data (pandas.DataFrame): The input data.

        Returns:
        pandas.DataFrame: The transformed data.
        """
        transformed_data = np.log(data)
        return transformed_data