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
    
    def create_outlier_report(data, file_name):
        """
        Creates an outlier report based on the data.

        Parameters:
        data (pandas.DataFrame): The input data.
        file_name (str): The desired name for the output file.
        """
        mean = data.mean()
        std = data.std()
        threshold = 3 * std
        outliers = data[(data < mean - threshold) | (data > mean + threshold)]
        outlier_count = len(outliers)

        output_file = f'{file_name}.txt'

        with open(output_file, 'w') as file:
            file.write(f'Outlier Report\n\n')
            file.write(f'Total Outliers: {outlier_count}\n\n')
            file.write(f'Outlier Details:\n')
            file.write(f'{outliers}\n')

        print(f'Outlier report created successfully at ../Reports/{output_file}')

    
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
    
