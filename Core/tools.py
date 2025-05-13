import pandas as pd

def create_subset(raw_data_path, log_center, lat_center, size_meters, size_entries_limit, label):
    """
    This method creates a subset of entries centered around a given point.
    """

    data = pd.read_csv(raw_data_path)
    return data


