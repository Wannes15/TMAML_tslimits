import numpy as np
import pandas as pd
from pytorch_forecasting import NaNLabelEncoder, TimeSeriesDataSet
import pyunpack
import wget

import os

import warnings
warnings.filterwarnings("ignore")

# Project root path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
  """Downloads a file from a url."""

  print('Pulling data from {} to {}'.format(url, output_path))
  wget.download(url, output_path)
  print('done')


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))

def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.

  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

  download_from_url(url, zip_path)

  unzip(zip_path, csv_path, data_folder)

  print('Done.')

# DOWNLOAD AND AGGREGATE ELECTRICITY DATA
def download_electricity():
  """Downloads electricity dataset from UCI repository."""

  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

  data_folder = f'{PROJECT_ROOT}/data/electricity/raw'
  os.makedirs(data_folder, exist_ok=True)
  csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
  zip_path = csv_path + '.zip'

  download_and_unzip(url, zip_path, csv_path, data_folder)

  print('Aggregating to hourly data')

  df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
  df.index = pd.to_datetime(df.index)
  df.sort_index(inplace=True)

  # Used to determine the start and end dates of a series
  output = df.resample('1h').mean().replace(0., np.nan)

  earliest_time = output.index.min()

  df_list = []
  for label in output:
    print('Processing {}'.format(label))
    srs = output[label]

    start_date = min(srs.fillna(method='ffill').dropna().index)
    end_date = max(srs.fillna(method='bfill').dropna().index)

    active_range = (srs.index >= start_date) & (srs.index <= end_date)
    srs = srs[active_range].fillna(0.)

    tmp = pd.DataFrame({'power_usage': srs})
    date = tmp.index
    tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
        date - earliest_time).days * 24
    tmp['days_from_start'] = (date - earliest_time).days
    tmp['categorical_id'] = label
    tmp['date'] = date
    tmp['id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

    df_list.append(tmp)

  output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

  output['categorical_id'] = output['id'].copy()
  output['hours_from_start'] = output['t']
  output['categorical_day_of_week'] = output['day_of_week'].copy()
  output['categorical_hour'] = output['hour'].copy()

  # Filter to match range used by other academic papers
  output = output[(output['days_from_start'] >= 1096)
                  & (output['days_from_start'] < 1346)].copy()

  output.to_csv(f"{PROJECT_ROOT}/data/electricity/processed/electricity_hourly.csv", index=False)

  print('Done.')


def create_base_electricity_dataset(
    electricity_df,
    max_encoder_length=168,
    max_prediction_length=24,
    min_encoder_length=0,
    min_prediction_length=24,
    randomize_length=True,
    predict_mode=False,
    target='power_usage'
    ):
    """Creates base TimeSeriesDataSet for electricity dataset for meta-learning.

    Based on the original TFT paper settings: 168h encoder (7 days), 24h
    prediction horizon (1 day). No normalization is applied; target defaults
    to 'power_usage' but should be 'log_power_usage' for the log1p meta-sets
    (see notebooks/electricity_meta_set_creation.ipynb), matching Favorita/M5's
    log_sales convention. Also used as the sole time_varying_unknown_real.

    Args:
        min_encoder_length: Set to 0 for meta-learning to allow variable-length windows
        min_prediction_length: Minimum length of the prediction horizon
        target: Target column name
    """
    df = electricity_df.copy()

    # TimeSeriesDataSet requires an integer time index.
    if df['time_idx'].dtype.kind != 'i':
        df['time_idx'] = df['time_idx'].astype(int)

    print(f"\nCreating TimeSeriesDataSet...")
    print(f"  {target}: mean={df[target].mean():.6f}, std={df[target].std():.6f}, range=[{df[target].min():.3f}, {df[target].max():.3f}]")

    electricity_train_dataset = TimeSeriesDataSet(
        data=df,
        time_idx='time_idx',
        target=target,
        group_ids=['traj_id'],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        randomize_length=randomize_length,
        predict_mode=predict_mode,

        time_varying_known_reals=[
            'hour', 'day_of_week', 'hours_from_start'
        ],

        time_varying_unknown_reals=[
            target
        ],

        static_categoricals=['traj_id'],

        # No automatic normalization - we normalize manually
        target_normalizer=None,

        categorical_encoders={
            "traj_id": NaNLabelEncoder(add_nan=True)
        },
        scalers={
            'hours_from_start': None,
            'day_of_week': None,
            'hour': None
        },
        add_encoder_length=False,
        add_relative_time_idx=False,
        add_target_scales=False,
    )

    print("Created TimeSeriesDataSet")
    return electricity_train_dataset
