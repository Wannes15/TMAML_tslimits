'''
code for processing Favorita dataset
process_favorita converts raw data into processed CSV
create_base_favorita_dataset creates the TimeSeriesDataSet for Favorita
'''

import py7zr
import zipfile

import glob
import datetime
import gc
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder

#########################################################################################
#Util functions
#########################################################################################
def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  
  if zip_path.endswith('.7z'):
    # Handle 7z files with py7zr
    with py7zr.SevenZipFile(zip_path, mode='r') as archive:
      archive.extractall(path=data_folder)
  else:
    # Handle regular zip files with zipfile
    with zipfile.ZipFile(zip_path, 'r') as archive:
      archive.extractall(path=data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))
  
#########################################################################################
#Preprocessing functions
#########################################################################################

# turns raw data into processed csv
def process_favorita():
    """Processes Favorita dataset.

    Makes use of the raw files should be manually downloaded from Kaggle @
        https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

    Args:
        config: Default experiment config for Favorita
    """

    url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

    data_folder = os.path.join(PROJECT_ROOT, "data", "favorita", "raw", "")

    # Save manual download to root folder to avoid deleting when re-processing.
    zip_file = os.path.join(data_folder,
                            'favorita-grocery-sales-forecasting.zip')

    if not os.path.exists(zip_file):
        raise ValueError(
            'Favorita zip file not found in {}!'.format(zip_file) +
            ' Please manually download data from Kaggle @ {}'.format(url))

    # Unpack main zip file
    outputs_file = os.path.join(data_folder, 'train.csv.7z')
    unzip(zip_file, outputs_file, data_folder)

    # Unpack individually zipped files
    for file in glob.glob(os.path.join(data_folder, '*.7z')):

        csv_file = file.replace('.7z', '')

        unzip(file, csv_file, data_folder)

    print('Unzipping complete, commencing data processing...')

    # Extract only a subset of data to save/process for efficiency
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2016, 12, 31)

    print('Regenerating data...')

    # load temporal data
    temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

    store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
    oil = pd.read_csv(os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
    items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

    temporal['date'] = pd.to_datetime(temporal['date'])

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        temporal = temporal[(temporal['date'] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal['date'] < end_date)]

    dates = temporal['date'].unique()

    # Add trajectory identifier
    temporal['traj_id'] = temporal['store_nbr'].apply(str) + '_' + temporal['item_nbr'].apply(str)
    temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(str)

    # Remove all IDs with negative returns
    print('Removing returns data')
    min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal['open'] = 1

    # Resampling
    print('Resampling to regular grid')
    resampled_dfs = []

    for traj_id, raw_sub_df in temporal.groupby('traj_id'):
        sub_df = raw_sub_df.set_index('date', drop=True).copy()

        sub_df = sub_df.resample('1d').last()

        sub_df['date'] = sub_df.index

        # Fill the traj_id for new rows
        sub_df['traj_id'] = traj_id 

        sub_df[['store_nbr', 'item_nbr', 'onpromotion']] = \
            sub_df[['store_nbr', 'item_nbr', 'onpromotion']].ffill()
        sub_df['open'] = sub_df['open'].fillna(0)

        sub_df['unit_sales'] = sub_df['unit_sales'].ffill()

        # Missing sales are forward filled
        sub_df['log_sales'] = np.log(np.maximum(sub_df['unit_sales'], 1e-8))

        # Recreate unique_id after filling
        sub_df['unique_id'] = sub_df['traj_id'] + '_' + sub_df['date'].apply(str)

        resampled_dfs.append(sub_df.reset_index(drop=True))

    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    print('Adding oil')
    oil.index = pd.to_datetime(oil.index)
    # Reindex oil to cover all dates and forward fill
    oil_reindexed = oil.reindex(pd.date_range(start=dates.min(), end=dates.max(), freq='D')).ffill()
    oil_df = pd.DataFrame({'date': oil_reindexed.index, 'oil': oil_reindexed.values})
    temporal = temporal.merge(oil_df, on='date', how='left')
    temporal['oil'] = temporal['oil'].fillna(-1)

    print('Adding store info')
    temporal = temporal.join(store_info, on='store_nbr', how='left')

    print('Adding item info')
    temporal = temporal.join(items, on='item_nbr', how='left')

    transactions['date'] = pd.to_datetime(transactions['date'])
    temporal = temporal.merge(
        transactions,
        left_on=['date', 'store_nbr'],
        right_on=['date', 'store_nbr'],
        how='left')
    temporal['transactions'] = temporal['transactions'].fillna(-1)

    # Additional date info
    temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
    temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
    temporal['month'] = pd.to_datetime(temporal['date'].values).month

    # Add holiday info
    print('Adding holidays')
    holiday_subset = holidays[holidays['transferred'].apply(
        lambda x: not x)].copy()
    holiday_subset.columns = [
        s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
    ]
    holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
    local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
    regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
    national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

    temporal['national_hol'] = temporal.merge(
        national_holidays, left_on=['date'], right_on=['date'],
        how='left')['description'].fillna('')
    temporal['regional_hol'] = temporal.merge(
        regional_holidays,
        left_on=['state', 'date'],
        right_on=['locale_name', 'date'],
        how='left')['description'].fillna('')
    temporal['local_hol'] = temporal.merge(
        local_holidays,
        left_on=['city', 'date'],
        right_on=['locale_name', 'date'],
        how='left')['description'].fillna('')

    temporal.sort_values('unique_id', inplace=True)

    save_dir = os.path.join(PROJECT_ROOT, 'data', 'favorita', 'processed_consecutive', '')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'favorita_processed_naSalesffill_2y.csv')

    print('Saving processed file to {}'.format(save_path))
    temporal.to_csv(save_path, index=False)
    pass

# cleans a bit
def favorita_clean_df():
    """Creates cleaned dataframe for Favorita dataset.
    Args:
        None
    """
    print("Cleaning Favorita dataframe...")
    data_path = os.path.join(PROJECT_ROOT, 'data', 'favorita', 'processed_consecutive', 'favorita_processed_naSalesffill_2y.csv')
    
    # Pre-define datatypes to avoid pandas defaulting to extremely memory-heavy objects
    dtypes = {
        'item_nbr': 'category', 'store_nbr': 'category', 'city': 'category', 
        'state': 'category', 'type': 'category', 'cluster': 'category', 
        'family': 'category', 'class': 'category', 'perishable': 'category', 
        'onpromotion': 'category', 'day_of_week': 'category',
        'national_hol': 'category', 'regional_hol': 'category', 'local_hol': 'category',
        'traj_id': 'category',
        'transactions': 'float32', 'oil': 'float32', 'day_of_month': 'float32', 
        'month': 'float32', 'open': 'float32', 'log_sales': 'float32', 'unit_sales': 'float32'
    }
    
    favorita_df = pd.read_csv(data_path, dtype=dtypes, usecols=lambda c: c != 'unique_id')
    
    print(f"Initial data shape: {favorita_df.shape}")
    print(f"Memory usage after load:\n{favorita_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Convert date column to datetime
    favorita_df['date'] = pd.to_datetime(favorita_df['date'])
    
    # Create time index (days since start)
    min_date = favorita_df['date'].min()
    favorita_df['time_idx'] = (favorita_df['date'] - min_date).dt.days.astype('int32')
    
    # Handle missing log_sales values (replace -inf with NaN, then handle appropriately)
    favorita_df['log_sales'] = favorita_df['log_sales'].replace([np.inf, -np.inf], np.nan)
    
    # Rename 'type' column to 'store_type' to avoid conflict with PyTorch's built-in type attribute
    if 'type' in favorita_df.columns:
        favorita_df['store_type'] = favorita_df['type']
        favorita_df = favorita_df.drop('type', axis=1)

    # Defensive dtype sanitation so downstream dataset creation does not need to coerce.
    categorical_columns = [
        'item_nbr', 'store_nbr', 'city', 'state', 'store_type', 'cluster',
        'family', 'class', 'perishable', 'onpromotion', 'day_of_week',
        'national_hol', 'regional_hol', 'local_hol', 'traj_id'
    ]

    # Ensure numeric columns are truly numeric and remove invalid time index rows.
    favorita_df['time_idx'] = pd.to_numeric(favorita_df['time_idx'], errors='coerce')
    before_rows = len(favorita_df)
    favorita_df.dropna(subset=['time_idx'], inplace=True)
    dropped_rows = before_rows - len(favorita_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with invalid time_idx")
    favorita_df['time_idx'] = favorita_df['time_idx'].astype('int32')

    for col in ['log_sales', 'transactions', 'oil', 'day_of_month', 'month', 'open', 'unit_sales']:
        if col in favorita_df.columns:
            favorita_df[col] = pd.to_numeric(favorita_df[col], errors='coerce')

    # Ensure categoricals are homogenous strings (avoid str/float comparison in encoders).
    for col in categorical_columns:
        if col in favorita_df.columns:
            favorita_df[col] = favorita_df[col].astype('string').fillna('__nan__').astype(str)
    
    # Force garbage collection to free dropped columns
    import gc
    gc.collect()

    print(f"Saving cleaned dataframe shape: {favorita_df.shape}")
    favorita_df.to_pickle(os.path.join(PROJECT_ROOT, 'data', 'favorita', 'processed_consecutive', 'favorita_cleaned_naSalesffill_2y.pkl'))
    return favorita_df


def create_base_favorita_dataset(
    favorita_df,
    max_encoder_length=90,
    max_prediction_length=30,
    min_encoder_length=90,
    anonymize_series_id=True,
    min_prediction_length=30,
    randomize_length=True,
    predict_mode=False
):
    """Creates base TimeSeriesDataSet for Favorita dataset for meta-learning.

    No normalization is applied; the target is only log-transformed upstream
    (favorita_clean_df). This function just creates the TimeSeriesDataSet.

    Based on the original Temporal Fusion Transformer paper settings for Favorita:
    - Encoder length: 90 days (3 months of historical data)
    - Prediction horizon: 30 days (1 month ahead)
    - Uses log-transformed sales as target
    Args:
        favorita_df: Cleaned Favorita dataframe (see favorita_clean_df)
        max_encoder_length: Maximum encoder length
        max_prediction_length: Prediction horizon
        min_encoder_length: Minimum encoder length (0 for meta-learning)
        anonymize_series_id: unused inside this function; the traj_id embedding
            shrink itself happens at model-build time in core/models/builders.py.
        min_prediction_length: Minimum prediction length
    """
    # store_nbr/city/state/store_type/cluster: constant within this single store, no signal.
    # item_nbr: unique per series, would give the model a per-series identifier to memorize.
    static_categoricals = ['traj_id', 'family', 'perishable', 'class']

    time_varying_known_categoricals = [
        'onpromotion', 'day_of_week', 'national_hol',
        'regional_hol', 'local_hol'
    ]
    categorical_columns = static_categoricals + time_varying_known_categoricals

    print(f"\nCreating Favorita TimeSeriesDataSet...")

    favorita_train_dataset = TimeSeriesDataSet(
        data=favorita_df,
        time_idx='time_idx',
        target='log_sales',
        group_ids=['traj_id'],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        randomize_length=randomize_length,
        predict_mode=predict_mode,

        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=[
            'day_of_month', 'month', 'open'
        ],
        time_varying_unknown_reals=[
            'transactions', 'oil', 'log_sales'
        ],

        allow_missing_timesteps=True,

        # No automatic normalization - target/reals are normalized upstream.
        target_normalizer=None,
        scalers={
            'transactions': None,
            'oil': None,
            'day_of_month': None,
            'month': None,
            'open': None,
        },
        
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in categorical_columns},

    )

    print(f"Successfully created Base Train TimeSeriesDataSet with {len(favorita_train_dataset)} samples")
    return favorita_train_dataset

