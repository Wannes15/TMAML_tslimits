'''
code for processing the M5 dataset (Walmart daily unit sales, Kaggle M5 Forecasting
Accuracy competition): process_m5 converts raw CSVs into a processed CSV,
M5_clean_df cleans/types it into a pickle, create_base_M5_dataset builds the
TimeSeriesDataSet.

No normalization is applied, only the log_sales transform (target_normalizer=None
throughout): per-entity normalization isn't possible in a genuine cold-start
scenario, where a novel series has no history yet.
'''

import os
import gc

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder


#########################################################################################
# Util functions
#########################################################################################
def reduce_mem_usage(df, verbose=True):
    """Downcast numeric columns to the smallest dtype that fits their value range.

    Worth it given M5's scale (sell_prices.csv alone is ~200MB, and the melted
    long-format sales table is ~30,490 series x 1,941 days ~= 59M rows).
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#########################################################################################
# Preprocessing functions
#########################################################################################

# turns raw data into processed csv
def process_m5():
    """Processes the M5 dataset.

    Raw files must be manually downloaded from Kaggle @
        https://www.kaggle.com/c/m5-forecasting-accuracy/data

    Uses sales_train_evaluation.csv (1941 days) rather than
    sales_train_validation.csv (1913 days): this project defines its own
    train/val/test splits via time_idx, not the competition's own split.
    """
    url = 'https://www.kaggle.com/c/m5-forecasting-accuracy/data'

    data_folder = os.path.join(PROJECT_ROOT, "data", "M5", "raw", "")

    sales_path = os.path.join(data_folder, 'sales_train_evaluation.csv')
    calendar_path = os.path.join(data_folder, 'calendar.csv')
    prices_path = os.path.join(data_folder, 'sell_prices.csv')

    for path in (sales_path, calendar_path, prices_path):
        if not os.path.exists(path):
            raise ValueError(
                'M5 raw file not found at {}!'.format(path) +
                ' Please manually download data from Kaggle @ {}'.format(url))

    print('Loading raw M5 files...')
    sales_df = pd.read_csv(sales_path)
    calendar_df = reduce_mem_usage(pd.read_csv(calendar_path))
    prices_df = reduce_mem_usage(pd.read_csv(prices_path))
    print('Sales: {} rows, {} cols. Calendar: {} rows. Sell prices: {} rows.'.format(
        *sales_df.shape, len(calendar_df), len(prices_df)))

    sales_df['traj_id'] = sales_df['store_id'].apply(str) + '_' + sales_df['item_id'].apply(str)

    id_cols = ['traj_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in sales_df.columns if c.startswith('d_')]

    print('Melting {} series x {} days to long format...'.format(len(sales_df), len(d_cols)))
    temporal = sales_df.melt(id_vars=id_cols, value_vars=d_cols, var_name='d', value_name='unit_sales')
    del sales_df
    gc.collect()

    # M5's melted grid is already complete (one row per d_ column per series,
    # sales=0 on no-sale days), so there are no gaps to resample/forward-fill.
    temporal['unit_sales'] = pd.to_numeric(temporal['unit_sales'], errors='coerce').fillna(0)
    temporal['log_sales'] = np.log(np.maximum(temporal['unit_sales'], 1e-8))

    print('Joining calendar...')
    calendar_cols = ['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
                      'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                      'snap_CA', 'snap_TX', 'snap_WI']
    temporal = temporal.merge(calendar_df[calendar_cols], on='d', how='left')
    temporal['date'] = pd.to_datetime(temporal['date'])

    # The 'd_N' column already gives a clean 0-based sequential day index.
    temporal['time_idx'] = temporal['d'].str.extract(r'(\d+)').astype(int) - 1

    print('Joining sell prices...')
    temporal = temporal.merge(
        prices_df[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
        on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    # groupby().transform(ffill) below errors on float16 (which reduce_mem_usage
    # may have downcast sell_price to), so force back to float32 first.
    temporal['sell_price'] = temporal['sell_price'].astype('float32')

    # A missing sell_price means the item wasn't yet stocked at that store, so
    # rows before its first observed price are structural zeros, not real
    # demand signal. Drop those leading rows entirely rather than filling them.
    release_idx = (
        temporal.loc[temporal['sell_price'].notna()]
        .groupby('traj_id')['time_idx'].min()
        .rename('release_time_idx')
    )
    temporal = temporal.merge(release_idx, on='traj_id', how='left')

    before_rows = len(temporal)
    temporal = temporal.dropna(subset=['release_time_idx'])
    print(f"Dropped {before_rows - len(temporal)} rows for traj_ids with no sell_price "
          f"ever (never actually stocked at that store)")

    before_rows = len(temporal)
    temporal = temporal[temporal['time_idx'] >= temporal['release_time_idx']].copy()
    print(f"Dropped {before_rows - len(temporal)} pre-release rows (item not yet stocked)")
    temporal = temporal.drop(columns=['release_time_idx'])

    # Remaining NaN sell_price gaps (after release) are data-quality gaps within the
    # item's active life — forward-fill only. No bfill: that would leak a later price
    # backward, which release-trimming is specifically meant to avoid.
    temporal['sell_price'] = temporal.groupby('traj_id')['sell_price'].transform(lambda s: s.ffill())

    print('Deriving snap_today (SNAP benefit day for this series\' own state)...')
    temporal['snap_today'] = 0
    for state, col in (('CA', 'snap_CA'), ('TX', 'snap_TX'), ('WI', 'snap_WI')):
        mask = temporal['state_id'] == state
        temporal.loc[mask, 'snap_today'] = temporal.loc[mask, col]
    temporal = temporal.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'])

    temporal.sort_values(['traj_id', 'time_idx'], inplace=True)

    save_dir = os.path.join(PROJECT_ROOT, 'data', 'M5', 'processed', '')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'm5_processed.csv')
    print('Saving processed file to {}'.format(save_path))
    temporal.to_csv(save_path, index=False)


# cleans a bit
def M5_clean_df():
    """Creates cleaned dataframe for the M5 dataset (dtype hygiene)."""
    print("Loading processed M5 dataframe with memory-optimized datatypes...")
    data_path = os.path.join(PROJECT_ROOT, 'data', 'M5', 'processed', 'm5_processed.csv')

    dtypes = {
        'item_id': 'category', 'dept_id': 'category', 'cat_id': 'category',
        'store_id': 'category', 'state_id': 'category', 'traj_id': 'category',
        'weekday': 'category', 'event_name_1': 'category', 'event_type_1': 'category',
        'event_name_2': 'category', 'event_type_2': 'category', 'snap_today': 'category',
        'unit_sales': 'float32', 'log_sales': 'float32', 'sell_price': 'float32',
        'month': 'float32', 'year': 'float32',
    }
    m5_df = pd.read_csv(data_path, dtype=dtypes)

    print(f"Initial data shape: {m5_df.shape}")
    print(f"Memory usage after load:\n{m5_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    m5_df['date'] = pd.to_datetime(m5_df['date'])

    m5_df['log_sales'] = m5_df['log_sales'].replace([np.inf, -np.inf], np.nan)

    categorical_columns = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'traj_id',
        'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_today',
    ]

    m5_df['time_idx'] = pd.to_numeric(m5_df['time_idx'], errors='coerce')
    before_rows = len(m5_df)
    m5_df.dropna(subset=['time_idx'], inplace=True)
    dropped_rows = before_rows - len(m5_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with invalid time_idx")
    m5_df['time_idx'] = m5_df['time_idx'].astype('int32')

    for col in ['log_sales', 'unit_sales', 'sell_price', 'month', 'year']:
        if col in m5_df.columns:
            m5_df[col] = pd.to_numeric(m5_df[col], errors='coerce')

    # Ensure categoricals are homogeneous strings (avoid str/float comparison in encoders).
    for col in categorical_columns:
        if col in m5_df.columns:
            m5_df[col] = m5_df[col].astype('string').fillna('__nan__').astype(str)

    gc.collect()

    print(f"Saving cleaned dataframe shape: {m5_df.shape}")
    save_path = os.path.join(PROJECT_ROOT, 'data', 'M5', 'processed', 'm5_cleaned.pkl')
    m5_df.to_pickle(save_path)
    return m5_df


##########################################################################################
# Aggregation
##########################################################################################
def aggregate_M5_to_state():
    """Aggregate the store-level cleaned M5 dataframe to state-level series.

    Store-level M5 series are too intermittent (too many exact-zero days) for
    reliable forecasting, so this sums sales across each state's ~3-4 stores
    instead, redefining traj_id as state_id + "_" + item_id. store_id is
    dropped from the output (create_base_M5_dataset detects its absence and
    excludes it from static_categoricals automatically).

    unit_sales is summed per (item, day), and log_sales is recomputed from
    that sum rather than averaged from the already-logged per-store values
    (log-sum != sum-of-logs). sell_price is a sales-weighted average across
    stores, falling back to a plain mean when every store had zero sales that
    day (sales-weighting is undefined at zero total weight). Everything else
    (dept_id/cat_id/calendar fields) is constant across a state's stores for
    a given day, so it's taken via first().
    """
    print("Loading store-level cleaned M5 dataframe...")
    data_path = os.path.join(PROJECT_ROOT, 'data', 'M5', 'processed', 'm5_cleaned.pkl')
    m5_df = pd.read_pickle(data_path)

    df = m5_df.copy()
    df['traj_id'] = df['state_id'].astype(str) + '_' + df['item_id'].astype(str)
    df['_price_x_sales'] = df['sell_price'] * df['unit_sales']

    print(f"Aggregating {df['traj_id'].nunique()} store-level trajectories "
          f"to state level...")
    agg = df.groupby(['traj_id', 'time_idx']).agg(
        unit_sales=('unit_sales', 'sum'),
        item_id=('item_id', 'first'),
        dept_id=('dept_id', 'first'),
        cat_id=('cat_id', 'first'),
        state_id=('state_id', 'first'),
        date=('date', 'first'),
        weekday=('weekday', 'first'),
        event_name_1=('event_name_1', 'first'),
        event_type_1=('event_type_1', 'first'),
        event_name_2=('event_name_2', 'first'),
        event_type_2=('event_type_2', 'first'),
        snap_today=('snap_today', 'first'),
        month=('month', 'first'),
        year=('year', 'first'),
        _price_x_sales=('_price_x_sales', 'sum'),
        _sell_price_mean=('sell_price', 'mean'),
    ).reset_index()

    agg['sell_price'] = np.where(
        agg['unit_sales'] > 0,
        agg['_price_x_sales'] / agg['unit_sales'],
        agg['_sell_price_mean'],
    )
    agg = agg.drop(columns=['_price_x_sales', '_sell_price_mean'])

    agg['log_sales'] = np.log(np.maximum(agg['unit_sales'], 1e-8))

    categorical_columns = [
        'item_id', 'dept_id', 'cat_id', 'state_id', 'traj_id',
        'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_today',
    ]
    for col in categorical_columns:
        agg[col] = agg[col].astype('string').fillna('__nan__').astype(str)

    agg.sort_values(['traj_id', 'time_idx'], inplace=True)
    agg = reduce_mem_usage(agg)

    print(f"State-level aggregated shape: {agg.shape} "
          f"({agg['traj_id'].nunique()} state-item trajectories)")
    save_path = os.path.join(PROJECT_ROOT, 'data', 'M5', 'processed', 'm5_state_aggregated.pkl')
    agg.to_pickle(save_path)
    return agg


##########################################################################################
# Modeling-scale fixes (must be applied upstream, once, before ANY reuse of the df)
##########################################################################################
def fix_m5_scale_for_modeling(m5_df):
    """Rebase `year` and recompute `log_sales` via log1p. Returns a new dataframe.

    Neither fix is present in the on-disk meta-set pickles, so this must be
    called on top of them. `year` is rebased to years since 2011, since fed
    raw it sits ~1000x larger than every other input with no auto-scaling.
    `log_sales` is recomputed as log1p(unit_sales) rather than
    log(max(unit_sales, 1e-8)): ~25% of CA state-level rows are zero-sales,
    and the old formula put those at a severe outlier (-18.42).

    Must be called on train_df/val_df themselves (see build_m5 in
    core/data/registry.py), since both are also reused directly elsewhere.
    """
    m5_df = m5_df.copy()
    m5_df['year'] = m5_df['year'] - 2011
    m5_df['log_sales'] = np.log1p(m5_df['unit_sales'])
    return m5_df


##########################################################################################
# TimeSeriesDataSet creation function
##########################################################################################
def create_base_M5_dataset(
    m5_df,
    max_encoder_length=90,
    max_prediction_length=28,
    min_encoder_length=90,
    anonymize_series_id=True,
    min_prediction_length=28,
    randomize_length=True,
    predict_mode=False
):
    """Creates base TimeSeriesDataSet for the M5 dataset for meta-learning.

    Data should already have fix_m5_scale_for_modeling() applied when passed
    in. This function does not normalize — see module docstring for why.

    Args:
        m5_df: Cleaned M5 dataframe
        max_encoder_length: Maximum encoder length
        max_prediction_length: Prediction horizon (M5's own horizon is 28 days)
        min_encoder_length: Minimum encoder length (0 for meta-learning)
        anonymize_series_id: unused inside this function; the traj_id embedding
            shrink itself happens at model-build time in core/models/builders.py.
        min_prediction_length: Minimum prediction length
    """
    # item_id omitted: unique per series, would give the model a per-series
    # identifier to memorize. store_id carries real store-level demand signal,
    # so it's included when present (aggregate_M5_to_state's output drops it).
    static_categoricals = ['traj_id', 'dept_id', 'cat_id', 'state_id']
    if 'store_id' in m5_df.columns:
        static_categoricals.insert(1, 'store_id')

    # snap_today is the SNAP flag for this series' own state (see process_m5).
    time_varying_known_categoricals = [
        'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_today',
    ]

    print(f"\nCreating M5 TimeSeriesDataSet...")

    categorical_columns = static_categoricals + time_varying_known_categoricals

    m5_dataset = TimeSeriesDataSet(
        data=m5_df,
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

        # sell_price is set by the retailer ahead of time, known for the horizon.
        time_varying_known_reals=['month', 'year', 'sell_price'],

        time_varying_unknown_reals=['log_sales'],

        allow_missing_timesteps=True,

        target_normalizer=None,
        scalers={
            'month': None,
            'year': None,
            'sell_price': None,
        },

        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,

        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in categorical_columns},
    )

    print(f"Successfully created Base M5 TimeSeriesDataSet with {len(m5_dataset)} samples")
    return m5_dataset
