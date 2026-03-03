import pandas as pd

DATA_PATH = "/Users/yuliia/Documents/Fraud-Detection/parquet/"

FRAUD_IDS = [
    11968000, 11970409, 11726701, 14827913,
    12411311, 12412748, 11812494, 12396334
]

FEATURES = [
    'num_of_trn', 
    'days_visits', 
    'gross_amount_mean', 
    'gross_amount_sum',
    'bonuses_accum_sum',
    'bonuses_used_sum',
    'num_of_waiters',
    'gross_amount_max',
    'first_last_trn_diff',
    'first_second_trn_diff',
    'first_third_trn_diff',
    'time_between_trn_median',
    'trn_per_day',
    'share_top_waiter',
    'share_bonus_trn',
    'share_bonus_after_first'
]

SKEWED = [
    'num_of_trn',
    'days_visits',
    'gross_amount_mean',
    'gross_amount_sum',
    'bonuses_accum_sum',
    'bonuses_used_sum',
    'num_of_waiters',
    'gross_amount_max',
    'first_last_trn_diff',
    'first_second_trn_diff',
    'first_third_trn_diff',
    'time_between_trn_median',
    'trn_per_day'
]


def load_data(activity_state = 1):
    df = pd.read_parquet(DATA_PATH + "processed_transactions.parquet", engine="pyarrow")
    client_data = pd.read_parquet(DATA_PATH + "client_level_features.parquet", engine="pyarrow")
    client_data = client_data[client_data['num_of_trn'] > activity_state]
    client_data = client_data.set_index('person_id')
    return df, client_data