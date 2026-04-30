import pandas as pd

DATA_PATH = "/Users/yuliia/Documents/Fraud-Detection/parquet/"

from parquet.fraud_ids import FRAUD_WAITER_IDS, FRAUD_IDS, FRAUD_WAITER_MONTH_IDS, FRAUD_WAITER_WEEK_IDS

FEATURES = [
    'share_top_waiter', 'gross_amount_mean_prcnt', 'num_of_trn_prcnt', 'first_last_trn_diff_prcnt'
]


def load_data(
    activity_state = 1, 
    days_visits = 1,
    num_of_trn = 1,
    place_num_of_waiters = 1,
    total_num_of_trn = 8,
    min_working_days = 2,
    num_of_trn_month = 10):
    df = pd.read_parquet(DATA_PATH + "processed_transactions.parquet", engine="pyarrow")
    client_data = pd.read_parquet(DATA_PATH + "client_level_features.parquet", engine="pyarrow")
    client_data = client_data[client_data['num_of_trn'] > activity_state]
    client_data = client_data[client_data['days_visits'] > days_visits]
    client_data['is_fraud'] = client_data['person_id'].isin(FRAUD_IDS)
    client_data = client_data.set_index('person_id')

    waiter_week_data = pd.read_parquet(DATA_PATH + "waiter_week_features.parquet", engine="pyarrow")
    waiter_week_data = waiter_week_data[waiter_week_data['num_of_trn'] > num_of_trn]
    waiter_week_data = waiter_week_data[waiter_week_data['place_num_of_waiters'] > place_num_of_waiters]
    waiter_week_data['is_fraud'] = waiter_week_data['waiter_week'].isin(FRAUD_WAITER_WEEK_IDS)
    waiter_week_data = waiter_week_data[waiter_week_data['working_days'] >= min_working_days]
    waiter_week_data = waiter_week_data.set_index('waiter_week')

    waiter_month_data = pd.read_parquet(DATA_PATH + "waiter_month_features.parquet", engine="pyarrow")
    waiter_month_data = waiter_month_data[waiter_month_data['num_of_trn'] > num_of_trn_month]
    waiter_month_data = waiter_month_data[waiter_month_data['place_num_of_waiters'] > place_num_of_waiters]
    waiter_month_data = waiter_month_data[waiter_month_data['working_days'] >= min_working_days]
    waiter_month_data['is_fraud'] = waiter_month_data['waiter_month'].isin(FRAUD_WAITER_MONTH_IDS)
    waiter_month_data = waiter_month_data.set_index('waiter_month')

    waiter_level_data = pd.read_parquet(DATA_PATH + "waiter_level_features.parquet", engine="pyarrow")
    waiter_level_data = waiter_level_data[waiter_level_data['num_of_trn'] > total_num_of_trn]
    waiter_level_data['is_fraud'] = waiter_level_data['waiter_id'].isin(FRAUD_WAITER_IDS)
    waiter_level_data = waiter_level_data.set_index('waiter_id')


    return df, client_data, waiter_week_data, waiter_month_data, waiter_level_data