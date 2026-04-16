import pandas as pd

DATA_PATH = "/Users/yuliia/Documents/Fraud-Detection/parquet/"

from parquet.fraud_ids import FRAUD_WAITER_IDS, FRAUD_IDS, FRAUD_WAITER_WEEK_IDS

FEATURES = [
    # ISO precision@100 = 0.27, OCSVM precision@100 = 0.23, LOF precision@100 = 0.01
    'bonus_trn_count',
    'share_top_waiter',
    'share_bonuses_used_top_waiter',
    'share_top_places',
    'num_of_trn_prcnt',
    'days_visits_prcnt',
    'gross_amount_mean_prcnt',
    'gross_amount_sum_prcnt',
    'bonuses_accum_sum_prcnt',
    'bonuses_used_sum_prcnt',
    'num_of_waiters_prcnt',
    'gross_amount_max_prcnt',
    'first_last_trn_diff_prcnt',
    'first_second_trn_diff_prcnt',
    'first_third_trn_diff_prcnt',
    'time_between_trn_median_prcnt',
    'trn_per_day_prcnt',
    'num_of_places_prcnt'

    # ISO precision@100 = 0.27, OCSVM precision@100 = 0.14, LOF precision@100 = 0.00
    # 'share_top_waiter',
    # 'bonuses_accum_sum',
    # 'bonuses_accum_sum_prcnt',
    # 'bonuses_used_sum',
    # 'share_bonuses_used_top_waiter',
    # 'num_of_waiters',
    # 'num_of_trn_prcnt',
    # 'num_of_waiters_prcnt',
    # 'first_last_trn_diff_prcnt',
    # 'days_visits_prcnt'

    # ISO precision@100 = 0.24, OCSVM precision@100 = 0.20, LOF precision@100 = 0.00
    # 'share_top_waiter',
    # 'bonuses_accum_sum_prcnt',
    # 'bonuses_used_sum',
    # 'gross_amount_sum',
    # 'share_bonuses_used_top_waiter',
    # 'num_of_waiters'


    # 'trn_per_day',
    # 'num_of_waiters',
    # 'share_top_waiter',
    # 'bonuses_accum_sum',
    # 'share_bonuses_used_top_waiter'

]


FEATURES_FOR_PERCENTILE = [
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
    'num_of_places'
]

PERCENTILE_FEATURES = [
    'num_of_trn_prcnt',
    'days_visits_prcnt',
    'gross_amount_mean_prcnt',
    'gross_amount_sum_prcnt',
    'bonuses_accum_sum_prcnt',
    'bonuses_used_sum_prcnt',
    'num_of_waiters_prcnt',
    'gross_amount_max_prcnt',
    'first_last_trn_diff_prcnt',
    'first_second_trn_diff_prcnt',
    'first_third_trn_diff_prcnt',
    'time_between_trn_median_prcnt',
    'trn_per_day_prcnt',
    'num_of_places_prcnt'
]

def load_data(
    activity_state = 1, 
    days_visits = 1,
    num_of_trn = 1,
    place_num_of_waiters = 1,
    total_num_of_trn = 8,
    min_working_days = 2):
    df = pd.read_parquet(DATA_PATH + "processed_transactions.parquet", engine="pyarrow")
    client_data = pd.read_parquet(DATA_PATH + "client_level_features.parquet", engine="pyarrow")
    client_data = client_data[client_data['num_of_trn'] > activity_state]
    client_data = client_data[client_data['days_visits'] > days_visits]
    client_data['is_fraud'] = client_data['person_id'].isin(FRAUD_IDS)
    client_data = client_data.set_index('person_id')

    for feat, prcnt_feat in zip(FEATURES_FOR_PERCENTILE, PERCENTILE_FEATURES):
        client_data[prcnt_feat] = client_data[feat].rank(pct=True)
    
    waiter_week_data = pd.read_parquet(DATA_PATH + "waiter_week_features.parquet", engine="pyarrow")
    waiter_week_data = waiter_week_data[waiter_week_data['num_of_trn'] > num_of_trn]
    waiter_week_data = waiter_week_data[waiter_week_data['place_num_of_waiters'] > place_num_of_waiters]
    waiter_week_data['is_fraud'] = waiter_week_data['waiter_week'].isin(FRAUD_WAITER_WEEK_IDS)
    waiter_week_data = waiter_week_data[waiter_week_data['working_days'] >= min_working_days]
    waiter_week_data = waiter_week_data.set_index('waiter_week')

    waiter_level_data = pd.read_parquet(DATA_PATH + "waiter_level_features.parquet", engine="pyarrow")
    waiter_level_data = waiter_level_data[waiter_level_data['num_of_trn'] > total_num_of_trn]
    waiter_level_data = waiter_level_data.set_index('waiter_id')


    return df, client_data, waiter_week_data, waiter_level_data