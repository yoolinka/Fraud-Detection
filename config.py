import pandas as pd

DATA_PATH = "/Users/yuliia/Documents/Fraud-Detection/parquet/"

FRAUD_IDS = [
    11968000, 11970409, 11726701, 14827913, 12411311, 11098795, 12412748, 11812494, 12396334, 11699175,
    10896313, 16376555, 16921722, 16846666, 16794470, 16440373, 11766135, 13969910, 12234377, 12861171, 
    13076915, 13337997, 13239788, 12614443, 12646735, 13119461, 13859241, 13119415, 3349318, 12830394, 
    12878021, 11973649, 12199729, 11973677, 12338391, 11962124, 12396486, 12199719, 12199704, 12199723, 
    12199714, 12284741, 12331474, 12284496, 12342012, 12963892
]

FRAUD_WAITER_IDS = [
    '539f5da6c76b7cd7fa3d859c_56',
    '539f5da6c76b7cd7fa3d859f_...',
    '539f5da6c76b7cd7fa3d859f_547',
    '539f5da6c76b7cd7fa3d859f_121',
    '542704afe4b07d8ca118acb2_18343',
    '539f5da6c76b7cd7fa3d859c_09',
    '5bbf439c4e928fc6872920fe_16694',
    '539f5da8c76b7cd7fa3d85dd_111',
    '539f5da8c76b7cd7fa3d85dd_7921',
    '539f5da8c76b7cd7fa3d85dd_3972',
    '539f5da8c76b7cd7fa3d85dd_9748',
    '539f5da8c76b7cd7fa3d85dd_1947',
    '54de0d442cdc51c13b68ed82_2045',
    '539f5da8c76b7cd7fa3d85d8_2045',
    '539f5da8c76b7cd7fa3d85d8_482109436',
    '65117ad169a6f4704ea03ecd_18368',
    '65117ad169a6f4704ea03ecd_2022',
    '601c0e3a4e92a83b9f84122c_1111',
    '601c0e3a4e92a83b9f84122c_7777',
    '601c0e3a4e92a83b9f84122c_122',
    '601c0e3a4e92a83b9f84122c_16467',
    '601c0e3a4e92a83b9f84122c_5252',
    '601c0e3a4e92a83b9f84122c_0597',
    '601c0e3a4e92a83b9f84122c_8874',
    '539f5da7c76b7cd7fa3d85af_6094',
    '539f5da7c76b7cd7fa3d85af_2197',
    '539f5da7c76b7cd7fa3d85af_6729',
    '601c0e3a4e92a83b9f84122c_7536',
    '602b89094e9293740b7f39a3_8885',
    '602b89094e9293740b7f39a3_66',
    '602b89094e9293740b7f39a3_1009',
    '601c0e3a4e92a83b9f84122c_15933',
]

FEATURES = [
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
]

SKEWED = [
    'bonus_trn_count'
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

def load_data(activity_state = 1, days_visits = 1):
    df = pd.read_parquet(DATA_PATH + "processed_transactions.parquet", engine="pyarrow")
    client_data = pd.read_parquet(DATA_PATH + "client_level_features.parquet", engine="pyarrow")
    client_data = client_data[client_data['num_of_trn'] > activity_state]
    client_data = client_data[client_data['days_visits'] > days_visits]
    client_data['is_fraud'] = client_data['person_id'].isin(FRAUD_IDS)
    client_data = client_data.set_index('person_id')

    for feat, prcnt_feat in zip(FEATURES_FOR_PERCENTILE, PERCENTILE_FEATURES):
        client_data[prcnt_feat] = client_data[feat].rank(pct=True)
    return df, client_data