"""Transaction-based feature extraction."""

from typing import List
import numpy as np
import pandas as pd



def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for unsupervised waiter fraud detection.
    Focuses on loyalty card abuse and peer-group normalization.
    Returns:
        DataFrame aggregated by waiter_id and transaction features
    """

    df = df.groupby('waiter_id').agg({
        'trn_id': 'count',
        'gross_amount': 'sum',
        'bonusses_used': 'sum',
        'bonusses_accum': 'sum',
        'discount_amount': 'sum',
        'bonus_used_flag': 'sum'
    })

    return df


def get_fraud_labels(df: pd.DataFrame, fraud_person_ids: List[int]) -> pd.DataFrame:
    """
    Get fraud labels for waiters based on fraud cards.
    Waiter is considered fraud if they used a fraud card more than 3 times.

    Args:
        df: Transaction dataframe
        fraud_person_ids: List of known fraud person_ids
        
    Returns:
        DataFrame with waiter_id and is_fraud_waiter label
    """

    fraud_cards = set(fraud_person_ids)

    fraud_waiters = (
        df[df['person_id'].isin(fraud_cards)]
        .groupby('waiter_id')
        .size()
        .reset_index(name='fraud_transaction_count')
    )

    fraud_waiters = fraud_waiters[fraud_waiters['fraud_transaction_count'] > 2]

    return fraud_waiters[['waiter_id']].assign(is_fraud_waiter=1)
