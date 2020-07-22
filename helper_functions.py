import pandas as pd
from typing import List
import numpy as np


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df

def optimize_memory(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))

def parse_training_request(request):
    raw_data = request.files['file']
    data_all = pd.read_csv(raw_data)
    np.random.seed(42)
    item_subset = list(np.random.choice(data_all['Item_ID'].unique(), size=10, replace=False))
    data_subset = data_all.loc[data_all['Item_ID'].isin(item_subset)].copy()

    data = data_subset

    cv_acc = request.form['cv_acc']
    project_id = request.form['project_id']

    return (data, cv_acc, project_id)

def get_top_features(xgb_model, n=5):
    feature_names = xgb_model._Booster.feature_names
    importances = xgb.feature_importances_
    imp_df = pd.DataFrame({'feature_name':feature_names,
                           'importance':importances
                          })
    imp_df = imp_df.sort_values('importance',
                                ascending=False
                               )
    return imp_df.head(n)