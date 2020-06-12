import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

class rms_pricing_model():
    def __init__(self, data_filepath):
        
        self.models = {}
        

        # Ingest Correctly Shaped Data
        sales_data = pd.read_csv(
            data_filepath,
            usecols=['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_', 'Price_'])

        # Optimize Memory Usage by Downcasting etc.
        sales_data = optimize_memory(sales_data)

        # Convert Data to Wide Format
        sales_data_wide = sales_data.set_index(
            ['Wk', 'Tier', 'Store',
             'Item_ID']).unstack(level=-1).reset_index().copy()
        sales_data_wide.columns = [
            ''.join(str(i) for i in col).strip()
            for col in sales_data_wide.columns.values
        ]
        sales_data_wide = sales_data_wide.sort_values(
            ['Tier', 'Store', 'Wk'], ascending=True).reset_index(drop=True)

        # # Remove Store,Weeks with Nan Sales for Some Items ( Not sure if we want to do this or replace with 0 )
        # # Drops about 421 rows, hence seems reasonable to remove
        sales_data_wide_clean = sales_data_wide.dropna(axis=0).copy()
        self.data = sales_data_wide_clean
        self.price_columns = [
        col for col in sales_data_wide_clean.columns if col.startswith('Price')
        ]

    def get_performance(self, item_id):

        sales_data_wide_clean = self.data.copy()

        target_colname = 'Qty_' + str(item_id)

        if target_colname not in sales_data_wide_clean.columns:
            print('Item Not Found in Dataset.')
            return None

        Price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]
        target_column = [target_colname]

        Week = sales_data_wide_clean['Wk'].copy()
        X = sales_data_wide_clean[Price_columns].copy()
        y = sales_data_wide_clean[target_column].copy()

        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(groups=Week)

        r2_total = 0
        mae_total = 0
        rmse_total = 0

        for train_index, test_index in logo.split(X, y, Week):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_train = np.asarray(y_train).ravel()
            y_test = np.asarray(y_test).ravel()

            test_model = XGBRegressor()
            test_model.fit(X_train, y_train)
            y_pred = test_model.predict(X_test)

            r2_total += r2_score(y_true=y_test, y_pred=y_pred)
            mae_total += mean_absolute_error(y_true=y_test, y_pred=y_pred)
            rmse_total += np.sqrt(
                mean_squared_error(y_true=y_test, y_pred=y_pred))

        item_id = target_column[0]
        avg_sales = y.mean().values[0]
        r2 = r2_total / n_splits
        mae = mae_total / n_splits
        mpe = mae / avg_sales
        rmse = rmse_total / n_splits

        outp = {
            'item_id': item_id,
            'avg_sales': avg_sales,
            'r2_score': r2,
            'mae_score': mae,
            'mpe_score': mpe,
            'rmse_score': rmse
        }

        return outp

    def get_model(self, item_id):

        sales_data_wide_clean = self.data.copy()

        target_colname = 'Qty_' + str(item_id)

        if target_colname not in sales_data_wide_clean.columns:
            print('Item Not Found in Dataset.')
            return None

        Price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]
        target_column = [target_colname]

        X = sales_data_wide_clean[Price_columns].copy()
        y = sales_data_wide_clean[target_column].copy()

        model = XGBRegressor()
        model.fit(X, y)
        
        self.models[item_id] = model

        return model
    
    def train_all_items(self, retrain=False):
        item_ids = [int(x.split('_')[1]) for x in self.price_columns]
        for item_id in item_ids:
            if retrain==False:
                if item_id not in self.models.keys():
                    self.get_model(item_id)
            else:
                self.get_model(item_id)