import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats
import pickle as p

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

import sys, json, requests
import operator
import random
import base64
import zlib

# < ---------------------- Configuration ---------------------->
master_data_input_path = 'data/data_all.csv'

class GA(object):

    #initialize variables and lists
    def __init__(self): 

        self.models = []
        self.price_std = []
        self.price_mean = []
        self.parents = []
        self.newparents = []
        self.bests = []
        self.best_p = [] 
        self.price_names = []
        self.iterated = 1
        self.population = 0
        self.epoch = 0
        self.best_price = []

        # increase max recursion for long stack
        iMaxStackSize = 15000
        sys.setrecursionlimit(iMaxStackSize)

    # create the initial population 
    def initialize(self):
        self.num_item = self.price_mean.shape[0]
        for i in range(self.population):
            parent = (np.random.standard_normal(self.num_item)*self.price_std*2 + self.price_mean).tolist()
            self.parents.append(parent)

    # set the details of this problem
    def properties(self, models, population, max_epoch, price_std, price_mean, price_names):

        self.models = models
        self.population = population
        self.epoch = max_epoch
        self.price_std = price_std
        self.price_mean = price_mean
        self.price_names = price_names
        self.initialize()

    # calculate the fitness function for X
    def fitness(self, prices):
        
        df = pd.DataFrame(prices, columns = self.price_names) 
        quantities = np.zeros((self.population, len(self.models)))
        for i in range(len(self.models)):
            quantities[:, i] = self.models[i].predict(df)
        fitness = np.zeros((self.population))
        for i in range(len(self.models)):
            fitness += prices[:, i]*quantities[:, i]
        
        return fitness
   
    # run generations of GA
    def evaluation(self):

        # loop through parents and calculate fitness
        best_pop = self.population // 2
        ft = self.fitness(np.asarray(self.parents))
        for i in range(len(self.parents)):
            parent = self.parents[i]
            self.bests.append((ft[i], parent))

        # sort the fitness list by fitness
        self.bests.sort(key=operator.itemgetter(0), reverse=True)
        self.best_p = self.bests[:best_pop]
        highest_fitness = self.best_p[0][0]
        highest_price = self.best_p[0][1]
        self.best_p = [x[1] for x in self.best_p]
        
        return highest_fitness, highest_price

    # mutate children after certain condition
    def mutation(self, ch):

        for i in range(len(ch)):
            if random.uniform(0, 1) > 0.95:
                ch[i] = np.random.standard_normal(1)[0]*self.price_std[i]*2 + self.price_mean[i]
        return ch

    # crossover two parents to produce two children by miixing them under random ration each time
    def crossover(self, ch1, ch2):

        threshold = random.randint(1, len(ch1)-1)
        tmp1 = ch1[threshold:]
        tmp2 = ch2[threshold:]
        ch1 = ch1[:threshold]
        ch2 = ch2[:threshold]
        ch1.extend(tmp2)
        ch2.extend(tmp1)

        return ch1, ch2

    # run the GA algorithm
    def run(self):
        if self.epoch < self.iterated:
            return self.best_price
        # run the evaluation once
        highest_fitness, highest_price = self.evaluation()
        newparents = []
        pop = len(self.best_p)
        
        print("{}th generation" .format(self.iterated))
        print("best solution so far: {}".format(highest_fitness))
        self.best_price = highest_price

        # randomly shuffle the best parents
        random.shuffle(self.best_p)
        for i in range(0, pop):
            if i < pop-1:
                r1 = self.best_p[i]
                r2 = self.best_p[i+1]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)
            else:
                r1 = self.best_p[i]
                r2 = self.best_p[0]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)

        # mutate the new children and potential parents to ensure global optima found
        for i in range(len(newparents)):
            newparents[i] = self.mutation(newparents[i])    

        self.parents = newparents
        self.bests = []
        self.best_p = []
        self.iterated += 1
        return self.run()

class rms_pricing_model():
    def __init__(self, data):
        
        self.models = {}
        

        # Ingest Correctly Shaped Data
        # sales_data = pd.read_csv(
        #     data_filepath,
        #     usecols=['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_', 'Price_'])

        # Ingest Correctly Shaped Data
        sales_data = data[['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_', 'Price_']].copy()

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

    def get_and_save_price_info(self, price_info_path):
        prices = self.data[self.price_columns]
        prices_std = prices.std(axis = 0, skipna = True)
        prices_mean = prices.mean(axis = 0, skipna = True)
        p.dump((prices_std, prices_mean, self.price_columns), open(price_info_path,'wb'))


    def get_performance(self, item_id):

        with open('keys.json') as f:
            HOST_KEY = json.load(f)['host_key']

        sales_data_wide_clean = self.data.copy()

        target_colname = 'Qty_' + str(item_id)

        if target_colname not in sales_data_wide_clean.columns:
            print('Item Not Found in Dataset.')
            return None

        Price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]
        target_column = target_colname

        Week = sales_data_wide_clean['Wk'].copy()
        X = sales_data_wide_clean[Price_columns].copy()
        y = sales_data_wide_clean[target_column].copy()

        payload = {
        'code':HOST_KEY,
        }

        data = {
            'X':X.to_json(),
            'y':y.to_json(),
            'Week':Week.to_json()
        }

        url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_2_cv'
        result = requests.get(url, params=payload, data=base64.b64encode(zlib.compress(json.dumps(data).encode('utf-8'))))
        print(item_id, 'cv status code', result.status_code)
        outp = result.json()
        outp['item_id'] = int(item_id)

        # logo = LeaveOneGroupOut()
        # n_splits = logo.get_n_splits(groups=Week)

        # r2_total = 0
        # mae_total = 0
        # rmse_total = 0

        # target_splits = 8
        # n_actual_splits = 0
        # nth_split = 0

        # for train_index, test_index in logo.split(X, y, Week):
            
        #     # probability to run this week on cross-validation
        #     # This logic ensures only 4 cv splits are done to save time
        #     cv_prob = max(0,(target_splits - n_actual_splits)/(n_splits-nth_split))
        #     nth_split += 1

        #     if np.random.random()>cv_prob:
        #         continue

        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #     y_train = np.asarray(y_train).ravel()
        #     y_test = np.asarray(y_test).ravel()

        #     test_model = XGBRegressor()
        #     test_model.fit(X_train, y_train)
        #     y_pred = test_model.predict(X_test)

        #     r2_total += r2_score(y_true=y_test, y_pred=y_pred)
        #     mae_total += mean_absolute_error(y_true=y_test, y_pred=y_pred)
        #     rmse_total += np.sqrt(
        #         mean_squared_error(y_true=y_test, y_pred=y_pred))

        #     n_actual_splits += 1

        # n_splits = n_actual_splits

        # item_id = item_id
        # avg_sales = y.mean().values[0]
        # r2 = r2_total / n_splits
        # mae = mae_total / n_splits
        # mpe = mae / avg_sales
        # rmse = rmse_total / n_splits

        # outp = {
        #     'item_id': item_id,
        #     'avg_sales': avg_sales,
        #     'r2_score': r2,
        #     'mae_score': mae,
        #     'mpe_score': mpe,
        #     'rmse_score': rmse
        # }

        return outp

    def get_all_performance(self):
        item_ids = [int(x.split('_')[1]) for x in self.price_columns]
        perf_df = pd.DataFrame(columns=[
            'item_id','avg_sales','r2_score',
            'mae_score','mpe_score','rmse_score']
            )
        for item_id in item_ids:
            item_perf = self.get_performance(item_id)
            perf_df = perf_df.append(item_perf, ignore_index=True)
        return perf_df

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
        target_column = target_colname

        X = sales_data_wide_clean[Price_columns].copy()
        y = sales_data_wide_clean[target_column].copy()

        model = XGBRegressor()
        model.fit(X, y)
        
        self.models[item_id] = model

        return model
    
    def train_all_items(self, retrain=True):
        item_ids = [int(x.split('_')[1]) for x in self.price_columns]
        for item_id in item_ids:
            if retrain==False:
                if item_id not in self.models.keys():
                    self.get_model(item_id)
            else:
                self.get_model(item_id)