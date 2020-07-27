import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats
import pickle as p

# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import LeaveOneGroupOut

import cvxpy as cp # for conflict detection
from deap import algorithms # for GA
from deap import base # for GA
from deap import creator # for GA
from deap import tools # for GA

import sys
import json
import requests
import operator
import random
import base64
import zlib
import os
from pathlib import Path
import gc
import shutil
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
log = logging.getLogger('fillet-flask.sub')


# Set current working directory
HOME = os.environ['HOME_SITE']
# HOME = ''

# Function Key required to call fillet-functions
# with open('keys.json') as f:
#       HOST_KEY = json.load(f)['host_key']
HOST_KEY = os.environ['FUNCTIONS_KEY']

# input: price_std, price_mean, price_names, constraints(in pre-specified json format), regressors (in dictionary), 
# population, generation, costs(optional), pre-set penalty constants, step(for prices), 
# random_seed for replication of results
def solve_cvx(eq, eq_s, ineq, ineq_s, mode):
    """
    eq, eq_s, ineq, ineq_s,: np.array
    p_c (list): price_columns
    p_r (dict): price_range
    mode (str): 'sum' or 'sum_squares'
    """
    import cvxpy as cp
    assert (np.sum(eq!=0)>0 or np.sum(ineq!=0)>0), 'Must at least have some constraints, cannot all be None.'
    x = cp.Variable((eq.shape[1], 1))
    if mode == 'sum':
        objective = cp.Minimize(cp.sum(x)) # any surrogate objective
    elif mode == 'sum_squares':
        objective = cp.Minimize(cp.sum_squares(x))
    else:
        return None
    constraints = []
    if np.sum(eq!=0)>0: # if all of them are zero, then no need to add
        constraints.append(eq@x-eq_s == 0)
    if np.sum(ineq!=0)>0:
        constraints.append(ineq@x-ineq_s >= 0)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return x.value, prob.status

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

def list_to_matrix(rules, product_to_idx, penalty):
    """
    rules (list): rules
    product_to_idx (dict)
    """
    matrix1 = np.zeros((len(rules), len(product_to_idx)))
    shifts1 = np.zeros((len(rules), 1))
    penalty1 = np.zeros((len(rules), 1))
    for k in range(len(rules)):
        products = rules[k]['products']
        scales = rules[k]['scales']
        scales = [float(s) for s in scales]
        for j, product in enumerate(products):
            matrix1[k, product_to_idx[product]] = float(scales[j])
        shifts1[k,0] = float(rules[k]['shift'])
        if rules[k]['penalty'] == -1:
            penalty1[k,0] = penalty
        else:
            penalty1[k,0] = rules[k]['penalty']*penalty # e.g. 4 * 10000
    return matrix1, shifts1, penalty1

# ====================================================

def GeneticAlgorithm(prices_std_list, prices_mean_list, price_columns, rules, regressors, population, generation, 
                        costs=None, penalty_hard_constant=1000000, penalty_soft_constant=1000, step=0.05, 
                        random_seed=1):
    log.info('SETTING UP GA ENVIRONMENT')
    # 1. Preprocess rules and price limits
    num_item = len(price_columns)
    product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_columns)}
    rule_list_old, price_range = rules
    rule_list = [rule for rule in rule_list_old if set(rule['products']).issubset(set(product_to_idx.keys()))] # filter out the ones not in price_columns
    print('{} out of {} rules contain products not in price_columns.'.format(len(rule_list_old)-len(rule_list), len(rule_list_old)))
    hard_rule_eq_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == 0)]
    hard_rule_small_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == 1)]
    hard_rule_large_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == 2)]
    hard_rule_smalleq_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == 3)]
    hard_rule_largeeq_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == 4)]
    soft_rule_eq_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == 0)]
    soft_rule_small_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == 1)]
    soft_rule_large_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == 2)]
    soft_rule_smalleq_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == 3)]
    soft_rule_largeeq_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == 4)]
    price_range_dic = {}
    for item in price_range:
        price_range_dic[item['item_id']] = [item['max'], item['min']]
    # 2. Find valid price vectors to start
    # 2.1. put hard equalities into matrix form
    matrix1, shifts1, penalty1 = list_to_matrix(hard_rule_eq_list, product_to_idx, penalty_hard_constant)
    # 2.2. put hard inequalities into matrix form
    matrix2_1, shifts2_1, penalty2_1 = list_to_matrix(hard_rule_small_list, product_to_idx, penalty_hard_constant)
    matrix2_1 = matrix2_1*(-1)
    shifts2_1 = shifts2_1*(-1)+0.0001
    matrix2_2, shifts2_2, penalty2_2 = list_to_matrix(hard_rule_large_list, product_to_idx, penalty_hard_constant)
    shifts2_2 = shifts2_2+0.0001
    matrix2_3, shifts2_3, penalty2_3 = list_to_matrix(hard_rule_smalleq_list, product_to_idx, penalty_hard_constant)
    matrix2_3 = matrix2_3*(-1)
    shifts2_3 = shifts2_3*(-1)
    matrix2_4, shifts2_4, penalty2_4 = list_to_matrix(hard_rule_largeeq_list, product_to_idx, penalty_hard_constant)
    # 2.2.2. adding price ranges
    prices = [i.split('_')[1] for i in price_columns]
    # 2.2.2.1 adding price floor
    matrix2_5 = np.zeros((2*len(prices), len(prices)))
    shifts2_5 = np.zeros((2*len(prices), 1))
    for i, product in enumerate(prices):
        matrix2_5[i, i] = 1.
        if int(product) not in price_range_dic.keys():
            print('product {} is not given price range, assumed to be within [0.5, 20].'.format(product))
            shifts2_5[i,0] = 0.5
        else:
            shifts2_5[i,0] = price_range_dic[int(product)][1]
    # 2.2.2.1 adding price cap
    for i, product in enumerate(prices):
        matrix2_5[len(prices)+i, i] = -1.
        if int(product) not in price_range_dic.keys():
            shifts2_5[len(prices)+i,0] = -20.
        else:
            shifts2_5[len(prices)+i,0] = -price_range_dic[int(product)][0]
    penalty2_5 = np.full((matrix2_5.shape[0],1), penalty_hard_constant)
    # 2.2.3. Put together hard inequality and price range
    matrix2 = np.vstack([matrix2_1, matrix2_2, matrix2_3, matrix2_4, matrix2_5])
    shifts2 = np.vstack([shifts2_1, shifts2_2, shifts2_3, shifts2_4, shifts2_5])
    penalty2 = np.vstack([penalty2_1, penalty2_2, penalty2_3, penalty2_4, penalty2_5])
    # 2.3. get 2 valid individuals from linear programming
    val_ind1, status1 = solve_cvx(matrix1, shifts1, matrix2, shifts2, 'sum')
    val_ind2, status2 = solve_cvx(matrix1, shifts1, matrix2, shifts2, 'sum_squares')
    print('status 1: {}'.format(status1))
    print('status 2: {}'.format(status2))
    print('val_ind1 shape: {}'.format(val_ind1.shape))
    print('val_ind2 shape: {}'.format(val_ind2.shape))
    assert status1 != 'infeasible' and status2 != 'infeasible', 'Hard constraints must have feasible region.'
    # 3. Put soft constraints into matrix form
    # 3.1. soft equality
    matrix3, shifts3, penalty3 = list_to_matrix(soft_rule_eq_list, product_to_idx, penalty_soft_constant)
    # 3.2. soft inequality
    matrix4_1, shifts4_1, penalty4_1 = list_to_matrix(soft_rule_small_list, product_to_idx, penalty_soft_constant)
    matrix4_1 = matrix4_1*(-1)
    shifts4_1 = shifts4_1*(-1)+0.0001
    matrix4_2, shifts4_2, penalty4_2 = list_to_matrix(soft_rule_large_list, product_to_idx, penalty_soft_constant)
    shifts4_2 = shifts4_2+0.0001
    matrix4_3, shifts4_3, penalty4_3 = list_to_matrix(soft_rule_smalleq_list, product_to_idx, penalty_soft_constant)
    matrix4_3 = matrix4_3*(-1)
    shifts4_3 = shifts4_3*(-1)
    matrix4_4, shifts4_4, penalty4_4 = list_to_matrix(soft_rule_largeeq_list, product_to_idx, penalty_soft_constant)
    matrix4 = np.vstack([matrix4_1, matrix4_2, matrix4_3, matrix4_4])
    shifts4 = np.vstack([shifts4_1, shifts4_2, shifts4_3, shifts4_4])
    penalty4 = np.vstack([penalty4_1, penalty4_2, penalty4_3, penalty4_4])
    # 4. Run GA using DEAP library
    log.info('SETTING UP GA RULESET')
    # 4.1. Define fitness function
    def evalObjective(individual, report=False):
        """
        returns:
        (revenue, penalty_): revenue of this individual and penalty from it violating the constraints
        """
        # # Calculating revenue
        # quantity = np.zeros((num_item))
        # individual = individual.round(2)
        # for code in regressors: # TODO: use multiple workers here to speedup the optimization process
        #     X = pd.DataFrame(individual.reshape(1, -1), columns=price_columns)
        #     X = X.reindex(sorted(X.columns), axis=1)
        #     quantity[product_to_idx[code]] = regressors[code].predict(X)

        # MULTITHREADED PREDICT
        pred_dict = {}
        quantity = np.zeros((num_item))
        f = lambda x: 0.05 * np.round(x/0.05)
        individual = f(individual)

        def opti_predict(code, individual):
            X = pd.DataFrame(individual.reshape(1, -1), columns=price_columns)
            X = X.reindex(sorted(X.columns), axis=1)
            pred_dict[code] = regressors[code].predict(X)

        opti_processes = []
        opti_results_ls = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            for code in regressors:
                opti_processes.append(executor.submit(opti_predict, code, individual))
        for task in as_completed(opti_processes):
            opti_results_ls.append(task.result())

        for code in regressors:
            quantity[product_to_idx[code]] = pred_dict[code]


        # Calculating constraint violation penalty
        output = individual.dot(quantity)
        temp1 = (matrix1.dot(individual.reshape(-1, 1)) - shifts1).round(2)
        mask1 = temp1 != 0
        penalty_1 = mask1.T.dot(penalty1)
        temp2 = (matrix2.dot(individual.reshape(-1, 1)) - shifts2).round(2)
        mask2 = temp2 < 0
        penalty_2 = mask2.T.dot(penalty2)
        temp3 = (matrix3.dot(individual.reshape(-1, 1)) - shifts3).round(2)
        mask3 = temp3 != 0
        penalty_3 = mask3.T.dot(penalty3)
        temp4 = (matrix4.dot(individual.reshape(-1, 1)) - shifts4).round(2)
        mask4 = temp4 < 0
        penalty_4 = mask4.T.dot(penalty4)
        if report:
            return [output, np.sum(mask1)+np.sum(mask2), np.sum(mask3)+np.sum(mask4)]
        if penalty_1.shape[0] > 0 and penalty_1.shape[1] > 0:
            output -= penalty_1[0,0]
        if penalty_2.shape[0] > 0 and penalty_2.shape[1] > 0:
            output -= penalty_2[0,0]
        if penalty_3.shape[0] > 0 and penalty_3.shape[1] > 0:
            output -= penalty_3[0,0]
        if penalty_4.shape[0] > 0 and penalty_4.shape[1] > 0:
            output -= penalty_4[0,0]
        return (output,)
    # 4.2. Initialize individuals and operations
    log.info('INITIALIZING INDIVIDUALS...')
    creator.create("RevenuePenalty", base.Fitness, weights=(1.,))
    creator.create("Individual", np.ndarray, fitness=creator.RevenuePenalty)
    toolbox = base.Toolbox()
    def get_individual(num_item, price_std, price_mean):
        return creator.Individual(np.random.standard_normal(num_item)*price_std*2 + price_mean)
    toolbox.register("individual", get_individual, num_item, np.array(prices_std_list), np.array(prices_mean_list))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalObjective)
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 4.3. Run the algoritm
    log.info('RUNNING GA...')
    random.seed(64)
    pop = toolbox.population(n=population)
    pop.append(creator.Individual(val_ind1.round(2).flatten()))
    pop.append(creator.Individual(val_ind2.round(2).flatten()))
    print('fitess of ind1: ',evalObjective(val_ind1.round(2).flatten()))
    print('fitess of ind2: ',evalObjective(val_ind2.round(2).flatten()))
#     hof = tools.ParetoFront(similar=np.array_equal)
    hof = tools.HallOfFame(2, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    print('GA started running...')
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generation, stats=stats,
                        halloffame=hof)
    log.info('GA COMPLETED.')
    return pop, stats, hof, evalObjective(hof[0], report=True)

# ====================================================


class rms_pricing_model():
    def __init__(self, data):

        self.models = {}

        # Ingest Correctly Shaped Data
        sales_data = data[['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_',
                           'Price_']].copy()

        # Optimize Memory Usage by Downcasting etc.
        sales_data = optimize_memory(sales_data)

        # Convert Data to Wide Format
        sales_data_wide = sales_data.set_index(
            ['Wk', 'Tier', 'Store',
             'Item_ID']).unstack(level=-1).reset_index().copy()

        # Clear unused variables to save memory
        del data
        del sales_data
        gc.collect()

        # Rename Columns
        sales_data_wide.columns = [
            ''.join(str(i) for i in col).strip()
            for col in sales_data_wide.columns.values
        ]

        # Sort df
        sales_data_wide = sales_data_wide.sort_values(
            ['Tier', 'Store', 'Wk'], ascending=True).reset_index(drop=True)

        # Remove Store,Weeks with Nan Sales for Items
        sales_data_wide_clean = sales_data_wide.dropna(axis=0).copy()

        # Save reshaped data to attibutes
        self.data = sales_data_wide_clean.copy()

        # Save price columns (features) to attributes
        self.price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]

        # Clear unused variables to save memory
        del sales_data_wide
        del sales_data_wide_clean
        gc.collect()

    def get_and_save_price_info(self, price_info_path):
        '''This function extracts the mean and variance
        of prices of the items in a dataset. The mean and
        variance is later used in price optimization initialization.
        '''
        prices = self.data[self.price_columns]
        prices_std = prices.std(axis=0, skipna=True)
        prices_mean = prices.mean(axis=0, skipna=True)
        p.dump((prices_std, prices_mean, self.price_columns),
               open(price_info_path, 'wb'))

    def get_performance(self, item_id, proj_id, modeltype='default'):
        '''This function calls fillet-functions to run a
        cross-validation check for one item and saves the results 
        to disk under the project folder. 
        '''

        with open(HOME + '/fillet_functions_api_endpoints.json') as f:
            fillet_func_urls = json.load(f)

        log.info(f'CROSS VALIDATING ITEM_ID {item_id} MODEL')

        # Copy object data
        sales_data_wide_clean = self.data.copy()

        # Specify target column of interest
        target_colname = 'Qty_' + str(item_id)
        target_column = [target_colname]

        # If item_id target column isn't in the columns
        if target_colname not in sales_data_wide_clean.columns:
            log.info(f'ITEM_ID {target_colname} NOT FOUND IN PROJECT {proj_id}')
            return None

        # Specify feature columns
        Price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]
        
        # Formalize Data Required for Cross Validation
        Week = sales_data_wide_clean[['Wk']].copy()
        X = sales_data_wide_clean[Price_columns].copy()
        X = X.reindex(sorted(X.columns), axis=1)
        y = sales_data_wide_clean[target_column].copy()

        payload = {
            'code': HOST_KEY,
        }

        # Create a temp path to stage CV Data
        temp_cv_path = f'temp/cv/{proj_id}/{item_id}'
        if not os.path.isdir(temp_cv_path):
            Path(temp_cv_path).mkdir(parents=True)

        # Send CV requests to fillet-functions until successful
        while True:
            try:
                # Stage files in temp folder
                X.to_parquet(temp_cv_path + '/X.parquet')
                y.to_parquet(temp_cv_path + '/y.parquet')
                Week.to_parquet(temp_cv_path + '/Wk.parquet')

                # Open staged files to send to fillet-functions
                files = {
                    'X_file': open(temp_cv_path + '/X.parquet', 'rb'),
                    'y_file': open(temp_cv_path + '/y.parquet', 'rb'),
                    'Wk_file': open(temp_cv_path + '/Wk.parquet', 'rb'),
                }

                # Send request to fillet-functions
                url = fillet_func_urls[modeltype]['cv']
                url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_2_cv'
                result = requests.get(url, params=payload, files=files)
                outp = result.json()

                # Add item_id to cv results
                outp['item_id'] = int(item_id)
                break
            except Exception as e:
                log.info(f'CV {item_id} FAILED. RETRYING...')
                log.info(f'{e}')
                time.sleep(60)
                pass

        # Clear unused variables to save memory
        del sales_data_wide_clean
        gc.collect()

        # Create folder to store intermediate cv results for the project
        cv_results_path = HOME + f'/projects/{proj_id}/cv/'
        if not os.path.isdir(cv_results_path):
            Path(cv_results_path).mkdir(parents=True)

        # Write cv results to project cv results folder
        with open(cv_results_path + f'{item_id}_cv_perf.json', 'w') as outfile:
            json.dump(outp, outfile)

        # Close staged files
        files['X_file'].close()
        files['y_file'].close()
        files['Wk_file'].close()

        # Clear unused variables to save memory
        del X
        del y
        del Week
        gc.collect()

        return outp

    def get_all_performance(self, proj_id, modeltype='default'):
        '''This function loops through items in a project,
        and calls get_performance to get and save CV results
        to the project folder. In order to speed up the
        processing, it makes up to 5 parallel calls to 
        fillet-functions via get_performance.
        '''

        # Get item_ids of a project
        item_ids = [int(x.split('_')[1]) for x in self.price_columns]
        
        # Create placeholder DataFrame
        perf_df = pd.DataFrame(columns=[
            'item_id', 'avg_sales', 'r2_score', 'mae_score', 'mpe_score',
            'rmse_score'
        ])

        # Send parallel calls to fillet-functions via get_performance
        processes_cv = []
        results_ls_cv = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for item_id in item_ids:
                processes_cv.append(
                    executor.submit(self.get_performance, item_id, proj_id))
        for task in as_completed(processes_cv):
            perf_df = perf_df.append(task.result(), ignore_index=True)

        # remove temp cv folder
        try:
            shutil.rmtree('temp/cv')
        except:
            pass

        # Check That All Models are Trained, Else Retrain
        proj_cv_path = HOME + f'/projects/{proj_id}/cv'
        cv_filenames = os.listdir(proj_cv_path)

        # Determine which CV results are completed and saved
        cv_models = [int(x.split('_')[0]) for x in cv_filenames]

        for item_id in item_ids:
            processes_incomplete = []
            results_ls_incomplete = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # If the item_id exist but cv_results do not
                if item_id not in cv_models:
                    # Retrain and Re-Save to Disk
                    processes_incomplete.append(
                        executor.submit(self.get_performance, item_id,
                                        proj_id, modeltype))
            for task in as_completed(processes_incomplete):
                results_ls_incomplete.append(task.result())

        return perf_df

    def get_model(self, item_id, proj_id, modeltype='default'):
        '''This functions makes a call to fillet-functions
        to train a price-demand model for a single item, and saves
        the learned model to the project folder.
        '''

        with open(HOME + '/fillet_functions_api_endpoints.json') as f:
            fillet_func_urls = json.load(f)

        log.info(f'TRAINING ITEM_ID {item_id} MODEL')

        # Copy object data
        sales_data_wide_clean = self.data.copy()

        # Specify target column of interest
        target_colname = 'Qty_' + str(item_id)
        target_column = [target_colname]

        # If item_id target column isn't in the columns
        if target_colname not in sales_data_wide_clean.columns:
            print('Item Not Found in Dataset.')
            return None

        # Specify feature columns
        Price_columns = [
            col for col in sales_data_wide_clean.columns
            if col.startswith('Price')
        ]
        
        # Formalize Data Required for Cross Validation
        X = sales_data_wide_clean[Price_columns].copy()
        X = X.reindex(sorted(X.columns), axis=1)
        y = sales_data_wide_clean[target_column].copy()

        # Clear unused variables to save memory
        del sales_data_wide_clean
        gc.collect()

        payload = {
            'code': HOST_KEY,
        }

        # Create temp file to stage train input data
        temp_train_path = f'temp/train/{proj_id}/{item_id}'
        if not os.path.isdir(temp_train_path):
            Path(temp_train_path).mkdir(parents=True)

        # Send train requests to fillet-functions until successful
        while True:
            try:
                # Stage files in temp folder
                X.to_parquet(temp_train_path + '/X.parquet')
                y.to_parquet(temp_train_path + '/y.parquet')

                # Open staged files to send to fillet-functions
                files = {
                    'X_file': open(temp_train_path + '/X.parquet', 'rb'),
                    'y_file': open(temp_train_path + '/y.parquet', 'rb')
                }
                # Send request to fillet-functions
                url = fillet_func_urls[modeltype]['train']
                # url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_1_train'
                result = requests.get(url, params=payload, files=files)
                
                # Load returned pickles binary content
                model = p.loads(result.content)
                break
            except Exception as e:
                log.info(f'TRAIN {item_id} FAILED. RETRYING...')
                log.info(f'{e}')
                time.sleep(60)
                pass

        model = p.loads(result.content)

        # Clear unused variables to save memory
        del X
        del y
        gc.collect()

        # Save model python object to object attributes 
        self.models[item_id] = model

        # Create project model folder to save model pickle files
        MODEL_PATH = HOME + f'/projects/{proj_id}/models/'
        if not os.path.isdir(MODEL_PATH):
            Path(MODEL_PATH).mkdir(parents=True)

        # Save model to project model folder
        with open(MODEL_PATH + f'model_{item_id}.p', 'wb') as f:
            p.dump(model, f)

        # Close staged files
        files['X_file'].close()
        files['y_file'].close()

        return model

    def train_all_items(self, proj_id, retrain=True, modeltype='default'):
        '''This function loops though all items in a project,
        and trains and saves each model to disk via get_model.
        In order to speed up the processing, it makes up to 5
        parallel calls to fillet-functions via get_model.
        '''

        # Get item_ids of a project
        item_ids = [int(x.split('_')[1]) for x in self.price_columns]

        # Send parallel calls to fillet-functions via get_model
        processes = []
        results_ls = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for item_id in item_ids:
                processes.append(
                    executor.submit(self.get_model, item_id, proj_id))
        for task in as_completed(processes):
            results_ls.append(task.result())

        # Check That All Models are Trained, Else Retrain
        proj_path = HOME + f'/projects/{proj_id}/'
        model_filenames = os.listdir(proj_path + 'models')

        # Retry training until all models are trained
        finished = 0
        while finished == 0:
            model_filenames = os.listdir(proj_path + 'models')
            
            # Determine which CV results are completed and saved
            trained_models = [
                int(x.split('_')[1].split('.')[0]) for x in model_filenames
            ]

            for item_id in item_ids:
                processes_incomplete = []
                results_ls_incomplete = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    # If the item_id exists but model does not
                    if item_id not in trained_models:
                        # Retrain and Re-Save to Disk
                        processes_incomplete.append(
                            executor.submit(self.get_model, item_id, proj_id, modeltype))
                for task in as_completed(processes_incomplete):
                    results_ls_incomplete.append(task.result())

            # Determine which models are completed and saved
            model_filenames = os.listdir(proj_path + 'models')
            trained_models = [
                int(x.split('_')[1].split('.')[0]) for x in model_filenames
            ]

            # Exit retry loop when all items have trained models
            if len(trained_models) == len(item_ids):
                finished = 1

        log.info(f'TRAINING COMPLETED FOR {len(item_ids)} ITEMS.')

        # remove temp train folder
        try:
            shutil.rmtree('temp/train')
        except:
            pass