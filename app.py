from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import optimize_floats, optimize_memory, get_top_features
from core_functions import rms_pricing_model, list_to_matrix, solve_cvx, GeneticAlgorithm
import numpy as np
import pickle as p

import json
import pandas as pd
import os
import requests
from pathlib import Path
import gc
import shutil
import zlib
# from xgboost import XGBRegressor
import itertools
import cvxpy as cp

import logging

app = Flask('fillet-flask')
logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s | %(levelname)-8s | %(name)-25s | %(threadName)-16s : %(message).80s'
)


# Set current working directory
HOME = os.environ['HOME_SITE']
# HOME = ''

# Function Key required to call fillet-functions
# with open('keys.json') as f:
#       HOST_KEY = json.load(f)['host_key']
HOST_KEY = os.environ['FUNCTIONS_KEY']


@app.route("/")
def hello():
    return "fillet-flask is alive."




@app.route('/train/', methods=['POST'])
def train():
    '''This function receives a dataset, a new project id,
    and a training algo name. A project is started, and
    a request is made to fillet-functions for a price->demand
    model to be trained for each item_id in the dataset. The 
    model is returned as .pkl and saved to disk in the project
    folder.
    '''

    # Retrieve request details
    app.logger.info('REQUEST DETAILS RETREIVAL')
    cv_acc = request.form['cv_acc']
    project_id = request.form['project_id']
    modeltype = request.form['modeltype']

    # Log Receive Train Request
    app.logger.info(f'TRAIN REQUEST RECEIVED PROJ_ID: {project_id}')

    # Load Parquet in Memory
    app.logger.info('ATTEMPTING DATA RETREIVAL')
    
    try:
        data_file = request.files['data']
    except Exception as e:
        app.logger.info('ERROR IN DATA RETREIVAL')
        app.logger.info(e)

    # Create temp staging folder
    temp_data_path = f'temp/staging/{project_id}'
    if not os.path.isdir(temp_data_path):
        Path(temp_data_path).mkdir(parents=True)
    
    # Save Parquet Binary to temp disk
    data_file.save(temp_data_path + '/data_staging.parquet')

    # Read Parquet into memory as DataFrame
    data_df = pd.read_parquet(temp_data_path + '/data_staging.parquet')

    # Log data load success.
    app.logger.info('DATA SUCCESSFULLY LOADED')

    # Delete content from temp folder
    try:
        shutil.rmtree(temp_data_path)
    except:
        pass

    # Initialize rms_pricing_model Object with data
    pdm = rms_pricing_model(data_df)

    # Clear unused variables to free up memory
    del data_file
    del data_df
    gc.collect()

    # Create project folder
    price_info_path = HOME + f'/projects/{project_id}/'
    if not os.path.isdir(price_info_path):
        Path(price_info_path).mkdir(parents=True)

    # Save price_info as pickle to project folder
    pdm.get_and_save_price_info(price_info_path + 'price_info.pkl')

    # Log successful save of price info
    app.logger.info('PRICE INFO SAVED')

    # Get all item_ids from price columns in dataset
    item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]

    # Save project properties to disk, for future referencing
    proj_properties = {'num_items': len(item_ids), 'items': item_ids}
    with open(price_info_path + 'proj_properties.json', 'w') as outfile:
        json.dump(proj_properties, outfile)

    # Trigger Train models on fillet-functions and save to disk
    pdm.train_all_items(proj_id=project_id, retrain=True, modeltype=modeltype)

    # Prepare output
    response_outp = {'result': 0, 'cv_acc': 0}
    response_outp['result'] = 'Success'


    # Get Model Feature Importances
    app.logger.info('GETTING FEATURE IMPORTANCES')
    importances_dict = {}
    for item in item_ids:
        item_model = pdm.models[item]
        item_imp_df = get_top_features(item_model, n=10)
        importances_dict[item] = item_imp_df.to_dict()

    # Save Model Importances to Disk
    with open(price_info_path + 'proj_fimportance.json', 'w') as outfile:
            json.dump(importances_dict, outfile)

    # Get Elasticities
    app.logger.info('GETTING ELASTICITY ESTIMATES')

    # Get Baseline Prices
    with open(price_info_path+'price_info.pkl', 'rb') as f:
        price_info = p.load(f)
        mean_prices = price_info[1]
        # Round to nearest 5 cent
        mean_prices = (mean_prices * 20).round() / 20
        mean_prices = pd.DataFrame(mean_prices).transpose()

    # Get Baseline Quantities
    baseline_quantities_dict = {}

    for item_id in item_ids:
        item_model = pdm.models[item_id]
        mean_prices = mean_prices[item_model._Booster.feature_names]
        pred = item_model.predict(mean_prices).round()
        baseline_quantities_dict[f'Qty_{item_id}'] = pred[0]

    elasticity_dict = {}

    # Get % Changes
    for item in item_ids:
        # If Prices go up by 10 cents...
        item_price_increase = mean_prices.copy()
        item_price_increase.loc[0,f'Price_{item}'] += 0.10
        # How much do quantities change?
        increase_quantities_dict = {}
        for item_j_id in item_ids:
            item_j_model = pdm.models[item_j_id]
            mean_prices = mean_prices[item_j_model._Booster.feature_names]
            pred = item_j_model.predict(item_price_increase).round()
            increase_quantities_dict[f'Qty_{item_j_id}'] = pred[0]
            
        # If Prices go up by 10 cents...
        item_price_decrease = mean_prices.copy()
        item_price_decrease.loc[0,f'Price_{item}'] -= 0.10
        # How much do quantities change?
        decrease_quantities_dict = {}
        for item_j_id in item_ids:
            item_j_model = pdm.models[item_j_id]
            mean_prices = mean_prices[item_j_model._Booster.feature_names]
            pred = item_j_model.predict(item_price_decrease).round()
            decrease_quantities_dict[f'Qty_{item_j_id}'] = pred[0]
            
        adjusted_quantities_df = pd.DataFrame([baseline_quantities_dict,
                                               increase_quantities_dict,
                                               decrease_quantities_dict
                                              ],
                                              index=['baseline_qty','increase_qty','decrease_qty']
                                             ).transpose()

        adjusted_quantities_df['increase_pct_chng'] = (adjusted_quantities_df['increase_qty'] - adjusted_quantities_df['baseline_qty']) / adjusted_quantities_df['baseline_qty']
        adjusted_quantities_df['decrease_pct_chng'] = (adjusted_quantities_df['decrease_qty'] - adjusted_quantities_df['baseline_qty']) / adjusted_quantities_df['baseline_qty']
        
        elasticity_dict[item] = adjusted_quantities_df.to_dict()

    # Save Model Importances to Disk
    with open(price_info_path + 'proj_elasticity_estimates.json', 'w') as outfile:
            json.dump(elasticity_dict, outfile)

    # User specified CV option
    if cv_acc == 'True':
        # Log CV Request
        app.logger.info('RUNNING OPTIONAL CROSS VALIDATION')
        # Trigger CV requests on fillet-functions and save to disk
        perf_df = pdm.get_all_performance(proj_id=project_id, modeltype=modeltype)
        # If train + cv completes within timeout, return cv results
        response_outp['cv_acc'] = perf_df.to_dict()
        # Save CV results to project folder
        with open(price_info_path + 'proj_cv_perf.json', 'w') as outfile:
            json.dump(perf_df.to_json(), outfile)

        # Log Successful CV
        app.logger.info(f'CROSS VALIDATION DONE')


    return jsonify(response_outp)


@app.route('/predict/', methods=['POST'])
def predict():
    '''This function receives a project id and set of n prices,
    then loads the n models in the existing project id and 
    uploads the models to fillet-function to get sales quantity
    estimates for the given set of prices.
    '''

    with open(HOME + '/fillet_functions_api_endpoints.json') as f:
        fillet_func_urls = json.load(f)

    

    # Retrieve request details
    prices = request.json['prices']
    project_id = request.json['project_id']

    app.logger.info(f'PREDICT REQUEST RECEIVED PROJECT {project_id}')

    # If the project exists, get its list of item_ids
    proj_path = HOME + f'/projects/{project_id}/'
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    # Otherwise, return error reponse
    except:
        app.logger.info('PROJECT {project_id} NOT FOUND')
        return jsonify({'error': 'project not found'})

    # "Price_" prefix is added to match feature/column names
    for k in list(prices.keys()):
        prices['Price_' + str(k)] = prices.pop(k)

    # Prices input converted to DataFrame, sorted, saved to json
    prices = pd.DataFrame(prices, index=[0])
    prices = prices.reindex(sorted(prices.columns), axis=1)
    prices_json = prices.to_json()

    # items iterator to retrieve saved models one by one
    items = [int(x.split('_')[1]) for x in prices.columns]

    # Open item_models and save them to list
    models_list = []
    for item in items:
        with open(HOME + f'/projects/{project_id}/models/model_{item}.p',
                  'rb') as f:
            model = p.load(f)
        models_list.append(model)

    pred_quantities_dict = {}

    app.logger.info('ATTEMPT LOCAL PREDICT REQUEST')

    for item in items:
        with open(HOME + f'/projects/{project_id}/models/model_{item}.p',
                  'rb') as f:
            model = p.load(f)
        prices = prices[model._Booster.feature_names]
        pred = model.predict(data=prices)
        pred_value = int(pred[0])
        pred_quantities_dict[f'Qty_{item}'] = pred_value

    app.logger.info('LOCAL PREDICT SUCCESS')
    app.logger.info('RETURNING RESPONSE')
    # Send qty_estimate in reponse
    response_outp = {'qty_estimates': pred_quantities_dict}
    return jsonify(response_outp)


@app.route('/query_progress/', methods=['POST'])
def query_progress():
    '''This function takes in a project_id and checks if there is
    an existing project_id. If there is a project previously
    created by a train request, responds with training progress.
    Otherwise returns error to prompt user to resend train request.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    app.logger.info(f'TRAINING PROGRESS QUERY FOR PROJ {project_id}')
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])
    except:
        return jsonify({'error': 'project not found'})

    # Project exists, attempt to locate existing trained models
    try:
        model_filenames = os.listdir(proj_path + 'models')
    except:
        return jsonify({'pct_complete': 0})

    # Trained models exist, determine how many are left to train.
    trained_models = set(
        [int(x.split('_')[1].split('.')[0]) for x in model_filenames])
    remaining_models = project_items - trained_models

    # Attempt to locate and count the number of cv results completed
    cv_path = HOME + f'/projects/{project_id}/cv'
    try:
        cv_dir = os.listdir(cv_path)
        cv_progress = len(cv_dir)
    except:
        cv_progress = 0

    # If CV is 100% complete, 'proj_cv_perf.json' also exists.
    cv_done = 0
    proj_dir = os.listdir(proj_path)
    if 'proj_cv_perf.json' in proj_dir:
        cv_done = 1
 
    # If Feature Importances are Complete 'proj_fimporance.json' also exists
    fi_done = 0
    if 'proj_fimportance.json' in proj_dir:
        fi_done = 1

    # If Elasticity Estimates are Complete 'proj_elasticity_estimates.json' also exists
    ee_done = 0
    if 'proj_elasticity_estimates.json' in proj_dir:
        ee_done = 1

    # Format response
    response_outp = {
        'pct_complete': round(
            (len(trained_models) / len(project_items)), 3) * 100,
        'project_items': list(project_items),
        'train_complete': list(trained_models),
        'cv_done': cv_done,
        'cv_progress': round(cv_progress / len(project_items), 3) * 100,
        'fi_done': fi_done,
        'ee_done': ee_done
    }
    return jsonify(response_outp)

@app.route('/batch_query_progress/', methods=['POST'])
def batch_query_progress():
    '''This function takes in a list of project_ids and checks if there is
    an existing project_id. If there is a project previously
    created by a train request, responds with training progress.
    Otherwise returns error to prompt user to resend train request.
    '''

    # Get request details
    project_id_ls = request.get_json()['project_id_ls']
    batch_outp = {}
    for project_id in project_id_ls:
        app.logger.info(f'TRAINING PROGRESS QUERY FOR PROJ {project_id}')
        
        # Attempt to locate and load in project from project_id
        proj_path = HOME + f'/projects/{project_id}/'
        try:
            with open(proj_path + 'proj_properties.json') as json_file:
                proj_properties = json.load(json_file)
            project_items = set(proj_properties['items'])
        except:
            batch_outp[project_id] = 'project not found'
            continue

        # Project exists, attempt to locate existing trained models
        try:
            model_filenames = os.listdir(proj_path + 'models')
        except:
            batch_outp[project_id] = 'training not started'
            continue

        # Trained models exist, determine how many are left to train.
        trained_models = set(
            [int(x.split('_')[1].split('.')[0]) for x in model_filenames])
        remaining_models = project_items - trained_models

        # Attempt to locate and count the number of cv results completed
        cv_path = HOME + f'/projects/{project_id}/cv'
        try:
            cv_dir = os.listdir(cv_path)
            cv_progress = len(cv_dir)
        except:
            cv_progress = 0

        # If CV is 100% complete, 'proj_cv_perf.json' also exists.
        cv_done = 0
        proj_dir = os.listdir(proj_path)
        if 'proj_cv_perf.json' in proj_dir:
            cv_done = 1

        # If Feature Importances are Complete 'proj_fimporance.json' also exists
        fi_done = 0
        if 'proj_fimportance.json' in proj_dir:
            fi_done = 1

        # If Elasticity Estimates are Complete 'proj_elasticity_estimates.json' also exists
        ee_done = 0
        if 'proj_elasticity_estimates.json' in proj_dir:
            ee_done = 1

        # Format response
        response_outp = {
            'pct_complete': round(
                (len(trained_models) / len(project_items)), 3) * 100,
            'project_items': list(project_items),
            'train_complete': list(trained_models),
            'cv_done': cv_done,
            'cv_progress': round(cv_progress / len(project_items), 3) * 100,
            'fi_done': fi_done,
            'ee_done': ee_done
        }
        batch_outp[project_id] = response_outp
    return jsonify(batch_outp)


@app.route('/get_cv_results/', methods=['POST'])
def get_cv_results():
    '''This function attempts to locate and return all
    completed cv results for a given project_id.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'
    try:

        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    except:
        return jsonify({'error': 'project not found'})

    # Attempt to locate and load in cv results
    proj_dir = os.listdir(proj_path)
    if 'proj_cv_perf.json' in proj_dir:
        with open(proj_path + 'proj_cv_perf.json') as json_file:
            cv_results = json.load(json_file)
        return jsonify(cv_results)

    # Check if any CV results have been saved so far
    try:
        num_cv_done = len(os.listdir(proj_path + 'cv'))
    except:
        return jsonify({'status': 'incomplete'})

    # If at least one CV results is available so far
    if num_cv_done > 0:
        # Create placeholder DataFrame
        perf_df = pd.DataFrame(columns=[
            'item_id', 'avg_sales', 'r2_score', 'mae_score', 'mpe_score',
            'rmse_score'
        ])
        # Append all found cv results to DataFrame
        for cv_json_filename in os.listdir(proj_path + 'cv'):
            with open(proj_path + 'cv/' + cv_json_filename) as json_file:
                perf_df = perf_df.append(json.load(json_file),
                                         ignore_index=True)
        # Return filled DataFrame of available CV results
        return jsonify(perf_df.to_json())

    else:
        return jsonify({'status': 'incomplete'})


@app.route('/get_feature_importances/', methods=['POST'])
def get_feature_importances():
    '''This function returns feature importances for all 
    item models in a project.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'

    # If the project exists, get its list of item_ids
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    # Otherwise, return error reponse
    except:
        return jsonify({'error': 'project not found'})

    # Check if Feature Importances Complete
    proj_dir = os.listdir(proj_path)
    if 'proj_fimportance.json' not in proj_dir:
        return jsonify({'error': 'feature importances not complete.'})

    else:
        with open(proj_path + 'proj_fimportance.json') as json_file:
            f_importances = json.load(json_file)

    return f_importances

@app.route('/get_elasticity_estimates/', methods=['POST'])
def get_elasticity_estimates():
    '''This function returns elasticity estimates for all 
    item models in a project.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'

    # If the project exists, get its list of item_ids
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    # Otherwise, return error reponse
    except:
        return jsonify({'error': 'project not found'})

    # Check if Feature Importances Complete
    proj_dir = os.listdir(proj_path)
    if 'proj_elasticity_estimates.json' not in proj_dir:
        return jsonify({'error': 'elasticity estimates not complete.'})

    else:
        with open(proj_path + 'proj_elasticity_estimates.json') as json_file:
            elasticity_estimates = json.load(json_file)

    return elasticity_estimates



@app.route('/detect_conflict/', methods=['POST'])
def detect_conflict():
    # 1. get data
    app.logger.info('DETECT CONFLICT REQUEST RECEIVED')
    constraints = request.get_json()['constraints']
    rule_list = constraints[0]
    hard_rule_list = [i for i in rule_list if i['penalty'] == -1]
    if len(hard_rule_list) == 0 and len(constraints[1]) == 0:
        return jsonify({'conflict':'No conflict.'})
    hard_rule_eq_list = [i for i in hard_rule_list if i['equality'] == 0]
    hard_rule_small_list = [i for i in hard_rule_list if i['equality'] == 1]
    hard_rule_large_list = [i for i in hard_rule_list if i['equality'] == 2]
    hard_rule_smalleq_list = [i for i in hard_rule_list if i['equality'] == 3]
    hard_rule_largeeq_list = [i for i in hard_rule_list if i['equality'] == 4]
    price_range = constraints[1]
    price_range_dic = {}
    for item in price_range:
        price_range_dic[item['item_id']] = [item['max'], item['min']]
    product_list_hard_rule = list(set(list(itertools.chain.from_iterable([rule['products'] for rule in hard_rule_list]))))
    product_list_price_range = list(price_range_dic.keys())
    product_list_price_range = [str(i) for i in product_list_price_range]
    product_list = list(set(product_list_hard_rule + product_list_price_range))
    product_to_idx = {column: i for i, column in enumerate(product_list)}
    # 2. Find valid price vectors to start
    # 2.1. put hard equalities into matrix form
    matrix1, shifts1, _ = list_to_matrix(hard_rule_eq_list, product_to_idx, 10)
    # 2.2. put hard inequalities into matrix form
    matrix2_1, shifts2_1, _ = list_to_matrix(hard_rule_small_list, product_to_idx, 10)
    matrix2_1 = matrix2_1*(-1)
    shifts2_1 = shifts2_1*(-1)+0.05
    matrix2_2, shifts2_2, _ = list_to_matrix(hard_rule_large_list, product_to_idx, 10)
    shifts2_2 = shifts2_2+0.05
    matrix2_3, shifts2_3, _ = list_to_matrix(hard_rule_smalleq_list, product_to_idx, 10)
    matrix2_3 = matrix2_3*(-1)
    shifts2_3 = shifts2_3*(-1)
    matrix2_4, shifts2_4, _ = list_to_matrix(hard_rule_largeeq_list, product_to_idx, 10)
    # 2.2.2. adding price ranges
    prices = product_list
    default_cap = 7.3 + 3.5 # mean + std
    default_floor = 7.3 - 3.5 # mean - std
    # 2.2.2.1 adding price floor
    matrix2_5 = np.zeros((2*len(prices), len(prices)))
    shifts2_5 = np.zeros((2*len(prices), 1))
    for i, product in enumerate(prices):
        matrix2_5[i, i] = 1.
        if int(product) not in price_range_dic.keys():
            print('product {} is not given price range, assumed to be within [{}, {}].'.format(product, default_floor, default_cap))
            shifts2_5[i,0] = default_floor
        else:
            shifts2_5[i,0] = price_range_dic[int(product)][1]
    # 2.2.2.1 adding price cap
    for i, product in enumerate(prices):
        matrix2_5[len(prices)+i, i] = -1.
        if int(product) not in price_range_dic.keys():
            shifts2_5[len(prices)+i,0] = -default_cap
        else:
            shifts2_5[len(prices)+i,0] = -price_range_dic[int(product)][0]
    # 2.2.3. Put together hard inequality and price range
    matrix2 = np.vstack([matrix2_1, matrix2_2, matrix2_3, matrix2_4, matrix2_5])
    shifts2 = np.vstack([shifts2_1, shifts2_2, shifts2_3, shifts2_4, shifts2_5])
    # 2.3. get 2 valid individuals from linear programming
    val_ind2, status2 = solve_cvx(matrix1, shifts1, matrix2, shifts2, 'sum_squares')
    # return the result
    if status2 == 'infeasible':
        return jsonify({'conflict':'Conflict exists'})
    else:
        return jsonify({'conflict':'No conflict'})



@app.route('/optimize/', methods=['POST'])
def optimize():
    app.logger.info('OPTIMIZE REQUEST RECEIVED')
    # get input
    project_id = request.get_json()['project_id']
    opti_id = request.get_json()['optimisation_id']
    constraints = request.get_json()['constraints']
    population =  request.get_json()['population']
    max_epoch = request.get_json()['max_epoch']

    app.logger.info(f'OPTIMIZE REQUEST RECEIVED PROJ {project_id}')

    price_info_path = HOME + f'/projects/{project_id}/price_info.pkl'
    model_path = HOME + f'/projects/{project_id}/models/'
    opti_path = HOME + f'/projects/{project_id}/{opti_id}/'

    if not os.path.isdir(opti_path):
        Path(opti_path).mkdir(parents=True)

    # load price information
    assert os.path.isfile(price_info_path), 'No price info file found.'
    price_std, price_mean, price_names = p.load(open(price_info_path, 'rb'))
    product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_names)}
    # load model
    assert os.path.isdir(model_path), 'No model directory found.'
    onlyfiles = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    onlyfiles = [f for f in onlyfiles if f.startswith('model_')]
    regressors = {}
    for file in onlyfiles:
        name = file.strip().split('.')[0].split('_')[1]
        regressors[name] = p.load(open(model_path + file, 'rb'))
    # run optimization
    app.logger.info(f'RUNNING OPTIMIZATION PROJ {project_id}')
    result = GeneticAlgorithm(price_std, price_mean, price_names, constraints, regressors, population, max_epoch)
                        # costs=None, penalty_hard_constant=1000000, penalty_soft_constant=100000, step=0.05, 
                        # random_seed=1
    response_outp = {}
    if result[0] == True:
        app.logger.info(f'OPTIMIZE REQUEST SUCCESS PROJ {project_id}')
        _, pop, stats, hof, report = result
        f = lambda x: 0.05 * np.round(x/0.05)
        best_ind = f(hof[0]).round(2)
    # Send result as Dict to avoid confusion
        response_outp['success'] = True
        response_outp['result'] = np.array(best_ind).tolist()
        response_outp['report'] = [float(i) for i in report]
        response_outp['report_info'] = ['estimated profit of the optimized price', # is None if cost is not provided completely for the products
                                        'estimated revenue of the optimized price', 
                                       'number of hard constraints (including price ranges) violated',
                                       'number of soft constraints (preferences) violated']
        response_outp['price_cols'] = price_names

        opti_results_path = opti_path + 'optimize_results.json'
        with open(opti_results_path, 'w') as outfile:
            json.dump(response_outp, outfile)

    else:
        app.logger.info(f'OPTIMIZE REQUEST FAILED PROJ {project_id}')
        if result[1] == 0:
            response_outp = {
                'success': False,
                'type': 0,
                'info':'Error: hard constraints and price ranges do not have a feasible region.'}
        elif result[1] == 1:
            response_outp = {
                'success': False,
                'type': 1,
                'info': 'Error: cost list is empty for profit calculation.'}
        elif result[1] == 2:
            response_outp = {
                'success': False,
                'type': 2,
                'info': 'Error: cost is missing for items {} for profit calculation.'.format(result[2])}

        opti_results_path = opti_path + 'optimize_results.json'
        with open(opti_results_path, 'w') as outfile:
            json.dump(response_outp, outfile)

    return jsonify(response_outp)

@app.route('/get_opti_results/', methods=['POST'])
def get_opti_results():
    '''This function attempts to locate and return all
    completed cv results for a given project_id.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    opti_id = request.get_json()['optimisation_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'
    try:

        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    except:
        return jsonify({'error': 'project not found'})

    opti_path = proj_path+f'{opti_id}/'

    # Attempt to locate and load in cv results
    try:
        opti_dir = os.listdir(opti_path)
    except:
        return jsonify({'error': 'optimisation_id not found'})
    
    if 'optimize_results.json' in opti_dir:
        with open(opti_path + 'optimize_results.json') as json_file:
            opti_results = json.load(json_file)
        return jsonify(opti_results)

    else:
        return jsonify({'status': 'incomplete'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')