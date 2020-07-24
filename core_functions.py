import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats
import pickle as p
import cvxpy as cp

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

import sys, json, requests
import operator
import random
import base64
import zlib
import os
from pathlib import Path

import logging
log = logging.getLogger('fillet-flask.sub')

# < ---------------------- Configuration ---------------------->




master_data_input_path = 'data/data_all.csv'

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

def GeneticAlgorithm(prices_std_list, prices_mean_list, price_columns, rules, regressors, population, generation, 
                        costs=None, penalty_hard_constant=1000000, penalty_soft_constant=10000, step=0.05, 
                        random_seed=1):
    # 1. Preprocess rules and price limits
    num_item = len(price_columns)
    product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_columns)}
    rule_list_old, price_range = rules
    rule_list = [rule for rule in rule_list_old if set(rule['products']).issubset(set(product_to_idx.keys()))] # filter out the ones not in price_columns
    print('{} out of {} rules contain products not in price_columns.'.format(len(rule_list_old)-len(rule_list), len(rule_list_old)))
    hard_rule_eq_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == True)]
    hard_rule_ineq_list = [i for i in rule_list if (i['penalty'] == -1 and i['equality'] == False)]
    soft_rule_eq_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == True)]
    soft_rule_ineq_list = [i for i in rule_list if (i['penalty'] != -1 and i['equality'] == False)]
    price_range_dic = {}
    for item in price_range:
        price_range_dic[item['item_id']] = [item['max'], item['min']]
    # 2. Find valid price vectors to start
    # 2.1. put hard equalities into matrix form
    equal_list = hard_rule_eq_list
    matrix1 = np.zeros((len(equal_list), len(price_columns)))
    shifts1 = []
    for k in range(len(equal_list)):
        products = equal_list[k]['products']
        scales = equal_list[k]['scales']
        scales = [float(s) for s in scales]
        shift = equal_list[k]['shift']
        for j, product in enumerate(products):
            matrix1[k, product_to_idx[product]] = float(scales[j])
        shifts1.append(shift)
    shifts1 = np.array(shifts1).reshape(-1, 1)
    penalty1 = np.full((matrix1.shape[0],1), penalty_hard_constant)
    # 2.1. put hard inequalities into matrix form
    inequal_list = hard_rule_ineq_list
    matrix2 = np.zeros((len(inequal_list)+2*len(price_columns), len(price_columns)))
    shifts2 = []
    # 2.2.1. adding hard constraints
    for k in range(len(inequal_list)):
        products = inequal_list[k]['products']
        scales = inequal_list[k]['scales']
        scales = [float(s) for s in scales]
        shift = inequal_list[k]['shift']
        for j, product in enumerate(products):
            matrix2[k, product_to_idx[product]] = float(scales[j])
        shifts2.append(shift)
    # 2.2.2. adding price ranges
    prices = [i.split('_')[1] for i in price_columns]
    # 2.2.2.1 adding price floor
    for i, product in enumerate(prices):
        matrix2[len(inequal_list)+i, i] = 1.
        if int(product) not in price_range_dic.keys():
            print('product {} is not given price range, assumed to be within [0, 20].'.format(product))
            shifts2.append(0.0)
        else:
            shifts2.append(price_range_dic[int(product)][1])
    # 2.2.2.1 adding price cap
    for i, product in enumerate(prices):
        matrix2[len(inequal_list)+len(price_columns)+i, i] = -1.
        if int(product) not in price_range_dic.keys():
            shifts2.append(-20.0)
        else:
            shifts2.append(-price_range_dic[int(product)][0])
    shifts2 = np.array(shifts2).reshape(-1, 1)
    penalty2 = np.full((matrix2.shape[0],1), penalty_hard_constant)
    # 2.4. get 2 valid individuals
    val_ind1, status1 = solve_cvx(matrix1, shifts1, matrix2, shifts2, 'sum')
    val_ind2, status2 = solve_cvx(matrix1, shifts1, matrix2, shifts2, 'sum_squares')
    print('val_ind1 shape: {}'.format(val_ind1.shape))
    print('val_ind2 shape: {}'.format(val_ind2.shape))
    print('status 1: {}'.format(status1))
    print('status 2: {}'.format(status2))
    assert status1 != 'infeasible' and status2 != 'infeasible', 'Hard constraints must have feasible region.'
    # 3. Run GA using DEAP library
    # 3.1. Define fitness function
    def evalObjective(individual):
        """
        returns:
        (revenue, penalty_): revenue of this individual and penalty from it violating the constraints
        """
        # Calculating revenue
        quantity = np.zeros((individual.shape[0]))
        product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_columns)}
        individual = individual.round(2)
        for code in regressors: # TODO: use multiple workers here to speedup the optimization process
            quantity[product_to_idx[code]] = regressors[code].predict(pd.DataFrame(individual.reshape(1, -1), columns=price_columns))
        output = individual.dot(quantity)
        # Calculating constraint violation penalty
        temp1 = (matrix1.dot(individual.reshape(-1, 1)) - shifts1).round(2)
        mask1 = temp1 != 0
        penalty_1 = mask1.T.dot(penalty1)
        temp2 = (matrix2.dot(individual.reshape(-1, 1)) - shifts2).round(2)
        mask2 = temp2 < 0
        penalty_2 = mask2.T.dot(penalty2)
        return (output - penalty_1[0,0] - penalty_2[0,0],)
    # 3.2. Initialize individuals and operations
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
    # 3.3. Run the algoritm
    random.seed(64)
    pop = toolbox.population(n=population)
    pop.append(creator.Individual(val_ind1.round(2).flatten()))
    pop.append(creator.Individual(val_ind2.round(2).flatten()))
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
    return pop, stats, hof


    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
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
		
		# print("{}th generation" .format(self.iterated))
		# print("best solution so far: {}".format(highest_fitness))
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
		target_column = [target_colname]

		Week = sales_data_wide_clean[['Wk']].copy()
		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		payload = {
		'code':HOST_KEY,
		}

		if not os.path.isdir('temp'):
			Path('temp').mkdir(parents=True)

		X.to_parquet('temp/X.parquet')
		y.to_parquet('temp/y.parquet')
		Week.to_parquet('temp/Wk.parquet')

		files = {'X_file': open('temp/X.parquet', 'rb'),
				 'y_file': open('temp/y.parquet', 'rb'),
				 'Wk_file': open('temp/Wk.parquet', 'rb'),
				 }

		# data = {
		# 	'X':X.to_json(),
		# 	'y':y.to_json(),
		# 	'Week':Week.to_json()
		# }

		url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_2_cv'
		# url = 'http://localhost:7071/api/fillet_func_2_cv'
		result = requests.get(
			url, params=payload,
			files=files
			# data=base64.b64encode(zlib.compress(json.dumps(data).encode('utf-8')))
			)
		outp = result.json()
		outp['item_id'] = int(item_id)

		files['X_file'].close()
		files['y_file'].close()
		files['Wk_file'].close()

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

	def get_model(self, item_id, proj_id):

		log.info(f'TRAINING ITEM_ID {item_id} MODEL')

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
		target_column = [target_colname]

		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		payload = {
		'code':HOST_KEY,
		}

		if not os.path.isdir('temp'):
			Path('temp').mkdir(parents=True)

		X.to_parquet('temp/X.parquet')
		y.to_parquet('temp/y.parquet')

		files = {'X_file': open('temp/X.parquet', 'rb'),
				 'y_file': open('temp/y.parquet', 'rb')}

		# data = {
		# 		'X':X.to_json(),
		# 		'y':y.to_json()
		# 	}

		url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_1_train'
		# url = 'http://localhost:7071/api/fillet_func_1_train'
		
		result = requests.get(url, params=payload,
			files=files
			)
		# result = requests.get(url, params=payload,
		# 	data=zlib.compress(json.dumps(data).encode('utf-8'))
		# 	)
		model_json = result.json()['model_json']  







		# model = XGBRegressor()
		# model.fit(X, y)
		
		self.models[item_id] = model_json

		MODEL_PATH = f'projects/{proj_id}/models/'
		if not os.path.isdir(MODEL_PATH):
			Path(MODEL_PATH).mkdir(parents=True)

		with open(MODEL_PATH+f'model_{item_id}.json','w') as f:
			json.dump(model_json,f)

		files['X_file'].close()
		files['y_file'].close()

		return model_json
	
	def train_all_items(self, proj_id, retrain=True):
		item_ids = [int(x.split('_')[1]) for x in self.price_columns]
		for item_id in item_ids:
			if retrain==False:
				if item_id not in self.models.keys():
					self.get_model(item_id,proj_id)
			else:
				self.get_model(item_id,proj_id)